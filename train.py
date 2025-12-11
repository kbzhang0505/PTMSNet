## Rethinking progressive low-light image enhancement: A frequency-aware tripartite multi-scale network
## Yingjian Li, Kaibing Zhang, Xuan Zhou, Zhouqiang Zhang, Sheng Hu
## https://doi.org/10.1016/j.neunet.2025.108351
import math
import os
import torch
import yaml
import torch.nn.functional as F

from utils import network_parameters, losses
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import numpy as np
import random
from transform.data_RGB import get_training_data, get_validation_data2
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import utils.losses
from model.PTMSNet import PTMSNet as PTMSNet
import argparse
from collections import OrderedDict

parser = argparse.ArgumentParser(description='Hyper-parameters for LLFormer')
parser.add_argument('-yml_path', default="configs/LOL/train/training_LOL.yaml", type=str)
parser.add_argument('--weights',
                    default='',
                    type=str,
                    help='Path to weights')
args = parser.parse_args()

## Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(2025)
np.random.seed(2025)
torch.manual_seed(2025)
torch.cuda.manual_seed_all(2025)


class Gradient_Net(nn.Module):
    def __init__(self, device):
        super(Gradient_Net, self).__init__()
        kernel_x = [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0).to(device)

        kernel_y = [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0).to(device)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        x = (0.3 * x[:, 0, :, :] + 0.59 * x[:, 1, :, :] + 0.11 * x[:, 2, :, :]).view(b, 1, h, w)
        grad_x = F.conv2d(x, self.weight_x)
        grad_y = F.conv2d(x, self.weight_y)
        gradient = torch.abs(grad_x) + torch.abs(grad_y)
        return gradient * 0.5


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()

    def forward(self, predict, target):
        b, c, h, w = target.shape
        target_view = target.view(b, c, h * w).permute(0, 2, 1)
        predict_view = predict.view(b, c, h * w).permute(0, 2, 1)
        target_norm = torch.nn.functional.normalize(target_view, dim=-1)
        predict_norm = torch.nn.functional.normalize(predict_view, dim=-1)
        cose_value = target_norm * predict_norm
        cose_value = torch.sum(cose_value, dim=-1)
        color_loss = torch.mean(1 - cose_value)

        return color_loss


class GradientLoss(nn.Module):
    def __init__(self, device):
        super(GradientLoss, self).__init__()
        self.loss = nn.L1Loss()
        self.gradient = Gradient_Net(device)

    def compute_gradient(self, img):
        gradimg = self.gradient(img)
        return gradimg

    def forward(self, predict, target):
        predict_grad = self.compute_gradient(predict)
        target_grad = self.compute_gradient(target)

        return self.loss(predict_grad, target_grad)


def Loss(predict, target, a, b, c):
    Charloss = losses.CharbonnierLoss()
    Colorloss = ColorLoss()
    SSIMloss = losses.SSIMLoss()
    L2 = nn.MSELoss()
    l1_loss = a * Charloss(predict, target)
    color_loss = b * Colorloss(predict, target)
    ssim_loss = c * SSIMloss(predict, target)
    sum_loss = l1_loss + L2(predict, target) + color_loss + ssim_loss
    if (math.isnan(l1_loss) or math.isnan(color_loss) or math.isnan(sum_loss)):
        print('error!!!')
    return sum_loss, l1_loss, color_loss, ssim_loss


def load_checkpoint(model, weights):
    checkpoint = torch.load(weights)
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)



def train():
    ## Load yaml configuration file
    yaml_file = args.yml_path

    with open(yaml_file, 'r') as config:
        opt = yaml.safe_load(config)
    print("load training yaml file: %s" % (yaml_file))

    Train = opt['TRAINING']
    OPT = opt['OPTIM']

    ## Build Model
    print('==> Build the model')
    model_restored = PTMSNet(inp_channels=3, out_channels=3, dim=16, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
                                  bias=False, LayerNorm_type='WithBias', attention=True)
    p_number = network_parameters(model_restored)
    model_restored.cuda()
    if args.weights != "":
        load_checkpoint(model_restored, args.weights)

    ## Training model path direction
    mode = opt['MODEL']['MODE']

    model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')
    utils.mkdir(model_dir)
    train_dir = Train['TRAIN_DIR']
    val_dir = Train['VAL_DIR']
    ## GPU
    gpus = ','.join([str(i) for i in opt['GPU']])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
    if len(device_ids) > 1:
        model_restored = nn.DataParallel(model_restored, device_ids=device_ids)

    ## Optimizer
    start_epoch = 1
    new_lr = float(OPT['LR_INITIAL'])
    optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    ## Scheduler (Strategy)
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                            eta_min=float(OPT['LR_MIN']))
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs,
                                       after_scheduler=scheduler_cosine)
    scheduler.step()

    ## Resume (Continue training by a pretrained model)
    if Train['RESUME']:
        path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
        utils.load_checkpoint(model_restored, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------')

    ## Loss

    ## DataLoaders
    print('==> Loading datasets')
    train_dataset = get_training_data(train_dir, {'patch_size': Train['TRAIN_PS']}, cutoff_freq=Train['cutoff_freq'])
    train_loader = MultiEpochsDataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                              shuffle=True, num_workers=2, drop_last=False)
    val_dataset = get_validation_data2(val_dir, {'patch_size': Train['VAL_PS']}, cutoff_freq=Train['cutoff_freq'])
    val_loader = MultiEpochsDataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=1,
                            drop_last=False)

    # Show the training configuration
    print(f'''==> Training details:
    ------------------------------------------------------------------
        Restoration mode:   {mode}
        Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
        Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
        Model parameters:   {p_number}
        Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
        Batch sizes:        {OPT['BATCH']}
        Learning rate:      {OPT['LR_INITIAL']}
        GPU:                {'GPU' + str(device_ids)}''')
    print('------------------------------------------------------------------')

    # Start training!
    print('==> Training start: ')
    best_psnr = 0
    best_ssim = 0
    best_lpips = 1
    best_epoch_psnr = 0
    best_epoch_ssim = 0
    best_epoch_lpips = 0
    total_start_time = time.time()

    ## Log
    log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
    utils.mkdir(log_dir)
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

    for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_l1_loss = 0
        epoch_color_loss = 0
        epoch_grad_loss = 0

        model_restored.train()
        for i, data in enumerate(tqdm(train_loader), 0):
            # Forward propagation
            for param in model_restored.parameters():
                param.grad = None
            target = data[0].cuda()
            input_ = data[1].cuda()
            SNR_map = data[3].cuda()
            restored = model_restored([input_, SNR_map])

            # Compute loss
            loss, l1_loss, color_loss, grad_loss = Loss(restored, target, 0.9, 0.6, 0.5)

            # Back propagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()
            epoch_l1_loss += l1_loss.item()
            epoch_color_loss += color_loss.item()
            epoch_grad_loss += grad_loss.item()

        ## Evaluation (Validation)
        if epoch % Train['VAL_AFTER_EVERY'] == 0:
            torch.cuda.empty_cache()
            model_restored.eval()
            psnr_val_rgb = []
            ssim_val_rgb = []
            for ii, data_val in enumerate(val_loader, 0):
                # torch.cuda.empty_cache()
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()
                SNR_map = data_val[3].cuda()
                h, w = target.shape[2], target.shape[3]
                with torch.no_grad():
                    restored = model_restored([input_, SNR_map])
                    restored = restored[:, :, :h, :w]

                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))
                    ssim_val_rgb.append(utils.torchSSIM(restored, target))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
            ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

            # Save the best PSNR model of validation
            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch_psnr = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restored.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_bestPSNR.pth"))
            print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
                epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

            # Save the best SSIM model of validation
            if ssim_val_rgb > best_ssim:
                best_ssim = ssim_val_rgb
                best_epoch_ssim = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restored.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_bestSSIM.pth"))
            print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
                epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))

            """
            # Save evey epochs of model
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
            """
            writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
            writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)
        scheduler.step()


        print("------------------------------------------------------------------")
        print(
            "Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tL1_Loss: {:.4f}\tColor_Loss: {:.4f}\tSSIM_Loss: {:.4f}\tLearningRate {:.6f}".format(
                epoch, time.time() - epoch_start_time,
                epoch_loss, epoch_l1_loss, epoch_color_loss, epoch_grad_loss, scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")

        # Save the last model
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_latest.pth"))

        writer.add_scalar('train/loss', epoch_loss, epoch)
        writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)
        del epoch_loss, epoch_l1_loss, epoch_color_loss, epoch_grad_loss, restored, data_val
    writer.close()

    total_finish_time = (time.time() - total_start_time)  # seconds
    print('Total training time: {:.1f} hours'.format((total_finish_time / 60 / 60)))


if __name__ == '__main__':
    train()
