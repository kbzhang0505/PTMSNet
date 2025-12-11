## Rethinking progressive low-light image enhancement: A frequency-aware tripartite multi-scale network
## Yingjian Li, Kaibing Zhang, Xuan Zhou, Zhouqiang Zhang, Sheng Hu
## https://doi.org/10.1016/j.neunet.2025.108351
import time

import torch
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import os
from collections import OrderedDict
from natsort import natsorted
from glob import glob

from skimage.util import img_as_ubyte

import utils
from transform.dataset_RGB import get_mask_fly
import cv2
import argparse
from model.PTMSNet import PTMSNet as PTMSNet


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

parser = argparse.ArgumentParser(description='Demo Low-light Image Enhancement')
parser.add_argument('--input_dir', default='./datasets/LOL/test/low/', type=str, help='Input images')
parser.add_argument('--target_dir', default='./datasets/LOL/test/high/', type=str, help='Input images')
parser.add_argument('--result_dir', default='./results/LOL', type=str,
                    help='Directory for results')
parser.add_argument('--weights',
                    default='',
                    type=str,
                    help='Path to weights')

args = parser.parse_args()


def save_img(filepath, img):
    cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


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


def cal(input, tar):
    PSNR = utils.torchPSNR(input, tar)
    SSIM = utils.torchSSIM(input, tar)
    return PSNR, SSIM


inp_dir = args.input_dir
out_dir = args.result_dir
tar_dir = args.target_dir

os.makedirs(out_dir, exist_ok=True)

files = natsorted(  # glob(os.path.join(inp_dir, '*.jpg')) +
    glob(os.path.join(inp_dir, '*.JPG'))
    + glob(os.path.join(inp_dir, '*.png')))
# + glob(os.path.join(inp_dir, '*.PNG')))
tar_files = natsorted(
    glob(os.path.join(tar_dir, '*.JPG'))
    + glob(os.path.join(tar_dir, '*.png'))
)

files = [[files[i], tar_files[i]] for i in range(len(tar_files))]

if len(files) == 0:
    raise Exception(f"No files found at {inp_dir}")

# Load corresponding models architecture and weights

model = PTMSNet(inp_channels=3, out_channels=3, dim=16,
                 heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', attention=True, )
model.cuda()

load_checkpoint(model, args.weights)
model.eval()

print('restoring images......')
time_list = []
mul = 16
index = 0
psnr_val_rgb = []
ssim_val_rgb = []
GTMean = False
# GTMean = False
for file_ in files:
    img = Image.open(file_[0]).convert('RGB')
    tar = Image.open(file_[1]).convert('RGB')
    input_ = TF.to_tensor(img).unsqueeze(0).cuda()
    tar_ = TF.to_tensor(tar).unsqueeze(0).cuda()

    # Pad the input if not_multiple_of 16
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    SNR_map = get_mask_fly(input_.cpu(), 15).unsqueeze(0).cuda()
    with torch.no_grad():
        time_start = time.time()
        restored = model([input_, SNR_map])

        if GTMean:
            restored_mean = torch.mean(restored)
            tar_mean = torch.mean(tar_)
            restored = torch.clip(restored * (tar_mean / restored_mean), 0, 1)
        time_list.append(float(time.time() - time_start))

    restored = torch.clamp(restored, 0, 1)
    restored = restored[:, :, :h, :w]
    psnr, ssim = cal(restored, tar_)
    psnr_val_rgb.append(psnr)
    ssim_val_rgb.append(ssim)
    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    restored = img_as_ubyte(restored[0])

    f = os.path.splitext(os.path.split(file_[0])[-1])[0]
    save_img((os.path.join(out_dir, f + f'_psnr_{round(float(psnr), 2)} ssim_{round(float(ssim), 4)}.png')), restored)
    index += 1
    print('%d/%d' % (index, len(files)))

ave_time = (sum(sorted(time_list)[:-1]) / (len(time_list) - 1)) * 1000
print('On average, each image takes {:.1f}ms'.format(ave_time))
print(f'PSNR:{sum(psnr_val_rgb) / len(psnr_val_rgb)}' + '\n' + f'SSIM:{sum(ssim_val_rgb) / len(ssim_val_rgb)}')
print(f"Files saved at {out_dir}")
print('finish !')
