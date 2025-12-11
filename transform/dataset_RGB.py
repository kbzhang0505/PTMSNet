import math
import os

# from matplotlib import pyplot as plt
from torch.utils.data import Dataset
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import random
import numpy as np
from utils.image_utils import load_img
import kornia

import cv2 as cv


#import torch.nn.functional as F
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


def visushrink_threshold(coeffs, sigma):
    """VisuShrink thresholding function."""
    threshold = sigma * np.sqrt(2 * np.log(len(coeffs)))
    return threshold


def denoise_image(img, cutoff_freq):
    # 傅里叶变换
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 构建低通滤波器
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2  # 中心位置
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - cutoff_freq:crow + cutoff_freq, ccol - cutoff_freq:ccol + cutoff_freq] = 1

    # 应用滤波器
    dft_shift_filtered = dft_shift * mask

    # 逆傅里叶变换
    f_ishift = np.fft.ifftshift(dft_shift_filtered)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    # 转换为 uint8 类型
    img_back = cv.normalize(img_back, None, 0, 255, cv.NORM_MINMAX)
    img_back = img_back.astype(np.uint8)

    return img_back


def get_mask(dark):
    dark = dark.unsqueeze(0)
    light = kornia.filters.gaussian_blur2d(dark, (5, 5), (1.5, 1.5))
    dark = dark[:, 0:1, :, :] * 0.299 + dark[:, 1:2, :, :] * 0.587 + dark[:, 2:3, :, :] * 0.114
    light = light[:, 0:1, :, :] * 0.299 + light[:, 1:2, :, :] * 0.587 + light[:, 2:3, :, :] * 0.114
    noise = torch.abs(dark - light)

    mask = torch.div(light, noise + 0.0001)

    batch_size = mask.shape[0]
    height = mask.shape[2]
    width = mask.shape[3]
    mask_max = torch.max(mask.view(batch_size, -1), dim=1)[0]
    mask_max = mask_max.view(batch_size, 1, 1, 1)
    mask_max = mask_max.repeat(1, 1, height, width)
    mask = mask * 1.0 / (mask_max + 0.0001)

    mask = torch.clamp(mask, min=0, max=1.0)

    # mask_np = mask.squeeze(0).squeeze(0).numpy()
    # plt.figure(figsize=(10, 8))
    # plt.subplot(1, 1, 1)
    # plt.imshow(mask_np, cmap='jet')
    # plt.colorbar(label='SNR (dB)')
    # plt.title('SNR Map')
    # plt.axis('off')
    #
    # plt.show()

    return mask.squeeze(0).float()


def get_mask_fly(image, cutoff_freq=15):
    # 生成模糊图像
    if len(image.size()) == 3:
        image = image[0:1, :, :] * 0.299 + image[1:2, :, :] * 0.587 + image[2:3, :, :] * 0.114
    elif len(image.size()) == 4:
        image = image.squeeze(0)
        image = image[0:1, :, :] * 0.299 + image[1:2, :, :] * 0.587 + image[2:3, :, :] * 0.114
    else:
        exit('please check the len of the image size, make sure that is 3 or 4!')
    image = (image * 255).squeeze(0).numpy().astype(np.uint8)
    blurred_image = denoise_image(image, cutoff_freq)

    # 转换为torch张量，并归一化至[0, 1]
    blurred_image_tensor = torch.tensor(blurred_image / 255.0, dtype=torch.float32)

    # 计算噪声
    noise = np.abs(image.astype(float) - blurred_image_tensor.numpy())

    # 计算掩码
    mask = blurred_image_tensor / (noise + 0.0001)

    # 归一化
    mask = mask / (torch.max(mask) + 0.0001)
    mask = torch.clamp(mask, min=0, max=1.0)
    if math.isnan(mask[0][0]):
        print('error!!!')

    # mask_np = mask.numpy()
    # plt.figure(figsize=(10, 8))
    # plt.subplot(1, 1, 1)
    # plt.imshow(mask_np, cmap='jet')
    # plt.colorbar(label='SNR (dB)')
    # plt.title('SNR Map')
    # plt.axis('off')
    #
    # plt.show()
    return mask.unsqueeze(0).float()


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, cutoff_freq=15):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))

        self.inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']
        self.cutoff_freq = cutoff_freq

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')

        w, h = tar_img.size
        padw = ps - w if w < ps else 0
        padh = ps - h if h < ps else 0

        # Reflect Pad in case image is smaller than patch_size
        if padw != 0 or padh != 0:
            inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
            tar_img = TF.pad(tar_img, (0, 0, padw, padh), padding_mode='reflect')

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        # in_np = inp_img

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Data Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]
        SNR_map = get_mask_fly(inp_img, self.cutoff_freq)
        # SNR_map = get_mask(inp_img)

        return tar_img, inp_img, filename, SNR_map


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))

        self.inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps = self.img_options['patch_size']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex
        ps = self.ps

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')
        # dark_np = np.array(inp_img)
        # Validate on center crop
        if self.ps is not None:
            inp_img = TF.center_crop(inp_img, (ps, ps))
            tar_img = TF.center_crop(tar_img, (ps, ps))

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]
        SNR_map = get_mask_fly(inp_img, 15)
        # SNR_map = get_mask(inp_img)

        return tar_img, inp_img, filename, SNR_map


class DataLoaderVal_(Dataset):
    def __init__(self, rgb_dir, img_options=None, rgb_dir2=None, cutoff_freq=15):
        super(DataLoaderVal_, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))

        self.inp_filenames = [os.path.join(rgb_dir, 'low', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, 'high', x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target
        self.mul = 16
        self.cutoff_freq = cutoff_freq

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path).convert('RGB')
        tar_img = Image.open(tar_path).convert('RGB')
        # dark_np = np.array(inp_img)
        #inp_img = TF.to_tensor(inp_img)
        #tar_img = TF.to_tensor(tar_img)
        w, h = inp_img.size
        #h, w = inp_img.shape[2], inp_img.shape[3]
        H, W = ((h + self.mul) // self.mul) * self.mul, ((w + self.mul) // self.mul) * self.mul
        padh = H - h if h % self.mul != 0 else 0
        padw = W - w if w % self.mul != 0 else 0
        inp_img = TF.pad(inp_img, (0, 0, padw, padh), padding_mode='reflect')
        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)
        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]
        # SNR_map = get_mask(inp_img)
        SNR_map = get_mask_fly(inp_img, self.cutoff_freq)

        return tar_img, inp_img, filename, SNR_map


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp).convert('RGB')

        inp = TF.to_tensor(inp)
        return inp, filename


class DataLoaderTest_(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTest_, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, 'low')))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, 'high')))

        self.clean_filenames = [os.path.join(rgb_dir, 'low', x) for x in clean_files if is_image_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, 'high', x) for x in noisy_files if is_image_file(x)]

        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        return clean, noisy, clean_filename, noisy_filename
