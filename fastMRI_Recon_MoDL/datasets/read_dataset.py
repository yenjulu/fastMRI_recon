import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math
import torch
import torch.nn as nn
import sigpy as sp

matplotlib.use('Agg')  # Or another backend suitable for your system
# sys.path.append(os.path.abspath('../'))
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from utils import psnr, ssim
from models.mri import fftc, ifftc
from models import mri

def plot_images(image_list, saved_name, save_dir='./plot'):

    num_images = len(image_list)
    rows = 2
    cols = math.ceil(num_images / rows)
    plt.figure(figsize=(16, 8)) 
    
    for i, (image) in enumerate(image_list):
            
        plt.subplot(rows, cols, i + 1)
        magnitude = np.abs(image)
        plt.imshow(magnitude, cmap='gray')
        # plt.title(f"{name}", fontsize = 26)
        plt.axis('off') 
    
    plt.savefig(f'{save_dir}/{saved_name}.png', dpi=600)
    plt.close()

dataset_path = '../../data/fastmri_dataset_small.hdf5'
index = 5
prefix = 'trn'

with h5.File(dataset_path, 'r') as f:
    r_gt, csm, kspace = f[prefix+'Org'][index], f[prefix+'Csm'][index], f[prefix+'Kspace'][index]

kspace[:, :, :] = kspace[:, :, :] / np.max(np.abs(kspace[:, :, :][:]))
# c_gt = np.sum(sp.ifft(kspace, axes=[-2, -1]), axis=-3)


mask = np.ones((396, 768)).astype(np.int8)
imask = sp.ifft(kspace, axes=[-2, -1])
c_gt = np.sum(imask * csm.conj(), axis=0) 
max_cgt_real = np.max(c_gt.real) 
max_cgt_imag = np.max(c_gt.imag) 
print(max_cgt_real, max_cgt_imag)
plot_images(image_list=[c_gt, r_gt], saved_name='c_gt, r_gt', save_dir='./plot')