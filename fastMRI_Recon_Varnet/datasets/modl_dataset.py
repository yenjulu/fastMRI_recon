import torch
from torch.utils.data import Dataset
import h5py as h5
import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt
import sigpy as sp
import scipy.io as sio

sys.path.append(os.path.abspath('../'))
from utils import * 
from models import mri


class fastmri_dataset(Dataset):
    def __init__(self, mode, dataset_path, mask_path, sigma=0.01):
        """
        :sigma: std of Gaussian noise to be added in the k-space
        """
        self.prefix = 'trn' if mode == 'train' else 'tst' if mode == 'val' else 'test'
        self.dataset_path = dataset_path
        self.mask_path = mask_path
        self.sigma = sigma
        
    def __getitem__(self, index):
        """
        :x0: zero-filled reconstruction (2 x nrow x ncol) - float32
        :gt: fully-sampled image (nrow x ncol) - float32
        :csm: coil sensitivity map (ncoil x nrow x ncol) - complex64
        :mask: undersample mask (nrow x ncol) - int8
        """
        with h5.File(self.dataset_path, 'r') as f:
            gt, csm, kspace = f[self.prefix+'Org'][index], f[self.prefix+'Csm'][index], f[self.prefix+'Kspace'][index]
            mask = sio.loadmat(self.mask_path)['mask']
          
        x0 = undersample_(gt, csm, mask, self.sigma)

        return torch.from_numpy(c2r(x0)), torch.from_numpy(gt), torch.from_numpy(csm), torch.from_numpy(mask)

    def __len__(self):
        with h5.File(self.dataset_path, 'r') as f:
            num_data = len(f[self.prefix+'Org'])
        return num_data


def undersample_(c_gt, csm, mask, sigma):

    ncoil, nrow, ncol = csm.shape
    csm = csm[None, ...]  # 4dim
    mask = mask[None, ...] # 3dim
    c_gt = c_gt[None, ...] # 3dim

    SenseOp = mri.SenseOp(csm, mask)

    b = SenseOp.fwd(c_gt)

    noise = torch.randn(b.shape) + 1j * torch.randn(b.shape)
    noise = noise * sigma / (2.**0.5)

    atb = SenseOp.adj(b + noise).squeeze(0).detach().numpy()

    return atb
