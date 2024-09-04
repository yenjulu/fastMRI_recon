from torch.utils.data import Dataset
import torch
import h5py as h5
import numpy as np
import scipy.io as sio
import random
import sigpy as sp

from proj_models import mri
from utils import r2c, c2r


def get_transformed_inputs(kspace_train, sens_maps, trn_mask, loss_mask):
    
    kspace_train[:, :, :] = kspace_train[:, :, :] / np.max(np.abs(kspace_train[:, :, :][:]))
        
    sub_kspace = kspace_train * np.tile(trn_mask[np.newaxis, ...], (16, 1, 1)) 
    ref_kspace = kspace_train * np.tile(loss_mask[np.newaxis, ...], (16, 1, 1))
    #print("dtype of sub_kspace:", sub_kspace.dtype)
    nw_input = mri.sense1(sub_kspace, sens_maps).astype(np.complex64) #  nrow x ncol 
   
    ref_kspace = c2r(ref_kspace, axis=0)  #  2 x ncoil x nrow x ncol
    nw_input = c2r(nw_input, axis=0)  # 2 x row x col
       
    return ref_kspace, nw_input, sens_maps

class ssdu_dataset(Dataset):
    def __init__(self, mode, dataset_path, mask_path):

        self.prefix = 'trn' if mode == 'train' else 'tst' if mode == 'val' else 'test'
        self.dataset_path = dataset_path
        self.mask_path = mask_path

    def __getitem__(self, index):
        """
        """
        with h5.File(self.dataset_path, 'r') as f:
            gt, csm, kspace = f[self.prefix+'Org'][index], f[self.prefix+'Csm'][index], f[self.prefix+'Kspace'][index]
        
              
        with h5.File(self.mask_path, 'r') as f:
            num_masks = len(f['trn_mask'])
            mask_index = random.randint(0, num_masks - 1)
            
            trn_mask = f['trn_mask'][mask_index]
            loss_mask = f['loss_mask'][mask_index]
        
        if (self.prefix == 'test'):
            mask_dir = 'data/mask_poisson_accelx8_396_768.mat'
            input_mask = sio.loadmat(mask_dir)['mask']
            trn_mask = input_mask
            loss_mask = input_mask
                   
        ref_kspace, nw_input, _  = get_transformed_inputs(kspace, csm, trn_mask, loss_mask)

        return torch.from_numpy(gt), torch.from_numpy(ref_kspace), torch.from_numpy(nw_input), torch.from_numpy(csm), torch.from_numpy(trn_mask), torch.from_numpy(loss_mask)
        
    def __len__(self):
        with h5.File(self.dataset_path, 'r') as f:
            num_data = len(f[self.prefix+'Csm'])  
        return num_data
