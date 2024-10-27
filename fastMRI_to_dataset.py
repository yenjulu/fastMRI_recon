import torch
from torch.utils.data import Dataset
import scipy.io as sio
import h5py
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import sigpy as sp
from sigpy.mri import samp, app
from scipy.io import savemat

from proj_models import mri
from proj_models.ssdu_masks import ssdu_masks

def gen_fastMRI_tst_dataset():
    '''
    this dataset contains only coil numbers = 16
    '''
    source_dir='data/brain/multicoil_test'
    output_file='data/fastmri_tst_dataset_ssdu.hdf5' 
    filenames_dir = 'data/filenames_test.mat'
    test_filenames = sio.loadmat(filenames_dir)['test_filenames']
    coil = 16
    x = 768  # (dim-2)
    y = 396  # (dim-3)
        
    with h5py.File(output_file, 'a') as h5_combined:          
        for filename in test_filenames:
            prefix = 'test'
            process_file(source_dir, filename, prefix, h5_combined, coil, x, y)

def gen_fastMRI_h5_dataset():
    '''
    this dataset contains only coil numbers = 16
    ''' 
    source_dir='data/brain/multicoil_train' 
    output_file = 'data/fastmri_dataset_small.hdf5'
    filenames_dir = 'filenames.mat'
    # Length of train_filenames: 135
    # Length of val_filenames: 34 
    train_filenames = sio.loadmat(filenames_dir)['train_filenames']
    val_filenames = sio.loadmat(filenames_dir)['val_filenames']
    coil = 16
    x = 768
    y = 396 
    
    with h5py.File(output_file, 'a') as h5_combined:
        for filename in train_filenames[:40]:
            prefix = 'trn'
            process_file(source_dir, filename, prefix, h5_combined, coil, x, y)
            
        for filename in val_filenames[:20]:
            prefix = 'tst'
            process_file(source_dir, filename, prefix, h5_combined, coil, x, y)
            
def process_file(source_dir, filename, prefix, h5_combined, coil, x, y):
    file_path = os.path.join(source_dir, filename)

    try:
        Csm, Org, Kspace  = fastMRI_to_dataset(file_path, x, y)
        Csm = Csm[:, :coil, :, :]
        Kspace = Kspace[:, :coil, :, :]

        for name, data in zip([prefix+'Kspace', prefix+'Csm', prefix+'Org'], [Kspace, Csm, Org]):
            print(f"Processing {name} with data shape: {data.shape if data is not None else 'None'}")
            if name in h5_combined:
                print(f"Updating existing dataset {name}")
                dataset = h5_combined[name]
                current_size = dataset.shape[0]
                new_size = current_size + data.shape[0]
                dataset.resize(new_size, axis=0)
                dataset[current_size:] = data
            else:
                print(f"Creating new dataset {name}")
                if name == f'{prefix}Csm':
                    h5_combined.create_dataset(name, data=data, maxshape=(None, coil, y, x), chunks=(1, coil, y, x))
                elif name == f'{prefix}Kspace':
                    h5_combined.create_dataset(name, data=data, maxshape=(None, coil, y, x), chunks=(1, coil, y, x))
                else:
                    h5_combined.create_dataset(name, data=data, maxshape=(None, y, x), chunks=(1, y, x))

    except Exception as e:
        print(f"An error occurred while processing {filename}: {e}")

def fastMRI_to_dataset(dataset_path, x, y):
    with h5py.File(dataset_path, 'r') as file:
      kspace = file['kspace'][:]   # <class 'numpy.ndarray'> Type: complex64

    N_slice, N_coil, _, _ = kspace.shape
    gt = sp.rss(sp.ifft(kspace, axes=[-2, -1]), axes=(-3))

    max_gt = np.max(gt)
    min_gt = np.min(gt)
    Org = 1 * (gt - min_gt) / (max_gt - min_gt)   

    device = sp.Device(0) if torch.cuda.is_available() else sp.cpu_device
    kspace_dev = sp.to_device(kspace, device=device)
    csm = []
    for s in range(N_slice):
        k = kspace_dev[s]
        c = app.EspiritCalib(k, device=device).run()
        c = sp.to_device(c)
        csm.append(c)

    Csm = np.array(csm)
    
    Org = np.transpose(Org, axes=(0, 2, 1))  # batch, 396, 768 
    kspace = np.transpose(kspace, axes=(0, 1, 3, 2))  # batch, coils, 396, 768 
    Csm = np.transpose(Csm, axes=(0, 1, 3, 2))  # batch, coils, 396, 768
    return Csm, Org, kspace

def gen_mask():
    mask_poisson = samp.poisson([396, 768], 8).astype(np.int8) 
    savemat('data/mask_poisson_accelx8_396_768.mat', {'mask': mask_poisson})

def gen_trn_loss_mask():
    ssdumask = ssdu_masks()
    output_file='data/trn_loss_mask_accelx8_ssdu.hdf5'
    # Keys: ['loss_mask', 'trn_mask']
    # Shape: (2500, 396, 768), Type: int8
    # Shape: (2500, 396, 768), Type: int8
    with h5py.File(output_file, 'a') as h5_combined:
        for _ in range(500):
            try:
                trn_mask, loss_mask = ssdumask.Gaussian_selection()
                trn_mask, loss_mask  = trn_mask[None, ...], loss_mask[None, ...]
                #print(trn_mask.shape) # (1, 396, 768)

                # Iterate over each dataset name and corresponding data
                for name, data in zip(['trn_mask', 'loss_mask'],
                                      [trn_mask, loss_mask]):
                    if name in h5_combined:
                        # Dataset exists, so resize it to fit the new data
                        dataset = h5_combined[name]
                        current_size = dataset.shape[0]
                        new_size = current_size + data.shape[0]
                        dataset.resize(new_size, axis=0)
                        # Append the new data
                        dataset[current_size:] = data
                    else:
                        # Dataset does not exist, so create it    
                        h5_combined.create_dataset(name, data=data, maxshape=(None, 396, 768), chunks=(1, 396, 768))    

            except Exception as e:
                print(f"An error occurred while processing : {e}")    
 
def random_set_train_val_files():
    directory = 'data/brain/multicoil_train'
    pattern = 'file_brain_AXT2_210_6001'
    
    files = []
    coils = []
    imgs = []
    count = 0
    
    for filename in os.listdir(directory):
        
        
        if filename.startswith(pattern):
            file_path = os.path.join(directory, filename)

            with h5py.File(file_path, 'r') as hdf:
                
                if count == 0:
                    keys = list(hdf.keys())
                    print("Keys: %s" % keys) # 'ismrmrd_header', 'kspace', 'reconstruction_rss'
                    
                    for key in keys:
                        dataset = hdf[key]
                        print(f"Shape: {dataset.shape}, Type: {dataset.dtype}") 
                
                if 'kspace' in hdf:
                    coil = hdf['kspace'].shape[1]
                    img = hdf['kspace'].shape[2:]
                    coils.append(coil)
                    imgs.append(img)
                    
                    if hdf['kspace'].shape[1] >= 16:
                        files.append(filename)
                        
            count = count + 1

    unique, counts = np.unique(imgs, return_counts=True)
    print(dict(zip(unique, counts)))  # {396: 172, 768: 172}
    
    unique, counts = np.unique(coils, return_counts=True)
    print(dict(zip(unique, counts)))  # {8: 2, 12: 1, 16: 97, 18: 2, 20: 70}

    np.random.shuffle(files)
    num_train = int(len(files) * 0.8)
    train_filenames = files[:num_train]
    val_filenames = files[num_train:]
    print(len(train_filenames), len(val_filenames))

    data_dict = {
        'train_filenames': train_filenames,
        'val_filenames': val_filenames
    }
    savemat('filenames.mat', data_dict)

def random_set_tst_files():
    directory = 'data/brain/multicoil_test'
    pattern = 'file_brain_AXT2_210_6001'
    files = []
    for filename in os.listdir(directory):
        if filename.startswith(pattern):
            file_path = os.path.join(directory, filename)

            with h5py.File(file_path, 'r') as hdf:
                if 'kspace' in hdf and hdf['kspace'].shape[1] >= 16:
                    files.append(filename)

    np.random.shuffle(files)
    test_filenames = files
    print(len(test_filenames))

    data_dict = {
        'test_filenames': test_filenames
    }
    savemat('filenames_test.mat', data_dict)

if __name__ == "__main__":

    gen_trn_loss_mask()
    # gen_fastMRI_h5_dataset()
    