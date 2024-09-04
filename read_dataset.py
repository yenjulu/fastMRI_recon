import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math
import torch
import torch.nn as nn
from torchsummary import summary
import sigpy as sp

matplotlib.use('Agg')  # Or another backend suitable for your system

from utils import psnr, ssim
from proj_models.mri import fftc, ifftc


def plot_images_comparison(image_list, saved_name, save_dir='./plot'):
    import math

    plt.figure(figsize=(16, 8)) 
    plt.subplot(1, 1, 1)
    
    y_pred = image_list[0]
    y_pred = np.rot90(y_pred, k=-1)
    y = image_list[1]
    y = np.rot90(y, k=-1)    

    plt.imshow(np.abs(y_pred-y), cmap='gray', vmin=np.abs(y).min(), vmax=np.abs(y).max())
    # plt.title(f"{name}", fontsize = 26)
    plt.axis('off') 
    
    plt.savefig(f'{save_dir}/{saved_name}.png', dpi=1000)
    plt.close()
        
def plot_images(image_list, saved_name, save_dir='./plot'):
    import math
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
          
def get_model_state_dict(filename):
    checkpoint = torch.load(filename)
    
    keys = list(checkpoint.keys())
    print("Keys: %s" % keys)

    optim_state_dict = checkpoint['optim_state_dict']
    model_state_dict = checkpoint['model_state_dict']

    for key, value in optim_state_dict.items():
        print(f"key: {key}")
        print(f"parameters: {value}")
        print(f"Shape of parameters: {value.shape}")
        print(f"Type of parameters: {type(value)}")

def model_parameters(model):
    total_params = 0

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            num_params = parameter.numel()  # return the number of elements in the tensor
            total_params += num_params
            print(f"{name} has {num_params} parameters")

    print(f"Total trainable parameters: {total_params}")



