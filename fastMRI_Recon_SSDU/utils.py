import math
import numpy as np
import random
import torch
import matplotlib.pyplot as plt


class Logger():
    def __init__(self, log_dir):
        self.log_dir = log_dir
    def write(self, log_message, verbose=True):
        with open(self.log_dir, 'a') as f:
            f.write(log_message)
            f.write('\n')
        if verbose:
            print(log_message)

def set_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

#math ================================

def torch_complex2real(input_data):
    """
    Parameters
    ----------
    input_data : tensor of shape nrow x ncol of complex dtype.

    Returns
    -------
    outputs concatenated real and imaginary parts as nrow x ncol x 2
    """
    return torch.view_as_real(input_data)

def torch_real2complex(input_data):
    """
    Parameters
    ----------
    input_data : tensor of shape nrow x ncol x 2

    Returns
    -------
    merges concatenated channels and outputs complex image of size nrow x ncol.
    """
    return torch.view_as_complex(input_data)

def torch_fftshift(input_x, axes=1):
    """
    Parameters
    ----------
    input_x : Tensor of shape ncoil x nrow x ncol
    axes : The axes along which to shift. The default is 1.

    """
    return torch_fftshift_flip2D(torch_fftshift_flip2D(input_x, axes=1), axes=2)

def torch_ifftshift(input_x, axes=1):
    """
    Parameters
    ----------
    input_x : Tensor of shape ncoil x nrow x ncol
    axes : The axes along which to shift. The default is 1.

    """
    return torch_ifftshift_flip2D(torch_ifftshift_flip2D(input_x, axes=1), axes=2)

def torch_fftshift_flip2D(input_data, axes=1):
    """
    Parameters
    ----------
    input_data : Tensor of shape ncoil x nrow x ncol
    axes : The axis along which to perform the shift. The default is 1.
    ------
    """

    nx = math.ceil(input_data.shape[1] / 2)
    ny = math.ceil(input_data.shape[2] / 2)

    if axes == 1:
        first_half = input_data[:, :nx, :]
        second_half = input_data[:, nx:, :]
    elif axes == 2:
        first_half = input_data[:, :, :ny]
        second_half = input_data[:, :, ny:]
    else:
        raise ValueError('Invalid axes for fftshift')

    return torch.cat([second_half, first_half], dim=axes)

def torch_ifftshift_flip2D(input_data, axes=1):
    """
    Parameters
    ----------
    input_data : Tensor of shape ncoil x nrow x ncol
    axes : The axis along which to perform the shift. The default is 1.
    ------
    """

    nx = math.floor(input_data.shape[1] / 2)
    ny = math.floor(input_data.shape[2] / 2)

    if axes == 1:
        first_half = input_data[:, :nx, :]
        second_half = input_data[:, nx:, :]
    elif axes == 2:
        first_half = input_data[:, :, :ny]
        second_half = input_data[:, :, ny:]
    else:
        raise ValueError('Invalid axes for ifftshift')

    return torch.cat([second_half, first_half], dim=axes)

def getSSIM(space_ref, space_rec):
    """
    Measures SSIM between the reference and the reconstructed images
    """

    space_ref = np.squeeze(space_ref)
    space_rec = np.squeeze(space_rec)
    space_ref = space_ref / np.amax(np.abs(space_ref))
    space_rec = space_rec / np.amax(np.abs(space_ref))
    data_range = np.amax(np.abs(space_ref)) - np.amin(np.abs(space_ref))

    return compare_ssim(space_rec, space_ref, data_range=data_range,
                        gaussian_weights=True,
                        use_sample_covariance=False)

def getPSNR(ref, recon):
    """
    Measures PSNR between the reference and the reconstructed images
    """

    mse = np.sum(np.square(np.abs(ref - recon))) / ref.size
    psnr = 20 * np.log10(np.abs(ref.max()) / (np.sqrt(mse) + 1e-10))

    return psnr

def fft(ispace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace : coil images of size nrow x ncol x ncoil.
    axes :   The default is (0, 1).
    norm :   The default is None.
    unitary_opt : The default is True.

    Returns
    -------
    transform image space to k-space.

    """

    kspace = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(ispace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * kspace.shape[axis]

        kspace = kspace / np.sqrt(fact)

    return kspace




def ifft(kspace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace : image space of size nrow x ncol x ncoil.
    axes :   The default is (0, 1).
    norm :   The default is None.
    unitary_opt : The default is True.

    Returns
    -------
    transform k-space to image space.

    """

    ispace = np.fft.ifftshift(np.fft.ifftn(np.fft.fftshift(kspace, axes=axes), axes=axes, norm=norm), axes=axes)

    if unitary_opt:

        fact = 1

        for axis in axes:
            fact = fact * ispace.shape[axis]

        ispace = ispace * np.sqrt(fact)

    return ispace

def ifft_torch(kspace, axes=(0, 1), norm=None, unitary_opt=True):

    ispace = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(kspace, dim=axes), dim=axes, norm=norm), dim=axes)

    # Apply scaling for unitarity if requested
    if unitary_opt:
        fact = torch.tensor([ispace.size(dim) for dim in axes]).prod().float().sqrt()

        ispace = ispace * fact
    
    return ispace

def norm(tensor, axes=(0, 1, 2), keepdims=True):
    """
    Parameters
    ----------
    tensor : It can be in image space or k-space.
    axes :  The default is (0, 1, 2).
    keepdims : The default is True.

    Returns
    -------
    tensor : applies l2-norm .

    """
    for axis in axes:
        tensor = np.linalg.norm(tensor, axis=axis, keepdims=True)

    if not keepdims: return tensor.squeeze()

    return tensor

def find_center_ind(kspace, axes=(1, 2, 3)):
    """
    Parameters
    ----------
    kspace : nrow x ncol x ncoil.
    axes :  The default is (1, 2, 3).

    Returns
    -------
    the center of the k-space

    """

    center_locs = norm(kspace, axes=axes).squeeze()

    return np.argsort(center_locs)[-1:]

def index_flatten2nd(ind, shape):
    """
    Parameters
    ----------
    ind : 1D vector containing chosen locations.
    shape : shape of the matrix/tensor for mapping ind.

    Returns
    -------
    list of >=2D indices containing non-zero locations

    """

    array = np.zeros(np.prod(shape))
    array[ind] = 1
    ind_nd = np.nonzero(np.reshape(array, shape))

    return [list(ind_nd_ii) for ind_nd_ii in ind_nd]

def sense1(input_kspace, sens_maps, axes=(0, 1)):
    """
    Parameters
    ----------
    input_kspace : nrow x ncol x ncoil
    sens_maps : nrow x ncol x ncoil

    axes : The default is (0,1).

    Returns
    -------
    sense1 image

    """

    image_space = ifft(input_kspace, axes=axes, norm=None, unitary_opt=True)
    Eh_op = np.conj(sens_maps) * image_space
    sense1_image = np.sum(Eh_op, axis=axes[-1] + 1)

    return sense1_image

def complex2real(input_data):
    """
    Parameters
    ----------
    input_data : row x col
    dtype :The default is np.float32.

    Returns
    -------
    output : row x col x 2

    """

    return np.stack((input_data.real, input_data.imag), axis=-1)

def real2complex(input_data):
    """
    Parameters
    ----------
    input_data : row x col x 2

    Returns
    -------
    output : row x col

    """

    return input_data[..., 0] + 1j * input_data[..., 1]


def fft_torch(ispace, axes=(0, 1), norm=None, unitary_opt=True):
    """
    Parameters
    ----------
    ispace : coil images of size nrow x ncol x ncoil.
    axes : The default is (0, 1).
    norm : The default is None.
    unitary_opt : The default is True.

    Returns
    -------
    Transform image space to k-space.
    """
    
    # Apply FFT
    kspace = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(ispace, dim=axes), dim=axes, norm=norm), dim=axes)

    # Apply scaling for unitarity if requested
    if unitary_opt:
        fact = torch.tensor([ispace.size(dim) for dim in axes]).prod().float().sqrt()

        kspace = kspace / fact
    
    return kspace


def c2r(complex_img, axis=0):
    """
    :input shape: row x col (complex64)
    :output shape: 2 x row x col (float32)
    """
    if isinstance(complex_img, np.ndarray):
        real_img = np.stack((complex_img.real, complex_img.imag), axis=axis)
    elif isinstance(complex_img, torch.Tensor):
        real_img = torch.stack((complex_img.real, complex_img.imag), axis=axis)
    else:
        raise NotImplementedError
    return real_img

def r2c(real_img, axis=0):
    """
    :input shape: 2 x row x col (float32)
    :output shape: row x col (complex64)
    """
    if axis == 0:
        complex_img = real_img[0] + 1j*real_img[1]
    elif axis == 1:
        complex_img = real_img[:,0] + 1j*real_img[:,1]
    else:
        raise NotImplementedError
    return complex_img

def fft_new(image, ndim, normalized=False):
    norm = "ortho" if normalized else None
    dims = tuple(range(-ndim, 0))

    image = torch.view_as_real(
        torch.fft.fftn(  # type: ignore
            torch.view_as_complex(image.contiguous()), dim=dims, norm=norm
        )
    )
    return image

def ifft_new(image, ndim, normalized=False):
    norm = "ortho" if normalized else None
    dims = tuple(range(-ndim, 0))
    image = torch.view_as_real(
        torch.fft.ifftn(  # type: ignore
            torch.view_as_complex(image.contiguous()), dim=dims, norm=norm
        )
    )
    return image

def roll(x, shift, dim):
    """
    Similar to np.roll but applies to PyTorch Tensors
    """
    if isinstance(shift, (tuple, list)):
        assert len(shift) == len(dim)
        for s, d in zip(shift, dim):
            x = roll(x, s, d)
        return x
    shift = shift % x.size(dim)
    if shift == 0:
        return x
    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)
    return torch.cat((right, left), dim=dim)

def fftshift(x, dim=None):
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [dim // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = x.shape[dim] // 2
    else:
        shift = [x.shape[i] // 2 for i in dim]
    return roll(x, shift, dim)

def ifftshift(x, dim=None):
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors
    """
    if dim is None:
        dim = tuple(range(x.dim()))
        shift = [(dim + 1) // 2 for dim in x.shape]
    elif isinstance(dim, int):
        shift = (x.shape[dim] + 1) // 2
    else:
        shift = [(x.shape[i] + 1) // 2 for i in dim]
    return roll(x, shift, dim)

def fft2(data):
    """
    Apply centered 2 dimensional Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The FFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = fft_new(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

def ifft2(data):
    """
    Apply centered 2-dimensional Inverse Fast Fourier Transform.
    Args:
        data (torch.Tensor): Complex valued input data containing at least 3 dimensions: dimensions
            -3 & -2 are spatial dimensions and dimension -1 has size 2. All other dimensions are
            assumed to be batch dimensions.
    Returns:
        torch.Tensor: The IFFT of the input.
    """
    assert data.size(-1) == 2
    data = ifftshift(data, dim=(-3, -2))
    data = ifft_new(data, 2, normalized=True)
    data = fftshift(data, dim=(-3, -2))
    return data

def ifft_fastmri(data):

    data = np.fft.ifftshift(data, axes=(-2, -1))
    data = np.fft.ifftn(data, axes=(-2, -1))
    data = np.fft.fftshift(data, axes=(-2, -1))
    
    return data

def complex_matmul(a, b):
    # function to multiply two complex variable in pytorch, the real/imag channel are in the third last two channels ((batch), (coil), 2, nx, ny).
    if len(a.size()) == 3:
        return torch.cat(((a[0] * b[0] - a[1] * b[1]).unsqueeze(0),
                          (a[0] * b[1] + a[1] * b[0]).unsqueeze(0)), dim=0)
    if len(a.size()) == 4:
        return torch.cat(((a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1]).unsqueeze(1),
                          (a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0]).unsqueeze(1)), dim=1)
    if len(a.size()) == 5:
        return torch.cat(((a[:, :, 0] * b[:, :, 0] - a[:, :, 1] * b[:, :, 1]).unsqueeze(2),
                          (a[:, :, 0] * b[:, :, 1] + a[:, :, 1] * b[:, :, 0]).unsqueeze(2)), dim=2)

def complex_conj(a):
    # function to multiply two complex variable in pytorch, the real/imag channel are in the last two channels.
    if len(a.size()) == 3:
        return torch.cat((a[0].unsqueeze(0), -a[1].unsqueeze(0)), dim=0)
    if len(a.size()) == 4:
        return torch.cat((a[:, 0].unsqueeze(1), -a[:, 1].unsqueeze(1)), dim=1)
    if len(a.size()) == 5:
        return torch.cat((a[:, :, 0].unsqueeze(2), -a[:, :, 1].unsqueeze(2)), dim=2)

#metrics ==================================================
def L1and2_loss(output_kspace, ref_kspace, scalar=0.5):
    
    epsilon = 1e-8
    l2_loss = torch.norm(ref_kspace - output_kspace, p=2) / (torch.norm(ref_kspace, p=2) + epsilon)
    l1_loss = torch.norm(ref_kspace - output_kspace, p=1) / (torch.norm(ref_kspace, p=1) + epsilon)
    return (1 - scalar) * l2_loss + (scalar * l1_loss)

def psnr_batch(y_batch, y_pred_batch):
    #calculate psnr for every batch and return mean
    mean_psnr = 0
    for batch_idx in range(y_batch.shape[0]):
        y = y_batch[batch_idx]
        y_pred = y_pred_batch[batch_idx]
        mean_psnr += psnr(y, y_pred, y.max())
    return mean_psnr / y_batch.shape[0]

def psnr(y, y_pred, MAX_PIXEL_VALUE=1.0):
    rmse_ = rmse(y, y_pred)
    if rmse_ == 0:
        return float('inf')
    return 20 * math.log10(MAX_PIXEL_VALUE/rmse_+1e-10)

def ssim_batch(y_batch, y_pred_batch):
    mean_ssim = 0
    for batch_idx in range(y_batch.shape[0]):
        y = y_batch[batch_idx]
        y_pred = y_pred_batch[batch_idx]
        mean_ssim += ssim(y, y_pred)
    return mean_ssim / y_batch.shape[0]

def ssim(y, y_pred):
    from skimage.metrics import structural_similarity
    return structural_similarity(y, y_pred, data_range=y.max() - y.min())

def mse(y, y_pred):
    return np.mean((y-y_pred)**2)

def rmse(y, y_pred):
    return math.sqrt(mse(y, y_pred))

def img_normalize(img):
    for i in range(img.shape[0]):
        max_img = np.max(img[i])
        min_img = np.min(img[i])
        img[i] = 1 * (img[i] - min_img) / (max_img - min_img)
    return img
#display =======================
def display_img(x, trn_mask, loss_mask, y, y_pred, y_dn, score=None):
    fig = plt.figure(figsize=(15,10))
    
    ax1 = plt.subplot2grid((2,6), (0,0), colspan=2)
    ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
    ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
    ax4 = plt.subplot2grid((2,6), (1,0), colspan=2)
    ax5 = plt.subplot2grid((2,6), (1,2), colspan=2)
    ax6 = plt.subplot2grid((2,6), (1,4), colspan=2)
    
    ax1.imshow(x, cmap='gray')
    ax1.set_title('zero-filled')
    ax2.imshow(trn_mask, cmap='gray')
    ax2.set_title('trn_mask')
    ax3.imshow(y, cmap='gray')
    ax3.set_title('GT')
    ax4.imshow(y_pred, cmap='gray')
    ax4.set_title('reconstruction')
    ax5.imshow(loss_mask, cmap='gray')
    ax5.set_title('loss_mask')
    ax6.imshow(y_dn, cmap='gray')
    ax6.set_title('recon_resnet')  
      
    if score:
        plt.suptitle('score: {:.4f}'.format(score))
    return fig