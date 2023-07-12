# Standard libraries
import itertools
import numpy as np

# PyTorch
import torch
import torch.nn as nn

# Local
from .utils import * 
from .decompression import *


# Others
import torch_dct as dct_pack
from monai.data.utils import dense_patch_slices



######################## 3D ########################
def block_splitting_3d(image, block_size = (8,8,8)):
    """ Splitting 3D volume into smaller cubes of block_size
    Input:
        image: [B,C,H,W,D]
        block_size: (Block_H, Block_W, Block_D)
    Output: 
        image_new (sub_blocks):  [B,C, N_Blocks, block_size[0], block_size[1], block_size[2] ]   where N_Blocks=(H*W*D)//np.prod(block_size)
    """

    assert isinstance(block_size, tuple) and len(block_size)==3, f"Block size should be a tuple of length 3: (Block_H, Block_W, Block_D). Instead got {block_size}"

    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B,C,H,W,D = image.shape

    image_size = (H,W,D)
    roi_size = block_size
    scan_interval = block_size
    slices = dense_patch_slices(image_size, roi_size, scan_interval)

    num_blocks = (H*W*D)//(np.prod(block_size))

    assert H%block_size[0]==0 , f"Height of image should be divisible by block_size[0]"
    assert W%block_size[1]==0 , f"Width of image should be divisible by block_size[1]"
    assert D%block_size[2]==0 , f"Depth of image should be divisible by block_size[2]"

    assert len(slices)==num_blocks , f"Number of 3D sub-blocks={len(slices)} of given image is not equal to expected number of sub-bloks={num_blocks}"

    image_new_shape = (B,C,num_blocks)+block_size
    image_new = torch.zeros(image_new_shape, dtype=torch.float32).to(device)

    for b_dim in range(B): # batch dimension
        for c_dim in range(C): # channel dimension
            for i, i_slice in enumerate(slices):
                image_new[b_dim,c_dim,i] = image[b_dim,c_dim][i_slice]

    return image_new



######################## 2D ########################
def block_splitting_2d(image, block_size = (8,8)):
    """ Splitting 2D spatial slice (at each depth) of input volume into smaller blocks of block_size
    Input:
        image: [B,C,D,H,W]
        block_size: (Block_H, Block_W)
    Output: 
        image_new (sub_blocks):  [B, C, D, N_Blocks, block_size[0], block_size[1] ]   where N_Blocks=(H*W)//np.prod(block_size)
    """

    assert isinstance(block_size, tuple) and len(block_size)==2, f"Block size should be a tuple of length 2: (Block_H, Block_W). Instead got {block_size}"
    
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B,C,D,H,W = image.shape

    image_size = (H,W)
    roi_size = block_size
    scan_interval = block_size
    slices = dense_patch_slices(image_size, roi_size, scan_interval)

    num_blocks = (H*W)//(np.prod(block_size))

    assert H%block_size[0]==0 , f"Height of image should be divisible by block_size[0]"
    assert W%block_size[1]==0 , f"Width of image should be divisible by block_size[1]"
  
    assert len(slices)==num_blocks , f"Number of 2D sub-blocks={len(slices)} of given image is not equal to expected number of sub-bloks={num_blocks}"

    image_new_shape = (B,C,D,num_blocks)+block_size
    image_new = torch.zeros(image_new_shape, dtype=torch.float32).to(device)

    for b_dim in range(B): # batch dimension
        for c_dim in range(C): # channel dimension
            for d_dim in range(D): # depth dimension
                for i, i_slice in enumerate(slices):
                    image_new[b_dim,c_dim,d_dim,i] = image[b_dim,c_dim,d_dim][i_slice]

    return image_new






######################## 3D ########################
def dct_3d(image):
    """ Discrete Cosine Transformation
    Input:
        Image     : [B, C, N_Blocks, Block_H, Block_W, Block_D] , 3D DCT is computed on the last three dimensions  
                    Values should be in the range of [0,255]

    Output:
        DCT(Image): [B, C, N_Blocks, Block_H, Block_W, Block_D]
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    # image = image - 128 # zero centering of pixels/voxels, from [0,255] to [-128,127]
    image = image - 0.5 # zero centering of pixels/voxels, from [0,1] to [-0.5,0.5]
    dct_image = dct_pack.dct_3d(image, 'ortho') # 3D DCT is applied on last three dimensions

    return dct_image



######################## 2D ########################
def dct_2d(image):
    """ Discrete Cosine Transformation
    Input:
        Image     : [B, C, D, N_Blocks, Block_H, Block_W] , 2D DCT is computed on the last two dimensions   
                    Values should be in the range of [0,255]

    Output:
        DCT(Image): [B, C, D, N_Blocks, Block_H, Block_W]
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)

    # image = image - 128 # zero centering of pixels/voxels, from [0,255] to [-128,127]
    image = image - 0.5 # zero centering of pixels/voxels, from [0,1] to [-0.5,0.5]

    dct_image = dct_pack.dct_2d(image, 'ortho') # 2D DCT is applied on the last two dimensions

    return dct_image


