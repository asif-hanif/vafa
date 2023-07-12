# Standard Libraries
import itertools
import numpy as np

# PyTorch
import torch
import torch.nn as nn

# Local
from .utils import *

# Others
import torch_dct as dct_pack
from monai.data.utils import dense_patch_slices


######################## 3D ########################
def block_merging_3d(image, new_image_shape):
    """ Mergse smaller cubes of block_size into big 3D volume

    Input:
        image (sub_blocks): [B,C, N_Blocks, block_size[0], block_size[1], block_size[2] ]   where N_Blocks=(H*W*D)//np.prod(block_size)
        new_image_shape   : desired shape of the new image [B,C,H,W,D]
    Output: 
        image_new:  [B,C,H,W,D]
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B,C,num_blocks,block_h,block_w,block_d = image.shape

    block_size = (block_h,block_w,block_d)

    H=new_image_shape[2]
    W=new_image_shape[3]
    D=new_image_shape[4]


    image_size = (H,W,D)
    roi_size = block_size
    scan_interval = block_size
    slices = dense_patch_slices( image_size, roi_size, scan_interval)

    assert len(slices)==num_blocks , f"Cannot merge sub-blocks of given image into new image of desired shape due to conflict in sizes."

     
    image_new = torch.zeros(new_image_shape, dtype=torch.float32).to(device) # [B,C,H,W,D]

    for b_dim in range(B): # batch dimension
        for c_dim in range(C): # channel dimension
            for i, i_slice in enumerate(slices):
                image_new[b_dim,c_dim][i_slice] = image[b_dim,c_dim][i]
                
    return image_new



######################## 2D ########################
def block_merging_2d(image, new_image_shape):
    """ Merges smaller 2D blocks of block_size into large 2D slice

    Input:
        image (sub_blocks): [B, C, D, N_Blocks, block_size[0], block_size[1] ]   where N_Blocks=(H*W)//np.prod(block_size)
        new_image_shape   : desired shape of the new image [B,C,D,H,W]
    Output: 
        image_new:  [B,C,D,H,W]

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B,C,D,num_blocks,block_h,block_w = image.shape

    block_size = (block_h,block_w)

    D=new_image_shape[2]
    W=new_image_shape[3]
    H=new_image_shape[4]


    image_size = (H,W)
    roi_size = block_size
    scan_interval = block_size
    slices = dense_patch_slices( image_size, roi_size, scan_interval)

    assert len(slices)==num_blocks , f"Cannot merge sub-blocks of given image into new image of desired shape due to conflict in sizes."

     
    image_new = torch.zeros(new_image_shape, dtype=torch.float32).to(device) # [B,C,H,W,D]

    for b_dim in range(B): # batch dimension
        for c_dim in range(C): # channel dimension
            for d_dim in range(D): # depth dimension
                for i, i_slice in enumerate(slices):
                    image_new[b_dim,c_dim,d_dim][i_slice] = image[b_dim,c_dim,d_dim][i]
                
    return image_new





######################## 3D ########################
def idct_3d(dct_image):
    """ Inverse Discrete Cosine Transformation
    Input:
        dct_image      : [B, C, N_Blocks, Block_H, Block_W, Block_D] , 3D IDCT is computed on the last three dimensions

    Output:
        IDCT(dct_image): [B, C, N_Blocks, Block_H, Block_W, Block_D]
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dct_image = dct_image.to(device)
    
    idct_image = dct_pack.idct_3d(dct_image, 'ortho').to(dtype=torch.float32, device=device) # 3D IDCT is applied on the last three dimensions
    
    return idct_image+128  # making pixel/voxel range from [-128,127] to [0,255]



######################## 2D ########################
def idct_2d(dct_image):
    """ Inverse Discrete Cosine Transformation
    Input:
        dct_image      : [B, C, D, N_Blocks, Block_H, Block_W] , 2D IDCT is computed on the last two dimensions

    Output:
        IDCT(dct_image): [B, C, D, N_Blocks, Block_H, Block_W]
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dct_image = dct_image.to(device)
    
    idct_image = dct_pack.idct_2d(dct_image, 'ortho').to(dtype=torch.float32, device=device) # 2D IDCT is applied on the last two dimensions
    
    return idct_image+128  # making pixel/voxel range from [-128,127] to [0,255]





######################## 2D or 3D ########################
def dequantize(quantized_dct_image, q_table):
    """ De-quantize the DCT coefficients with q_table
    Input:
        quantized_dct_image : [B, C, N_Blocks, Block_H, Block_W, Block_D]
        q_table             : [B, C, N_Blocks, Block_H, Block_W, Block_D]
    
    Output:

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    quantized_dct_image =  quantized_dct_image.to(device)
    q_table = q_table.to(device)

    dequantitize_dct_image = quantized_dct_image * q_table 
    return dequantitize_dct_image
    




################################################################################
################################################################################










