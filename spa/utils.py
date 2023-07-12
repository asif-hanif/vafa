import numpy as np
import einops
import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from .compression import block_splitting_3d




######################################################################################################################
#                                                    SPA-3D 
######################################################################################################################

def perturb_dct_coefs(blocks_dct, perturbation):

    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blocks_dct = blocks_dct.to(device)       # [B, C, N_Blocks, Block_H, Block_W, Block_D]
    perturbation = perturbation.to(device)   # [B, C, N_Blocks, Block_H, Block_W, Block_D]
    
    perturbation[:,:,:,0,0,0]=1.0 # do not perturb DC coefficient

    return blocks_dct*perturbation



def perturb_dct_coefs_band(blocks_dct, perturbation, masks=None, freq_band=None):

    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blocks_dct = blocks_dct.to(device)       # [B, C, N_Blocks, Block_H, Block_W, Block_D]
    perturbation = perturbation.to(device)   # [B, C, N_Blocks, Block_H, Block_W, Block_D]
    
    perturbation[masks.get(freq_band)]=1     # coefs located at True location in mask will not be perturbed
    perturbation[:,:,:,0,0,0]=1.0 # do not perturb DC coefficient

    return blocks_dct*perturbation



def ssim_loss_fn(images, adv_images, patch_size=None):
    if patch_size is None:
        im1 = images.permute(0,1,4,2,3)      # [B,C,H,W,D]  --> [B,C,D,H,W]
        im2 = adv_images.permute(0,1,4,2,3)  # [B,C,H,W,D]  --> [B,C,D,H,W]
        ssim_loss = 1-ssim(im1, im2, data_range=1, nonnegative_ssim=True,size_average=True, win_size=3)
    else:
        im1 = block_splitting_3d(images, patch_size).squeeze(1).permute(0,1,4,2,3)      # [B,C=1,H,W,D]  --> [B, C=1, N_Blocks, Block_H, Block_W, Block_D] --> Remove C Dim and Permute Dims --> [B, N_Blocks, Block_D, Block_H, Block_W]
        im2 = block_splitting_3d(adv_images, patch_size).squeeze(1).permute(0,1,4,2,3)  # [B,C=1,H,W,D]  --> [B, C=1, N_Blocks, Block_H, Block_W, Block_D] --> Remove C Dim and Permute Dims --> [B, N_Blocks, Block_D, Block_H, Block_W]
        ssim_loss = 1-ssim(im1, im2, data_range=1, nonnegative_ssim=True,size_average=True, win_size=3)
    
    return ssim_loss


def l1_loss_fn(images, adv_images, patch_size=None):
    l1_loss_func = torch.nn.L1Loss(reduction='mean')

    if patch_size is None:
        l1_loss = l1_loss_func(adv_images,images)
    else:
        im1 = block_splitting_3d(images, patch_size)      # [B,C=1,H,W,D]  --> [B, C=1, N_Blocks, Block_H, Block_W, Block_D] 
        im2 = block_splitting_3d(adv_images, patch_size)  # [B,C=1,H,W,D]  --> [B, C=1, N_Blocks, Block_H, Block_W, Block_D] 
        l1_loss = l1_loss_func(im2,im1)

    return l1_loss


def l2_loss_fn(images, adv_images, patch_size=None):
    l2_loss_func = torch.nn.MSELoss(reduction='mean')

    if patch_size is None:
        l2_loss = l2_loss_func(adv_images,images)
    else:
        im1 = block_splitting_3d(images, patch_size)      # [B,C=1,H,W,D]  --> [B, C=1, N_Blocks, Block_H, Block_W, Block_D] 
        im2 = block_splitting_3d(adv_images, patch_size)  # [B,C=1,H,W,D]  --> [B, C=1, N_Blocks, Block_H, Block_W, Block_D] 
        l2_loss = l2_loss_func(im2,im1)

    return l2_loss



def get_masks(patch_size):
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    side_len = patch_size[0]
    matrix_ones = np.ones(patch_size[:2], dtype=np.float32)

    low_mask = np.zeros(patch_size, dtype=np.float32)
    for counter, z in enumerate(range(side_len//2)):
        low_mask[:,:,z] = np.flip((np.triu(matrix_ones, side_len//2+counter)),1)

    high_mask = np.zeros(patch_size, dtype=np.float32)
    for counter, z in enumerate(range(side_len-1, (side_len//2)-1, -1)):
        high_mask[:,:,z] = np.flip((np.triu(matrix_ones, side_len//2+counter)),0)
 
    middle_mask = np.where((low_mask+high_mask)==1, 0 , 1)

    return torch.from_numpy(low_mask).to(device), torch.from_numpy(middle_mask).to(device), torch.from_numpy(high_mask).to(device)



def drop_dct_coefs(blocks_perturbed_dct, gate):
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # shape of 'gate': [B, C, N_Blocks, Block_H, Block_W, Block_D, 2]

    b,c,n,h,w,d = blocks_perturbed_dct.shape # [B, C, N_Blocks, Block_H, Block_W, Block_D]

    one_hot_mask = torch.nn.functional.gumbel_softmax(gate, tau=0.01, hard=True)                            # [B, C, N_Blocks, Block_H, Block_W, Block_D, 2]
    rep_tensor = einops.repeat(torch.tensor([0,1]), 'cl -> b c n h w d cl', b=b, c=c, n=n, h=h, w=w, d=d)   # [B, C, N_Blocks, Block_H, Block_W, Block_D, 2]
    binary_mask = torch.sum(one_hot_mask * rep_tensor, dim=-1)                                              # [B, C, N_Blocks, Block_H, Block_W, Block_D]  (it contains either 1 or 0 at each location)

    binary_mask[:,:,:,0,0,0]=1.0 # do not drop DC coefficient
    print(binary_mask.numel(), binary_mask.numel()-binary_mask.count_nonzero())
    return blocks_perturbed_dct*binary_mask.to(device)





######################################################################################################################
#                                                    SPA-2D 
######################################################################################################################

def perturb_dct_coefs_2d(blocks_dct, perturbation):

    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blocks_dct = blocks_dct.to(device)       # [B, C, D, N_Blocks, Block_H, Block_W]
    perturbation = perturbation.to(device)   # [B, C, D, N_Blocks, Block_H, Block_W]
    
    perturbation[:,:,:,:,0,0]=1.0 # do not perturb DC coefficient

    return blocks_dct*perturbation



def perturb_dct_coefs_band_2d(blocks_dct, perturbation, masks=None, freq_band=None):

    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    blocks_dct = blocks_dct.to(device)       # [B, C, D, N_Blocks, Block_H, Block_W]
    perturbation = perturbation.to(device)   # [B, C, D, N_Blocks, Block_H, Block_W]
    
    perturbation[masks.get(freq_band)]=1     # coefs located at True location in mask will not be perturbed
    perturbation[:,:,:,:,0,0]=1.0 # do not perturb DC coefficient

    return blocks_dct*perturbation



def get_masks_2d(patch_size):
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

    side_len = patch_size[0]
    matrix_ones = np.ones(patch_size, dtype=np.float32)

    low_mask = np.flip((np.triu(matrix_ones, side_len//2)),1).copy()
    
    high_mask = np.fliplr(np.rot90(low_mask)).copy()
   
    middle_mask = np.where((low_mask+high_mask)==1, 0 , 1).astype(dtype=np.float32)

    return torch.from_numpy(low_mask).to(device), torch.from_numpy(middle_mask).to(device), torch.from_numpy(high_mask).to(device)
