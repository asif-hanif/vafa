import json
import os
import sys
import time
import math
import io
import numpy as np 
import warnings
import torch 
import torch.nn as nn 
import torch.optim as optim 

from  torchattacks.attack import Attack  

from .utils import *
from .compression import *
from .decompression import *

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import einops
import warnings


#############################################################################################################
#                                                 DFB (3D)                                                  #
#############################################################################################################

class DFB(Attack):
    r"""  
    Arguments:
        - model (nn.Module): model to attack.
        - steps (int): number of steps. (DEFALUT: 20)
        - batch_size (int): batch size
        - q_max: bound for quantization table
        - targeted: True for targeted attack
    Shapes:
        - images: [B,C,H,W,D] normalized to [0,1]. B=BatchSize, C=Number of Channels,  H=Height,  W=Width, D=Depth
        - labels: [B,C,H,W,D]
    """

    def __init__(self, model, loss_fn, batch_size=1, channels=1, height=96, width=96, depth=96, block_size=(8,8,8), freq_band=None, verbose=True):
        super(DFB, self).__init__("DFB", model)
        self.verbose = verbose

        self.block_size = tuple(block_size) 
        self.loss_fn = loss_fn

        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width  = width
        self.depth  = depth

     

        self.freq_band = freq_band # which frequency band to be dropped: [ 'low', 'middle', 'high']
        assert freq_band in ['low', 'middle', 'high'] , f"Unknown freqeuncy band: '{freq_band}'. Valid options for 'freq_band' are ['low', 'middle', 'high']"
       

        assert self.height%block_size[0]==0 , f"Height of image should be divisible by block_size[0]"
        assert self.width%block_size[1]==0  , f"Width of image should be divisible by block_size[1]"
        assert self.depth%block_size[2]==0  , f"Depth of image should be divisible by block_size[2]"


        # initialize perturbation tensor
        num_blocks = (self.height*self.width*self.depth)//(np.prod(self.block_size))


        ## masks to perturb DCT coefficients
        low_mask, middle_mask, high_mask = get_masks(self.block_size) # get masks for low, middle and high frequency locations in 3D DCT matrix
        
        # DC coefficients mask values (DC coefficients will not be dropped)
        low_mask[0,0,0] = 0
        middle_mask[0,0,0] = 0
        high_mask[0,0,0] = 0

        
        low_mask = einops.repeat(low_mask, 'h w d-> b c n h w d', b=batch_size, c=channels, n=num_blocks).bool()       # [Block_H, Block_W, Block_D] --> [B, C, N_Blocks, Block_H, Block_W, Block_D]
        middle_mask = einops.repeat(middle_mask, 'h w d-> b c n h w d', b=batch_size, c=channels, n=num_blocks).bool() # [Block_H, Block_W, Block_D] --> [B, C, N_Blocks, Block_H, Block_W, Block_D]
        high_mask = einops.repeat(high_mask, 'h w d-> b c n h w d', b=batch_size, c=channels, n=num_blocks).bool()     # [Block_H, Block_W, Block_D] --> [B, C, N_Blocks, Block_H, Block_W, Block_D]
        

        self.masks = {}
        self.masks["low"]=low_mask
        self.masks["middle"]=middle_mask
        self.masks["high"]=high_mask


    def forward(self, images, labels):
        r"""
        images: [B,C,H,W,D] normalized to [0,1]
        labels: [B,C,H,W,D]
        """
        

        images   = images.clone().detach().to(self.device)       #  [B,C,H,W,D]
        labels   = labels.clone().detach().to(self.device)       #  [B,C,H,W,D]

        B,C,H,W,D = images.shape

        blocks = block_splitting_3d(images, self.block_size)      # [B, C, H, W, D] --> [B, C, N_Blocks, Block_H, Block_W, Block_D]
        blocks_dct = dct_3d(blocks)                               # [B, C, N_Blocks, Block_H, Block_W, Block_D] , 3D DCT is computed on last three dimensions 

        blocks_dct[self.masks.get(self.freq_band)]=0.0            # drop frequency band

        blocks_idct = idct_3d(blocks_dct)                         # [B, C, N_Blocks, Block_H, Block_W, Block_D] , 3D IDCT is computed on last three dimensions 
        adv_images = block_merging_3d(blocks_idct, images.shape)  # [B, C, N_Blocks, Block_H, Block_W, Block_D] --> [B,C,H,W,D]


        logits = self.model(adv_images) # logits: [B,NumClasses,H,W,D] , passing adversarial images through the model 
        
        dice_loss  = self.loss_fn(logits, labels)
        ssim_loss = ssim_loss_fn(images, adv_images)
        total_loss = dice_loss + ssim_loss  


        # predicted labels
        _, pred_labels = torch.max(logits.data, 1, keepdim=True)  
   
        success_rate = ((pred_labels != labels).sum()/labels.numel()).cpu().detach().numpy()

        # print losses and success rate
        if self.verbose: print(f"Dice_Loss: {dice_loss.item():0.5f}    {'SSIM_Loss: '+str(round(ssim_loss.item(),5))}   Total_Loss: {total_loss.item():0.5f}  Success_Rate: {round(success_rate*100,2)} (%)")     


        adv_images = torch.clamp(adv_images, min=0, max=1.0).detach()
       
        return adv_images.detach(), pred_labels, 












#############################################################################################################
#                                                 DFB (2D)                                                  #
#############################################################################################################

class DFB_2D(Attack):
    r"""  
    Arguments:
        - model (nn.Module): model to attack.
        - steps (int): number of steps. (DEFALUT: 20)
        - batch_size (int): batch size
        - q_max: bound for quantization table
        - targeted: True for targeted attack
    Shapes:
        - images: [B,C,H,W,D] normalized to [0,1]. B=BatchSize, C=Number of Channels,  H=Height,  W=Width, D=Depth
        - labels: [B,C,H,W,D]
    """

    def __init__(self, model, loss_fn, batch_size=1, channels=1, height=96, width=96, depth=96, block_size=(8,8), freq_band=None, verbose=True):
        super(DFB_2D, self).__init__("DFB_2D", model)
        self.verbose = verbose

        self.block_size = tuple(block_size) 
        self.loss_fn = loss_fn

        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width  = width
        self.depth  = depth

     

        self.freq_band = freq_band # which frequency band to be dropped: [ 'low', 'middle', 'high']
        assert freq_band in ['low', 'middle', 'high'] , f"Unknown freqeuncy band: '{freq_band}'. Valid options for 'freq_band' are ['low', 'middle', 'high']"
       

        assert self.height%block_size[0]==0 , f"Height of image should be divisible by block_size[0]"
        assert self.width%block_size[1]==0  , f"Width of image should be divisible by block_size[1]"
        assert isinstance(self.block_size, tuple) and len(self.block_size)==2, f"Block size should be a tuple of length 2: (Block_H, Block_W). Instead got {self.block_size}"


        num_blocks = (self.height*self.width)//(np.prod(self.block_size))


        ## masks to perturb DCT coefficients
        low_mask, middle_mask, high_mask = get_masks_2d(self.block_size) # get masks for low, middle and high frequency locations in 3D DCT matrix
        
        # DC coefficients mask values  (DC coefficients will not be dropped)
        low_mask[0,0] = 0
        middle_mask[0,0] = 0
        high_mask[0,0] = 0

        
        low_mask = einops.repeat(low_mask, 'h w -> b c d n h w', b=batch_size, c=channels, d=self.depth, n=num_blocks).bool()       # [Block_H, Block_W] --> [B, C, D, N_Blocks, Block_H, Block_W]
        middle_mask = einops.repeat(middle_mask, 'h w -> b c d n h w', b=batch_size, c=channels, d=self.depth, n=num_blocks).bool() # [Block_H, Block_W] --> [B, C, D, N_Blocks, Block_H, Block_W]
        high_mask = einops.repeat(high_mask, 'h w -> b c d n h w', b=batch_size, c=channels, d=self.depth, n=num_blocks).bool()     # [Block_H, Block_W] --> [B, C, D, N_Blocks, Block_H, Block_W]
        

        self.masks = {}
        self.masks["low"]=low_mask
        self.masks["middle"]=middle_mask
        self.masks["high"]=high_mask


    def forward(self, images, labels):
        r"""
        images: [B,C,H,W,D] normalized to [0,1]
        labels: [B,C,H,W,D]
        """
        

        images   = images.clone().detach().to(self.device)       #  [B,C,H,W,D]
        labels   = labels.clone().detach().to(self.device)       #  [B,C,H,W,D]

        B,C,H,W,D = images.shape
        images = images.permute(0,1,4,2,3)                       #  [B,C,H,W,D] --> [B,C,D,H,W]


        blocks = block_splitting_2d(images, self.block_size)      # [B, C, D, H, W] --> [B, C, D, N_Blocks, Block_H, Block_W]
        blocks_dct = dct_2d(blocks)                               # [B, C, D, N_Blocks, Block_H, Block_W] , 2D DCT is computed on last 2 dimensions  
            
        blocks_dct[self.masks.get(self.freq_band)]=0.0            # drop frequency band

        blocks_idct = idct_2d(blocks_dct)                         # [B, C, D, N_Blocks, Block_H, Block_W] , 2D IDCT is computed on last 2 dimensions 
        adv_images = block_merging_2d(blocks_idct, images.shape)  # [B, C, D, N_Blocks, Block_H, Block_W] --> [B,C,D,H,W]
        
        adv_images = adv_images.permute(0,1,3,4,2)                # [B,C,D,H,W] -->  [B,C,H,W,D]


        logits = self.model(adv_images) # logits: [B,NumClasses,H,W,D] , passing adversarial images through the model 
        
        dice_loss  = self.loss_fn(logits, labels)
        ssim_loss = ssim_loss_fn(images, adv_images)
        total_loss = dice_loss + ssim_loss  


        # predicted labels
        _, pred_labels = torch.max(logits.data, 1, keepdim=True)  
   
        success_rate = ((pred_labels != labels).sum()/labels.numel()).cpu().detach().numpy()

        # print losses and success rate
        if self.verbose: print(f"Dice_Loss: {dice_loss.item():0.5f}    {'SSIM_Loss: '+str(round(ssim_loss.item(),5))}   Total_Loss: {total_loss.item():0.5f}  Success_Rate: {round(success_rate*100,2)} (%)")     


        adv_images = torch.clamp(adv_images, min=0, max=1.0).detach()
       
        return adv_images.detach(), pred_labels, 



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    