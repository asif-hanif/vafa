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
#                                                 SPA (3D)                                                  #
#############################################################################################################

class SPA(Attack):
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

    def __init__(self, model, loss_fn, batch_size=1, channels=1, height=96, width=96, depth=96, steps=10, rho=0.2, block_size=(8,8,8), lambda_dice=1, use_ssim_loss=False, lambda_ssim=1, targeted=False, freq_band='all', print_every=10, verbose=True):
        super(SPA, self).__init__("SPA", model)
        self.verbose = verbose

        self.steps = steps
        self.targeted = targeted
        self.block_size = tuple(block_size) 
        self.lambda_dice = lambda_dice
        self.use_ssim_loss = use_ssim_loss
        self.lambda_ssim = lambda_ssim
        self.loss_fn = loss_fn

        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width  = width
        self.depth  = depth

        self.rho = rho

        self.freq_band = freq_band # which frequency band to be perturbed: ['all', 'low', 'middle', 'high']
        assert freq_band in ['all', 'low', 'middle', 'high'] , f"Unknown freqeuncy band: '{freq_band}'. Valid options for 'freq_band' are ['all', 'low', 'middle', 'high']"
        self.print_every = print_every  # print losses and success_rate after 'print_every' steps


        assert self.height%block_size[0]==0 , f"Height of image should be divisible by block_size[0]"
        assert self.width%block_size[1]==0  , f"Width of image should be divisible by block_size[1]"
        assert self.depth%block_size[2]==0  , f"Depth of image should be divisible by block_size[2]"


        # initialize perturbation tensor
        num_blocks = (self.height*self.width*self.depth)//(np.prod(self.block_size))
        perturbation_shape = (self.batch_size,self.channels,num_blocks)+self.block_size     # [B, C, N_Blocks, Block_H, Block_W, Block_D]
        # self.perturbation = torch.full( perturbation_shape, 1 , dtype = torch.float32)                                         # initialize perturbation with 1
        self.perturbation = torch.from_numpy(np.random.uniform(low=1-rho, high=1+rho, size=perturbation_shape).astype('float32') )      # initialize perturbation with samples drawn from uniform distribution


        ## masks to perturb DCT coefficients
        low_mask, middle_mask, high_mask = get_masks(self.block_size) # get masks for low, middle and high frequency locations in 3D DCT matrix
        self.low_mask = einops.repeat(low_mask, 'h w d-> b c n h w d', b=batch_size, c=channels, n=num_blocks).bool()       # [Block_H, Block_W, Block_D] --> [B, C, N_Blocks, Block_H, Block_W, Block_D]
        self.middle_mask = einops.repeat(middle_mask, 'h w d-> b c n h w d', b=batch_size, c=channels, n=num_blocks).bool() # [Block_H, Block_W, Block_D] --> [B, C, N_Blocks, Block_H, Block_W, Block_D]
        self.high_mask = einops.repeat(high_mask, 'h w d-> b c n h w d', b=batch_size, c=channels, n=num_blocks).bool()     # [Block_H, Block_W, Block_D] --> [B, C, N_Blocks, Block_H, Block_W, Block_D]
        
        mask_l = torch.logical_or(self.middle_mask, self.high_mask) # mask for perturbing only low_freq coefs
        mask_m = torch.logical_or(self.low_mask, self.high_mask)    # mask for perturbing only middle_freq coefs
        mask_h = torch.logical_or(self.low_mask, self.middle_mask)  # mask for perturbing only high_freq coefs

        self.masks = {}
        self.masks["low"]=mask_l
        self.masks["middle"]=mask_m
        self.masks["high"]=mask_h


    def forward(self, images, labels):
        r"""
        images: [B,C,H,W,D] normalized to [0,1]
        labels: [B,C,H,W,D]
        """
        

        if self.verbose and (images.max()>1 or images.min()<0) : warnings.warn(f"SPA Attack: Image values are expected to be in the range of [0,1], instead found [min,max]=[{images.min().item()} , {images.max().item()}]")

        images   = images.clone().detach().to(self.device)       #  [B,C,H,W,D]
        labels   = labels.clone().detach().to(self.device)       #  [B,C,H,W,D]

        
        B,C,H,W,D = images.shape


        optimizer = torch.optim.Adam([self.perturbation], lr= 0.01)
        
        
        for i in range(self.steps):
            self.perturbation.requires_grad = True

            blocks = block_splitting_3d(images, self.block_size)      # [B, C, H, W, D] --> [B, C, N_Blocks, Block_H, Block_W, Block_D]
            blocks_dct = dct_3d(blocks)                               # [B, C, N_Blocks, Block_H, Block_W, Block_D] , 3D DCT is computed on last three dimensions 
            
            if self.verbose and i==0: print(f"SPA: Perturbing '{self.freq_band}' frequency band(s)")
            if self.freq_band == 'all':
                blocks_perturbed_dct = perturb_dct_coefs(blocks_dct, self.perturbation)
            else:
                blocks_perturbed_dct = perturb_dct_coefs_band(blocks_dct, self.perturbation, masks=self.masks, freq_band=self.freq_band)


            blocks_idct = idct_3d(blocks_perturbed_dct)                   # [B, C, N_Blocks, Block_H, Block_W, Block_D] , 3D IDCT is computed on last three dimensions 
            merged_blocks = block_merging_3d(blocks_idct, images.shape)   # [B, C, N_Blocks, Block_H, Block_W, Block_D] --> [B,C,H,W,D]


            # adversarially perturbed images
            adv_images = merged_blocks  # [B, C, H, W, D]


            logits = self.model(adv_images) # logits: [B,NumClasses,H,W,D] , passing adversarial images through the model 


            if self.targeted: # for targetted attack
                if self.verbose and i==0: print('SPA: Using Targeted Attack ...\n')
                dice_loss  = self.loss_fn(logits, labels)
            else: # for un-targetted attack
                dice_loss  = -1*self.loss_fn(logits, labels)



            if self.use_ssim_loss:
                if self.verbose and i==0: print(f'SPA: Using SSIM Loss (LambdaSSIM={self.lambda_ssim})  ...')
                ssim_loss = ssim_loss_fn(images, adv_images)
                # ssim_loss = ssim_loss_fn(images, adv_images, patch_size=self.block_size); if self.verbose and i==0: print(f'SPA: Using Block-wise SSIM Loss (LambdaSSIM={self.lambda_ssim})  ...')



            total_loss = self.lambda_dice*dice_loss + (self.lambda_ssim*ssim_loss if self.use_ssim_loss else 0.0) 
            optimizer.zero_grad()
            total_loss.backward()

           
            ## to see statistics of learnable perturbation over steps
            # vals = self.perturbation + 0.0
            # print(f"Steps={i} : min={vals.min():0.5f} , max={vals.max():0.5f} , mean={vals.mean():0.5f} , std={vals.std():0.5f} , percentage(vals<1.0)={(vals<1.0).sum().item()/vals.numel()*100:0.2f} (%) , percentage(vals>1.0)={(vals>1.0).sum().item()/vals.numel()*100:0.2f} (%)   ")
            ## num_vals=vals.numel() , num(vals<1.0)=(vals<1.0).sum().item(), num(vals>1.0)=(vals>1.0).sum().item() 
            

            # update perturbation matrix
            self.perturbation = self.perturbation - 0.1*torch.sign(self.perturbation.grad)
            self.perturbation = torch.clamp(self.perturbation, 1-self.rho, 1+self.rho ).detach()  # it limits the values of `perturbation` to [1-rho, 1+rho]
            

            # predicted labels
            _, pred_labels = torch.max(logits.data, 1, keepdim=True)  


            # calculate success rate
            if self.targeted:
                success_rate = ((pred_labels == labels).sum()/labels.numel()).cpu().detach().numpy()
            else:
                success_rate = ((pred_labels != labels).sum()/labels.numel()).cpu().detach().numpy()


            # print losses and success rate
            if self.verbose and (i==0 or (i+1)%self.print_every == 0): print(f"Step: {str(i+1).zfill(2)}    Dice_Loss: {dice_loss.item():0.5f}    {'SSIM_Loss: '+str(round(ssim_loss.item(),5)) if self.use_ssim_loss else ''}   Total_Loss: {total_loss.item():0.5f}  Success_Rate: {round(success_rate*100,2)} (%)")     


            if success_rate >= 1:
                print(f"Ending at Step={i+1} with SuccessRate={success_rate*100:0.2f} (%)")
                adv_images = torch.clamp(adv_images, min=0, max=1.0).detach()
                return adv_images, pred_labels, self.perturbation.detach()       


        adv_images = torch.clamp(adv_images, min=0, max=1.0).detach()
       
        return adv_images, pred_labels, self.perturbation.detach()









#############################################################################################################
#                                                 SPA (2D)                                                  #
#############################################################################################################


class SPA_2D(Attack):
    r"""  
    Arguments:
        - model (nn.Module): model to attack.
        - steps (int): number of steps. (DEFALUT: 20)
        - batch_size (int): batch size
        - q_max: bound for quantization table
        - targeted: True for targeted attack
    Shapes:
        - images: [B,C,H,W,D] normalized to [0,1]. B = BatchSize, C = Number of Channels,  H = Height,  W = Width, D=Depth
        - labels: [B,C,H,W,D]
    """

    def __init__(self, model, loss_fn, batch_size=1, channels=1, height=96, width=96, depth=96, steps=10, rho=0.2, block_size=(8,8), lambda_dice=1, use_ssim_loss=False, lambda_ssim=1, targeted=False, freq_band='all', print_every=10, verbose=True):
        super(SPA_2D, self).__init__("SPA_2D", model)
        self.verbose = verbose

        self.steps = steps
        self.targeted = targeted
        self.block_size = tuple(block_size) 
        self.lambda_dice = lambda_dice
        self.use_ssim_loss = use_ssim_loss
        self.lambda_ssim = lambda_ssim
        self.loss_fn = loss_fn

        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width  = width
        self.depth  = depth

        self.rho = rho

        self.freq_band = freq_band # which frequency band to be perturbed: ['all', 'low', 'middle', 'high']
        assert freq_band in ['all', 'low', 'middle', 'high'] , f"Unknown freqeuncy band: '{freq_band}'. Valid options for 'freq_band' are ['all', 'low', 'middle', 'high']"
        self.print_every = print_every  # print losses and success_rate after 'print_every' steps

        assert self.height%block_size[0]==0 , f"Height of image should be divisible by block_size[0]"
        assert self.width%block_size[1]==0  , f"Width of image should be divisible by block_size[1]"
        assert isinstance(self.block_size, tuple) and len(self.block_size)==2, f"Block size should be a tuple of length 2: (Block_H, Block_W). Instead got {self.block_size}"
    


        # initialize perturbation tensor
        num_blocks = (self.height*self.width)//(np.prod(self.block_size))
        perturbation_shape =  (self.batch_size,self.channels,self.depth,num_blocks)+self.block_size  # [B, C, D, N_Blocks, Block_H, Block_W]
        # self.perturbation = torch.full( perturbation_shape, 1 , dtype = torch.float32)                                            # initialize perturbation with 1
        self.perturbation = torch.from_numpy(np.random.uniform(low=1-rho, high=1+rho, size=perturbation_shape).astype('float32') )  # initialize perturbation with samples drawn from uniform distribution


        ## masks to perturb DCT coefficients
        low_mask, middle_mask, high_mask = get_masks_2d(self.block_size) # get masks for low, middle and high frequency locations in 3D DCT matrix
        self.low_mask = einops.repeat(low_mask, 'h w -> b c d n h w', b=batch_size, c=channels, d=self.depth, n=num_blocks).bool()       # [Block_H, Block_W] --> [B, C, D, N_Blocks, Block_H, Block_W]
        self.middle_mask = einops.repeat(middle_mask, 'h w -> b c d n h w', b=batch_size, c=channels, d=self.depth, n=num_blocks).bool() # [Block_H, Block_W] --> [B, C, D, N_Blocks, Block_H, Block_W]
        self.high_mask = einops.repeat(high_mask, 'h w -> b c d n h w', b=batch_size, c=channels, d=self.depth, n=num_blocks).bool()     # [Block_H, Block_W] --> [B, C, D, N_Blocks, Block_H, Block_W]
        
        mask_l = torch.logical_or(self.middle_mask, self.high_mask) # mask for perturbing only low_freq coefs
        mask_m = torch.logical_or(self.low_mask, self.high_mask)    # mask for perturbing only middle_freq coefs
        mask_h = torch.logical_or(self.low_mask, self.middle_mask)  # mask for perturbing only high_freq coefs

        self.masks = {}
        self.masks["low"]=mask_l
        self.masks["middle"]=mask_m
        self.masks["high"]=mask_h


 
    def forward(self, images, labels):
        r"""
        images: [B,C,H,W,D] normalized to [0,1]
        labels: [B,C,H,W,D]
        """
        
        if self.verbose and (images.max()>1 or images.min()<0) : warnings.warn(f"SPA Attack: Image values are expected to be in the range of [0,1], instead found [min,max]=[{images.min().item()} , {images.max().item()}]")

        images   = images.clone().detach().to(self.device)       #  [B,C,H,W,D]
        labels   = labels.clone().detach().to(self.device)       #  [B,C,H,W,D]

        
        B,C,H,W,D = images.shape
        images = images.permute(0,1,4,2,3)   #  [B,C,H,W,D] --> [B,C,D,H,W]

        optimizer = torch.optim.Adam([self.perturbation], lr= 0.01)

        
        for i in range(self.steps):
            self.perturbation.requires_grad = True

            blocks = block_splitting_2d(images, self.block_size)      # [B, C, D, H, W] --> [B, C, D, N_Blocks, Block_H, Block_W]
            blocks_dct = dct_2d(blocks)                               # [B, C, D, N_Blocks, Block_H, Block_W] , 2D DCT is computed on last 2 dimensions  
            
            if self.verbose and i==0: print(f"SPA-2D: Perturbing '{self.freq_band}' frequency band(s)")
            if self.freq_band == 'all':
                blocks_perturbed_dct = perturb_dct_coefs_2d(blocks_dct, self.perturbation)
            else:
                blocks_perturbed_dct = perturb_dct_coefs_band_2d(blocks_dct, self.perturbation, masks=self.masks, freq_band=self.freq_band)


            blocks_idct = idct_2d(blocks_perturbed_dct)                   # [B, C, D, N_Blocks, Block_H, Block_W] , 2D IDCT is computed on last 2 dimensions 
            merged_blocks = block_merging_2d(blocks_idct, images.shape)   # [B, C, D, N_Blocks, Block_H, Block_W] --> [B,C,D,H,W]


            # adversarially perturbed images
            adv_images = merged_blocks # [B,C,D,H,W]
            adv_images = adv_images.permute(0,1,3,4,2) # [B,C,D,H,W] -->  [B,C,H,W,D]


            logits = self.model(adv_images) # logits: [B,NumClasses,H,W,D] , passing adversarial images through the model 


            if self.targeted: # for targetted attack
                if self.verbose and i==0: print('SPA-2D: Using Targeted Attack ...\n')
                dice_loss  = self.loss_fn(logits, labels)
            else: # for un-targetted attack
                dice_loss  = -1*self.loss_fn(logits, labels)



            if self.use_ssim_loss:
                if self.verbose and i==0: print(f'SPA-2D: Using SSIM Loss (LambdaSSIM={self.lambda_ssim})  ...')
                ssim_loss = ssim_loss_fn(images, adv_images)



            total_loss = self.lambda_dice*dice_loss + (self.lambda_ssim*ssim_loss if self.use_ssim_loss else 0.0) 
            optimizer.zero_grad()
            total_loss.backward()

           
            ## to see statistics of learnable perturbation over steps
            # vals = self.perturbation + 0.0
            # print(f"Steps={i} : min={vals.min():0.5f} , max={vals.max():0.5f} , mean={vals.mean():0.5f} , std={vals.std():0.5f} , percentage(vals<1.0)={(vals<1.0).sum().item()/vals.numel()*100:0.2f} (%) , percentage(vals>1.0)={(vals>1.0).sum().item()/vals.numel()*100:0.2f} (%)   ")
            ## num_vals=vals.numel() , num(vals<1.0)=(vals<1.0).sum().item(), num(vals>1.0)=(vals>1.0).sum().item() 
            

            # update perturbation matrix
            self.perturbation = self.perturbation - 0.1*torch.sign(self.perturbation.grad)
            self.perturbation = torch.clamp(self.perturbation, 1-self.rho, 1+self.rho ).detach()  # it limits the values of `perturbation` to [1-rho, 1+rho]
            

            # predicted labels
            _, pred_labels = torch.max(logits.data, 1, keepdim=True)  


            # calculate success rate
            if self.targeted:
                success_rate = ((pred_labels == labels).sum()/labels.numel()).cpu().detach().numpy()
            else:
                success_rate = ((pred_labels != labels).sum()/labels.numel()).cpu().detach().numpy()


            # print losses and success rate
            if self.verbose and (i==0 or (i+1)%self.print_every == 0): print(f"Step: {str(i+1).zfill(2)}    Dice_Loss: {dice_loss.item():0.5f}    {'SSIM_Loss: '+str(round(ssim_loss.item(),5)) if self.use_ssim_loss else ''}   Total_Loss: {total_loss.item():0.5f}  Success_Rate: {round(success_rate*100,2)} (%)")     


            if success_rate >= 1:
                print(f"Ending at Step={i+1} with SuccessRate={success_rate*100:0.2f} (%)")
                adv_images = torch.clamp(adv_images, min=0, max=1.0).detach()
                return adv_images, pred_labels, self.perturbation.detach()       


        adv_images = torch.clamp(adv_images, min=0, max=1.0).detach()
       
        return adv_images, pred_labels, self.perturbation.detach()



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    