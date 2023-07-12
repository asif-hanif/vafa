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

    def __init__(self, model, loss_fn, batch_size=1, channels=1, height=96, width=96, depth=96, steps=20, rho=0.2, block_size=(8,8,8), lambda_dice=1, use_ssim_loss=False, lambda_ssim=1,  use_l1_loss=False, lambda_l1=1, use_gate=False, targeted=False, verbose=True):
        super(SPA, self).__init__("SPA", model)
        self.verbose = verbose

        self.steps = steps
        self.targeted = targeted
        self.block_size = tuple(block_size) 
        self.lambda_dice = lambda_dice
        self.use_ssim_loss = use_ssim_loss
        self.lambda_ssim = lambda_ssim
        self.use_l1_loss = use_l1_loss
        self.lambda_l1 = lambda_l1
        self.use_gate = use_gate
        self.loss_fn = loss_fn

        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width  = width
        self.depth  = depth

        self.rho = rho
        self.rho_range = torch.tensor([1-rho, 1+rho])

        assert self.height%block_size[0]==0 , f"Height of image should be divisible by block_size[0]"
        assert self.width%block_size[1]==0  , f"Width of image should be divisible by block_size[1]"
        assert self.depth%block_size[2]==0  , f"Depth of image should be divisible by block_size[2]"


        # initialize perturbation tensor
        num_blocks = (self.height*self.width*self.depth)//(np.prod(self.block_size))
        perturbation_shape = (self.batch_size,self.channels,num_blocks)+self.block_size     # [B, C, N_Blocks, Block_H, Block_W, Block_D]
        self.perturbation = torch.full( perturbation_shape, 1 , dtype = torch.float32)      # initialize perturbation with 1
        # self.perturbation = torch.from_numpy(np.random.uniform(low=-2, high=2, size=perturbation_shape).astype('float32') )                # initialize 

        if use_gate: self.gate = torch.full( perturbation_shape+(2,), 10 , dtype = torch.float32)                # initialize, [B, C, N_Blocks, Block_H, Block_W, Block_D, 2]
        # if use_gate: self.gate = torch.from_numpy(np.random.uniform(low=-1, high=1, size=perturbation_shape+(2,)).astype('float32') )                # initialize 


        low_mask, middle_mask, high_mask = get_masks(self.block_size) # get masks for low, middle and high frequency locations in DCT matrix
        self.low_mask = einops.repeat(low_mask, 'h w d-> b c n h w d', b=batch_size, c=channels, n=num_blocks) # [Block_H, Block_W, Block_D] --> [B, C, N_Blocks, Block_H, Block_W, Block_D]
        self.middle_mask = einops.repeat(middle_mask, 'h w d-> b c n h w d', b=batch_size, c=channels, n=num_blocks) # [Block_H, Block_W, Block_D] --> [B, C, N_Blocks, Block_H, Block_W, Block_D]
        self.high_mask = einops.repeat(high_mask, 'h w d-> b c n h w d', b=batch_size, c=channels, n=num_blocks) # [Block_H, Block_W, Block_D] --> [B, C, N_Blocks, Block_H, Block_W, Block_D]
 

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
            if self.use_gate: self.gate.requires_grad = True

            blocks = block_splitting_3d(images*255, self.block_size)      # [B, C, H, W, D] --> [B, C, N_Blocks, Block_H, Block_W, Block_D]
            blocks_dct = dct_3d(blocks)                                   # [B, C, N_Blocks, Block_H, Block_W, Block_D] , 3D DCT is computed on last three dimensions 
            
            # blocks_perturbed_dct = perturb_dct_coefs(blocks_dct, self.perturbation, self.rho)
            # blocks_perturbed_dct = perturb_dct_coefs_band(blocks_dct, self.perturbation, self.rho, mask=self.low_mask)

            if self.use_gate: blocks_perturbed_dct = drop_dct_coefs(blocks_dct, self.gate)
            blocks_perturbed_dct = perturb_dct_coefs(blocks_perturbed_dct, self.perturbation, self.rho)




            blocks_idct = idct_3d(blocks_perturbed_dct)                   # [B, C, N_Blocks, Block_H, Block_W, Block_D] , 3D IDCT is computed on last three dimensions 
            merged_blocks = block_merging_3d(blocks_idct, images.shape)   # [B, C, N_Blocks, Block_H, Block_W, Block_D] --> [B,C,H,W,D]

            # adversarially perturbed images
            adv_images = merged_blocks  # [B, C, H, W, D]
            adv_images = adv_images/255.0

            logits = self.model(adv_images) # logits: [B,NumClasses,H,W,D] , passing adversarial images through the model 

            if self.targeted: # for targetted attack
                if self.verbose and i==0: print('SPA: Using Targeted Attack ...\n')
                dice_loss  = self.loss_fn(logits, labels)
            else: # for un-targetted attack
                dice_loss  = -1*self.loss_fn(logits, labels)



            if self.use_ssim_loss:
                if self.verbose and i==0: print(f'SPA: Using SSIM Loss (LambdaSSIM={self.lambda_ssim})  ...')
                ssim_loss = ssim_loss_fn(images, adv_images, patch_size=self.block_size)
            

            if self.use_l1_loss:
                if self.verbose and i==0: print(f'SPA: Using L1 Loss (LambdaL1={self.lambda_l1})  ...')
                l1_loss = l1_loss_fn(images, adv_images)
                # l1_loss = l1_loss_fn(images, adv_images, patch_size=self.block_size)
               

      


            total_loss = self.lambda_dice*dice_loss + (self.lambda_ssim*ssim_loss if self.use_ssim_loss else 0.0) + (self.lambda_l1*l1_loss if self.use_l1_loss else 0.0)
            optimizer.zero_grad()
            total_loss.backward()

            
            # update perturbation matrix
            self.perturbation = self.perturbation - torch.sign(self.perturbation.grad)
            self.perturbation = self.perturbation.detach()
            # self.perturbation = torch.clamp(self.perturbation, torch.logit(self.rho_range[0]-self.rho), torch.logit(self.rho_range[1]-self.rho) ).detach()
            

            if self.use_gate:
                if self.verbose and (i==0 or (i+1)%10 == 0): print(f"SPA: Using gate ...")
                self.gate = self.gate - torch.sign(self.gate.grad)
                self.gate = self.gate.detach()



            # vals = torch.sigmoid(self.perturbation) + self.rho 
            # print(f"Stets={i} : min={vals.min()} , max={vals.max()} , mean={vals.mean()} , std={vals.std()}")

            _, pred_labels = torch.max(logits.data, 1, keepdim=True)


            if self.targeted:
                success_rate = ((pred_labels == labels).sum()/labels.numel()).cpu().detach().numpy()
            else:
                success_rate = ((pred_labels != labels).sum()/labels.numel()).cpu().detach().numpy()

            if self.verbose and (i==0 or (i+1)%10 == 0): print('Step: ', i+1, "  Loss: ", round(total_loss.item(),5), "  Current Success Rate: ", round(success_rate*100,4), '%' )     
                

            if success_rate >= 1:
                print('Ending at Step={} with Success Rate={}'.format(i+1, success_rate))
                adv_images = torch.clamp(adv_images, min=0, max=1.0).detach()
                return adv_images, pred_labels, self.perturbation.detach()       


        adv_images = torch.clamp(adv_images, min=0, max=1.0).detach()
       
        return adv_images, pred_labels, self.perturbation.detach()





















#############################################################################################################
#                                                 InfoDrop 2D                                               #
#############################################################################################################


# Note: This is re-implementation and adaptation of AdvDrop (https://arxiv.org/pdf/2108.09034.pdf) attack involving 2D DCT on each 2D spatial slice of volumetric image.
class InfoDrop2D(Attack):
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

    def __init__(self, model,loss_fn, batch_size=1, channels=1, height=96, width=96, depth=96, steps=20, q_min=5, q_max=10, block_size=(8,8), targeted=False, verbose=True):
        super(InfoDrop2D, self).__init__("InfoDrop2D", model)
        self.verbose = verbose

        self.steps = steps
        self.targeted = targeted
        self.block_size = tuple(block_size) 
        self.loss_fn = loss_fn

        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width  = width
        self.depth  = depth


        # learnable quantization table values will be limited to the following range  
        self.q_range = [q_min, q_max]


        # parameters for differential quantization
        self.alpha_range = [0.1, 1e-20]
        self.alpha = torch.tensor(self.alpha_range[0])
        self.alpha_interval = torch.tensor((self.alpha_range[1] - self.alpha_range[0])/ self.steps)
        
        assert self.height%block_size[0]==0 , f"Height of image should be divisible by block_size[0]"
        assert self.width%block_size[1]==0  , f"Width of image should be divisible by block_size[1]"
        assert isinstance(self.block_size, tuple) and len(self.block_size)==2, f"Block size should be a tuple of length 2: (Block_H, Block_W). Instead got {self.block_size}"
    

        # initialize quantization table
        num_blocks = (self.height*self.width)//(np.prod(self.block_size))
        q_tables_shape = (self.batch_size,self.channels,self.depth,num_blocks)+self.block_size    # [B, C, D, N_Blocks, Block_H, Block_W]
        self.q_tables = torch.full( q_tables_shape, q_max, dtype = torch.float32)                 # initialize the q_tables with q_max
        
 
    def forward(self, images, labels):
        r"""
        images: [B,C,H,W,D] normalized to [0,1]
        labels: [B,C,H,W,D]
        """

        if self.verbose and (images.max()>1 or images.min()<0) : warnings.warn(f"InfoDrop-2D Attack: Image values are expected to be in the range of [0,1], instead found [min,max]=[{images.min().item()} , {images.max().item()}]")

        images   = images.clone().detach().to(self.device)    #  [B,C,H,W,D]
        labels   = labels.clone().detach().to(self.device)    #  [B,C,H,W,D]

        B,C,D,H,W = images.shape

        images = images.permute(0,1,4,2,3)   #  [B,C,H,W,D] --> [B,C,D,H,W]

        self.alpha = self.alpha.to(self.device)
        self.alpha_interval = self.alpha_interval.to(self.device)

        optimizer = torch.optim.Adam([self.q_tables], lr= 0.01)
        
        
        for i in range(self.steps):
            # set requires_grad for q_tables 
            self.q_tables.requires_grad = True
            blocks = block_splitting_2d(images*255, self.block_size)                  # [B, C, D, H, W] --> [B, C, D, N_Blocks, Block_H, Block_W]
            blocks_dct = dct_2d(blocks)                                               # [B, C, D, N_Blocks, Block_H, Block_W] , 2D DCT is computed on last 2 dimensions 
            blocks_dct_qunatized = quantize(blocks_dct, self.q_tables, self.alpha)    # [B, C, D, N_Blocks, Block_H, Block_W]
            blocks_dct_dequnatized = dequantize(blocks_dct_qunatized, self.q_tables)  # [B, C, D, N_Blocks, Block_H, Block_W]
            blocks_idct = idct_2d(blocks_dct_dequnatized)                             # [B, C, D, N_Blocks, Block_H, Block_W] , 2D IDCT is computed on last 2 dimensions 
            merged_blocks = block_merging_2d(blocks_idct, images.shape)               # [B, C, D, N_Blocks, Block_H, Block_W] --> [B,C,D,H,W]


            # adversarially perturbed images
            adv_images = merged_blocks # [B,C,D,H,W]
            adv_images = adv_images.permute(0,1,3,4,2) # [B,C,D,H,W] -->  [B,C,H,W,D]


            ## normalize each (H,W) slice to [0,1] range
            maxx = torch.amax(adv_images, dim=(2,3), keepdim=True)
            minn = torch.amin(adv_images, dim=(2,3), keepdim=True)
            adv_images = (adv_images - minn)/(maxx-minn)


            logits = self.model(adv_images) # logits: [B,NumClasses,H,W,D] , passing adversarial images through the model 

            if self.targeted: # for targetted attack
                if self.verbose and i==0: print('InfoDrop-2D: Using Targeted Attack ...\n')
                adv_loss  = self.loss_fn(logits, labels)
            else: # for un-targetted attack
                adv_loss  = -1*self.loss_fn(logits, labels)


            total_loss = adv_loss 
            optimizer.zero_grad()
            total_loss.backward()

            
            # update qunatization tables
            self.q_tables = self.q_tables.detach() - torch.sign(self.q_tables.grad)
            self.q_tables = torch.clamp(self.q_tables, self.q_range[0], self.q_range[1]).detach()
            
            self.alpha += self.alpha_interval

            _, pred_labels = torch.max(logits.data, 1, keepdim=True)

            if self.targeted:
                success_rate = ((pred_labels == labels).sum()/labels.numel()).cpu().detach().numpy()
            else:
                success_rate = ((pred_labels != labels).sum()/labels.numel()).cpu().detach().numpy()

            if self.verbose and (i==0 or (i+1)%10 == 0): print('Step: ', i+1, "  Loss: ", round(total_loss.item(),5), "  Current Success Rate: ", round(success_rate*100,4), '%' )     
                
            if success_rate >= 1:
                print('Ending at Step={} with Success Rate={}'.format(i+1, success_rate))
                adv_images = torch.clamp(adv_images, min=0, max=1.0).detach()
                return adv_images, pred_labels, self.q_tables.detach()       


        adv_images = torch.clamp(adv_images, min=0, max=1.0).detach()
       
        return adv_images, pred_labels, self.q_tables.detach()





if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    