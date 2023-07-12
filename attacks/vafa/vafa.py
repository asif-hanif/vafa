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

import warnings


#############################################################################################################
#                                                 VAFA 3D                                                   #
#############################################################################################################

class VAFA(Attack):
    r"""  
    Arguments:
        - model (nn.Module): model to attack
        - loss_fn: loss function
        - batch_size (int): batch size
        - channels (int): number of channels in the image
        - height (int): height of the image
        - width (int): width of the image
        - depth (int): depth of the image
        - steps (int): number of steps 
        - q_max: upper bound on the value of quantization table
        - q_min: lower bound on the value of quantization table
        - block_size (tuple): block/patch size (Block_H, Block_W, Block_D)
        - use_ssim_loss: True when SSIM_Loss is used in the optimization objective
        - targeted: True for targeted attack
    Shapes:
        - images: [B,C,H,W,D] normalized to [0,1]. B=BatchSize, C=Number of Channels,  H=Height,  W=Width, D=Depth
        - labels: [B,C,H,W,D]
    """

    def __init__(self, model,loss_fn, batch_size=1, channels=1, height=96, width=96, depth=96, steps=20, q_min=5, q_max=10,  block_size=(8,8,8), use_ssim_loss=False, targeted=False, verbose=True):
        super(VAFA, self).__init__("VAFA", model)
        self.verbose = verbose

        self.steps = steps
        self.targeted = targeted
        self.block_size = tuple(block_size) 
        self.use_ssim_loss = use_ssim_loss
        self.loss_fn = loss_fn

        self.batch_size = batch_size
        self.channels = channels
        self.height = height
        self.width  = width
        self.depth  = depth



        # parameters for differentiable quantization
        self.alpha_range = [0.1, 1e-20]
        self.alpha = torch.tensor(self.alpha_range[0])
        self.alpha_interval = torch.tensor((self.alpha_range[1] - self.alpha_range[0])/ self.steps)
        
        assert self.height%block_size[0]==0 , f"Height of image should be divisible by block_size[0]"
        assert self.width%block_size[1]==0  , f"Width of image should be divisible by block_size[1]"
        assert self.depth%block_size[2]==0  , f"Depth of image should be divisible by block_size[2]"


        # initialize quantization table
        num_blocks = (self.height*self.width*self.depth)//(np.prod(self.block_size))
        q_tables_shape = (self.batch_size,self.channels,num_blocks)+self.block_size    # [B, C, N_Blocks, Block_H, Block_W, Block_D]
        self.q_tables = torch.full( q_tables_shape, q_max, dtype = torch.float32)      # initialize q_tables with q_max
        
        # learnable quantization table values will be limited to the following range: [min_value, max_value] 
        self.q_range = [q_min, q_max]

 
    def forward(self, images, labels):
        r"""
        images: [B,C,H,W,D] normalized to [0,1]
        labels: [B,C,H,W,D]
        """

        if self.verbose and (images.max()>1 or images.min()<0) : warnings.warn(f"InfoDrop-3D Attack: Image values are expected to be in the range of [0,1], instead found [min,max]=[{images.min().item()} , {images.max().item()}]")

        images   = images.clone().detach().to(self.device)       #  [B,C,H,W,D]
        labels   = labels.clone().detach().to(self.device)       #  [B,C,H,W,D]

        B,C,H,W,D = images.shape

        self.alpha = self.alpha.to(self.device)
        self.alpha_interval = self.alpha_interval.to(self.device)

        optimizer = torch.optim.Adam([self.q_tables], lr= 0.01)
        
        
        for i in range(self.steps):
            # set requires_grad for q_tables 
            self.q_tables.requires_grad = True
            blocks = block_splitting_3d(images*255, self.block_size)                     # [B, C, H, W, D] --> [B, C, N_Blocks, Block_H, Block_W, Block_D]
            blocks_dct = dct_3d(blocks)                                                  # [B, C, N_Blocks, Block_H, Block_W, Block_D] , 3D DCT is computed on last three dimensions 
            blocks_dct_qunatized = quantize(blocks_dct, self.q_tables, self.alpha)       # [B, C, N_Blocks, Block_H, Block_W, Block_D]
            blocks_dct_dequnatized = dequantize(blocks_dct_qunatized, self.q_tables)     # [B, C, N_Blocks, Block_H, Block_W, Block_D]
            blocks_idct = idct_3d(blocks_dct_dequnatized)                                # [B, C, N_Blocks, Block_H, Block_W, Block_D] , 3D IDCT is computed on last three dimensions 
            merged_blocks = block_merging_3d(blocks_idct, images.shape)                  # [B, C, N_Blocks, Block_H, Block_W, Block_D] --> [B,C,H,W,D]


            # adversarially perturbed images
            adv_images = merged_blocks  # [B, C, H, W, D]


            ## normalize each (H,W) slice to [0,1] range
            maxx = torch.amax(adv_images, dim=(2,3), keepdim=True)
            minn = torch.amin(adv_images, dim=(2,3), keepdim=True)
            adv_images = (adv_images - minn)/(maxx-minn)


            logits = self.model(adv_images) # logits: [B,NumClasses,H,W,D] , passing adversarial images through the model 

            if self.targeted: # for targetted attack
                if self.verbose and i==0: print('InfoDrop-3D: Using Targeted Attack ...\n')
                adv_loss  = self.loss_fn(logits, labels)
            else: # for un-targetted attack
                adv_loss  = -1*self.loss_fn(logits, labels)


            if self.use_ssim_loss:
                if self.verbose and i==0: print('InfoDrop-3D: Using Block-wise SSIM Loss ...\n')
                im1 = block_splitting_3d(images, self.block_size).squeeze(1).permute(1,0,4,2,3)      # [B,C=1,H,W,D]  --> [B, C=1, N_Blocks, Block_H, Block_W, Block_D] --> Remove C Dim and Permute Dims --> [N_Blocks, B, Block_D, Block_H, Block_W]
                im2 = block_splitting_3d(adv_images, self.block_size).squeeze(1).permute(1,0,4,2,3)  # [B,C=1,H,W,D]  --> [B, C=1, N_Blocks, Block_H, Block_W, Block_D] --> Remove C Dim and Permute Dims --> [N_Blocks, B, Block_D, Block_H, Block_W]
                ssim_loss = 1-ssim(im1, im2, data_range=1, nonnegative_ssim=True,size_average=True, win_size=3)
                adv_loss = adv_loss + ssim_loss


            total_cost = adv_loss 
            optimizer.zero_grad()
            total_cost.backward()

            
            # update qunatization tables
            self.q_tables = self.q_tables.detach() - torch.sign(self.q_tables.grad)
            self.q_tables = torch.clamp(self.q_tables, self.q_range[0], self.q_range[1]).detach()
            
            self.alpha += self.alpha_interval

            _, pred_labels = torch.max(logits.data, 1, keepdim=True)



            if self.targeted:
                success_rate = ((pred_labels == labels).sum()/labels.numel()).cpu().detach().numpy()
            else:
                success_rate = ((pred_labels != labels).sum()/labels.numel()).cpu().detach().numpy()

            if self.verbose and (i==0 or (i+1)%10 == 0): print('Step: ', i+1, "  Loss: ", round(total_cost.item(),5), "  Current Success Rate: ", round(success_rate*100,4), '%' )     
                
            if success_rate >= 1:
                print('Ending at Step={} with Success Rate={}'.format(i+1, success_rate))
                adv_images = torch.clamp(adv_images, min=0, max=1.0).detach()
                return adv_images, pred_labels, self.q_tables.detach()       


        adv_images = torch.clamp(adv_images, min=0, max=1.0).detach()
       
        return adv_images, pred_labels, self.q_tables.detach()




#############################################################################################################
#                                                 VAFA 2D                                                   #
#############################################################################################################


# Note: This is re-implementation and adaptation of AdvDrop (https://arxiv.org/pdf/2108.09034.pdf) attack involving 2D DCT on each 2D spatial slice of volumetric image.

class VAFA_2D(Attack):
    r"""
    Arguments:
        - model (nn.Module): model to attack
        - loss_fn: loss function
        - batch_size (int): batch size
        - channels (int): number of channels in the image
        - height (int): height of the image
        - width (int): width of the image
        - depth (int): depth of the image
        - steps (int): number of steps 
        - q_max: upper bound on the value of quantization table
        - q_min: lower bound on the value of quantization table
        - block_size (tuple): block/patch size (Block_H, Block_W)
        - targeted: True for targeted attack
    Shapes:
        - images: [B,C,H,W,D] normalized to [0,1]. B=BatchSize, C=Number of Channels,  H=Height,  W=Width, D=Depth
        - labels: [B,C,H,W,D]
    """

    def __init__(self, model,loss_fn, batch_size=1, channels=1, height=96, width=96, depth=96, steps=10, q_min=5, q_max=20, block_size=(8,8), targeted=False, verbose=True):
        super(VAFA_2D, self).__init__("VAFA_2D", model)
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


        # parameters for differentiable quantization
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

        B,C,H,W,D = images.shape

        images = images.permute(0,1,4,2,3)   #  [B,C,H,W,D] --> [B,C,D,H,W]

        self.alpha = self.alpha.to(self.device)
        self.alpha_interval = self.alpha_interval.to(self.device)

        optimizer = torch.optim.Adam([self.q_tables], lr= 0.01)
        
        blocks = block_splitting_2d(images*255, self.block_size)                  # [B, C, D, H, W] --> [B, C, D, N_Blocks, Block_H, Block_W]
        
        for i in range(self.steps):
            # set requires_grad for q_tables 
            self.q_tables.requires_grad = True
            
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


            total_cost = adv_loss 
            optimizer.zero_grad()
            total_cost.backward()

            
            # update qunatization tables
            self.q_tables = self.q_tables.detach() - torch.sign(self.q_tables.grad)
            self.q_tables = torch.clamp(self.q_tables, self.q_range[0], self.q_range[1]).detach()
            
            self.alpha += self.alpha_interval

            _, pred_labels = torch.max(logits.data, 1, keepdim=True)

            if self.targeted:
                success_rate = ((pred_labels == labels).sum()/labels.numel()).cpu().detach().numpy()
            else:
                success_rate = ((pred_labels != labels).sum()/labels.numel()).cpu().detach().numpy()

            if self.verbose and (i==0 or (i+1)%10 == 0): print('Step: ', i+1, "  Loss: ", round(total_cost.item(),5), "  Current Success Rate: ", round(success_rate*100,4), '%' )     
                
            if success_rate >= 1:
                print('Ending at Step={} with Success Rate={}'.format(i+1, success_rate))
                adv_images = torch.clamp(adv_images, min=0, max=1.0).detach()
                return adv_images, pred_labels, self.q_tables.detach()       


        adv_images = torch.clamp(adv_images, min=0, max=1.0).detach()
       
        return adv_images, pred_labels, self.q_tables.detach()





if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    