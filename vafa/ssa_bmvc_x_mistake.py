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


from torch.autograd import Variable as V



def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result





#############################################################################################################
#                                                 SSA 2D                                                   #
#############################################################################################################


# Note: This is re-implementation and adaptation of SSA (https://arxiv.org/pdf/2207.05382.pdf) attack involving 2D DCT on each 2D spatial slice of volumetric image.
class SSA_2D(Attack):
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

    def __init__(self, model,loss_fn, batch_size=1, channels=1, height=96, width=96, depth=96, steps=10, rho=0.5, eps=16/255.0, num_spectrum_transforms=10, block_size=(96,96), targeted=False, verbose=True):
        super(SSA_2D, self).__init__("SSA_2D", model)
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


        # number of augmentations 
        self.num_spectrum_transforms = num_spectrum_transforms

        self.rho = rho
        self.eps = eps
        self.sigma = eps
        self.alpha = eps/steps


        assert self.height%block_size[0]==0 , f"Height of image should be divisible by block_size[0]"
        assert self.width%block_size[1]==0  , f"Width of image should be divisible by block_size[1]"
        assert isinstance(self.block_size, tuple) and len(self.block_size)==2, f"Block size should be a tuple of length 2: (Block_H, Block_W). Instead got {self.block_size}"
    


        num_blocks = (self.height*self.width)//(np.prod(self.block_size))
        blocks_shape = (self.batch_size,self.channels,self.depth,num_blocks)+self.block_size    # [B, C, D, N_Blocks, Block_H, Block_W]

 
    def forward(self, images, labels):
        r"""
        images: [B,C,H,W,D] normalized to [0,1]
        labels: [B,C,H,W,D]
        """
        

        if self.verbose and (images.max()>1 or images.min()<0) : warnings.warn(f"InfoDrop-2D Attack: Image values are expected to be in the range of [0,1], instead found [min,max]=[{images.min().item()} , {images.max().item()}]")

        images   = images.clone().detach().to(self.device)    #  [B,C,H,W,D]
        labels   = labels.clone().detach().to(self.device)    #  [B,C,H,W,D]

        B,C,D,H,W = images.shape

        images_min = clip_by_tensor(images - self.eps, 0.0, 1.0)
        images_max = clip_by_tensor(images + self.eps, 0.0, 1.0)


        for i in range(self.steps):
            noise = 0
            for j in range(self.num_spectrum_transforms):
                images = images.permute(0,1,4,2,3)   #  [B,C,H,W,D] --> [B,C,D,H,W]
                gauss_noise = torch.randn(images.shape,device=self.device) * (self.sigma)
                blocks = block_splitting_2d((images+gauss_noise)*255, self.block_size)          # [B, C, D, H, W] --> [B, C, D, N_Blocks, Block_H, Block_W]
                blocks_dct = dct_2d(blocks)                                                     # [B, C, D, N_Blocks, Block_H, Block_W] , 2D DCT is computed on last 2 dimensions 
                spectrum_perturbation = (torch.rand_like(blocks_dct, device=self.device) * 2 * self.rho + 1 - self.rho) # mask ~ Unif[1-rho , 1+rho]
                blocks_idct = idct_2d(blocks_dct*spectrum_perturbation)                         # [B, C, D, N_Blocks, Block_H, Block_W] , 2D IDCT is computed on last 2 dimensions 
                merged_blocks = block_merging_2d(blocks_idct, images.shape)                     # [B, C, D, N_Blocks, Block_H, Block_W] --> [B,C,D,H,W]


                # adversarially perturbed images
                images = merged_blocks/255.0 # [B,C,D,H,W]
                images = images.permute(0,1,3,4,2) # [B,C,D,H,W] -->  [B,C,H,W,D]

                images = V(images, requires_grad = True)

                logits = self.model(images) # logits: [B,NumClasses,H,W,D] , passing adversarial images through the model 

                if self.targeted: # for targetted attack
                    if self.verbose and i==0: print('InfoDrop-2D: Using Targeted Attack ...\n')
                    adv_loss  = -1*self.loss_fn(logits, labels)
                else: # for un-targetted attack
                    adv_loss  = self.loss_fn(logits, labels)


                total_cost = adv_loss 
                total_cost.backward()

                noise = noise + images.grad.data


            noise = noise/self.num_spectrum_transforms        

            images = images + self.alpha*torch.sign(noise)

            images = clip_by_tensor(images, images_min, images_max)


            with torch.no_grad():
                logits = self.model(images)
                _, pred_labels = torch.max(logits.data, 1, keepdim=True)


            if self.targeted:
                success_rate = ((pred_labels == labels).sum()/labels.numel()).cpu().detach().numpy()
            else:
                success_rate = ((pred_labels != labels).sum()/labels.numel()).cpu().detach().numpy()

            if self.verbose and (i==0 or (i+1)%10 == 0): print('Step: ', i+1, "  Loss: ", round(total_cost.item(),5), "  Current Success Rate: ", round(success_rate*100,4), '%' )     
                
            if success_rate >= 1:
                print('Ending at Step={} with Success Rate={}'.format(i+1, success_rate))
                return images, pred_labels       


        return images.detach(), pred_labels


 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    