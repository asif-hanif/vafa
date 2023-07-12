import numpy as np 
import json
import os
import sys
import time
import math
import io
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torchvision import models  
import torchvision.datasets as dsets 
import torchvision.transforms as transforms  
from  torchattacks.attack import Attack  
from .utils import *
from .compression import *
from .decompression import *
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

# from torch.autograd import Variable as V

def normalize(images):
    # normalize each (H,W) slice to [0,1] range
    maxx = torch.amax(images, dim=(2,3),keepdim=True)
    minn = torch.amin(images, dim=(2,3),keepdim=True)
    images = (images - minn)/(maxx-minn)
    return images



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




class InfoDrop(Attack):
    r"""    
    Distance Measure : l_inf bound on quantization table
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (DEFALUT: 40)
        batch_size (int): batch size
        q_size: bound for quantization table
        targeted: True for targeted attack
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.   
    """


    def __init__(self, model,loss_fn, height=96, width=96, depth=96, steps=40, batch_size = 20, block_size=8, q_size=10, num_classes=None, targeted=False, use_rel_loss=False, rel_loss_lambda=1.0, use_ssim=False):
        super(InfoDrop, self).__init__("InfoDrop", model)
        self.loss_fn = loss_fn
        self.steps = steps
        self.targeted = targeted
        self.batch_size = batch_size
        self.height = height
        self.width  = width
        self.depth  = depth 

        self.sigma = 16.0
        self.rho = 0.5
        self.max_epsilon = 16.0

        if num_classes is not None: self.num_classes = num_classes   
        else: raise Exception("Specify the number of classes in the dataset.")
        
        self.use_rel_loss = use_rel_loss
        self.rel_loss_lambda = rel_loss_lambda
        
        self.use_ssim = use_ssim
        # value for quantization range
        self.factor_range = [5, q_size]
        # differential quantization
        self.alpha_range = [0.1, 1e-20]
        self.alpha = torch.tensor(self.alpha_range[0])
        self.alpha_interval = torch.tensor((self.alpha_range[1] - self.alpha_range[0])/ self.steps)
        block_n = np.ceil(height / block_size) * np.ceil(height / block_size) 
        q_ini_table = np.empty((batch_size,int(block_n),block_size,block_size), dtype = np.float32)
        
        q_ini_table.fill(q_size)

        # each channel of input will have its own quatization table
        self.q_tables = {}
        for d in range(self.depth):  self.q_tables[d] = torch.from_numpy(q_ini_table)
    


     
    def forward(self, images, labels):
        r"""
        images: [B,C,H,W,D]
        """

        B,C,H,W,D = images.shape

        q_table = None
        
        self.alpha = self.alpha.to(self.device)
        self.alpha_interval = self.alpha_interval.to(self.device)

        images   = images.clone().detach().to(self.device)
        labels   = labels.clone().detach().to(self.device)
        

        images_min = clip_by_tensor(images - self.max_epsilon/255.0, 0.0, 1.0)
        images_max = clip_by_tensor(images + self.max_epsilon/255.0, 0.0, 1.0)


        optimizer = torch.optim.Adam([self.q_tables[d] for d in range(self.depth)], lr= 0.01)
        
        components = {}
        for d in range(self.depth): components[d] = images[:,0,:,:,d] # [B,H,W]  # assuming C=1

        
        for i in range(self.steps):

            # set requires_grad for q_table of each depth channel
            for d in range(self.depth): self.q_tables[d].requires_grad = True
            
            # gauss_noise = (torch.randn(1, 1, 96, 96, 96)*(self.sigma/255)).to(self.device) # gauss_noise ~ N[0, sigma]
            # mask = (torch.rand_like(images) * 2 * self.rho + 1 - self.rho).to(self.device) # mask ~ Unif[1-rho , 1+rho]

            gauss_noise = torch.zeros(images.shape).to(self.device)
            mask = torch.ones(images.shape).to(self.device)

            upresults = {}
            for k in components.keys():
                comp = block_splitting( (components[k]+gauss_noise[:,0,:,:,k])*255 )
                comp = dct_8x8(comp)
                comp = quantize(comp, self.q_tables[k], self.alpha)
                comp = dequantize(comp, self.q_tables[k])
                comp = idct_8x8(comp*mask[:,0,:,:,k].reshape(1,144,8,8))
                merge_comp = block_merging(comp, self.height, self.width)
                upresults[k] = merge_comp/255


            # adversarially perturbed images
            images_new = torch.cat([upresults[d].unsqueeze(3) for d in range(self.depth)],dim=3).unsqueeze(1)  # [B,H,W,D] --> [B,C=1,H,D,W]

                   
            # logits = self.model(normalize(images_new)) # passing adversarial images through model
            logits = self.model(images_new) # passing adversarial images through model

            if self.targeted: # for targetted attack
                if i==0: print('\nUsing Targeted Attack ...\n')
                targ_labels = torch.zeros(images.shape, dtype=torch.int64).to(self.device)
                # rand_labels = torch.from_numpy(np.random.randint(0, self.num_classes, size = images.shape))
                adv_loss  = self.loss_fn(logits, targ_labels)
            else: # for un-targetted attack
                adv_loss  = -1*self.loss_fn(logits, labels)


            total_cost = adv_loss 
            optimizer.zero_grad()
            # images_new.retain_grad()
            total_cost.backward()

            self.alpha += self.alpha_interval


            for k in self.q_tables.keys():
                self.q_tables[k] = self.q_tables[k].detach() -  torch.sign(self.q_tables[k].grad)
                self.q_tables[k] = torch.clamp(self.q_tables[k], self.factor_range[0], self.factor_range[1]).detach()

            
            # noise = images_new.grad.data
            # noise = (self.max_epsilon/(255*self.steps))*torch.sign(noise)
            # for d in range(self.depth): components[d] = clip_by_tensor(components[d]+noise[:,0,:,:,d], images_min[:,0,:,:,d], images_max[:,0,:,:,d])
            

            _, pred_labels = torch.max(logits.data, 1, keepdim=True)

            if self.targeted:
                suc_rate = ((pred_labels == labels).sum()/labels.numel()).cpu().detach().numpy()
            else:
                suc_rate = ((pred_labels != labels).sum()/labels.numel()).cpu().detach().numpy()


            if (i+1)%10 == 0:     
                print('Step: ', i+1, "  Loss: ", round(total_cost.item(),5), "  Current Success Rate: ", round(suc_rate*100,4), '%' )

            # if suc_rate >= 1:
            #     print('End at step {} with Success Rate {}'.format(i+1, suc_rate))
            #     q_images = torch.clamp(images_new, min=0, max=255.0).detach()
            #     return q_images, pred_labels, q_table       
        
        
        # q_images = torch.cat([components[d].unsqueeze(3) for d in range(self.depth)],dim=3).unsqueeze(1)

        return torch.clamp(images_new.detach(),0,1), pred_labels, q_table



def save_img(img, img_name, save_dir):
    create_dir(save_dir)
    img_path = os.path.join(save_dir, img_name)
    img_pil = Image.fromarray(img.astype(np.uint8))
    img_pil.save(img_path)
    
    
def pred_label_and_confidence(model, input_batch, labels_to_class):
    input_batch = input_batch.cuda()
    with torch.no_grad():
        out = model(input_batch)
    _, index = torch.max(out, 1)

    percentage = torch.nn.functional.softmax(out, dim=1) * 100
    # print(percentage.shape)
    pred_list = []
    for i in range(index.shape[0]):
        pred_class = labels_to_class[index[i]]
        pred_conf =  str(round(percentage[i][index[i]].item(),2))
        pred_list.append([pred_class, pred_conf])
    return pred_list

   

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    