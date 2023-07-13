# Standard Libraries
import os
import numpy as np

# PyTorch
import torch
import torch.nn as nn



# Differentiable Rounding Function (Source:  https://github.com/RjDuan/AdvDrop/blob/main/utils.py)
def phi_diff(x, alpha):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = x.to(device)

    alpha = torch.where(alpha >= 2.0, torch.tensor([2.0]).cuda(), alpha)

    s = 1/(1-alpha).to(device)

    k = torch.log(2/alpha -1).to(device) 

    phi_x = torch.tanh((x - (torch.floor(x) + 0.5)) * k) * s

    x_ = (phi_x + 1)/2 + torch.floor(x)

    return x_



# (Source:  https://github.com/RjDuan/AdvDrop/blob/main/utils.py)
def diff_round(x): 
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    return torch.round(x) + (x - torch.round(x))**3

