# code adapted from: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/


import torch
import warnings

def gaussain_noise(images, std=8/255, device=None, verbose=True):

    if verbose:
        print(f"\nGN: std={std*255}\n")
        if images.max()>1 or images.min()<0 : warnings.warn(f"GN Attack: Image values are expected to be in the range of [0,1], instead found [min,max]=[{images.min().item()} , {images.max().item()}]")

    images = images.clone().detach().to(device)
    adv_images = images + std*torch.randn_like(images)
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    return adv_images
