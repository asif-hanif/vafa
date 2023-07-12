# code adapted from: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/


import torch
import torch.nn as nn
import warnings

def fast_gradient_sign_method_l_inf(model, images, labels, loss_fn, eps=8/255, device=None, targeted=False, verbose=True):
    if verbose:
        print(f"\nFGSM: eps={eps*255} , targeted={targeted}\n")
        if images.max()>1 or images.min()<0 : warnings.warn(f"FGSM Attack: Image values are expected to be in the range of [0,1], instead found [min,max]=[{images.min().item()} , {images.max().item()}]")

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)

    images.requires_grad = True
    logits = model(images)

    # calculate loss
    if targeted:
        loss = -loss_fn(logits, labels)
    else:
        loss = loss_fn(logits, labels)

    # update adversarial images
    grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]

    adv_images = images + eps*grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    return adv_images

