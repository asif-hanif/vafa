# code adapted from: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/


import torch
import torch.nn as nn
import warnings

def basic_iterative_method_l_inf(model, images, labels, loss_fn, steps=20, alpha=2/255, eps=8/255, device=None, targeted=False, verbose=True):

    if verbose:
        print(f"\nBIM: alpha={alpha} , eps={eps*255} , steps={steps} , targeted={targeted}\n")
        if steps == 0: steps = int(min(eps*255 + 4, 1.25*eps*255))

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    ori_images = images.clone().detach()

    for i in range(steps):

        images.requires_grad = True

        logits = model(images)

        # calculate loss
        if targeted:
            loss = -1*loss_fn(logits, labels)
        else:
            loss = loss_fn(logits, labels)

        if verbose:
            if i==0 or (i+1)%10 == 0: print("Step:", str(i+1).zfill(3), " ,   Loss:", f"{round(loss.item(),5):3.5f}" )

        # update adversarial images
        grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]

        adv_images = images + alpha*grad.sign()
        
        a = torch.clamp(ori_images - eps, min=0)
        b = (adv_images >= a).float()*adv_images + (adv_images < a).float()*a
        c = (b > ori_images+eps).float()*(ori_images+eps) + (b <= ori_images + eps).float()*b
        
        images = torch.clamp(c, max=1).detach()
        

    return images    








