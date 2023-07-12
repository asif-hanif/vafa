import torch
import warnings

def projected_gradient_descent_l_inf(model, images, labels, loss_fn, steps=20, alpha=2/255, eps=8/255, random_start=True, device=None, targeted=False, verbose=True):
    
    if verbose:
        print(f"\nPGD: alpha={alpha} , eps={eps*255} , steps={steps} , targeted={targeted}\n")
        if images.max()>1 or images.min()<0 : warnings.warn(f"PGD Attack: Image values are expected to be in the range of [0,1], instead found [min,max]=[{images.min().item()} , {images.max().item()}]")

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    adv_images = images.clone().detach()

    if random_start:
        # starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps,eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()


    for i in range(steps):

        adv_images.requires_grad = True

        adv_logits = model(adv_images)

        # calculate loss
        if targeted:
            loss = -1*loss_fn(adv_logits, labels)
        else:
            loss = loss_fn(adv_logits, labels)

        if verbose: 
            if i==0 or (i+1)%10 == 0: print("Step:", str(i+1).zfill(3), " ,   Loss:", f"{round(loss.item(),5):3.5f}" )

        # update adversarial images
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha*grad.sign()

        delta = torch.clamp(adv_images - images, min=-eps, max=eps)

        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images







