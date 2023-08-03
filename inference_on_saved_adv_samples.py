# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import json
from datetime import datetime


import numpy as np
import torch
from unetr import UNETR



from utils.get_args import get_args 
from utils.data_utils import get_loader_btcv
from utils.data_utils import get_loader_acdc
from utils.utils import MyOutput
from utils.utils import print_attack_info
from utils.utils import get_folder_name


from attacks import vafa
from attacks.pgd import projected_gradient_descent_l_inf as pgd_l_inf
from attacks.fgsm import fast_gradient_sign_method_l_inf as fgsm_l_inf
from attacks.bim import basic_iterative_method_l_inf as bim_l_inf
from attacks.gn import gaussain_noise as gn


import monai
from monai.inferers import sliding_window_inference
from monai.utils.misc import fall_back_tuple
from monai.data.utils import dense_patch_slices


from monai.metrics import DiceMetric
from monai.metrics import HausdorffDistanceMetric
from monai.transforms import AsDiscrete
from monai.utils.enums import MetricReduction
from monai.data import decollate_batch


from collections import defaultdict
import nibabel as nib


from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

import lpips
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization



def get_slices(input_shape, roi_size ):
    # input_shape = (B,C,H,W,D)
    # roi_size = (roi_x, roi_y, roi_z)
    num_spatial_dims = len(input_shape) - 2  
    image_size = input_shape[2:]
    roi_size = fall_back_tuple(roi_size, image_size)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size[i], roi_size[i]) for i in range(num_spatial_dims))
    scan_interval = roi_size
    # store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    return slices


    
def main():

    now_start = datetime.now() 
    print('\n\n')                                                   
    print(f"HostName     =  {os.uname()[1]}")                   
    print(f'Time & Date  =  {now_start.strftime("%I:%M %p")} , {now_start.strftime("%d_%b_%Y")}\n\n')

    args = get_args()
    args.test_mode = True
    
    assert args.use_pretrained, " '--use_pretrained' needs to be mentioned"
    assert args.pretrained_path, "'--pretrained_path' needs to be specified" 


    # folder containing saving adversarial images 
    folder_name = get_folder_name(args)


    adv_imgs_dir_ext = os.path.join(args.adv_images_dir, ""  if args.no_sub_dir_adv_images else folder_name)

    if args.dataset == 'btcv':
        data_loader = get_loader_btcv(args)
    else: 
        raise ValueError(f"Unsupported Dataset: '{args.dataset}' .")

    print(f"\nDataset = {args.dataset.upper()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    if args.model_name == "unet-r":
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate)
    else:
        raise ValueError("Unsupported model " + str(args.model_name))
    
    
        
    pretrained_path  = args.pretrained_path

    print(f"\nModel   = {args.model_name.upper()} ")
    print(f"\nLoading Model Weights from:  {pretrained_path}\n")
    checkpoint_dict = torch.load(pretrained_path)
    model.load_state_dict(checkpoint_dict["model_state_dict"] if "model_state_dict" in checkpoint_dict.keys() else  checkpoint_dict["state_dict"])


    model.eval()
    model.to(device)

    loss_fn = monai.losses.DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0.0, smooth_dr=1e-6)

    transform_true_label = AsDiscrete(to_onehot=args.out_channels, n_classes=args.out_channels)
    transform_pred_label = AsDiscrete(argmax=True, to_onehot=args.out_channels, n_classes=args.out_channels)

    dice_score_monai = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)
    hd95_score_monai = HausdorffDistanceMetric(include_background=True, distance_metric='euclidean', percentile=95, directed=False, reduction=MetricReduction.MEAN, get_not_nans=True)


    dice_organ_dict_clean = {}
    dice_organ_dict_adv   = {}

    hd95_organ_dict_clean = {}
    hd95_organ_dict_adv   = {}

    lpips_alex_dict = {}

    voxel_success_rate_list    = []
    print("\n\n")

    for i, batch in enumerate(data_loader):
        # if i >0: break

        # get clean images
        val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())  # Image [Min,Max]=[0,1]

        img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
        lbl_name = batch["label_meta_dict"]["filename_or_obj"][0].split("/")[-1]


        ## load adv-image
        adv_val_inputs_path = os.path.join(adv_imgs_dir_ext, "imagesTsAdv", f"adv_{img_name}")
        print(f"Image Name = {img_name}\nLoading Adversarial Image :{adv_val_inputs_path}")
        adv_val_inputs = nib.load(adv_val_inputs_path).get_fdata()/255.0  # Image Shape=[H,W,D]   [Min,Max]=[0,1]
        adv_val_inputs = torch.tensor(adv_val_inputs).unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32) # Image Shape=[B,C,H,W,D]   [Min,Max]=[0,1]
   

        # TO DO: think about overlaping regions

        # inference on whole volume of input data
        with torch.no_grad():
            # inference on clean inputs
            val_logits       = sliding_window_inference(val_inputs, (96, 96, 96), 12, model, overlap=args.infer_overlap)
            val_scores       = torch.softmax(val_logits, 1).cpu().numpy()
            val_labels_clean = np.argmax(val_scores, axis=1).astype(np.uint8)

            # inference on adversarial inputs
            val_logits_adv  = sliding_window_inference(adv_val_inputs, (96, 96, 96), 12 , model, overlap=args.infer_overlap)
            val_scores_adv  = torch.softmax(val_logits_adv, 1).cpu().numpy()
            val_labels_adv  = np.argmax(val_scores_adv, axis=1).astype(np.uint8)
             
            # ture labels
            val_labels  = val_labels.cpu().numpy().astype(np.uint8)[0]
        

            ## Ground Truth
            val_true_labels_list    = decollate_batch(batch["label"].cuda())
            val_true_labels_convert = [transform_true_label(val_label_tensor) for val_label_tensor in val_true_labels_list]

            ## Clean Predictions
            val_clean_pred_labels_list    = decollate_batch(val_logits)
            val_clean_pred_labels_convert = [transform_pred_label(val_pred_tensor) for val_pred_tensor in val_clean_pred_labels_list]

            ## Adv Predictions
            val_adv_pred_labels_list    = decollate_batch(val_logits_adv)
            val_adv_pred_labels_convert = [transform_pred_label(val_pred_tensor) for val_pred_tensor in val_adv_pred_labels_list]


            ## MONAI DICE Score
            dice_clean = dice_score_monai(y_pred=val_clean_pred_labels_convert, y=val_true_labels_convert)
            dice_adv   = dice_score_monai(y_pred=val_adv_pred_labels_convert, y=val_true_labels_convert)

            dice_organ_dict_clean[img_name] = dice_clean[0].tolist()
            dice_organ_dict_adv[img_name] = dice_adv[0].tolist()


            ## MONAI HD95 Score
            hd95_score_clean = hd95_score_monai(y_pred=val_clean_pred_labels_convert, y=val_true_labels_convert)
            hd95_score_adv   = hd95_score_monai(y_pred=val_adv_pred_labels_convert, y=val_true_labels_convert)

            hd95_organ_dict_clean[img_name] = hd95_score_clean[0].tolist()
            hd95_organ_dict_adv[img_name] = hd95_score_adv[0].tolist()

 
            img = val_inputs[0,0].permute(2,0,1).unsqueeze(1).float().cpu()
            adv = adv_val_inputs[0,0].permute(2,0,1).unsqueeze(1).float().cpu()
            lpips_alex_dict[img_name] = 1-loss_fn_alex((2*img-1),(2*adv-1)).view(-1,).mean().item()


            voxel_suc_rate = (val_labels_clean!=val_labels_adv).sum()/np.prod(val_labels_clean.shape)
            voxel_success_rate_list.append(voxel_suc_rate)

            print(f"Adv Attack Success Rate (voxel): {round(voxel_suc_rate*100,3)}  (%)")
            print(f"Mean Organ Dice (Clean): {round(np.nanmean(dice_organ_dict_clean[img_name])*100,2):.2f} (%)        Mean Organ HD95 (Clean): {round(np.nanmean(hd95_organ_dict_clean[img_name]),2)}")                
            print(f"Mean Organ Dice (Adv)  : {round(np.nanmean(dice_organ_dict_adv[img_name])*100,2):.2f} (%)        Mean Organ HD95 (Adv)  : {round(np.nanmean(hd95_organ_dict_adv[img_name]),2)}")
            print(f"LPIPS_Alex: {round(lpips_alex_dict[img_name],4)}")

            print("\n\n")


    dice_clean_all = []
    dice_adv_all   = []
    for key in dice_organ_dict_clean.keys(): dice_clean_all.append(np.nanmean(dice_organ_dict_clean[key]))
    for key in dice_organ_dict_adv.keys(): dice_adv_all.append(np.nanmean(dice_organ_dict_adv[key]))

    hd95_clean_all = []
    hd95_adv_all   = []
    for key in hd95_organ_dict_clean.keys(): hd95_clean_all.append(np.nanmean(hd95_organ_dict_clean[key]))
    for key in hd95_organ_dict_adv.keys(): hd95_adv_all.append(np.nanmean(hd95_organ_dict_adv[key]))



    print("\n", "".join(["#"]*130), "\n", "".join(["#"]*130))
    
    print(f"\n Model = {args.model_name.upper()} \n")
    print(" Model Weights Path:" , pretrained_path)
    print(f"\n Dataset = {args.dataset.upper()}")

    print(f"\n Path of Adversarial Images = {adv_imgs_dir_ext}")


    print("\n Attack Info:")
    print_attack_info(args)

    print('\n')
    print(f" Overall Mean Dice (Clean): {round(np.mean(dice_clean_all)*100,3):0.3f}  (%)" )
    print(f" Overall Mean Dice (Adv)  : {round(np.mean(dice_adv_all)*100,3):0.3f}  (%)" )
    
    print('\n')
    print(f" Overall Mean HD95 (Clean): {round(np.mean(hd95_clean_all),3):0.3f}" )
    print(f" Overall Mean HD95 (Adv)  : {round(np.mean(hd95_adv_all),3):0.3f}" )


 
    lpips_alex_all = []
    for key in lpips_alex_dict.keys(): lpips_alex_all.append(lpips_alex_dict[key])

    print('\n')
    print(f" Overall LPIPS_Alex: {round(np.mean(lpips_alex_all),4):0.4f}")


    now_end = datetime.now()
    print(f'\n Time & Date  =  {now_end.strftime("%I:%M %p")} , {now_end.strftime("%d_%b_%Y")}\n')

    duration = now_end - now_start
    duration_in_s = duration.total_seconds() 

    days    = divmod(duration_in_s, 86400)       # Get days (without [0]!)
    hours   = divmod(days[1], 3600)              # Use remainder of days to calc hours
    minutes = divmod(hours[1], 60)               # Use remainder of hours to calc minutes
    seconds = divmod(minutes[1], 1)              # Use remainder of minutes to calc seconds

    print(f" Total Time =>   {int(days[0])} Days : {int(hours[0])} Hours : {int(minutes[0])} Minutes : {int(seconds[0])} Seconds \n\n")

    print("", "".join(["#"]*130), "\n", "".join(["#"]*130),"\n")
    print(" Done!\n")




if __name__ == "__main__":
    main()
