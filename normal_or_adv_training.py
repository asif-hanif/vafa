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
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed


from unetr import UNETR
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from trainer import run_training

from utils.get_args import get_args 
from utils.data_utils import get_loader_btcv
from utils.data_utils import get_loader_acdc
from utils.utils import MyOutput
from utils.utils import print_attack_info
from utils.utils import get_folder_name

from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction


def main():
    now_start = datetime.now() 

    args = get_args()
    args.amp = not args.noamp
    args.now_start = now_start

    if args.adv_training_mode:
        # folder for saving model
        folder_name = get_folder_name(args)
    else:
        folder_name = ""


    args.folder_name  = folder_name

    if args.resume:
        save_model_dir_ext = os.path.join(args.checkpoint_dir, "" if args.no_sub_dir_model else args.folder_name) 
        args.logdir = save_model_dir_ext
    else:
        save_model_dir_ext = os.path.join(args.save_model_dir, "" if args.no_sub_dir_model else args.folder_name) 
        args.logdir = save_model_dir_ext


    # folder will not be created if either debugging or resuming
    if not (args.debugging or args.resume):
        # create folder for saving results
        os.mkdir(save_model_dir_ext)
        
        # save argparse file content
        with open(f"{os.path.join(save_model_dir_ext, 'args.json')}", 'wt') as f:
            json.dump(vars(args),f, indent=4, default=str)


    # log will not be saved if debugging
    if not args.debugging:
        # keep the terminal output on console and also save it to a file
        sys.stdout = MyOutput(f"{os.path.join(save_model_dir_ext, 'log.out' )}")


    print("\n\n", "".join(["#"]*130), "\n", "".join(["#"]*130), "\n\n""")                                                   
    print(f"HostName     =  {os.uname()[1]}")                   
    print(f'Time & Date  =  {now_start.strftime("%I:%M %p")} , {now_start.strftime("%d_%b_%Y")}\n\n')


    if args.adv_training_mode:
        print(f"Adversarial-Training of '{args.model_name.upper()}' Model under following Attack:")
        print_attack_info(args)
    else:
        print(f"\nTraining the '{args.model_name.upper()}'  Model ... ")



    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("\nNum. of GPUs = ", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)


def main_worker(gpu, args):

    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)

    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)

    args.gpu = gpu

    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False


    if args.dataset == 'btcv':
        print("\nDataset = BTCV\n")
        loader = get_loader_btcv(args)
    else: 
        raise ValueError(f"Unsupported Dataset: '{args.dataset}' .")

    

    print("\nRank =", args.rank, " ,  GPU =", args.gpu)

    if args.rank == 0: print("BatchSize:", args.batch_size, " ,  Epochs:", args.max_epochs, "\n")

    inf_size = [args.roi_x, args.roi_y, args.roi_z]


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
    

    start_epoch = 0
    best_acc = 0


    if args.use_pretrained:
        pretrained_path  = args.pretrained_path
        checkpoint_dict = torch.load(pretrained_path)
        model.load_state_dict(checkpoint_dict["model_state_dict"] if "model_state_dict" in checkpoint_dict.keys() else  checkpoint_dict["state_dict"])
        print(f"\nLoading Pre-trained Model Weights from:  {pretrained_path}\n")


    if args.resume:
        checkpoint_dir_ext = os.path.join(args.checkpoint_dir, "" if args.no_sub_dir_model else args.folder_name) 
        if args.resume_latest: checkpoint_path  = os.path.join(checkpoint_dir_ext, 'model_latest.pt')
        if args.resume_best: checkpoint_path  = os.path.join(checkpoint_dir_ext, 'model_best.pt')

        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint_dict["model_state_dict"] if "model_state_dict" in checkpoint_dict.keys() else  checkpoint_dict["state_dict"])

        if args.resume_but_restart:
            start_epoch = 0
            best_acc = 0
        else:
            start_epoch = checkpoint_dict["epoch"]+1
            best_acc = checkpoint_dict["best_acc"]

        print(f"\nResuming Training ...")
        print(f"Resume Checkpoint Path: {checkpoint_path}")
        print(f"Resume={args.resume}\nRestart={args.resume_but_restart}")
        print(f"Start Epoch={start_epoch}")
        if "epoch_acc" in checkpoint_dict.keys(): print(f"Accuracy (at Epoch={start_epoch-1})={checkpoint_dict['epoch_acc']:0.6f}")
        print(f"Best Accuracy={best_acc:0.6f}\n")
        

        pretrained_path = checkpoint_path
    

    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap)



    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Model Parameters = {model_total_params:,}\n")

    model.cuda(args.gpu)


    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)


    ## optimizer
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight)
    else:
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))


    # load optimizer state if resume
    if args.resume and not args.resume_but_restart: 
        print(f"Loading optimizer state_dict from: {pretrained_path}")
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"] if "optimizer_state_dict" in checkpoint_dict.keys() else  checkpoint_dict["optimizer"])


    ## scheduler
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs)
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    else:
        scheduler = None


    # load scheduler state if resume
    if args.resume and not args.resume_but_restart:
        print(f"Loading scheduler state_dict from: {pretrained_path}")
        scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"] if "scheduler_state_dict" in checkpoint_dict.keys() else  checkpoint_dict["scheduler"])


    dice_loss = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr)

    post_label = AsDiscrete(to_onehot=args.out_channels, n_classes=args.out_channels)
    post_pred  = AsDiscrete(argmax=True, to_onehot=args.out_channels, n_classes=args.out_channels)
    dice_acc   = DiceMetric(include_background=True, reduction=MetricReduction.MEAN, get_not_nans=True)



    accuracy = run_training(
        model=model,
        train_loader=loader[0],
        val_loader=loader[1],
        optimizer=optimizer,
        loss_func=dice_loss,
        acc_func=dice_acc,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        best_acc = best_acc,
        post_label=post_label,
        post_pred=post_pred)



    print("\n", "".join(["#"]*130), "\n", "".join(["#"]*130))

    if args.adv_training_mode:
        print(f"\n Adversarial-Training of '{args.model_name.upper()}' Model completed under following Attack:")
        print_attack_info(args)

        print(" Model Weights Loaded from Path before Adversarial Training:" , pretrained_path)
        print(" Adversarially Trained Model Weights Saved at Path:" , args.logdir)


    now_end = datetime.now()
    print(f'\nTime & Date  =  {now_end.strftime("%I:%M %p")} , {now_end.strftime("%d_%b_%Y")}\n')

    duration = now_end - args.now_start
    duration_in_s = duration.total_seconds() 

    days    = divmod(duration_in_s, 86400)       # Get days (without [0]!)
    hours   = divmod(days[1], 3600)              # Use remainder of days to calc hours
    minutes = divmod(hours[1], 60)               # Use remainder of hours to calc minutes
    seconds = divmod(minutes[1], 1)              # Use remainder of minutes to calc seconds

    print(f"Total Time =>   {int(days[0])} Days : {int(hours[0])} Hours : {int(minutes[0])} Minutes : {int(seconds[0])} Seconds \n\n")

    print("\n", "".join(["#"]*130), "\n", "".join(["#"]*130))

    return accuracy

    


if __name__ == "__main__":
    main()