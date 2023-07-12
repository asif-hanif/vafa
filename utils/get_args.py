import argparse 

def get_args():
    parser = argparse.ArgumentParser(description="Segmentation Pipeline")

    parser.add_argument("--use_pretrained", action="store_true", help="model will be initialized from saved pre-trained checkpoint.")
    parser.add_argument("--pretrained_path", default="", type=str, help="full path of pre-trained checkpoint")
    parser.add_argument("--resume", action="store_true", help="resume training from a checkpoint")
    parser.add_argument("--resume_latest", action="store_true", help="resume training from latest checkpoint")
    parser.add_argument("--resume_best", action="store_true", help="resume training from best checkpoint")
    parser.add_argument("--resume_but_restart", action="store_true", help="resume training from the checkpoint but set start_epoch=0")

    parser.add_argument("--logdir", default="None", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--data_dir", default="None", type=str, help="dataset directory")
    parser.add_argument("--json_list", default="None", type=str, help="dataset json file")
    parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")


    parser.add_argument("--max_epochs", default=5000, type=int, help="max number of training epochs")
    parser.add_argument("--val_every", default=100, type=int, help="validation frequency")
    parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
    parser.add_argument("--sw_batch_size", default=1, type=int, help="number of sliding window batch size (during inference)")
    parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
    parser.add_argument("--reg_weight", default=1e-5, type=float, help="regularization weight")
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--noamp", action="store_true", help="do not use amp for training")
    parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
    parser.add_argument("--warmup_epochs", default=50, type=int, help="number of warmup epochs")
    

    parser.add_argument("--distributed", action="store_true", help="start distributed training")
    parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--workers", default=8, type=int, help="number of workers")


    parser.add_argument("--model_name", default="None", type=str, help="model name")
    parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
    parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
    parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
    parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
    parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
    parser.add_argument("--feature_size", default=16, type=int, help="feature size dimention")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
    parser.add_argument("--res_block", action="store_true", help="use residual blocks")
    parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory (Swin-UNETR)")


    parser.add_argument("--dataset", default=None, type=str, help="name of the dataset")
    parser.add_argument("--use_normal_dataset", action="store_true", help="use monai Dataset class")
    parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
    parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
    parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
    parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
    parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    


    # attack related arguments
    parser.add_argument("--attack_name",  default=None, type=str,  help="name of adversarial attack")
    parser.add_argument("--steps", default=20, type=int,  help="number of iterations to generate adversarial example")
    parser.add_argument("--alpha", default=0.01, type=float,  help="step size for update during attack")
    parser.add_argument("--eps", default=4, type=float,  help="perturbation budget on the scale of [0,255]")
    parser.add_argument("--std", default=4, type=float,  help="standard deviation for gaussian noise on the scale of [0,255]")
    parser.add_argument("--targeted", action='store_true',  help="if targeted attack is to be chosen")

    # vafa attack related arguments
    parser.add_argument("--q_max", default=20, type=float,  help="upper bound on quantization table values in VAFA attack")

    # ssa attack related arguments
    parser.add_argument("--num_spectrum_augs", default=10, type=int,  help="number of spectrum augmentations in SSA attack")

 
    # advdrop related arguments
    parser.add_argument("--rho", default=0.2, type=float,  help="budget of freq-domain perturbation matrix")
    parser.add_argument("--use_ssim_loss", action='store_true',  help="if SSIM loss is to be used in adversarial loss")
    parser.add_argument("--lambda_dice", default=0.01, type=float,  help="lambda for adversarial dice loss")
    parser.add_argument("--lambda_ssim", default=1, type=float,  help="lambda for ssim loss")
    parser.add_argument("--block_size",  default=[8,8,8] , type=int, nargs="+",  help="DCT block size")
    parser.add_argument("--freq_band",  default='all', type=str, choices=['all', 'low', 'middle', 'high'], help="which frequency band should be perturbed in SPA")


    # different modes
    parser.add_argument("--gen_train_adv_mode", action='store_true',  help="if adversarial versions of train samples are to be generated")
    parser.add_argument("--gen_val_adv_mode", action='store_true',  help="if adversarial versions of validation/test samples are to be generated")
    parser.add_argument("--adv_training_mode", action='store_true',  help="if adversarial training is to be performed. adv-images are created during training.")
    parser.add_argument("--freq_reg_mode", action='store_true',  help="adversarial training with frequency regularization term in loss function...")

    # directories
    parser.add_argument("--adv_images_dir", default="", type=str, help="parent directory containing adversarial images")
    parser.add_argument("--save_adv_images_dir", default=None, type=str, help="parent directory to save adversarial images")
    parser.add_argument("--save_model_dir", default=None, type=str, help="parent directory to save model finetuned on adversarial images")
    parser.add_argument("--no_sub_dir_model", action='store_true', help="if mentioned, sub-folder will not be searched in parent direcotry containing model checkpoint")
    parser.add_argument("--no_sub_dir_adv_images", action='store_true', help="if mentioned, sub-folder will not be searched in parent direcotry containing adv-images")

    parser.add_argument("--debugging", action='store_true', help="if mentioned, folders would not be created and results will not be saved.")
    parser.add_argument("--exp_name", default=None, type=str,  help="experiment name: sub-folder in parent directory for saving results")

    args = parser.parse_args()

    args.block_size = tuple(args.block_size)

    # sanity checks on arguments
    assert not (args.resume_latest and not args.resume), "To resume from last checkpoint, '--resume' has to be also True"
    assert not (args.resume_best and not args.resume), "To resume from best checkpoint, '--resume' has to be also True"
    assert not (args.resume_latest and args.resume_best) , "'--resume_latest' and '--resume_best' are mutually exclusive. Use either of them."
    assert not (args.resume and not bool(args.checkpoint_dir) ), "To resume from a checkpoint, '--checkpoint_dir' must be provided"
    assert not (args.freq_reg_mode and not args.adv_training_mode ), "To use frequency-regularization in adversarial training, '--adv_training_mode' must be mentioned"
   
    return args
