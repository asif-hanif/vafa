# Frequency Domain Adversarial Training for Robust Volumetric Medical Segmentation (MICCAI'23)

> [**Frequency Domain Adversarial Training for Robust Volumetric Medical Segmentation**](https://link.springer.com/chapter/10.1007/978-3-031-43895-0_43)<br>
> [Asif Hanif](https://scholar.google.com/citations?hl=en&user=6SO2wqUAAAAJ), 
[Muzammal Naseer](https://scholar.google.ch/citations?user=tM9xKA8AAAAJ&hl=en),
[Salman Khan](https://salman-h-khan.github.io),
[Mubarak Shah](https://scholar.google.com/citations?user=p8gsO3gAAAAJ&hl=en)
and [Fahad Shahbaz Khan](https://scholar.google.es/citations?user=zvaeYnUAAAAJ&hl=en)


[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2307.07269)
[![video](https://img.shields.io/badge/Video-Presentation-F9D371)](miscellaneous/to_be_announced.md)
[![slides](https://img.shields.io/badge/Presentation-Slides-B762C1)](miscellaneous/to_be_announced.md)



<hr />

| ![main figure](/media/vafa_vaft.png)|
|:--| 
| **Overview of Adversarial Frequency Attack and Training**: A model trained on voxel-domain adversarial attacks is vulnerable to frequency-domain adversarial attacks. In our proposed adversarial training method, we generate adversarial samples by perturbing their frequency-domain representation using a novel module named "Frequency Perturbation". The model is then updated while minimizing the dice loss on clean and adversarially perturbed images. Furthermore, we propose a frequency consistency loss to improve the model performance. |


> **Abstract:** <p style="text-align: justify;">*It is imperative to ensure the robustness of deep learning models in critical applications such as, healthcare. While recent advances in deep learning have improved the performance of volumetric medical image segmentation models, these models cannot be deployed for real-world applications immediately due to their vulnerability to adversarial attacks. We present a 3D frequency domain adversarial attack for volumetric medical image segmentation models and demonstrate its advantages over conventional input or voxel domain attacks. Using our proposed attack, we introduce a novel frequency domain adversarial training approach for optimizing a robust model against voxel and frequency domain attacks.  Moreover, we propose frequency consistency loss to regulate our frequency domain adversarial training that achieves a better tradeoff between model's performance on clean and adversarial samples.* </p>
<hr />

## Brief Description
In the context of 2D natural images, it has been recently observed that frequency-domain based adversarial attacks are more effective against the defenses that are primarily designed to *undo* the impact of pixel-domain adversarial noise in natural images. Motivated by this observation in 2D natural images, here we explore the effectiveness of frequency-domain based adversarial attacks in the regime of volumetric medical image segmentation and aim to obtain a volumetric medical image segmentation model that is robust against adversarial attacks. To achieve this goal, we propose a *min-max* objective for adversarial training of volumetric medical image segmentation model in frequency-domain. 

</br>

> **Volumetric Adversaral Frequency Attack (VAFA)**: For *maximization* step, we introduce **V**olumetric **A**dversarial **F**requency **A**ttack - **VAFA** which operates in the frequency-domain of the data (unlike other prevalent voxel-domain attacks) and explicitly takes into account the 3D nature of the volumetric medical data to achieve higher fooling rate. The proposed **VAFA** transforms the 3D patches of input volumetric medical image into frequency-domain by employing 3D discrete cosine transform (3D-DCT) and perturbs the DCT coefficients via a learnable *quantization* table and then converts the perturbed frequency-domain data back into voxel-domain through inverse 3D-DCT. To preserve structural information in adversarial sample, we incorporate SSIM loss along with adversarial loss which helps us attain better SSIM and LPIPS. We optimize following objective to generate adversarial sample:

```math
\begin{equation} 
\begin{gathered}
\underset{ \boldsymbol{\mathrm{q}} }{\mathrm{maximize}}~~ \mathcal{L}_{\mathrm{dice}} (\mathcal{M}_{\theta}({\mathrm{X}}^{\prime}), {\mathrm{Y}}) - \mathcal{L}_{\mathrm{ssim}}({\mathrm{X}},{\mathrm{X}}^{\prime}) \\
\mathrm{s.t.}~~ \|\boldsymbol{\mathrm{q}}\|_{\infty} \le q_{\mathrm{max}},
\end{gathered}
\end{equation}
```

> where $`\mathrm{X}`$ is clean image and $`\mathrm{X}^{\prime} = \mathcal{D}_{I}\big(~\varphi(\mathcal{D}({\mathrm{X}}),\boldsymbol{\mathrm{q}})~\big)`$ is adversarial image. $`\mathcal{D}(\cdot)~\text{and}~\mathcal{D}_{I}(\cdot)`$ are 3D-DCT and 3D-IDCT functions respectively. $`\boldsymbol{\mathrm{q}}`$ is learnable quatization table and $`\varphi(\cdot)`$ is a function which performs three operations: quantization, rounding and de-quatization of DCT coefficients. $`\mathcal{L}_{\mathrm{dice}}(\cdot)~\text{and}~\mathcal{L}_{\mathrm{ssim}}(\cdot)`$ are dice loss and structural similarity loss functions respectively. For further details, please check our paper.

</br>

> **Volumetric Adversaral Frequency Training (VAFT)**: For *minimization* step, we propose **V**olumetric **A**dversarial **F**requency **T**raining - **VAFT** to obtain a model that is robust to adversarial attacks. In VAFT, we update model parameters on clean and adversarial (obtained via VAFA) samples and further introduce a novel *frequency consistency loss* to keep frequency representation of the logits of clean and adversarial samples close to each other for a better accuracy tradeoff. We solve following objective during adversarial training:

```math
\begin{equation}
 \underset{ \theta }{\mathrm{minimize}}~ \mathcal{L}_{\text{dice}} (\mathcal{M}_{\theta}({\mathrm{X}}), {\mathrm{Y}})+  \mathcal{L}_{\text{dice}} (\mathcal{M}_{\theta}({\mathrm{X}}^{\prime}), {\mathrm{Y}}) + \mathcal{L}_{_{\mathrm{fr}}}(\mathcal{M}_{\theta}({\mathrm{X}}),\mathcal{M}_{\theta}({\mathrm{X}}^{\prime})) 
\end{equation}
```
where $`\mathrm{X}^{\prime}`$ is obtained from VAFA attack and $`\mathcal{L}_{_{\mathrm{fr}}}(\mathcal{M}_{\theta}({\mathrm{X}}),\mathcal{M}_{\theta}({\mathrm{X}}^{\prime})) = \|\mathcal{D}(\mathcal{M}_{\theta}({\mathrm{X}}))-\mathcal{D}(\mathcal{M}_{\theta}({\mathrm{X}}^{\prime}))\|_{_1}`$ is frequency consistency loss.

</br>

<hr />



## Updates :rocket:
- **July 13, 2023** : Released pre-trained clean and adversarially trained (under VAFA attack) [UNETR](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf) model checkpoints. 
- **July 10, 2023** : Released code for attacking [UNETR](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf) model with support for [Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) dataset.
- **May 25, 2023** : Early acceptance in [MICCAI 2023](https://conferences.miccai.org/2023/en/) (top 14%) &nbsp;&nbsp; :confetti_ball:


<br>

## Installation :wrench:
1. Create conda environment
```shell
conda create --name vafa python=3.8
conda activate vafa
```
2. Install PyTorch and other dependencies
```shell
pip install -r requirements.txt
```

## VAFA Attack
Code of VAFA attack can be accessed [here](attacks/vafa/vafa.py).

## Dataset
We conducted experiments on two volumetric medical image segmentation datasets: [Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789), [ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html). Synapse contains 14 classes (including background) and ACDC contains 4 classes (including background). We follow the same dataset preprocessing as in [nnFormer](https://github.com/282857341/nnFormer). 

The dataset folders for Synapse should be organized as follows: 

```
DATASET_SYNAPSE/
    ├── imagesTr/
        ├── img0001.nii.gz
        ├── img0002.nii.gz
        ├── img0003.nii.gz
        ├── ...  
    ├── labelsTr/
        ├── label0001.nii.gz
        ├── label0002.nii.gz
        ├── label0003.nii.gz
        ├── ...  
    ├── dataset_synapse_18_12.json
 ```

File `dataset_synapse_18_12.json` contains train-val split (created from train files) of Synapse datatset. There are 18 train images and 12 validation images. File `dataset_synapse_18_12.json` can be accessed [here](miscellaneous/dataset_synapse_18_12.json). Place this file in datatset parent folder. Pre-processed Synapse dataset can be downloaded from the following link as well.

| Dataset | Link |
|:-- |:-- |
| BTCV-Synapse (18-12 split) | [Download](https://drive.google.com/file/d/1-Tst3l2kMrC0rlNGDM9CwvRk_2KRFXOo/view?usp=sharing) |

You can use the command `tar -xzf btcv-synapse.tar.gz` to un-compress the file.

</br>

## Model
We use [UNETR](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf) model with following parameters:
```python
model = UNETR(
    in_channels=1,
    out_channels=14,
    img_size=(96,96,96),
    feature_size=16,
    hidden_size=768,
    mlp_dim=3072,
    num_heads=12,
    pos_embed="perceptron",
    norm_name="instance",
    conv_block=True,
    res_block=True,
    dropout_rate=0.0)

```

We also used [UNETR++](https://arxiv.org/abs/2212.04497) in our experiments but its code is not in a presentable form. Therefore, we are not including support for UNETR++ model in this repository. 

Clean and adversarially trained (under VAFA attack) [UNETR](https://openaccess.thecvf.com/content/WACV2022/papers/Hatamizadeh_UNETR_Transformers_for_3D_Medical_Image_Segmentation_WACV_2022_paper.pdf) models can be downloaded from the links given below. Place these models in a directory and give full path of the model (including name of the model e.g. `/folder_a/folder_b/model.pt`) in argument `--pretrained_path` to attack that model.

| Dataset | Model | Link |
|:-- |:-- |:-- | 
|Synapse | Clean UNETR $(\mathcal{M})$ | [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/asif_hanif_mbzuai_ac_ae/EaaTHPv6MGZGnDdwDYQRO9YBTGE3_87veLEXDG1V4uHjaw?e=XyLc61)|
|Synapse | Adversarially Trained (under VAFA) UNETR $(\mathcal{M}_{{\mathrm{VAFA-FR}}})$  | [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/asif_hanif_mbzuai_ac_ae/EVji-stXEFVChGViXw2se1kBFO1SPR4H1F2FGJKWYR-QLQ?e=ATeSnN)|
|Synapse | Adversarially Trained (under VAFA, **without** Frequecny Regularization Eq. 4 and dice loss on clean images ) UNETR $(\mathcal{M}_{{\mathrm{VAFA}}})$ | [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/asif_hanif_mbzuai_ac_ae/EflJdtvNAA1AsbHAVOsVlCcBqXsEz1uNep8iEphSO_bFWA?e=q4BYS0)|



## Launch VAFA Attack on the Model
```shell
python generate_adv_samples.py --model_name unet-r --feature_size=16 --infer_overlap=0.5 \
--dataset btcv --data_dir=<PATH_OF_DATASET> \
--json_list=dataset_synapse_18_12.json \
--use_pretrained \
--pretrained_path=<PATH_OF_PRETRAINED_MODEL>  \
--gen_val_adv_mode \
--save_adv_images_dir=<PATH_TO_SAVE_ADV_TEST_IMAGES> \
--attack_name vafa-3d --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss --debugging
```
If adversarial images are not intended to be saved, use `--debugging` argument. If `--use_ssim_loss` is not mentioned, SSIM loss will not be used in the adversarial objective (Eq. 2). If adversarial versions of train images are inteded to be generated, mention argument `--gen_train_adv_mode` instead of `--gen_val_adv_mode`.

For VAFA attack on each 2D slice of volumetric image, use : `--attack_name vafa-2d --q_max 20 --steps 20 --block_size 32 32 --use_ssim_loss`

Use following arguments when launching pixel/voxel domain attacks:

[PGD](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.pgd):&nbsp;&nbsp;&nbsp;        `--attack_name pgd --steps 20 --eps 4 --alpha 0.01`

[FGSM](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.fgsm):             `--attack_name fgsm --steps 20 --eps 4 --alpha 0.01`

[BIM](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.bim):&nbsp;&nbsp;&nbsp;        `--attack_name bim --steps 20 --eps 4 --alpha 0.01`

[GN](https://adversarial-attacks-pytorch.readthedocs.io/en/latest/attacks.html#module-torchattacks.attacks.gn):&nbsp;&nbsp;&nbsp;&nbsp;   `--attack_name gn --steps 20 --eps 4 --alpha 0.01 --std 4`

## Launch Adversarial Training (VAFT) of the Model
```shell
python run_normal_or_adv_training.py --model_name unet-r --in_channels 1 --out_channel 14 --feature_size=16 --batch_size=3 --max_epochs 5000 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
--save_checkpoint \
--dataset btcv --data_dir=<PATH_OF_DATASET> \
--json_list=dataset_synapse_18_12.json \
--use_pretrained \
--pretrained_path=<PATH_OF_PRETRAINED_MODEL>  \
--save_model_dir=<PATH_TO_SAVE_ADVERSARIALLY_TRAINED_MODEL> \
--val_every 15 \
--adv_training_mode --freq_reg_mode \
--attack_name vafa-3d --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss 
```

Arugument `--adv_training_mode` in conjunction with `--freq_reg_mode` performs adversarial training with dice loss on clean images, adversarial images and frequency regularization term (Eq. 4) in the objective function (Eq. 3). For vanilla adversarial training (i.e. dice loss on adversarial images), use only `--adv_training_mode`. For normal training of the model, do not mention these two arguments. 


## Inference on the Model with already saved Adversarial Images
If adversarial images have already been saved and one wants to do inference on the model using saved adversarial images, use following command:

```shell
python inference_on_saved_adv_samples.py --model_name unet-r --in_channels 1 --out_channel 14 --feature_size=16 --infer_overlap=0.5 \
--dataset btcv --data_dir=<PATH_OF_DATASET> \
--json_list=dataset_synapse_18_12.json \
--use_pretrained \
--pretrained_path=<PATH_OF_PRETRAINED_MODEL>  \
--adv_images_dir=<PATH_OF_SAVED_ADVERSARIAL_IMAGES> \ 
--attack_name vafa-3d --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss 
```

Attack related arguments are used to automatically find the sub-folder containing adversarial images. Sub-folder should be present in parent folder path specified by `--adv_images_dir` argument.  If `--no_sub_dir_adv_images` is mentioned, sub-folder will not be searched and images are assumed to be present directly in the parent folder path specified by `--adv_images_dir` argument. Structure of dataset folder should be same as specified in [Datatset](##dataset) section.


## Citation
If you find our work, this repository, or pretrained models useful, please consider giving a star :star: and citation.
```bibtex
@inproceedings{hanif2023frequency,
  title={Frequency Domain Adversarial Training for Robust Volumetric Medical Segmentation},
  author={Hanif, Asif and Naseer, Muzammal and Khan, Salman and Shah, Mubarak and Khan, Fahad Shahbaz},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={457--467},
  year={2023},
  organization={Springer}
}
```

<hr />

## Contact
Should you have any question, please create an issue on this repository or contact at **asif.hanif@mbzuai.ac.ae**

<hr />

<!---
## Our Related Works
  --->
