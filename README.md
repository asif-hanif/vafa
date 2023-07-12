# Frequency Domain Adversarial Training for Robust Volumetric Medical Segmentation ([MICCAI'23](https://conferences.miccai.org/2023/en/))

<hr />

| ![main figure](/media/vafa_vaft.png)|
|:--| 
| **Overview of Adversarial Frequency Attack and Training**: A model trained on voxel-domain adversarial attacks is vulnerable to frequency-domain adver- sarial attacks. In our proposed adversarial training method, we generate adversarial samples by perturbing their frequency-domain representation using a novel module named "Frequency Perturbation". The model is then updated while minimizing the dice loss on clean and adversarially perturbed images. Furthermore, we propose a frequency consistency loss to improve the model performance. |


> **Abstract:** <p style="text-align: justify;">*It is imperative to ensure the robustness of deep learning models in critical applications such as, healthcare. While recent advances in deep learning have improved the performance of volumetric medical image segmentation models, these models cannot be deployed for real-world applications immediately due to their vulnerability to adversarial attacks. We present a 3D frequency domain adversarial attack for volumetric medical image segmentation models and demonstrate its advantages over conventional input or voxel domain attacks. Using our proposed attack, we introduce a novel frequency domain adversarial training approach for optimizing a robust model against voxel and frequency domain attacks.  Moreover, we propose frequency consistency loss to regulate our frequency domain adversarial training that achieves a better tradeoff between model's performance on clean and adversarial samples.* </p>
<hr />

## Brief Description
In the context of 2D natural images, it has been recently observed that frequency-domain based adversarial attacks are more effective against the defenses that are primarily designed to *undo* the impact of pixel-domain adversarial noise in natural images. Motivated by this observation in 2D natural images, here we explore the effectiveness of frequency-domain based adversarial attacks in the regime of volumetric medical image segmentation and aim to obtain a volumetric medical image segmentation model that is robust against adversarial attacks. To achieve this goal, we propose a *min-max* objective for adversarial training of volumetric medical image segmentation model in frequency-domain. 

> **Volumetric Adversaral Frequency Attack (VAFA)**: For *maximization* step, we introduce **V**olumetric **A**dversarial **F**requency **A**ttack - **VAFA** which operates in the frequency-domain of the data (unlike other prevalent voxel-domain attacks) and explicitly takes into account the 3D nature of the volumetric medical data to achieve higher fooling rate. The proposed **VAFA** transforms the 3D patches of input volumetric medical image into frequency-domain by employing 3D discrete cosine transform (3D-DCT) and perturbs the DCT coefficients via a learnable *quantization* table and then converts the perturbed frequency-domain data back into voxel-domain through inverse 3D-DCT. To preserve structural information in adversarial sample, we incorporate SSIM loss along with adversarial loss which helps us attain better SSIM and LPIPS. 

> **Volumetric Adversaral Frequency Training (VAFT)**: For *minimization* step, we propose **V**olumetric **A**dversarial **F**requency **T**raining - **VAFT** to obtain a model that is robust to adversarial attacks. In VAFT, we update model parameters on clean and adversarial (obtained via VAFA) samples and further introduce a novel *frequency consistency loss* to keep frequency representation of logits of clean and adversarial samples close to each other for a better accuracy tradeoff.
<hr />

## Updates :loudspeaker:
- **July 10, 2023** : 
- **May 25, 2023** : Early acceptance in [MICCAI 2023](https://conferences.miccai.org/2023/en/)  &nbsp;&nbsp; :confetti_ball:


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

## Datatset
We follow the same dataset preprocessing as in [nnFormer](https://github.com/282857341/nnFormer). We conducted experiments on two datasets: Synapse, ACDC

The dataset folders for Synapse should be organized as follows: 

```
DATASET_SYNAPSE/
  ├── imagesTr/
  ├── imagesTs/
  ├── labelsTr/
  ├── labelsTs/
  ├── dataset_synapse_18_12.json
 ```
File `dataset_synapse_18_12.json` contains train-test split of Synapse datatset. There are 18 train images and 12 test images. File can be accessed here. 

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
    pos_embed=perceptron,
    norm_name=instance,
    conv_block=True,
    res_block=True,
    dropout_rate=0.0)

```

## Launch VAFA Attack on the Model
```shell
python unetr_gen_train_or_val_adv.py --feature_size=16 --infer_overlap=0.5 \
--data_dir=<PATH_OF_DATASET> \
--json_list=dataset_synapse.json \
--use_pretrained \
--pretrained_path=<PATH_OF_PRETRAINED_MODEL>  \
--gen_val_adv_mode \
--save_adv_images_dir=<PATH_TO_SAVE_ADV_TEST_IMAGES>
--attack_name vafa-3d --q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss
```
If adversarial images are not intended to be saved, use `--debugging` argument.

## Lanuch Adversarial Training (VAFT) of the Model
```shell
python unetr_adv_training.py --feature_size=16 --batch_size=4 --optim_lr=1e-4 --lrschedule=warmup_cosine --infer_overlap=0.5 \
--save_checkpoint \
--data_dir=<PATH_OF_DATASET> \
--json_list=dataset_synapse.json \
--use_pretrained \
--pretrained_path=<PATH_OF_PRETRAINED_MODEL>  \
--adv_training_freq_reg_mode \
--attack_name vafa \
--q_max 20 --steps 20 --block_size 32 32 32 --use_ssim_loss \
--save_model_dir=<PATH_TO__SAVE_ADVERSARIALLY_TRAINED_MODEL> \
--val_every 1
```
