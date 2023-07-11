# Frequency Domain Adversarial Training for Robust Volumetric Medical Segmentation ([MICCAI'23](https://conferences.miccai.org/2023/en/))

<hr />

![main figure](/media/vafa_vaft.png)
> **Abstract:** <p style="text-align: justify;">*It is imperative to ensure the robustness of deep learning models in critical applications such as, healthcare. While recent advances in deep learning have improved the performance of volumetric medical image segmentation models, these models cannot be deployed for real-world applications immediately due to their vulnerability to adversarial attacks. We present a 3D frequency domain adversarial attack for volumetric medical image segmentation models and demonstrate its advantages over conventional input or voxel domain attacks. Using our proposed attack, we introduce a novel frequency domain adversarial training approach for optimizing a robust model against voxel and frequency domain attacks.  Moreover, we propose frequency consistency loss to regulate our frequency domain adversarial training that achieves a better tradeoff between model's performance on clean and adversarial samples.* </p>
<hr />


## Brief Description 
In the context of 2D natural images, it has been recently observed that frequency-domain based adversarial attacks are more effective against the defenses that are primarily designed to ``undo'' the impact of pixel-domain adversarial noise in natural images. Motivated by this observation in 2D natural images, here we explore the effectiveness of frequency-domain based adversarial attacks in the regime of volumetric medical image segmentation and aim to obtain a volumetric medical image segmentation model that is robust against adversarial attacks. To achieve this goal, we propose a *min-max* objective for adversarial training of volumetric medical image segmentation model in frequency-domain. 

**(1) Volumetric Adversaral Frequency Attack (VAFA)**: For *maximization* step, we introduce **V**olumetric **A**dversarial **F**requency **A**ttack - **VAFA** which operates in the frequency-domain of the data (unlike other prevalent voxel-domain attacks) and explicitly takes into account the 3D nature of the volumetric medical data to achieve higher fooling rate. The proposed \textbf{VAFA} transforms the 3D patches of input volumetric medical image into frequency-domain by employing 3D discrete cosine transform (3D-DCT) and perturbs the DCT coefficients via a learnable \textit{quantization} table and then converts the perturbed frequency-domain data back into voxel-domain through inverse 3D-DCT. To preserve structural information in adversarial sample, we incorporate SSIM loss along with adversarial loss which helps us attain better SSIM and LPIPS. 

**(2) Volumetric Adversaral Frequency Training (VAFT)**: For *minimization* step, we propose **V**olumetric **A**dversarial **F**requency **T**raining - **VAFT** to obtain a model that is robust to adversarial attacks. In VAFT, we update model parameters on clean and adversarial (obtained via VAFA) samples and further introduce a novel *frequency consistency loss* to keep frequency representation of logits of clean and adversarial samples close to each other for a better accuracy tradeoff.


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


## Model




