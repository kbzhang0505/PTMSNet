## Rethinking progressive low-light image enhancement: A frequency-aware tripartite multi-scale network (Neural Networks 2026)


[Yingjian Li](https://orcid.org/0009-0000-9869-014X), [Kaibing Zhang](https://orcid.org/0000-0002-3770-017X), [Xuan Zhou](https://orcid.org/0000-0002-6991-8077), [Zhouqiang Zhang](https://orcid.org/0000-0002-0477-3508), [Sheng Hu](https://orcid.org/0000-0001-8669-4469)

<img src="Figure\Fig1.png" alt="Fig1" style="zoom:67%;" />





- [x] 
- [x] Inference code and pre-trained checkpoints
- [x] Training code and data loader
- [x] Result 

>**Abstract:** Low-light images often suffer from complex degradation factors such as reduced contrast, color distortion, and noise.
> Recently, progressive networks have gained great popularity in low-light image enhancement due to  their strong ability to recover image details and improve overall quality. 
> However, most existing progressive methods typically rely on a single-scale fusion strategy, which limits their capacity to capture rich multi-scale information effectively. 
> To overcome this limitation, we propose a novel Progressive Tripartite Multi-Scale Network (PTMSNet) for low-light image enhancement. 
> Our network uniquely enhances multi-scale interactions among different branches across multiple scales to achieve comprehensive feature fusion. 
> To be more specific, we design a Parallel Hybrid Module (PHM) that integrates a Transformer branch with a Convolutional Neural Network (CNN) branch in parallel. 
> This design allows the two branches to learn complementary feature representations and effectively preserve local textures and structural information. 
> Meanwhile, we introduce a Frequency-Aware Fusion Module (FAFM), which leverages global frequency-domain information to generate a more accurate Signal-to-Noise Ratio (SNR) map than previous Gaussian-based SNR fusion. 
> Finally, a joint loss function which integrates a pixel-wise fidelity loss, a structural similarity loss, and a chromatic consistency loss is further employed to optimize the model parameters. 
> Extensive evaluations on three benchmark datasets demonstrate that newly proposed PTMSNet significantly outperforms state-of-the-art (SOTA) competitors, including several progressive predecessors. 
> Moreover, the generalization experiments on the ExDark dataset and low-light images captured in real-world scenarios further highlight robust object detection capability under low-light conditions detection capability and practical applicability in photometric measurement systems with unstable illumination. 
> The code will be made publicly available at https://github.com/kbzhang0505/PTMSNet. 


## Introduction

This repository hosts the code for the paper "Rethinking Progressive Low-light Image Enhancement: A Frequency-Aware Tripartite Multi-Scale Network". PTMSNet is an innovative network designed for low-light image enhancement. By leveraging multi-scale feature interaction, the Parallel Hybrid Module (PHM), and the Frequency-Aware Fusion Module (FAFM), it effectively balances local details and global structures. PTMSNet demonstrates superior performance across multiple low-light image benchmark datasets (e.g., LOL, MIT-5K) and low-light object detection tasks.



### Current Status & Code Availability

The repository is currently in the **pre-release stage**. The full code (including model implementations, training scripts, and testing configurations) will be publicly released **after the paper is officially accepted by the journal** (expected timeline: within 1 week of acceptance).



### **LOLv1**：

<img src="Figure\Fig3.png" alt="Fig3" style="zoom:67%;" />

<img src="Figure\Fig4.png" alt="Fig4" style="zoom:67%;" />



### **LOLv2-Real**：

<img src="Figure\Fig2.png" alt="Fig2" style="zoom:67%;" />



### **LOLv2-Synthetic**：

<img src="Figure\Fig7.png" alt="Fig7" style="zoom:67%;" />



### **MIT-Adobe-FiveK**：

<img src="Figure\Fig5.png" alt="Fig5" style="zoom:67%;" />



### **Unpaired dataset**：

<img src="Figure\Fig6.png" alt="Fig6" style="zoom:67%;" />



## Validated Superiority

PTMSNet significantly outperforms state-of-the-art (SOTA) methods in the following tasks (see the paper and the `results` folder in the repository for details):

- **Low-light Image Enhancement**:     Achieves a PSNR of 24.801 (SSIM 0.861) on the LOLv1 dataset and a PSNR of     22.569 (SSIM 0.857) on the LOLv2-real dataset;
- **Generalization Capability**:     Achieves an average NIQE of 3.779 on unpaired datasets (e.g., DICM, LIME),     outperforming compared methods;
- **Downstream Tasks**: Achieves a     low-light object detection mAP of 74.89% when combined with YOLOv6 (on the     ExDark dataset).


## Get Started
### Dependencies and Installation
1. Create Conda Environment 
```
conda create -n PTMSNet python=3.8
conda activate PTMSNet
conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 -c pytorch
pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm
```
2. Clone Repo
```
git clone https://github.com/kbzhang0505/PTMSNet.git
```

3. Install warmup scheduler

```
cd PTMSNet
cd pytorch-gradual-warmup-lr; python setup.py install; cd ..
```

### Dataset
You can use the following links to download the datasets

1. LOL [[Link](https://daooshee.github.io/BMVC2018website/)]
2. LOLv2-real [[Baidu drive](https://pan.baidu.com/s/1AokwhWJ-X4BBiB9u4ChOMg?pwd=42fp)]
3. LOLv2-synthetic [[Baidu drive](https://pan.baidu.com/s/1q4uFA-YZzfdthV2r6yeTzQ?pwd=sycy)]
4. MIT-Adobe FiveK [[Google drive](https://drive.google.com/drive/folders/1c33pXjeqX-Fwxc_yAozGQ0UlsSZ4IOqn?usp=drive_link) | [Baidu drive](https://pan.baidu.com/s/1z4sBVXdn8eJv1VpSI0LohA?pwd=yvhi)]




### Train

1. To download training and testing data


2. To train LLFormer, run
```bash
python train.py -yml_path your_config_path
```
```
You need to modify the config for your own training environment
```

3. We provide pre-trained checkpoints for four datasets, which can be obtained as follows:
   1. LOLv1 checkpoints [[Baidu drive](https://pan.baidu.com/s/1Y2NsENnsg1JU71pFxgTS7g?pwd=f41j)]
   2. LOLv2-real checkpoints [[Baidu drive](https://pan.baidu.com/s/1jo-yO2K-6iU2YzIVcALlIg?pwd=hmhy)]
   3. MIT-Adobe FiveK checkpoints [[Baidu drive](https://pan.baidu.com/s/1YugWKxTXEcaDqgGKRU077Q?pwd=2uy3)]




## Contact Us

For collaborations or inquiries, please contact the corresponding author:
 Kaibing Zhang (Email: zhangkaibing@xpu.edu.cn)

⭐ Click Star/Watch to follow the repository; you will be notified immediately when the code is released!

## Citations
If UHDLOL benchmark and LLFormer help your research or work, please consider citing:

```
@article{LI2026PTMSNet,
title = {Rethinking progressive low-light image enhancement: A frequency-aware tripartite multi-scale network},
journal = {Neural Networks},
volume = {196},
pages = {108351},
year = {2026},
author = {Yingjian Li and Kaibing Zhang and Xuan Zhou and Zhouqiang Zhang and Sheng Hu},
}
```





## Our Related Works
- Multi-Branch and Progressive Network for Low-Light Image Enhancement, TIP 2023. [Paper](https://ieeexplore.ieee.org/document/10102793) | [Code](https://github.com/kbzhang0505/MBPNet)

---

## Reference Repositories
This implementation is based on / inspired by:
- HWMNet: https://github.com/FanChiMao/HWMNet
- Restormer: https://github.com/swz30/Restormer
- LLFlow: https://github.com/wyf0912/LLFlow
- BasicSR: https://github.com/XPixelGroup/BasicSR
- LLFormer: https://github.com/TaoWangzj/LLFormer

---

<details>
<summary>statistics</summary>

![visitors](https://visitor-badge.laobi.icu/badge?page_id=TaoWangzj/LLFormer)


</details>



## Notes

Current experimental results are for academic exchange only; the full code will be finalized and released upon paper acceptance.


