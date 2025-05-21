# PTMSNet: Progressive Tripartite Multi-Scale Network for Low-Light Image Enhancement

<img src="Figure\Fig1.png" alt="Fig1" style="zoom:67%;" />





- [x] 
- [ ] Inference code and checkpoint
- [ ] Training code and data loader
- [x] Result 



## Introduction

This repository hosts the code for the paper "Rethinking Progressive Low-light Image Enhancement: A Frequency-Aware Tripartite Multi-Scale Network". PTMSNet is an innovative network designed for low-light image enhancement. By leveraging multi-scale feature interaction, the Parallel Hybrid Module (PHM), and the Frequency-Aware Fusion Module (FAFM), it effectively balances local details and global structures. PTMSNet demonstrates superior performance across multiple low-light image benchmark datasets (e.g., LOL, MIT-5K) and low-light object detection tasks.



### Current Status & Code Availability

The repository is currently in the **pre-release stage**. The full code (including model implementations, training scripts, and testing configurations) will be publicly released **after the paper is officially accepted by the journal** (expected timeline: within 1 week of acceptance).



### LOLv1：

<img src="Figure\Fig3.png" alt="Fig3" style="zoom:67%;" />

<img src="Figure\Fig4.png" alt="Fig4" style="zoom:67%;" />



### **LOLv2-Real**:

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



## Future Plans

- **Code Release**: Full code     (PyTorch implementation) will be uploaded immediately after paper     acceptance, supporting one-click training/testing;
- **Feedback**: Welcome to submit     bugs or improvement suggestions via Issues; these will be prioritized     after acceptance.



## Contact Us

For collaborations or inquiries, please contact the corresponding author:
 Kaibing Zhang (Email: zhangkaibing@xpu.edu.cn)

⭐ Click Star/Watch to follow the repository; you will be notified immediately when the code is released!



## Notes

Current experimental results are for academic exchange only; the full code will be finalized and released upon paper acceptance.
