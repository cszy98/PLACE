# PLACE: Adaptive Layout-Semantic Fusion for Semantic Image Synthesis (CVPR 2024)

### Introduction

The source code for our paper "PLACE: Adaptive Layout-Semantic Fusion for Semantic Image Synthesis" (CVPR 2024)

[**[Project Page]**](https://cszy98.github.io/PLACE/)  [**[Code]**](https://github.com/cszy98/PLACE/tree/main)  [**[Paper]**](https://arxiv.org/abs/2403.01852)

### Overview

![overview](resources/overview.png)

### Quick Start

#### Installation

```
git clone 
cd PLACE
conda env create -f environment.yaml
conda activate PLACE
```

#### Data Preparation

Please follow the dataset preparation process in [FreestyleNet](https://github.com/essunny310/FreestyleNet).

#### Running

The pre-trained models can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1b5pC52hasLwm1gOkc9LmdIyxZjrdlNWC?usp=drive_link) and should be put into the `ckpt` folder.

After the dataset and pre-trained models are prepared, you may evaluate the model with the following scripts:

```
# evaluate on the ADE20K dataset
./run_inference_ADE20K.sh
# evaluate on the COCO-Stuff dataset
./run_inference_COCO.sh
```

For out-of-distribution synthesis, you just need to modify the `ADE20K` or `COCO` dictionary in the `dataset.py`

### Citation

```
@article{lv2024place,
  title={PLACE: Adaptive Layout-Semantic Fusion for Semantic Image Synthesis},
  author={Lv, Zhengyao and Wei, Yuxiang and Zuo, Wangmeng and Kwan-Yee K. Wong},
  journal={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

### Contact

Please send mail to cszy98@gmail.com
