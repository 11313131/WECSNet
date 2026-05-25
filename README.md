
This is the official Pytorch/Pytorch implementation of the papers: <br/>


# WECSNet: A Wavelet‑Enhanced Cross‑Strip Network for Remote Sensing Object Detection







## Introduction

The master branch is built on MMRotate which works with **PyTorch 1.6+**.

WECSNet backbone code is placed under mmrotate/models/backbones/, and the train/test configure files are placed under configs/wecsnet/ 

## Pretrained Weights of Backbones


Imagenet 300-epoch pre-trained WECSNet-tiny backbone: [Download](https://github.com/lwCVer/LEGNet/releases/download/pre-train/LWEGNet_tiny.pth)


Imagenet 300-epoch pre-trained WECSNet-small backbone: [Download](https://github.com/lwCVer/LEGNet/releases/download/pre-train/LWEGNet_small.pth)


## Results and Models

DOTA1.0

|           Model            |  mAP  | Angle | training mode | Batch Size |                                                 Configs                                                  |
|:--------------------------:|:-----:| :---: |---------------|:----------:|:--------------------------------------------------------------------------------------------------------:|
| WECSNet-Tiny (1024,1024,200) | 78.85 | le90  | single-scale  |    2\*4    |  [orcnn_wecsnet_tiny_dota10_test_ss_e36.py](./configs/wecsnet/orcnn_wecsnet_tiny_dota10_test_ss_e36.py)  |
| WECSNet-Small (1024,1024,200) | 80.11 | le90  | single-scale  |    2\*4    | [orcnn_wecsnet_small_dota10_test_ss_e36.py](./configs/wecsnet/orcnn_wecsnet_small_dota10_test_ss_e36.py) |


DOTA1.5

|             Model             |  mAP  | Angle | training mode | Batch Size |                                                 Configs                                                  |
|:-----------------------------:|:-----:| :---: |---| :------: |:--------------------------------------------------------------------------------------------------------:|
| WECSNet-Small (1024,1024,200) | 72.60 | le90  | single-scale |    2\*4     | [orcnn_wecsnet_small_dota10_test_ss_e36.py](./configs/wecsnet/orcnn_wecsnet_small_dota15_test_ss_e36.py) |

FAIR-v1.0

|         Model         |  mAP  | Angle | training mode | Batch Size |                                                 Configs                                                  |
| :----------------------: |:-----:| :---: |---| :------: |:--------------------------------------------------------------------------------------------------------:|
| WECSNet-Small (1024,1024,500) | 48.57 | le90  | multi-scale |    2\*4     | [orcnn_wecsnet_small_fairv1_test_ms_e12.py](./configs/wecsnet/orcnn_wecsnet_small_fairv1_test_ms_e12.py) |

DIOR-R 

|                    Model                     |  mAP  | Batch Size |                                               Configs                                                |
| :------------------------------------------: |:-----:|:----------:|:----------------------------------------------------------------------------------------------------:|
|                   WECSNet-Small                  | 71.40 |    2\*4    | [orcnn_wecsnet_small_dior_test_ss_e36.py](./configs/wecsnet/orcnn_wecsnet_small_dior_test_ss_e36.py) |

## Installation

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instruction.



```shell
conda create -n WECSNet-Det python=3.8 -y
conda activate WECSNet-Det
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/index.html
pip install pytorch_wavelets
pip install mmdet
cd WECSNet
pip install -v -e .
```

## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
We provide [colab tutorial](demo/MMRotate_Tutorial.ipynb), and other tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [useful tools](docs/en/tutorials/useful_tools.md)







```
## License
Licensed under a [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) for Non-commercial use only. 
Any commercial use should get formal permission first.
