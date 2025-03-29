
This is the PyTorch implementation of ``FCN+: Global Receptive Convolution Makes FCN Great Again'' which has been accepted to NeuroComputing. [NeuroComputing version](https://www.sciencedirect.com/science/article/abs/pii/S0925231225003273), [ArXiv version](https://arxiv.org/abs/2303.04589).

Our FCN+ is built upon MMSegmentation. The installation of this repository is the same as MMSegmentation, so we attach the README.md from MMSegmentation below.

We implement our global receptive convolution as a standard convolution followed by shifting operation (please refer to Sec. 3.4 of our paper). For the shifting operation, please see `class GlobalShiftV2Portion` in `mmseg/models/plugins/shift.py`. Its usage can be found in `shift_exps/configs_ade/fcn_r101-plugs_shift_s4_p05_c1-d8_512x512-scale_1024_320-ratio_05_20-160k_ade20k.py`:
```python

model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        depth=101,
        plugins=[
            dict(
                cfg=dict(type='GlobalShift', scale=4, portion=0.5),
                stages=(False, False, False, True),
                position='after_conv1'),
        ]))

```

To start the training process, run the following command:
```bash
MASTER_PORT=1246 GPUS=8 GPUS_PER_NODE=8 CPUS_PER_TASK=2 tools/slurm_train.sh mediaa mmseg_2 shift_exps/configs_ade/fcn_r101-plugs_shift_s4_p05_c1-d8_512x512-scale_1024_320-ratio_05_20-320k_ade20k.py --seed=0
```

## Citation
```
@article{REN2025129655,
title = {FCN+: Global receptive convolution makes FCN great again},
journal = {Neurocomputing},
volume = {631},
pages = {129655},
year = {2025},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2025.129655},
url = {https://www.sciencedirect.com/science/article/pii/S0925231225003273},
author = {Xiaoyu Ren and Zhongying Deng and Jin Ye and Junjun He and Dongxu Yang},
}
```



<div align="center">
  <img src="resources/mmseg-logo.png" width="600"/>
</div>
<br />

[![PyPI](https://img.shields.io/pypi/v/mmsegmentation)](https://pypi.org/project/mmsegmentation)
[![docs](https://img.shields.io/badge/docs-latest-blue)](https://mmsegmentation.readthedocs.io/en/latest/)
[![badge](https://github.com/open-mmlab/mmsegmentation/workflows/build/badge.svg)](https://github.com/open-mmlab/mmsegmentation/actions)
[![codecov](https://codecov.io/gh/open-mmlab/mmsegmentation/branch/master/graph/badge.svg)](https://codecov.io/gh/open-mmlab/mmsegmentation)
[![license](https://img.shields.io/github/license/open-mmlab/mmsegmentation.svg)](https://github.com/open-mmlab/mmsegmentation/blob/master/LICENSE)

Documentation: https://mmsegmentation.readthedocs.io/

## Introduction

MMSegmentation is an open source semantic segmentation toolbox based on PyTorch.
It is a part of the OpenMMLab project.

The master branch works with **PyTorch 1.3 to 1.5**.

![demo image](resources/seg_demo.gif)

### Major features

- **Unified Benchmark**

  We provide a unified benchmark toolbox for various semantic segmentation methods.

- **Modular Design**

  We decompose the semantic segmentation framework into different components and one can easily construct a customized semantic segmentation framework by combining different modules.

- **Support of multiple methods out of box**

  The toolbox directly supports popular and contemporary semantic segmentation frameworks, *e.g.* PSPNet, DeepLabV3, PSANet, DeepLabV3+, etc.

- **High efficiency**

  The training speed is faster than or comparable to other codebases.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog

v0.5.1 was released in 11/08/2020.
Please refer to [changelog.md](docs/changelog.md) for details and release history.

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/model_zoo.md).

Supported backbones:
- [x] ResNet
- [x] ResNeXt
- [x] [HRNet](configs/hrnet/README.md)
- [x] [ResNeSt](configs/resnest/README.md)

Supported methods:
- [x] [FCN](configs/fcn)
- [x] [PSPNet](configs/pspnet)
- [x] [DeepLabV3](configs/deeplabv3)
- [x] [PSANet](configs/psanet)
- [x] [DeepLabV3+](configs/deeplabv3plus)
- [x] [UPerNet](configs/upernet)
- [x] [NonLocal Net](configs/nonlocal_net)
- [x] [EncNet](configs/encnet)
- [x] [CCNet](configs/ccnet)
- [x] [DANet](configs/danet)
- [x] [GCNet](configs/gcnet)
- [x] [ANN](configs/ann)
- [x] [OCRNet](configs/ocrnet)
- [x] [Fast-SCNN](configs/fastscnn)
- [x] [Semantic FPN](configs/sem_fpn)
- [x] [PointRend](configs/point_rend)
- [x] [EMANet](configs/emanet)
- [x] [DNLNet](configs/dnlnet)
- [x] [Mixed Precision (FP16) Training](configs/fp16/README.md)

## Installation

Please refer to [INSTALL.md](docs/install.md) for installation and dataset preparation.

## Get Started

Please see [getting_started.md](docs/getting_started.md) for the basic usage of MMSegmentation.
There are also tutorials for [adding new dataset](docs/tutorials/new_dataset.md), [designing data pipeline](docs/tutorials/data_pipeline.md), and [adding new modules](docs/tutorials/new_modules.md).

A Colab tutorial is also provided. You may preview the notebook [here](demo/MMSegmentation_Tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/master/demo/MMSegmentation_Tutorial.ipynb) on Colab.

## Contributing

We appreciate all contributions to improve MMSegmentation. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMSegmentation is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new semantic segmentation methods.
