1.  method introduction

This method is based on image segmentation of deep learning, which is used for wheat area segmentation.

The main models are U-Net and FPN.

2.  Environment

 Python 3.8

 PyTorch

 segmentation_models_pytorch

 OpenCV

 NumPy

3.  Installation

pip install torch torchvision

pip install segmentation-models-pytorch

pip install opencv-python

pip install numpy

4.  Data

Original image and mask file

5.  code structure

WheatDataset

Model: Using segmentation_models_pytorch to build U-net and FPN

Training

Evaluation: IoU F1-score…

6.  Key parameters

> batch size: 4
>
> learning rate: 1e-3
>
> epoch: 25
>
> threshold: 0.6

7.  Note:

FPN model needs that the input image size can be divisible by 32.

The size of image and mask must same.

The different thresholds will influence the outcom.
