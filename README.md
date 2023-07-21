# EfficientNet_EarlyExit
This repository contains scripts for training multi-exit EfficientNet model on ImageNet.

with help from:

EfficientNet paper
https://arxiv.org/abs/1905.11946

PyTorch EfficientNet repository
https://pytorch.org/vision/main/models/efficientnet.html

This is done as a part of a research project about early exiting and partitioning. 
It contains a multi-exit version of the EfficientNEt model. In the multi-exit case, easier samples that do not need the entire model for classification, can take the earlier exits and save time in the inference process. 
This way we accelerate the inference process without losing much accuracy.
The codes are in PyTorch.
