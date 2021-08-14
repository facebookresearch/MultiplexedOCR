# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn

from multiplexer.layers.res_blocks import BasicBlock, res_layer
from multiplexer.modeling.roi_heads.mask_head.roi_cropper_builder import build_roi_cropper


class Resnet18RectV2FeatureExtractor(nn.Module):
    """
    Resnet18 rect feature extractor V2
    - Major changes compared to V0:
        - Added bn0 layer
    """

    def __init__(self, cfg):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(Resnet18RectV2FeatureExtractor, self).__init__()

        block = BasicBlock
        layers = [2, 2, 2, 2]

        # final output: N, C, 48//16, <=320//8
        self.cropper = build_roi_cropper(
            cfg=cfg,
            height=cfg.MODEL.ROI_MASK_HEAD.CROPPER_RESOLUTION_H,
            width=cfg.MODEL.ROI_MASK_HEAD.CROPPER_RESOLUTION_W,
        )

        # Use a batch norm  layer to normalize the inputs instead of using
        # a fixed mean and variance
        self.bn0 = nn.BatchNorm2d(3, eps=1e-5, momentum=0.1)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = res_layer(block, 64, 64, 64, layers[0])
        self.layer2 = res_layer(
            block,
            64,
            128,
            128,
            layers[1],
            stride=2,
        )

        # We want to preserve horizontal resolution.
        # Hence stride 2 for height but stride 1 for width.
        self.layer3 = res_layer(
            block,
            128,
            256,
            256,
            layers[2],
            stride=(2, 1),
        )

        # Final one is just stride 1
        self.layer4 = res_layer(
            block,
            256,
            512,
            512,
            layers[3],
            stride=1,
        )

    def forward(self, x, proposals):
        x = self.cropper(x, proposals)

        x = self.bn0(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
