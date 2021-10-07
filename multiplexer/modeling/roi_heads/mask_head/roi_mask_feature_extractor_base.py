# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


from torch import nn
from torch.nn import functional as F

from multiplexer.layers import Conv2d
from multiplexer.modeling.poolers import Pooler


class MaskRCNNFPNFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg):
        """
        Arguments:
            num_classes (int): number of output classes
            input_size (int): number of channels of the input once it's flattened
            representation_size (int): size of the intermediate representation
        """
        super(MaskRCNNFPNFeatureExtractor, self).__init__()

        # resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        if cfg.MODEL.CHAR_MASK_ON or cfg.SEQUENCE.SEQ_ON:
            resolution_h = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_H
            resolution_w = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_W
        else:
            resolution_h = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
            resolution_w = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_MASK_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution_h, resolution_w),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = cfg.MODEL.BACKBONE.OUT_CHANNELS
        self.pooler = pooler

        layers = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS

        next_feature = input_size
        self.blocks = []
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "mask_fcn{}".format(layer_idx)
            module = Conv2d(next_feature, layer_features, 3, stride=1, padding=1)
            # Caffe2 implementation uses MSRAFill, which in fact
            # corresponds to kaiming_normal_ in PyTorch
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            nn.init.constant_(module.bias, 0)
            self.add_module(layer_name, module)
            next_feature = layer_features
            self.blocks.append(layer_name)

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))

        return x
