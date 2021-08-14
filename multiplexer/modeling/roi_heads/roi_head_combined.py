# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List

import torch

from multiplexer.modeling.roi_heads.box_head import build_roi_box_head
from multiplexer.modeling.roi_heads.mask_head import build_roi_mask_head
from multiplexer.structures import BoxList

from .build import ROI_HEADS_REGISTRY


@ROI_HEADS_REGISTRY.register()
class CombinedROIHead(torch.nn.Module):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg):
        super(CombinedROIHead, self).__init__()

        # flags to deprecate: cfg.MODEL.RPN_ONLY, cfg.MODEL.MASK_ON, cfg.SEQUENCE.SEQ_ON
        self.box = build_roi_box_head(cfg)
        self.mask = build_roi_mask_head(cfg)

        self.cfg = cfg.clone()

        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor

    def forward(self, features: List[torch.Tensor], proposals: List[BoxList], targets=None):
        losses = {}

        # TODO rename x to roi_box_features, if it doesn't increase memory consumption
        x, detections, loss_box = self.box(features, proposals, targets)
        losses.update(loss_box)
        if self.cfg.MODEL.MASK_ON or self.cfg.SEQUENCE.SEQ_ON:
            mask_features = features
            # optimization: during training, if we share the feature extractor between
            # the box and the mask heads,
            # then we can reuse the features already computed
            if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
                mask_features = x
            # During training, self.box() will return
            # the unaltered proposals as "detections"
            # this makes the API consistent during training and testing

            x, detections, loss_mask = self.mask(mask_features, detections, targets)
            if loss_mask is not None:
                losses.update(loss_mask)
        return x, detections, losses
