#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from multiplexer.modeling.meta_arch.rcnn import GeneralizedRCNN
# from multiplexer.modeling.detector.seg_rcnn import SegRCNN
# from multiplexer.modeling.detector.cropped_rcnn import CroppedRCNN

from multiplexer.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")


# _DETECTION_META_ARCHITECTURES = {
#     # "CroppedRCNN": CroppedRCNN,
#     "GeneralizedRCNN": GeneralizedRCNN,
#     # "SegRCNN": SegRCNN,
# }

def build_model(cfg):
    meta_arch = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)
    return meta_arch(cfg)