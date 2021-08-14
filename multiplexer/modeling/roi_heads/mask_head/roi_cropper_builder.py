# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from .roi_cropper_dynamic import DynamicRotatedROICropper
from .roi_cropper_horizontal import HorizontalROICropper
from .roi_cropper_rotated import RotatedROICropper

_ROI_CROPPERS = {
    "DynamicRotatedROICropper": DynamicRotatedROICropper,
    "HorizontalROICropper": HorizontalROICropper,
    "RotatedROICropper": RotatedROICropper,
}


def build_roi_cropper(cfg, height, width):
    cropper = _ROI_CROPPERS[cfg.MODEL.ROI_MASK_HEAD.ROI_CROPPER]
    return cropper(height=height, width=width)
