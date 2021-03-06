# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from multiplexer.modeling.box_regression import Box2BoxTransform
from multiplexer.modeling.roi_heads.box_head.roi_box_post_processor_base import BaseBoxPostProcessor
from multiplexer.modeling.roi_heads.box_head.roi_box_post_processor_rotated import (
    RotatedBoxPostProcessor,
)

_ROI_BOX_POST_PROCESSOR = {
    "BaseBoxPostProcessor": BaseBoxPostProcessor,
    "RotatedBoxPostProcessor": RotatedBoxPostProcessor,
}


def make_roi_box_post_processor(cfg):
    # use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG

    box_post_processor = _ROI_BOX_POST_PROCESSOR[cfg.MODEL.ROI_BOX_HEAD.POST_PROCESSOR]

    postprocessor = box_post_processor(
        score_thresh, nms_thresh, detections_per_img, box2box_transform, cfg
    )
    return postprocessor
