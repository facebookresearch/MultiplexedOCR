from multiplexer.modeling.box_coder import BoxCoder
from multiplexer.modeling.roi_heads.box_head.roi_box_post_processor_base import (
    BaseBoxPostProcessor,
    BaseBoxPostProcessorForTorchscript,
)
from multiplexer.modeling.roi_heads.box_head.roi_box_post_processor_rotated import (
    RotatedBoxPostProcessor,
)

_ROI_BOX_POST_PROCESSOR = {
    "BaseBoxPostProcessor": BaseBoxPostProcessor,
    "RotatedBoxPostProcessor": RotatedBoxPostProcessor,
}


def make_roi_box_post_processor(cfg):
    # use_fpn = cfg.MODEL.ROI_HEADS.USE_FPN

    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH
    nms_thresh = cfg.MODEL.ROI_HEADS.NMS
    detections_per_img = cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG

    box_post_processor = _ROI_BOX_POST_PROCESSOR[cfg.MODEL.ROI_BOX_HEAD.POST_PROCESSOR]

    if (
        cfg.MODEL.TORCHSCRIPT_ONLY
        and cfg.MODEL.ROI_BOX_HEAD.POST_PROCESSOR == "BaseBoxPostProcessor"
    ):
        box_post_processor = BaseBoxPostProcessorForTorchscript

    postprocessor = box_post_processor(score_thresh, nms_thresh, detections_per_img, box_coder, cfg)
    return postprocessor
