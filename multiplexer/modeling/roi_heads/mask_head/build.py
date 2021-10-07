from multiplexer.modeling.matcher import Matcher
from multiplexer.utils.registry import Registry

ROI_MASK_HEAD_REGISTRY = Registry("ROI_MASK_HEAD")


def build_roi_mask_head(cfg):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_MASK_HEAD.NAME`.
    """

    matcher = Matcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD,
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    mask_head = ROI_MASK_HEAD_REGISTRY.get(cfg.MODEL.ROI_MASK_HEAD.NAME)(
        cfg=cfg,
        proposal_matcher=matcher,
        discretization_size=(
            cfg.MODEL.ROI_MASK_HEAD.RESOLUTION_H,
            cfg.MODEL.ROI_MASK_HEAD.RESOLUTION_W,
        ),
        language=cfg.MODEL.LANGUAGE,
    )

    return mask_head
