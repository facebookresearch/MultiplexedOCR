from multiplexer.utils.registry import Registry

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")


def build_roi_heads(cfg):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg)
