from multiplexer.utils.registry import Registry

ROI_BOX_HEAD_REGISTRY = Registry("ROI_BOX_HEAD")


def build_roi_box_head(cfg):
    """
    Build ROIHeads defined by `cfg.MODEL.BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    box_head = ROI_BOX_HEAD_REGISTRY.get(name)(cfg)

    if cfg.MODEL.ROI_BOX_HEAD.FROZEN:
        for p in box_head.parameters():
            p.requires_grad = False

    return box_head
