import torch

from multiplexer.modeling.roi_heads.mask_head import build_roi_mask_head

from .build import ROI_HEADS_REGISTRY


@ROI_HEADS_REGISTRY.register()
class MaskROIHead(torch.nn.Module):
    def __init__(self, cfg):
        super(MaskROIHead, self).__init__()
        self.mask = build_roi_mask_head(cfg)

    def forward(self, features, proposals, targets=None):
        return self.mask(features, proposals, targets)
