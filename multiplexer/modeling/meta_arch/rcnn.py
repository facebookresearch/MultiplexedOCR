from torch import nn

from multiplexer.modeling.backbone import build_backbone

from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    = rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        if cfg.MODEL.SEG_ON:
            self.proposal = build_segmentation(cfg)
        else:
            self.proposal = build_rpn(cfg)
        if cfg.MODEL.TRAIN_DETECTION_ONLY:
            self.roi_heads = None
        else:
            self.roi_heads = build_roi_heads(cfg)

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        if self.cfg.MODEL.SEG_ON and not self.training:
            (proposals, seg_results), fuse_feature = self.proposal(images, features, targets)
        else:
            if self.cfg.MODEL.SEG_ON:
                (proposals, proposal_losses), fuse_feature = self.proposal(
                    images, features, targets
                )
            else:
                in_features = features
                if self.cfg.MODEL.FPN.USE_PRETRAINED:
                    in_features = list(features.values())
                proposals, proposal_losses = self.proposal(images, in_features, targets)
        if self.roi_heads is not None:
            if self.cfg.MODEL.SEG_ON and self.cfg.MODEL.SEG.USE_FUSE_FEATURE:
                x, result, detector_losses = self.roi_heads(fuse_feature, proposals, targets)
            else:
                in_features = features
                if self.cfg.MODEL.FPN.USE_PRETRAINED:
                    in_features = list(features.values())
                x, result, detector_losses = self.roi_heads(in_features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            # x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            if self.roi_heads is not None:
                losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses
        else:
            if self.cfg.MODEL.SEG_ON:
                return result, proposals, seg_results
            else:
                return result
