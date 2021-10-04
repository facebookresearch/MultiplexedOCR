# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch

from multiplexer.modeling.roi_heads.box_head.loss import make_roi_box_loss_evaluator
from multiplexer.modeling.roi_heads.box_head.roi_box_feature_extractors import (
    make_roi_box_feature_extractor,
)
from multiplexer.modeling.roi_heads.box_head.roi_box_post_processor_builder import (
    make_roi_box_post_processor,
)
from multiplexer.modeling.roi_heads.box_head.roi_box_predictors import make_roi_box_predictor

from .build import BOX_HEAD_REGISTRY


@BOX_HEAD_REGISTRY.register()
class ROIBoxHead(torch.nn.Module):
    """
    Generic Box Head class.
    """

    def __init__(self, cfg):
        super(ROIBoxHead, self).__init__()
        self.cfg = cfg
        self.feature_extractor = make_roi_box_feature_extractor(cfg)
        self.predictor = make_roi_box_predictor(cfg)
        self.post_processor = make_roi_box_post_processor(cfg)
        self.loss_evaluator = make_roi_box_loss_evaluator(cfg)

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the subsampled proposals
                are returned. During testing, the predicted boxlists are returned
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """

        if self.training:
            # Faster R-CNN subsamples during training the proposals with a fixed
            # positive / negative ratio
            with torch.no_grad():
                proposals = self.loss_evaluator.subsample(proposals, targets)

        # extract features that will be fed to the final classifier. The
        # feature_extractor generally corresponds to the pooler + heads
        x = self.feature_extractor(features, proposals)
        # final classifier that converts the features into predictions
        class_logits, box_regression = self.predictor(x)
        if not self.training:
            if self.cfg.MODEL.ROI_BOX_HEAD.INFERENCE_USE_BOX:
                result = self.post_processor((class_logits, box_regression), proposals)
                # print(result[0].get_field('masks'))
                return x, result, {}
            else:
                return x, proposals, {}

        loss_classifier, loss_box_reg = self.loss_evaluator([class_logits], [box_regression])
        if self.cfg.MODEL.ROI_BOX_HEAD.USE_REGRESSION:
            return (
                x,
                proposals,
                {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg},
            )
        else:
            return (x, proposals, {"loss_classifier": loss_classifier})
