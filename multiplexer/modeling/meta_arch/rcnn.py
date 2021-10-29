import torch
from torch import nn

from multiplexer.modeling.backbone import build_backbone
from multiplexer.modeling.proposal_generator import build_proposal_generator
from multiplexer.modeling.roi_heads import build_roi_heads

# from multiplexer.modeling.segmentation.seg_module_builder import build_segmentation
from multiplexer.structures.image_list import to_image_list
from multiplexer.structures.word_result import WordResult
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

        self.proposal = build_proposal_generator(cfg)
        # if cfg.MODEL.SEG_ON:
        #     self.proposal = build_segmentation(cfg)
        # else:
        #     self.proposal = build_rpn(cfg)
        if cfg.MODEL.TRAIN_DETECTION_ONLY:
            self.roi_heads = None
        else:
            self.roi_heads = build_roi_heads(cfg)

        if self.cfg.MODEL.SEG_ON and self.cfg.MODEL.SEG.USE_FUSE_FEATURE:
            self.in_feature_type = "fuse_feature"
        elif self.cfg.MODEL.FPN.USE_PRETRAINED:
            self.in_feature_type = "pretrained"
        else:
            self.in_feature_type = "features"

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

        proposal_out = self.forward_proposal(images, features, targets)
        return self.forward_roi_heads(proposal_out, targets)

    def forward_proposal(self, images, features, targets=None):
        fuse_feature = None
        proposals = None
        proposal_losses = None
        seg_results = None

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

        if self.in_feature_type == "image":
            in_features = images.tensors
        elif self.in_feature_type == "fuse_feature":
            in_features = fuse_feature
        elif self.in_feature_type == "pretrained":
            in_features = list(features.values())
        else:
            in_features = features

        return {
            "in_features": in_features,
            "proposals": proposals,
            "proposal_losses": proposal_losses,
            "seg_results": seg_results,
        }

    def forward_roi_heads(self, proposal_out, targets=None):
        if self.roi_heads is not None:
            x, result, detector_losses = self.roi_heads(
                proposal_out["in_features"], proposal_out["proposals"], targets
            )
        else:
            # RPN-only models don't have roi_heads
            # x = features
            result = proposal_out["proposals"]
            detector_losses = {}

        if self.training:
            losses = {}
            if self.roi_heads is not None:
                losses.update(detector_losses)
            losses.update(proposal_out["proposal_losses"])
            return losses
        else:
            cpu_device = torch.device("cpu")
            
            prediction_dict = {
                "global_prediction": None,
            }

            if self.cfg.MODEL.TRAIN_DETECTION_ONLY:
                prediction_dict["global_prediction"] = [obj.to(cpu_device) for obj in result]
                assert len(proposal_out["seg_results"]["scores"]) == 1
                prediction_dict["scores"] = proposal_out["seg_results"]["scores"][0].to(cpu_device).tolist()
                # Add dummy word result list
                word_result_list = []
                for _ in range(len(prediction_dict["scores"])):
                    word_result = WordResult()
                    word_result.seq_word = ""
                    word_result_list.append(word_result)
                prediction_dict["word_result_list"] = word_result_list
            else:
                prediction_dict["global_prediction"] = [obj.to(cpu_device) for obj in result[0]]

                prediction_dict["word_result_list"] = result[1]["word_result_list"]
            
            return prediction_dict
