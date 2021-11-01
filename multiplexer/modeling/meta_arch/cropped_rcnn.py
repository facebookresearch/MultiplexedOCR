from multiplexer.structures.image_list import to_image_list

from .build import META_ARCH_REGISTRY
from .rcnn import GeneralizedRCNN


@META_ARCH_REGISTRY.register()
class CroppedRCNN(GeneralizedRCNN):
    def __init__(self, cfg):
        super(CroppedRCNN, self).__init__(cfg)
        self.in_feature_type = "image"
        
    def add_score_field(self, proposal_out):
        for i in range(len(proposal_out["proposals"])):
            # DETECTION_THRESHOLD = 0.3
            # scores = proposal_out["seg_results"]["scores"][i]
            # keep_indices = torch.where(scores > DETECTION_THRESHOLD)[0]
            # print(keep_indices)
            # proposal_out["proposals"][i] = proposal_out["proposals"][i][keep_indices]
            # proposal_out["seg_results"]["rotated_boxes"][i] = \
            # proposal_out["seg_results"]["rotated_boxes"][i][keep_indices]
            # proposal_out["seg_results"]["polygons"][i] = \
            # proposal_out["seg_results"]["polygons"][i][keep_indices]
            # proposal_out["seg_results"]["preds"][i] = \
            # proposal_out["seg_results"]["preds"][i][keep_indices]
            # proposal_out["seg_results"]["scores"][i] = \
            # proposal_out["seg_results"]["scores"][i][keep_indices]
            proposal_out["proposals"][i].add_field(
                "scores", proposal_out["seg_results"]["scores"][i]
            )

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

        # TODO: unify training and testing
        if self.training:
            x, result, losses = self.roi_heads(images.tensors, targets, targets)
            return losses
        else:
            features = self.backbone(images.tensors)
            proposal_out = self.forward_proposal(images, features, targets)
            self.add_score_field(proposal_out)
            return self.forward_roi_heads(proposal_out, targets)

