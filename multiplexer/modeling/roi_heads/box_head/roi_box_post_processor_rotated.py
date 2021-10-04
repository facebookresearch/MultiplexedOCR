import torch
import torch.nn.functional as F

from multiplexer.modeling.roi_heads.box_head.roi_box_post_processor_base import BaseBoxPostProcessor
from multiplexer.structures.bounding_box import BoxList
from multiplexer.structures.boxlist_ops import boxlist_nms, cat_boxlist


class RotatedBoxPostProcessor(BaseBoxPostProcessor):
    """
    From a set of classification scores, box regression and proposals,
    computes the post-processed boxes, and applies NMS to obtain the
    final results
    """

    def __init__(
        self,
        score_thresh=0.05,
        nms=0.5,
        detections_per_img=100,
        box2box_transform=None,
        cfg=None,
    ):
        """
        Arguments:
            score_thresh (float)
            nms (float)
            detections_per_img (int)
            box2box_transform (Box2BoxTransform)
        """
        super(RotatedBoxPostProcessor, self).__init__(
            score_thresh, nms, detections_per_img, box2box_transform, cfg
        )

    def forward(self, x, boxes):
        """
        Arguments:
            x (tuple[tensor, tensor]): x contains the class logits
                and the box_regression from the model.
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for ech image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra fields labels and scores
        """
        class_logits, box_regression = x
        class_prob = F.softmax(class_logits, -1)

        # TODO think about a representation of batch of boxes
        image_shapes = [box.size for box in boxes]
        boxes_per_image = [len(box) for box in boxes]
        if (
            self.cfg.MODEL.SEG.USE_SEG_POLY
            or self.cfg.MODEL.ROI_BOX_HEAD.USE_MASKED_FEATURE
            or self.cfg.MODEL.ROI_MASK_HEAD.USE_MASKED_FEATURE
        ):
            masks = [box.get_field("masks") for box in boxes]
            rotated_boxes_5d = [box.get_field("rotated_boxes_5d") for box in boxes]

        if self.cfg.MODEL.ROI_BOX_HEAD.USE_REGRESSION:
            concat_boxes = torch.cat([a.bbox for a in boxes], dim=0)
            proposals = self.box2box_transform.decode(
                box_regression.view(sum(boxes_per_image), -1), concat_boxes
            )
            proposals = proposals.split(boxes_per_image, dim=0)
        else:
            proposals = boxes
        num_classes = class_prob.shape[1]
        class_prob = class_prob.split(boxes_per_image, dim=0)

        results = []
        if (
            self.cfg.MODEL.SEG.USE_SEG_POLY
            or self.cfg.MODEL.ROI_BOX_HEAD.USE_MASKED_FEATURE
            or self.cfg.MODEL.ROI_MASK_HEAD.USE_MASKED_FEATURE
        ):
            for prob, boxes_per_img, image_shape, mask, rotated_box_5d in zip(
                class_prob, proposals, image_shapes, masks, rotated_boxes_5d
            ):
                boxlist = self.prepare_boxlist(
                    boxes_per_img, prob, image_shape, mask, rotated_box_5d
                )
                if self.cfg.MODEL.ROI_BOX_HEAD.USE_REGRESSION:
                    boxlist = boxlist.clip_to_image(remove_empty=False)
                    boxlist = self.filter_results(boxlist, num_classes)
                results.append(boxlist)
        else:
            for prob, boxes_per_img, image_shape in zip(class_prob, proposals, image_shapes):
                boxlist = self.prepare_boxlist(boxes_per_img, prob, image_shape)
                if self.cfg.MODEL.ROI_BOX_HEAD.USE_REGRESSION:
                    boxlist = boxlist.clip_to_image(remove_empty=False)
                    boxlist = self.filter_results(boxlist, num_classes)
                results.append(boxlist)
        return results

    def prepare_boxlist(self, boxes, scores, image_shape, mask=None, rotated_box_5d=None):
        """
        Returns BoxList from `boxes` and adds probability scores information
        as an extra field
        `boxes` has shape (#detections, 4 * #classes), where each row represents
        a list of predicted bounding boxes for each of the object classes in the
        dataset (including the background class). The detections in each row
        originate from the same object proposal.
        `scores` has shape (#detection, #classes), where each row represents a list
        of object detection confidence scores for each of the object classes in the
        dataset (including the background class). `scores[i, j]`` corresponds to the
        box at `boxes[i, j * 4:(j + 1) * 4]`.
        """
        if not self.cfg.MODEL.ROI_BOX_HEAD.USE_REGRESSION:
            scores = scores.reshape(-1)
            boxes.add_field("scores", scores)
            return boxes
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        boxlist = BoxList(boxes, image_shape, mode="xyxy")
        boxlist.add_field("scores", scores)
        if mask is not None:
            boxlist.add_field("masks", mask, check_consistency=False)
        if rotated_box_5d is not None:
            boxlist.add_field("rotated_boxes_5d", rotated_box_5d, check_consistency=False)
        return boxlist

    def filter_results(self, boxlist, num_classes):
        """Returns bounding-box detection results by thresholding on scores and
        applying non-maximum suppression (NMS).
        """
        # unwrap the boxlist to avoid additional overhead.
        # if we had multi-class NMS, we could perform this directly on the boxlist
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)
        # if (
        #     self.cfg.MODEL.SEG.USE_SEG_POLY
        #     or self.cfg.MODEL.ROI_BOX_HEAD.USE_MASKED_FEATURE
        #     or self.cfg.MODEL.ROI_MASK_HEAD.USE_MASKED_FEATURE
        # ):
        masks = boxlist.get_field("masks")
        rotated_boxes_5d = boxlist.get_field("rotated_boxes_5d")

        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > self.score_thresh
        for j in range(1, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4 : (j + 1) * 4]
            masks_j = masks[inds]
            rotated_boxes_5d_j = rotated_boxes_5d[inds]

            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class.add_field("masks", masks_j)
            boxlist_for_class.add_field("rotated_boxes_5d", rotated_boxes_5d_j)

            boxlist_for_class = boxlist_nms(boxlist_for_class, self.nms, score_field="scores")
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j, dtype=torch.int64, device=device)
            )

            result.append(boxlist_for_class)

        result = cat_boxlist(result)
        number_of_detections = len(result)

        # Limit to max_per_image detections **over all classes**
        if number_of_detections > self.detections_per_img > 0:
            cls_scores = result.get_field("scores")
            image_thresh, _ = torch.kthvalue(
                cls_scores.cpu(), number_of_detections - self.detections_per_img + 1
            )
            keep = cls_scores >= image_thresh.item()
            keep = torch.nonzero(keep).squeeze(1)
            result = result[keep]
        return result
