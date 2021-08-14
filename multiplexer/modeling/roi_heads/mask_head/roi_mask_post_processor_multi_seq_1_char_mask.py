# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from torch import nn
from torch.nn import functional as F

from multiplexer.structures import BoxList


class MultiSeq1CharMaskPostProcessor(nn.Module):
    """
    From the results of the CNN, post process the masks
    by taking the mask corresponding to the class with max
    probability (which are of fixed size and directly output
    by the CNN) and return the masks in the mask field of the BoxList.

    If a masker object is passed, it will additionally
    project the masks in the image according to the locations in boxes,
    """

    def __init__(self, cfg, masker=None):
        super(MultiSeq1CharMaskPostProcessor, self).__init__()
        self.masker = masker
        self.cfg = cfg

    def forward(self, pred_output, boxes):
        """
        Arguments:
            pred_output["mask_logits"] (Tensor): the mask logits
            pred_output["char_mask_logits"] (Tensor): the char mask logits
            boxes (list[BoxList]): bounding boxes that are used as
                reference, one for each image

        Returns:
            results (list[BoxList]): one BoxList for each image, containing
                the extra field mask
        """
        mask_logits = pred_output["mask_logits"]
        char_mask_logits = pred_output["char_mask_logits"]

        boxes_per_image = [len(box) for box in boxes]
        results = []

        if mask_logits is not None:
            mask_prob = mask_logits.sigmoid()
            mask_prob = mask_prob.squeeze(dim=1)[:, None]
            if self.masker:
                mask_prob = self.masker(mask_prob, boxes)

            mask_prob = mask_prob.split(boxes_per_image, dim=0)

            for prob, box in zip(mask_prob, boxes):
                bbox = BoxList(box.bbox, box.size, mode="xyxy")
                for field in box.fields():
                    bbox.add_field(field, box.get_field(field))
                bbox.add_field("mask", prob)
                results.append(bbox)
        else:
            for box in boxes:
                bbox = BoxList(box.bbox, box.size, mode="xyxy")
                for field in box.fields():
                    bbox.add_field(field, box.get_field(field))
                results.append(bbox)

        if self.cfg.MULTIPLEXER.TEST.RUN_ALL_HEADS:
            char_results = {
                "char_mask": None,
                "boxes": boxes[0].bbox.detach().cpu().numpy(),
                "seq_outputs_list": pred_output["seq_outputs_list"],
                "seq_scores_list": pred_output["seq_scores_list"],
                "detailed_seq_scores_list": pred_output["detailed_seq_scores_list"],
                "language_probs": pred_output["language_probs"],
            }
        else:
            char_results = {
                "word_result_list": pred_output["word_result_list"],
            }

        if char_mask_logits is not None:
            char_mask_softmax = F.softmax(char_mask_logits, dim=1)
            char_results["char_mask"] = char_mask_softmax.cpu().numpy()
        else:
            char_results["char_mask"] = None

        return [results, char_results]
