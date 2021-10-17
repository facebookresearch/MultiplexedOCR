#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


import torch
from torch import nn

from multiplexer.layers import cat

# from multiplexer.layers import Conv2d
from multiplexer.modeling.matcher import Matcher
from multiplexer.structures import LanguageList, pairwise_iou
from multiplexer.structures.bounding_box import BoxList
from multiplexer.utils.languages import get_language_config, lang_code_to_char_map_class


from .build import ROI_MASK_HEAD_REGISTRY
from .loss import make_roi_mask_loss_evaluator
from .roi_mask_feature_extractor_builder import make_roi_mask_feature_extractor

# from .inference import make_roi_mask_post_processor
from .roi_mask_post_processor_builder import make_roi_mask_post_processor
from .roi_mask_predictor_builder import make_roi_mask_predictor


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, has_bias=False):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


# TODO
def project_char_masks_on_boxes(
    segmentation_masks,
    segmentation_char_masks,
    proposals,
    discretization_size,
    encode_language="en_num_36",
    num_chars=36,
    gt_languages=None,
    char_map_class=None,
):
    """
    Given segmentation masks and the bounding boxes corresponding
    to the location of the masks in the image, this function
    crops and resizes the masks in the position defined by the
    boxes. This prepares the masks for them to be fed to the
    loss computation as the targets.

    Arguments:
        segmentation_masks: an instance of SegmentationMask
        proposals: an instance of BoxList
        char_map_class: the char_map_class corresponding to the encode_language
    """
    masks = []
    char_masks = []
    char_mask_weights = []
    decoder_targets = []
    word_targets = []
    M_H, M_W = discretization_size[0], discretization_size[1]
    device = proposals.bbox.device
    proposals = proposals.convert("xyxy")
    assert segmentation_masks.size == proposals.size, "{}, {}".format(segmentation_masks, proposals)
    assert segmentation_char_masks.size == proposals.size, "{}, {}".format(
        segmentation_char_masks, proposals
    )
    # TODO put the proposals on the CPU, as the representation for the
    # masks is not efficient GPU-wise (possibly several small tensors for
    # representing a single instance mask)
    proposals = proposals.bbox.to(torch.device("cpu"))
    for segmentation_mask, segmentation_char_mask, proposal, gt_language in zip(
        segmentation_masks, segmentation_char_masks, proposals, gt_languages
    ):
        # crop the masks, resize them to the desired resolution and
        # then convert them to the tensor representation,
        # instead of the list representation that was used
        cropped_mask = segmentation_mask.crop(proposal)
        scaled_mask = cropped_mask.resize((M_W, M_H))
        mask = scaled_mask.convert(mode="mask")
        masks.append(mask)
        cropped_char_mask = segmentation_char_mask.crop(proposal)
        scaled_char_mask = cropped_char_mask.resize((M_W, M_H))
        (
            char_mask,
            char_mask_weight,
            decoder_target,
            word_target,
        ) = scaled_char_mask.convert_seq_char_mask(
            encode_language=encode_language,
            num_chars=num_chars,
            gt_language=gt_language,
            char_map_class=char_map_class,
        )
        assert torch.all(word_target >= -2), "Big Negative word_target detected! {}".format(
            word_target
        )
        assert torch.all(word_target <= 100000), "Huge word_target detected! {}".format(word_target)
        char_masks.append(char_mask)
        char_mask_weights.append(char_mask_weight)
        decoder_targets.append(decoder_target)
        word_targets.append(word_target)

    if len(masks) == 0:
        stacked_masks = torch.empty(0, dtype=torch.float32, device=device)
        stacked_char_masks = torch.empty(0, dtype=torch.long, device=device)
        stacked_char_mask_weights = torch.empty(0, dtype=torch.float32, device=device)
        stacked_decoder_targets = torch.empty(0, dtype=torch.long, device=device)
        stacked_word_targets = None  # need verification
    else:
        stacked_masks = torch.stack(masks, dim=0).to(device, dtype=torch.float32)
        stacked_char_masks = torch.stack(char_masks, dim=0).to(device, dtype=torch.long)
        stacked_char_mask_weights = torch.stack(char_mask_weights, dim=0).to(
            device, dtype=torch.float32
        )
        stacked_decoder_targets = torch.stack(decoder_targets, dim=0).to(device, dtype=torch.long)
        stacked_word_targets = torch.stack(word_targets, dim=0).to(device, dtype=torch.long)

    return {
        "masks": stacked_masks,
        "char_masks": stacked_char_masks,
        "char_mask_weights": stacked_char_mask_weights,
        "decoder_targets": stacked_decoder_targets,
        "word_targets": stacked_word_targets,
    }


@ROI_MASK_HEAD_REGISTRY.register()
class V1ROIMaskHead(torch.nn.Module):
    def __init__(self, cfg, proposal_matcher, discretization_size, language="en_num_36"):
        super(V1ROIMaskHead, self).__init__()
        self.language = language
        self.proposal_matcher = proposal_matcher
        self.discretization_size = discretization_size
        self.cfg = cfg.clone()
        self.feature_extractor = make_roi_mask_feature_extractor(cfg)
        self.predictor = make_roi_mask_predictor(cfg)
        self.post_processor = make_roi_mask_post_processor(cfg)
        self.loss_evaluator = make_roi_mask_loss_evaluator(cfg)
        self.char_maps = {}

    # def filter_ignored_targets(self, targets):
    #     filtered_targets = []
    #     for targets_per_image in targets:
    #         lang_list = targets_per_image.get_field("languages").languages
    #         matched = [i for i in range(len(lang_list)) if lang_list[i] != "none"]
    #         if random.random() < 0.001:
    #             num_filtered = len(targets_per_image) - len(matched)
    #             print(
    #                 "Filtered {} out of {} targets in ROIMaskHead".format(
    #                     num_filtered, len(targets_per_image)
    #                 )
    #             )
    #         filtered_targets.append(targets_per_image[matched])
    #     return filtered_targets

    def keep_only_positive_boxes(self, boxes):
        """
        Given a set of BoxList containing the `labels` field,
        return a set of BoxList for which `labels > 0`.

        Arguments:
            boxes (list of BoxList)
        """
        assert isinstance(boxes, (list, tuple))
        assert isinstance(boxes[0], BoxList)
        assert boxes[0].has_field("labels")

        batch_size_per_im = self.cfg.MODEL.ROI_MASK_HEAD.MASK_BATCH_SIZE_PER_IM

        positive_boxes = []
        positive_inds = []
        for boxes_per_image in boxes:
            labels = boxes_per_image.get_field("labels")
            inds_mask = labels > 0
            inds = torch.where(inds_mask)[0]
            if len(inds) > batch_size_per_im:
                new_inds = inds[:batch_size_per_im]
                inds_mask[inds[batch_size_per_im:]] = 0
            else:
                new_inds = inds
            positive_boxes.append(boxes_per_image[new_inds])
            positive_inds.append(inds_mask)
        return positive_boxes, positive_inds

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = pairwise_iou(target, proposal)
        # match_quality_matrix = boxlist_polygon_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)
        # Mask RCNN needs "labels" and "masks "fields for creating the targets
        target = target.copy_with_fields(["labels", "masks", "char_masks", "languages"])
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds
        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets, encode_language):
        masks = []
        char_masks = []
        char_mask_weights = []
        decoder_targets = []
        word_targets = []
        gt_language_targets = []

        if encode_language not in self.char_maps:
            char_map_class = lang_code_to_char_map_class[encode_language]
            char_map_class.init(char_map_path=self.cfg.CHAR_MAP.DIR)  # only init once
            self.char_maps[encode_language] = char_map_class
        else:
            char_map_class = self.char_maps[encode_language]

        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # this can probably be removed, but is left here for clarity
            # and completeness
            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0

            # mask scores are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            target_masks = matched_targets.get_field("masks")
            positive_masks = target_masks[positive_inds]

            target_char_masks = matched_targets.get_field("char_masks")
            positive_char_masks = target_char_masks[positive_inds]

            positive_proposals = proposals_per_image[positive_inds]

            target_languages = matched_targets.get_field("languages")
            positive_languages = target_languages[positive_inds]

            targets_per_image = project_char_masks_on_boxes(
                positive_masks,
                positive_char_masks,
                positive_proposals,
                self.discretization_size,
                encode_language=encode_language,
                # num_chars=self.cfg.MODEL.ROI_MASK_HEAD.CHAR_NUM_CLASSES,
                num_chars=get_language_config(self.cfg, encode_language).NUM_CHAR,
                gt_languages=positive_languages,
                char_map_class=char_map_class,
            )

            masks_per_image = targets_per_image["masks"]
            char_masks_per_image = targets_per_image["char_masks"]
            char_mask_weights_per_image = targets_per_image["char_mask_weights"]
            decoder_targets_per_image = targets_per_image["decoder_targets"]
            word_targets_per_image = targets_per_image["word_targets"]

            masks.append(masks_per_image)
            char_masks.append(char_masks_per_image)
            char_mask_weights.append(char_mask_weights_per_image)
            decoder_targets.append(decoder_targets_per_image)
            word_targets.append(word_targets_per_image)

            gt_language_targets.append(positive_languages)

        return {
            "masks": masks,
            "char_masks": char_masks,
            "char_mask_weights": char_mask_weights,
            "decoder_targets": decoder_targets,
            "word_targets": word_targets,
            "gt_language_targets": gt_language_targets,
        }

    def feature_mask(self, x, proposals):
        masks = []
        for proposal in proposals:
            segmentation_masks = proposal.get_field("masks")
            boxes = proposal.bbox.to(torch.device("cpu"))
            for segmentation_mask, box in zip(segmentation_masks, boxes):
                cropped_mask = segmentation_mask.crop(box)
                scaled_mask = cropped_mask.resize(
                    (
                        self.cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_W,
                        self.cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION_H,
                    )
                )
                mask = scaled_mask.convert(mode="mask")
                masks.append(mask)
        if len(masks) == 0:
            return x
        masks = torch.stack(masks, dim=0).to(x.device, dtype=torch.float32)
        soft_ratio = self.cfg.MODEL.ROI_MASK_HEAD.SOFT_MASKED_FEATURE_RATIO
        if soft_ratio > 0:
            if soft_ratio < 1.0:
                x = x * (soft_ratio + (1 - soft_ratio) * masks.unsqueeze(1))
            else:
                x = x * (1.0 + soft_ratio * masks.unsqueeze(1))
        else:
            x = x * masks.unsqueeze(1)
        return x

    def forward(self, features, proposals, targets=None):
        """
        Arguments:
            features (list[Tensor]): feature-maps from possibly several levels
            proposals (list[BoxList]): proposal boxes
            targets (list[BoxList], optional): the ground-truth targets.

        Returns:
            x (Tensor): the result of the feature extractor
            proposals (list[BoxList]): during training, the original proposals
                are returned. During testing, the predicted boxlists are returned
                with the `mask` field set
            losses (dict[Tensor]): During training, returns the losses for the
                head. During testing, returns an empty dict.
        """
        if self.training:
            # during training, only focus on positive boxes
            all_proposals = proposals

            # In some cases, e.g., we don't run box head, the proposal boxes
            # that are not matched to gt boxes haven't been filtered and
            # don't have the "labels" field, so we need to run the matcher here.
            for boxes_per_image, targets_per_image in zip(proposals, targets):
                if not boxes_per_image.has_field("labels"):
                    matched_targets = self.match_targets_to_proposals(
                        boxes_per_image, targets_per_image
                    )
                    matched_idxs = matched_targets.get_field("matched_idxs")

                    labels_per_image = matched_targets.get_field("labels")
                    labels_per_image = labels_per_image.to(dtype=torch.int64)

                    neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
                    labels_per_image[neg_inds] = 0

                    boxes_per_image.add_field("labels", labels_per_image)

            proposals, positive_inds = self.keep_only_positive_boxes(proposals)
            if all(len(proposal) == 0 for proposal in proposals):
                return None, None, None
        if self.training and self.cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            x = features
            x = x[torch.cat(positive_inds, dim=0)]
        else:
            x = self.feature_extractor(features, proposals)
            if self.cfg.MODEL.ROI_MASK_HEAD.USE_MASKED_FEATURE:
                x = self.feature_mask(x, proposals)
        if self.training:
            # kept_indices = self.filter_ignored_targets(targets)
            dict_targets = self.prepare_targets(proposals, targets, encode_language=self.language)

            mask_targets = dict_targets["masks"]
            char_mask_targets = dict_targets["char_masks"]
            char_mask_weights = dict_targets["char_mask_weights"]
            decoder_targets = dict_targets["decoder_targets"]
            word_targets = dict_targets["word_targets"]
            gt_language_targets = dict_targets["gt_language_targets"]

            decoder_targets = cat(decoder_targets, dim=0)
            word_targets = cat(word_targets, dim=0)
            gt_language_targets = LanguageList.concat(gt_language_targets)

            assert torch.all(decoder_targets >= -2), "\n".join(
                [
                    "Weird decoder_targets detected:",
                    "decoder_targets = {}".format(decoder_targets),
                ]
            )

        # During inferencing
        if not self.training:
            if x.numel() > 0:
                pred_output = self.predictor(x)
                result = self.post_processor(pred_output=pred_output, boxes=proposals)
                return x, result, {}
            else:
                return None, None, {}

        # During training

        # Prepare targets for SEQ_ON mode:
        if self.cfg.SEQUENCE.SEQ_ON:
            decoder_targets_dict = {}
            word_targets_dict = {}
            for language in set(self.cfg.SEQUENCE.LANGUAGES + self.cfg.SEQUENCE.LANGUAGES_ENABLED):
                # TODO: optimize prepare_targets so we only need to call it once
                targets_dict = self.prepare_targets(proposals, targets, encode_language=language)

                decoder_targets_one = targets_dict["decoder_targets"]
                word_targets_one = targets_dict["word_targets"]

                decoder_targets_one = cat(decoder_targets_one, dim=0)
                word_targets_one = cat(word_targets_one, dim=0)

                assert torch.all(decoder_targets_one >= -2), "\n".join(
                    [
                        "Weird decoder_targets_one detected:",
                        "decoder_targets_one = {}".format(decoder_targets_one),
                    ]
                )
                decoder_targets_dict[language] = decoder_targets_one
                word_targets_dict[language] = word_targets_one

        if self.cfg.MODEL.CHAR_MASK_ON:
            if self.cfg.SEQUENCE.SEQ_ON:
                pred_output = self.predictor(
                    x,
                    decoder_targets=decoder_targets,
                    word_targets=word_targets,
                    gt_language_targets=gt_language_targets,
                )
                # mask_logits, char_mask_logits, seq_outputs
                loss_mask, loss_char_mask = self.loss_evaluator(
                    proposals,
                    pred_output["mask_logits"],
                    pred_output["char_mask_logits"],
                    mask_targets,
                    char_mask_targets,
                    char_mask_weights,
                )
                return (
                    x,
                    all_proposals,
                    {
                        "loss_mask": loss_mask,
                        "loss_char_mask": loss_char_mask,
                        "loss_seq": pred_output["seq_outputs"],
                    },
                )
            else:
                pred_output = self.predictor(x)

                loss_mask, loss_char_mask = self.loss_evaluator(
                    proposals,
                    pred_output["mask_logits"],
                    pred_output["char_mask_logits"],
                    mask_targets,
                    char_mask_targets,
                    char_mask_weights,
                )
                return (
                    x,
                    all_proposals,
                    {"loss_mask": loss_mask, "loss_char_mask": loss_char_mask},
                )
        else:
            if self.cfg.SEQUENCE.SEQ_ON:
                if self.cfg.MODEL.MASK_ON:
                    # losses from sequence heads
                    pred_output = self.predictor(
                        x,
                        decoder_targets=decoder_targets_dict,
                        word_targets=word_targets_dict,
                        gt_language_targets=gt_language_targets,
                    )
                    # add the mask loss
                    if pred_output["mask_logits"] is not None:
                        pred_output["loss_seq_dict"]["loss_mask"] = self.loss_evaluator(
                            proposals, pred_output["mask_logits"], mask_targets
                        )

                    return (x, all_proposals, pred_output["loss_seq_dict"])
                else:
                    pred_output = self.predictor(
                        x,
                        decoder_targets=decoder_targets,
                        word_targets=word_targets,
                        gt_language_targets=gt_language_targets,
                    )
                    return (x, all_proposals, {"loss_seq": pred_output["seq_outputs"]})
            else:
                pred_output = self.predictor(x)
                loss_mask = self.loss_evaluator(proposals, pred_output["mask_logits"], targets)
                return x, all_proposals, {"loss_mask": loss_mask}
