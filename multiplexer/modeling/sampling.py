# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from multiplexer.layers import nonzero_tuple

__all__ = ["subsample_labels"]


def subsample_labels_old(labels: torch.Tensor, num_samples: int, positive_fraction: float):
    """
        Arguments:
            labels: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.
    """
    pos_idx = []
    neg_idx = []
    for matched_idxs_per_image in labels:
        positive = torch.nonzero(matched_idxs_per_image >= 1).squeeze(1)
        negative = torch.nonzero(matched_idxs_per_image == 0).squeeze(1)

        num_pos = int(num_samples * positive_fraction)
        # protect against not enough positive examples
        num_pos = min(positive.numel(), num_pos)
        num_neg = num_samples - num_pos
        # protect against not enough negative examples
        num_neg = min(negative.numel(), num_neg)

        # randomly select positive and negative examples
        perm1 = torch.randperm(positive.numel())[:num_pos]
        perm2 = torch.randperm(negative.numel())[:num_neg]

        pos_idx_per_image = positive[perm1]
        neg_idx_per_image = negative[perm2]

        # create binary mask from indices
        pos_idx_per_image_mask = torch.zeros_like(
            matched_idxs_per_image, dtype=torch.uint8
        )
        neg_idx_per_image_mask = torch.zeros_like(
            matched_idxs_per_image, dtype=torch.uint8
        )
        pos_idx_per_image_mask[pos_idx_per_image] = 1
        neg_idx_per_image_mask[neg_idx_per_image] = 1

        pos_idx.append(pos_idx_per_image_mask)
        neg_idx.append(neg_idx_per_image_mask)

    return pos_idx, neg_idx


def subsample_labels(
    labels: torch.Tensor, num_samples: int, positive_fraction: float, bg_label: int
):
    """
    Return `num_samples` (or fewer, if not enough found)
    random samples from `labels` which is a mixture of positives & negatives.
    It will try to return as many positives as possible without
    exceeding `positive_fraction * num_samples`, and then try to
    fill the remaining slots with negatives.
    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.
    Returns:
        pos_idx, neg_idx (Tensor):
            1D vector of indices. The total length of both is `num_samples` or fewer.
    """
    positive = nonzero_tuple((labels != -1) & (labels != bg_label))[0]
    negative = nonzero_tuple(labels == bg_label)[0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = min(positive.numel(), num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = min(negative.numel(), num_neg)

    # randomly select positive and negative examples
    perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

    pos_idx = positive[perm1]
    neg_idx = negative[perm2]
    return pos_idx, neg_idx
