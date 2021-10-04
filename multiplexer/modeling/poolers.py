# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch
from torch import nn

from multiplexer.layers import ROIAlign, cat


class LevelMapper(object):
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.
    """

    def __init__(self, k_min, k_max, canonical_scale=224, canonical_level=4, eps=1e-6):
        """
        Arguments:
            k_min (int)
            k_max (int)
            canonical_scale (int)
            canonical_level (int)
            eps (float)
        """
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists):
        """
        Arguments:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.sqrt(cat([boxlist.area() for boxlist in boxlists]))
        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0 + self.eps))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return target_lvls.to(torch.int64) - self.k_min


class Pooler(nn.Module):
    """
    Pooler for Detection with or without FPN.
    It currently hard-code ROIAlign in the implementation,
    but that can be made more generic later on.
    Also, the requirement of passing the scales is not strictly necessary, as they
    can be inferred from the size of the feature map / size of original image,
    which is available thanks to the BoxList.
    """

    def __init__(self, output_size, scales, sampling_ratio, pooler_type="ROIAlign"):
        """
        Arguments:
            output_size (list[tuple[int]] or list[int]): output size for the pooled region
            scales (list[flaot]): scales for each Pooler
            sampling_ratio (int): sampling ratio for ROIAlign
        """
        super(Pooler, self).__init__()
        poolers = []
        for scale in scales:
            if pooler_type == "ROIAlign":
                poolers.append(
                    ROIAlign(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                )
            elif pooler_type == "ROIAlignV2":
                from detectron2.layers.roi_align import ROIAlign as ROIAlignV2

                poolers.append(
                    ROIAlignV2(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                )
            elif pooler_type == "ROIAlignRotated":
                from detectron2.layers.roi_align_rotated import ROIAlignRotated

                poolers.append(
                    ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                )
        self.pooler_type = pooler_type
        self.poolers = nn.ModuleList(poolers)
        self.output_size = output_size
        # get the levels in the feature map by leveraging the fact that the network always
        # downsamples by a factor of 2 at each level.
        lvl_min = -math.log2(scales[0])
        lvl_max = -math.log2(scales[-1])
        self.map_levels = LevelMapper(lvl_min, lvl_max)

    def convert_to_roi_format(self, boxes):
        concat_boxes = cat([b.bbox for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [torch.full((len(b), 1), i, dtype=dtype, device=device) for i, b in enumerate(boxes)],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)
        return rois

    def convert_rbox_to_roi_format(self, boxes):
        # torch.cat([b.get_field("rotated_boxes_5d").tensor for b in boxes])
        concat_boxes = cat([b.get_field("rotated_boxes_5d").tensor for b in boxes], dim=0)
        device, dtype = concat_boxes.device, concat_boxes.dtype
        ids = cat(
            [torch.full((len(b), 1), i, dtype=dtype, device=device) for i, b in enumerate(boxes)],
            dim=0,
        )
        rois = torch.cat([ids, concat_boxes], dim=1)

        return rois

    def forward(self, x, boxes):
        """
        Arguments:
            x (list[Tensor]): feature maps for each level
            boxes (list[BoxList]): boxes to be used to perform the pooling operation.
        Returns:
            result (Tensor)
        """
        num_levels = len(self.poolers)
        if self.pooler_type == "ROIAlign" or self.pooler_type == "ROIAlignV2":
            rois = self.convert_to_roi_format(boxes)
        else:
            assert self.pooler_type == "ROIAlignRotated"
            rois = self.convert_rbox_to_roi_format(boxes)

        if num_levels == 1:
            return self.poolers[0](x[0], rois)
        levels = self.map_levels(boxes)
        num_rois = len(rois)
        num_channels = x[0].shape[1]
        output_size_h = self.output_size[0]
        output_size_w = self.output_size[1]

        dtype, device = x[0].dtype, x[0].device
        result = torch.zeros(
            (num_rois, num_channels, output_size_h, output_size_w),
            dtype=dtype,
            device=device,
        )

        for level, (per_level_feature, pooler) in enumerate(zip(x, self.poolers)):
            idx_in_level = torch.nonzero(levels == level).squeeze(1)
            rois_per_level = rois[idx_in_level]
            result[idx_in_level] = pooler(per_level_feature, rois_per_level)

        return result
