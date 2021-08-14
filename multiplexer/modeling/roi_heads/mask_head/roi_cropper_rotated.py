# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import numpy as np
import torch
from torch import nn


class RotatedROICropper(nn.Module):
    def __init__(self, height=48, width=320):
        super(RotatedROICropper, self).__init__()
        self.height = height
        self.width = width
        self.dst_pts = np.array(
            [
                [0, self.height],
                [0, 0],
                [self.width, 0],
                [self.width, self.height],
            ],
            dtype="float32",
        )

    def forward(self, x, proposals):
        device = x.device

        patches = []

        assert len(x) == len(proposals)  # == number of images in the batch

        for k in range(len(proposals)):
            # .contiguous() is necessary if we use the C++ op of warp_perspective
            # which maps the address directly
            # NCHW -> CHW -> HWC
            cv2_img = x[k].permute(1, 2, 0).contiguous().cpu().numpy()
            rotated_boxes = proposals[k].get_field("rotated_boxes_5d")
            # for box in proposals[k].rotated_boxes:
            for box in rotated_boxes:
                rect = ((box[0], box[1]), (box[2], box[3]), -box[4])
                # obtain the corner points of rotated box
                src_pts = cv2.boxPoints(rect)

                # in case we need to keep the same width and height:
                # dst_pts = np.array(
                # [[0, box[3]], [0, 0], [box[2], 0], [box[2], box[3]]], dtype="float32")

                transform_mat = cv2.getPerspectiveTransform(src_pts, self.dst_pts)

                cv2_crop_img = cv2.warpPerspective(
                    cv2_img, transform_mat, (self.width, self.height)
                )

                patches.append(
                    torch.tensor(cv2_crop_img, device=device).permute(2, 0, 1).unsqueeze(0)
                )

        return torch.vstack(patches)
