# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import cv2
import numpy as np
import torch
from torch import nn


class DynamicRotatedROICropper(nn.Module):
    def __init__(self, height=48, width=320):
        super(DynamicRotatedROICropper, self).__init__()
        self.height = height
        self.min_output_width = 16
        self.max_output_width = width  # used as max width instead of fixed width
        self.horizontal_stretch_factor = 1.5
        self.hxstretch = self.height * self.horizontal_stretch_factor

    def forward(self, x, proposals):
        device = x.device

        patches = []

        assert len(x) == len(proposals)  # == number of images in the batch

        resize_w_list = []

        batch_width = self.min_output_width

        for k in range(len(proposals)):
            rotated_boxes = proposals[k].get_field("rotated_boxes_5d")
            resize_w = torch.clamp(
                torch.round(
                    self.hxstretch * rotated_boxes.tensor[:, 2] / rotated_boxes.tensor[:, 3]
                ).int(),
                self.min_output_width,
                self.max_output_width,
            )
            max_width = torch.max(resize_w)
            batch_width = max(batch_width, max_width)
            resize_w_list.append(resize_w)

        for k in range(len(proposals)):
            # .contiguous() is necessary if we use the C++ op of warp_perspective
            # which maps the address directly
            # NCHW -> CHW -> HWC
            cv2_img = x[k].permute(1, 2, 0).contiguous().cpu().numpy()
            rotated_boxes = proposals[k].get_field("rotated_boxes_5d")
            # for box in proposals[k].rotated_boxes:
            for i, box in enumerate(rotated_boxes):
                if isinstance(box, torch.Tensor):
                    box = box.tolist()
                rect = ((box[0], box[1]), (box[2], box[3]), -box[4])
                # obtain the corner points of rotated box
                src_pts = cv2.boxPoints(rect)

                # in case we need to keep the same width and height:
                # dst_pts = np.array(
                # [[0, box[3]], [0, 0], [box[2], 0], [box[2], box[3]]], dtype="float32")
                dst_pts = np.array(
                    [
                        [0, self.height],
                        [0, 0],
                        [resize_w_list[k][i], 0],
                        [resize_w_list[k][i], self.height],
                    ],
                    dtype="float32",
                )

                transform_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)

                cv2_crop_img = cv2.warpPerspective(
                    cv2_img, transform_mat, (int(resize_w_list[k][i]), self.height)
                )

                # HwC -> CHw -> CHW (with 0-padding) -> NCHW (N=1)
                patches.append(
                    torch.cat(
                        (
                            torch.tensor(cv2_crop_img, device=device).permute(2, 0, 1),
                            torch.zeros(
                                3, self.height, batch_width - resize_w_list[k][i], device=device
                            ),
                        ),
                        dim=2,
                    ).unsqueeze(0)
                )

        return torch.vstack(patches)
