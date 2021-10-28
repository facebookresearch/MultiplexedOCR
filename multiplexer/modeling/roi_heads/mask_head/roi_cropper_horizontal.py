import torch
import torch.nn.functional as F
from torch import nn


class HorizontalROICropper(nn.Module):
    def __init__(self, height=48, width=320):
        super(HorizontalROICropper, self).__init__()
        self.height = height
        self.width = width

    def forward(self, x, proposals):
        patches = []

        for i in range(len(proposals)):
            max_h = x.shape[2] - 1
            max_w = x.shape[3] - 1
            bboxes = proposals[i].bbox.int()
            bboxes[:, 0] = torch.clamp(bboxes[:, 0], 0, max_w)
            bboxes[:, 1] = torch.clamp(bboxes[:, 1], 0, max_h)
            bboxes[:, 2] = torch.clamp(bboxes[:, 2], 0, max_w)
            bboxes[:, 3] = torch.clamp(bboxes[:, 3], 0, max_h)
            bboxes[:, 2] = torch.max(bboxes[:, 2], bboxes[:, 0])
            bboxes[:, 3] = torch.max(bboxes[:, 3], bboxes[:, 1])
            for box in bboxes:
                patches.append(
                    F.interpolate(
                        x[i : i + 1, :, box[1] : box[3] + 1, box[0] : box[2] + 1],
                        (self.height, self.width),
                    )
                )

        return torch.vstack(patches)
