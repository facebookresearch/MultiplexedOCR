import math

import torch


class RotatedBoxList(object):
    def __init__(self, tensor, image_size):
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        self.tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        self.size = image_size  # (image_width, image_height)

    def transpose(self, method):
        return self

    def crop(self, box, keep_ind=None):
        if keep_ind is not None:
            self.tensor = [self.tensor[i] for i in keep_ind]
        return self

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))
        ratio_width, ratio_height = ratios
        self.scale(ratio_width, ratio_height)
        self.size = size
        return self

    def scale(self, scale_x, scale_y):
        """
        Scale the rotated box with horizontal and vertical scaling factors
        Note: when scale_factor_x != scale_factor_y,
        the rotated box does not preserve the rectangular shape when the angle
        is not a multiple of 90 degrees under resize transformation.
        Instead, the shape is a parallelogram (that has skew)
        Here we make an approximation by fitting a rotated rectangle to the parallelogram.
        """

        self.tensor[:, 0] *= scale_x
        self.tensor[:, 1] *= scale_y
        theta = self.tensor[:, 4] * math.pi / 180.0
        c = torch.cos(theta)
        s = torch.sin(theta)

        self.tensor[:, 2] *= torch.sqrt((scale_x * c) ** 2 + (scale_y * s) ** 2)
        self.tensor[:, 3] *= torch.sqrt((scale_x * s) ** 2 + (scale_y * c) ** 2)
        self.tensor[:, 4] = torch.atan2(scale_x * s, scale_y * c) * 180 / math.pi

    def set_size(self, size):
        pass

    def rotate(self, angle, r_c, start_h, start_w):
        return self

    def __iter__(self):
        return iter(self.tensor)

    def __getitem__(self, item):
        if isinstance(item, (int, slice)):
            selected_rotated_boxes = [self.tensor[item]]
        else:
            # advanced indexing on a single dimension
            selected_rotated_boxes = []
            # original_item_for_debug = item.clone()  # DEBUG
            if isinstance(item, torch.Tensor) and item.dtype == torch.uint8:
                item = item.nonzero()
                item = item.squeeze(1) if item.numel() > 0 else item
                item = item.tolist()
            for i in item:
                assert i < len(self.tensor), "i = {}, len(self.tensor) = {}".format(
                    i, len(self.tensor)
                )
                selected_rotated_boxes.append(self.tensor[i])
        return RotatedBoxList(torch.stack(selected_rotated_boxes), self.size)

    def __len__(self):
        return len(self.tensor)

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "image_size={}, ".format(self.size)
        s += "tensor={}, ".format(self.tensor)
        s += ")"
        return s
