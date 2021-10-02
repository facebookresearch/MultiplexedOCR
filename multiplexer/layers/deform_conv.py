# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn

from .wrappers import Conv2d, _NewEmptyTensorOp


class DFConv2d(nn.Module):
    """Deformable convolutional layer"""

    def __init__(
        self,
        in_channels,
        out_channels,
        with_modulated_dcn=True,
        kernel_size=3,
        stride=1,
        groups=1,
        dilation=1,
        deformable_groups=1,
        bias=False,
    ):
        super(DFConv2d, self).__init__()
        if isinstance(kernel_size, (list, tuple)):
            assert len(kernel_size) == 2
            offset_base_channels = kernel_size[0] * kernel_size[1]
        else:
            offset_base_channels = kernel_size * kernel_size
        if with_modulated_dcn:
            from d2ocr.layers import ModulatedDeformConv

            offset_channels = offset_base_channels * 3  # default: 27
            conv_block = ModulatedDeformConv
        else:
            from d2ocr.layers import DeformConv

            offset_channels = offset_base_channels * 2  # default: 18
            conv_block = DeformConv
        self.offset = Conv2d(
            in_channels,
            deformable_groups * offset_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation,
            groups=1,
            dilation=dilation,
        )
        for layer in [
            self.offset,
        ]:
            nn.init.kaiming_uniform_(layer.weight, a=1)
            torch.nn.init.constant_(layer.bias, 0.0)

        self.conv = conv_block(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=groups,
            deformable_groups=deformable_groups,
            bias=bias,
        )
        self.with_modulated_dcn = with_modulated_dcn
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = dilation
        self.dilation = dilation

    def forward(self, x):
        if x.numel() > 0:
            if not self.with_modulated_dcn:
                offset = self.offset(x)
                x = self.conv(x, offset)
            else:
                offset_mask = self.offset(x)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, -9:, :, :].sigmoid()
                x = self.conv(x, offset, mask)
            return x
        # get output shape
        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(
                x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
            )
        ]
        output_shape = [x.shape[0], self.conv.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)
