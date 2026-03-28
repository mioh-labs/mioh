# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
import torchvision
from torch.nn import init as init
from torch.nn.modules.utils import _pair, _single
import math


class MPSDeformConvUnavailableError(RuntimeError):
    pass


def _torchvision_deform_conv2d(x, offset, weight, bias, stride, padding, dilation, mask):
    return torchvision.ops.deform_conv2d(
        x, offset, weight, bias, stride, padding, dilation, mask
    )


def _mps_deform_conv2d(x, offset, weight, bias, stride, padding, dilation, mask):
    try:
        from mps_deform_conv import deform_conv2d as mps_deform_conv2d
    except Exception as e:
        raise MPSDeformConvUnavailableError(str(e)) from e

    try:
        return mps_deform_conv2d(
            x,
            offset,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            mask=mask,
        )
    except Exception as e:
        raise MPSDeformConvUnavailableError(str(e)) from e


def dispatch_deform_conv2d(x, offset, weight, bias, stride, padding, dilation, mask):
    if getattr(getattr(x, "device", None), "type", None) == "mps":
        try:
            return _mps_deform_conv2d(x, offset, weight, bias, stride, padding, dilation, mask)
        except MPSDeformConvUnavailableError:
            pass
    return _torchvision_deform_conv2d(x, offset, weight, bias, stride, padding, dilation, mask)

class ModulatedDeformConv2d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 deform_groups=1,
                 bias=True):
        super(ModulatedDeformConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deform_groups = deform_groups
        self.with_bias = bias
        # enable compatibility with nn.Conv2d
        self.transposed = False
        self.output_padding = _single(0)

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, *self.kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.init_weights()

    def init_weights(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.zero_()

        if hasattr(self, 'conv_offset'):
            self.conv_offset.weight.data.zero_()
            self.conv_offset.bias.data.zero_()

    def forward(self, x, offset, mask):
        pass
