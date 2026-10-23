"""CELF-Net: a dense mosaic detector whose instances are defined by their
grid, as in CELF (a mosaic = a connected region with one grid).

Input: 960x544 RGB (the frame resized to 960x540, padded 2 rows top and
bottom), 0..1. Output at 1/4 resolution (136x240), 8 channels:
  0      mosaic logit
  1, 2   log grid periods x and y, (log P - LOG_P0), P in input pixels
         (cells are not square in every source: e.g. 30x26 or 40x30)
  3, 4   cos, sin of the local x grid phase 2 pi (x - ph_x) / P_x
  5, 6   cos, sin of the local y grid phase
  7      logit of a boundary between two different mosaics
The local phase is what a convolution can see (translation equivariant);
the decoder turns it into the grid's absolute phase with the pixel's own
coordinates, and pixels with the same grid form one instance.

The temporal variant (in_ch=8) also takes the previous frame and its own
previous result (mosaic, boundaries between mosaics): what was one mosaic
stays one, what was two stays two, as CELF keeps them.

The object variant (out 11, in 9) defines instances by object, not grid
(frame-anchored studios give touching mosaics one grid): three more
output channels
  8      logit of an object centre (one peak per mosaic object)
  9, 10  offset from the pixel to its object's centre, x and y, in output
         pixels / OFFSET_SCALE
and a ninth input, the centres of the previous result (Gaussian peaks).

Convolutions, batch norm and ReLU only (Neural Engine friendly).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_P0 = math.log(12.0)
STRIDE = 4
IN_W, IN_H = 960, 544


def conv_bn(cin, cout, k=3, s=1, groups=1, act=True, d=1):
    layers = [nn.Conv2d(cin, cout, k, s, d * (k // 2), dilation=d, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, s=1, e=4, d=1):
        super().__init__()
        hid = cin * e
        self.residual = s == 1 and cin == cout
        self.body = nn.Sequential(
            conv_bn(cin, hid, 1),
            conv_bn(hid, hid, 3, s, groups=hid, d=d),
            conv_bn(hid, cout, 1, act=False))

    def forward(self, x):
        y = self.body(x)
        return x + y if self.residual else y


def stage(cin, cout, n, d=1):
    return nn.Sequential(InvertedResidual(cin, cout, 2), *[InvertedResidual(cout, cout, 1, d=d) for _ in range(n - 1)])


class CELFNet(nn.Module):
    def __init__(self, width=1.0, in_ch=3, out_ch=8):
        super().__init__()
        c = [int(round(v * width)) for v in (16, 24, 48, 96, 160)]
        f = int(round(64 * width))
        self.in_ch = in_ch
        self.stem = conv_bn(in_ch, c[0], 3, 2)                               # 1/2
        self.s4 = stage(c[0], c[1], 2)                                        # 1/4
        self.s8 = stage(c[1], c[2], 3)                                        # 1/8
        self.s16 = stage(c[2], c[3], 4)                                       # 1/16
        self.s32 = nn.Sequential(stage(c[3], c[4], 2), InvertedResidual(c[4], c[4], d=2),
                                 InvertedResidual(c[4], c[4], d=4))           # 1/32, wide context
        self.lat = nn.ModuleList([conv_bn(ci, f, 1) for ci in (c[1], c[2], c[3], c[4])])
        self.smooth = nn.ModuleList([nn.Sequential(conv_bn(f, f, 3, groups=f), conv_bn(f, f, 1)) for _ in range(3)])
        self.head = nn.Sequential(InvertedResidual(f, f), InvertedResidual(f, f), nn.Conv2d(f, out_ch, 1))

    def forward(self, x):
        x = self.stem(x)
        a = self.s4(x); b = self.s8(a); c = self.s16(b); d = self.s32(c)
        p = self.lat[3](d)
        for i, feat in zip((2, 1, 0), (c, b, a)):
            p = self.lat[i](feat) + F.interpolate(p, size=feat.shape[-2:], mode="nearest")
            p = self.smooth[i](p)
        return self.head(p)



OFFSET_SCALE = 16.0
