"""MLX reconstruction/upsample path used by LADA's BasicVSR++ model."""

from __future__ import annotations

import mlx.core as mx

from .alignment import _conv2d_nchw
from .backbone import residual_blocks_with_input_conv_forward


def reconstruction_forward(
    x: mx.array,
    lq: mx.array,
    tensors: dict[str, mx.array],
    *,
    num_blocks: int = 5,
) -> mx.array:
    """Run LADA BasicVSR++ reconstruction and add the original LQ frame."""

    reconstruction_tensors = _strip_prefix(tensors, "reconstruction.")
    out = residual_blocks_with_input_conv_forward(x, reconstruction_tensors, num_blocks=num_blocks)
    out = _conv2d_nchw(
        out,
        tensors["upsample1.upsample_conv.weight"],
        tensors["upsample1.upsample_conv.bias"],
        padding=1,
    )
    out = _pixel_shuffle_nchw(out, 2)
    out = mx.maximum(out, out * 0.1)
    out = _conv2d_nchw(
        out,
        tensors["upsample2.upsample_conv.weight"],
        tensors["upsample2.upsample_conv.bias"],
        padding=1,
    )
    out = _pixel_shuffle_nchw(out, 2)
    out = mx.maximum(out, out * 0.1)
    out = _conv2d_nchw(out, tensors["conv_hr.weight"], tensors["conv_hr.bias"], padding=1)
    out = mx.maximum(out, out * 0.1)
    out = _conv2d_nchw(out, tensors["conv_last.weight"], tensors["conv_last.bias"], padding=1)
    return out + lq


def _pixel_shuffle_nchw(x: mx.array, scale: int) -> mx.array:
    batch, channels, height, width = x.shape
    out_channels = channels // (scale * scale)
    x = mx.reshape(x, (batch, out_channels, scale, scale, height, width))
    x = mx.transpose(x, (0, 1, 4, 2, 5, 3))
    return mx.reshape(x, (batch, out_channels, height * scale, width * scale))


def _strip_prefix(tensors: dict[str, mx.array], prefix: str) -> dict[str, mx.array]:
    return {name[len(prefix):]: value for name, value in tensors.items() if name.startswith(prefix)}
