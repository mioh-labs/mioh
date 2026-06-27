"""MLX feature extraction path used by LADA's BasicVSR++ model."""

from __future__ import annotations

import mlx.core as mx

from .alignment import _conv2d_nchw
from .backbone import residual_blocks_with_input_conv_forward


def feature_extract_forward(
    x: mx.array,
    tensors: dict[str, mx.array],
    *,
    num_blocks: int = 5,
) -> mx.array:
    """Run the LADA BasicVSR++ feature extractor with NCHW MLX tensors."""

    out = _conv2d_nchw(x, tensors["0.weight"], tensors["0.bias"], stride=2, padding=1)
    out = mx.maximum(out, out * 0.1)
    out = _conv2d_nchw(out, tensors["2.weight"], tensors["2.bias"], stride=2, padding=1)
    out = mx.maximum(out, out * 0.1)
    residual_tensors = _strip_prefix(tensors, "4.")
    return residual_blocks_with_input_conv_forward(out, residual_tensors, num_blocks=num_blocks)


def _strip_prefix(tensors: dict[str, mx.array], prefix: str) -> dict[str, mx.array]:
    return {name[len(prefix):]: value for name, value in tensors.items() if name.startswith(prefix)}
