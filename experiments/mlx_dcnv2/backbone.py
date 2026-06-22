"""MLX BasicVSR++ residual backbone blocks."""

from __future__ import annotations

import mlx.core as mx

from .alignment import _conv2d_nchw


def residual_blocks_with_input_conv_forward(
    x: mx.array,
    tensors: dict[str, mx.array],
    *,
    num_blocks: int,
) -> mx.array:
    """Run BasicVSR++ `ResidualBlocksWithInputConv` with NCHW MLX tensors."""

    out = _conv2d_nchw(x, tensors["main.0.weight"], tensors["main.0.bias"], padding=1)
    out = mx.maximum(out, out * 0.1)
    for block_index in range(num_blocks):
        identity = out
        out = _conv2d_nchw(
            out,
            tensors[f"main.2.{block_index}.conv1.weight"],
            tensors[f"main.2.{block_index}.conv1.bias"],
            padding=1,
        )
        out = mx.maximum(out, 0)
        out = _conv2d_nchw(
            out,
            tensors[f"main.2.{block_index}.conv2.weight"],
            tensors[f"main.2.{block_index}.conv2.bias"],
            padding=1,
        )
        out = identity + out
    return out
