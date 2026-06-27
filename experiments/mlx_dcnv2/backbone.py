"""MLX BasicVSR++ residual backbone blocks."""

from __future__ import annotations

import mlx.core as mx


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


def prepare_backbone_tensors_nhwc(tensors: dict[str, mx.array]) -> dict[str, mx.array]:
    """Pre-shape PyTorch conv tensors for MLX NHWC execution.

    The NCHW compatibility path is convenient for correctness work, but it
    reshapes/transposes at every call site. The NHWC path lets benchmarks
    measure the cost profile we would want in a real all-MLX propagation loop.
    """

    prepared: dict[str, mx.array] = {}
    for name, value in tensors.items():
        if name.endswith(".weight") and len(value.shape) == 4:
            prepared[name] = mx.transpose(value, (0, 2, 3, 1))
        elif name.endswith(".bias") and len(value.shape) == 1:
            prepared[name] = mx.reshape(value, (1, 1, 1, value.shape[0]))
        else:
            prepared[name] = value
    return prepared


def residual_blocks_with_input_conv_forward_nhwc(
    x: mx.array,
    tensors: dict[str, mx.array],
    *,
    num_blocks: int,
) -> mx.array:
    """Run `ResidualBlocksWithInputConv` with NHWC activations and OHWI weights."""

    out = mx.conv2d(x, tensors["main.0.weight"], padding=1)
    out = out + tensors["main.0.bias"]
    out = mx.maximum(out, out * 0.1)
    for block_index in range(num_blocks):
        identity = out
        out = mx.conv2d(out, tensors[f"main.2.{block_index}.conv1.weight"], padding=1)
        out = out + tensors[f"main.2.{block_index}.conv1.bias"]
        out = mx.maximum(out, 0)
        out = mx.conv2d(out, tensors[f"main.2.{block_index}.conv2.weight"], padding=1)
        out = out + tensors[f"main.2.{block_index}.conv2.bias"]
        out = identity + out
    return out


def _conv2d_nchw(
    x: mx.array,
    weight_oihw: mx.array,
    bias: mx.array | None,
    *,
    padding: int = 0,
    stride: int = 1,
) -> mx.array:
    x_nhwc = mx.transpose(x, (0, 2, 3, 1))
    weight_ohwi = mx.transpose(weight_oihw, (0, 2, 3, 1))
    y = mx.conv2d(x_nhwc, weight_ohwi, stride=stride, padding=padding)
    if bias is not None:
        y = y + mx.reshape(bias, (1, 1, 1, bias.shape[0]))
    return mx.transpose(y, (0, 3, 1, 2))
