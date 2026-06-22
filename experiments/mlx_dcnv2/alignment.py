"""MLX pieces of BasicVSR++ second-order deformable alignment."""

from __future__ import annotations

import mlx.core as mx

from .deform_conv2d import deform_conv2d_forward


def second_order_deformable_alignment_forward(
    x: mx.array,
    extra_feat: mx.array,
    flow_1: mx.array,
    flow_2: mx.array,
    tensors: dict[str, mx.array],
    *,
    max_residue_magnitude: float = 10.0,
) -> mx.array:
    """Run the BasicVSR++ second-order deformable alignment module in MLX."""

    offset, mask = compute_second_order_offset_mask(
        extra_feat,
        flow_1,
        flow_2,
        tensors,
        max_residue_magnitude=max_residue_magnitude,
    )
    return deform_conv2d_forward(
        x,
        offset,
        tensors["weight"],
        tensors.get("bias"),
        stride=1,
        padding=1,
        dilation=1,
        mask=mask,
    )


def compute_second_order_offset_mask(
    extra_feat: mx.array,
    flow_1: mx.array,
    flow_2: mx.array,
    tensors: dict[str, mx.array],
    *,
    max_residue_magnitude: float = 10.0,
) -> tuple[mx.array, mx.array]:
    """Compute BasicVSR++ second-order alignment offset and modulation mask.

    Inputs and outputs use PyTorch-compatible NCHW layout.  We convert only at
    the `mx.conv2d` boundary because MLX expects NHWC input and OHWI weights.
    """

    x = mx.concatenate([extra_feat, flow_1, flow_2], axis=1)
    for layer in (0, 2, 4):
        x = _conv2d_nchw(
            x,
            tensors[f"conv_offset.{layer}.weight"],
            tensors[f"conv_offset.{layer}.bias"],
            padding=1,
        )
        x = mx.maximum(x, x * 0.1)
    x = _conv2d_nchw(
        x,
        tensors["conv_offset.6.weight"],
        tensors["conv_offset.6.bias"],
        padding=1,
    )
    o1, o2, mask = mx.split(x, 3, axis=1)
    offset = max_residue_magnitude * mx.tanh(mx.concatenate([o1, o2], axis=1))
    offset_1, offset_2 = mx.split(offset, 2, axis=1)
    offset_1 = offset_1 + mx.tile(_flip_flow_channels(flow_1), (1, offset_1.shape[1] // 2, 1, 1))
    offset_2 = offset_2 + mx.tile(_flip_flow_channels(flow_2), (1, offset_2.shape[1] // 2, 1, 1))
    return mx.concatenate([offset_1, offset_2], axis=1), mx.sigmoid(mask)


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


def _flip_flow_channels(flow: mx.array) -> mx.array:
    return mx.take(flow, mx.array([1, 0]), axis=1)
