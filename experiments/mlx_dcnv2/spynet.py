"""MLX SPyNet subset used by LADA BasicVSR++ inference."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .alignment import _conv2d_nchw
from .flow_warp import flow_warp


def spynet_basic_module_forward(x: mx.array, tensors: dict[str, mx.array]) -> mx.array:
    """Run one LADA SPyNet basic module."""

    out = x
    for layer in range(5):
        out = _conv2d_nchw(
            out,
            tensors[f"basic_module.{layer}.conv.weight"],
            tensors[f"basic_module.{layer}.conv.bias"],
            padding=3,
        )
        if layer < 4:
            out = mx.maximum(out, 0)
    return out


def spynet_compute_flow(ref: mx.array, supp: mx.array, tensors: dict[str, mx.array]) -> mx.array:
    """Run LADA SPyNet compute_flow for inputs already sized to multiples of 32."""

    ref_pyramid = [(ref - tensors["mean"]) / tensors["std"]]
    supp_pyramid = [(supp - tensors["mean"]) / tensors["std"]]
    for _ in range(5):
        ref_pyramid.append(avg_pool2d_nchw(ref_pyramid[-1], kernel_size=2, stride=2))
        supp_pyramid.append(avg_pool2d_nchw(supp_pyramid[-1], kernel_size=2, stride=2))
    ref_pyramid = ref_pyramid[::-1]
    supp_pyramid = supp_pyramid[::-1]

    batch, _, height, width = ref_pyramid[-1].shape
    flow = mx.zeros((batch, 2, height // 32, width // 32), dtype=ref.dtype)
    for level in range(6):
        if level == 0:
            flow_up = flow
        else:
            flow_up = interpolate_bilinear_nchw(flow, scale_factor=2, align_corners=True) * 2.0
        warped = flow_warp(supp_pyramid[level], mx.transpose(flow_up, (0, 2, 3, 1)), padding_mode="border")
        module_tensors = _spynet_module_tensors(tensors, level)
        flow = flow_up + spynet_basic_module_forward(
            mx.concatenate([ref_pyramid[level], warped, flow_up], axis=1),
            module_tensors,
        )
    return flow


def spynet_forward(ref: mx.array, supp: mx.array, tensors: dict[str, mx.array]) -> mx.array:
    """Run LADA SPyNet forward with resize-to-multiple-of-32 handling."""

    _, _, height, width = ref.shape
    width_up = width if width % 32 == 0 else 32 * (width // 32 + 1)
    height_up = height if height % 32 == 0 else 32 * (height // 32 + 1)
    ref_up = interpolate_bilinear_nchw_to_size(ref, size=(height_up, width_up), align_corners=False)
    supp_up = interpolate_bilinear_nchw_to_size(supp, size=(height_up, width_up), align_corners=False)
    flow = interpolate_bilinear_nchw_to_size(
        spynet_compute_flow(ref_up, supp_up, tensors),
        size=(height, width),
        align_corners=False,
    )
    scale = mx.array([width / width_up, height / height_up], dtype=flow.dtype).reshape(1, 2, 1, 1)
    return flow * scale


def avg_pool2d_nchw(x: mx.array, *, kernel_size: int, stride: int) -> mx.array:
    x_nhwc = mx.transpose(x, (0, 2, 3, 1))
    y = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)(x_nhwc)
    return mx.transpose(y, (0, 3, 1, 2))


def interpolate_bilinear_nchw(x: mx.array, *, scale_factor: int, align_corners: bool) -> mx.array:
    x_nhwc = mx.transpose(x, (0, 2, 3, 1))
    y = nn.Upsample(scale_factor=scale_factor, mode="linear", align_corners=align_corners)(x_nhwc)
    return mx.transpose(y, (0, 3, 1, 2))


def interpolate_bilinear_nchw_to_size(
    x: mx.array,
    *,
    size: tuple[int, int],
    align_corners: bool,
) -> mx.array:
    out_h, out_w = size
    params = mx.array([1 if align_corners else 0, out_h, out_w], dtype=mx.int32)
    output_shape = (x.shape[0], x.shape[1], out_h, out_w)
    (out,) = _BILINEAR_RESIZE_NCHW_KERNEL(
        inputs=[x, params],
        output_shapes=[output_shape],
        output_dtypes=[x.dtype],
        grid=(output_shape[0] * output_shape[1] * out_h * out_w, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out


def _spynet_module_tensors(tensors: dict[str, mx.array], level: int) -> dict[str, mx.array]:
    prefix = f"basic_module.{level}."
    return {name[len(prefix):]: value for name, value in tensors.items() if name.startswith(prefix)}


_BILINEAR_RESIZE_NCHW_KERNEL = mx.fast.metal_kernel(
    name="mlx_bilinear_resize_nchw_float32",
    input_names=["x", "params"],
    output_names=["out"],
    source="""
        uint elem = thread_position_in_grid.x;
        uint batch = x_shape[0];
        uint channels = x_shape[1];
        uint out_h = uint(params[1]);
        uint out_w = uint(params[2]);
        uint total = batch * channels * out_h * out_w;
        if (elem >= total) {
            return;
        }

        uint ox = elem % out_w;
        uint oy = (elem / out_w) % out_h;
        uint c = (elem / (out_h * out_w)) % channels;
        uint n = elem / (channels * out_h * out_w);

        uint in_h = x_shape[2];
        uint in_w = x_shape[3];
        bool align_corners = params[0] != 0;

        float sample_y;
        float sample_x;
        if (align_corners) {
            sample_y = (out_h > 1) ? float(oy) * float(in_h - 1) / float(out_h - 1) : 0.0f;
            sample_x = (out_w > 1) ? float(ox) * float(in_w - 1) / float(out_w - 1) : 0.0f;
        } else {
            sample_y = (float(oy) + 0.5f) * float(in_h) / float(out_h) - 0.5f;
            sample_x = (float(ox) + 0.5f) * float(in_w) / float(out_w) - 0.5f;
        }

        sample_y = min(max(sample_y, 0.0f), float(in_h - 1));
        sample_x = min(max(sample_x, 0.0f), float(in_w - 1));

        int y0 = int(floor(sample_y));
        int x0 = int(floor(sample_x));
        int y1 = min(y0 + 1, int(in_h - 1));
        int x1 = min(x0 + 1, int(in_w - 1));
        float wy1 = sample_y - float(y0);
        float wx1 = sample_x - float(x0);
        float wy0 = 1.0f - wy1;
        float wx0 = 1.0f - wx1;

        float v00 = float(x[((n * channels + c) * in_h + uint(y0)) * in_w + uint(x0)]);
        float v01 = float(x[((n * channels + c) * in_h + uint(y0)) * in_w + uint(x1)]);
        float v10 = float(x[((n * channels + c) * in_h + uint(y1)) * in_w + uint(x0)]);
        float v11 = float(x[((n * channels + c) * in_h + uint(y1)) * in_w + uint(x1)]);

        out[elem] = v00 * wy0 * wx0 + v01 * wy0 * wx1 + v10 * wy1 * wx0 + v11 * wy1 * wx1;
    """,
)
