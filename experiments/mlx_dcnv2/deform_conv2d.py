"""Reference MLX-facing DCNv2 forward implementation.

This module intentionally starts with a correctness-first reference path.  The
next step is replacing the hot loop with ``mx.fast.metal_kernel`` while keeping
these tests as the contract against ``torchvision.ops.deform_conv2d``.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import numpy as np


def deform_conv2d_forward(
    x: mx.array,
    offset: mx.array,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    mask: mx.array | None = None,
) -> mx.array:
    """Run modulated deformable convolution with MLX array inputs.

    The implementation uses an MLX custom Metal kernel for float32 arrays and
    falls back to a correctness-first NumPy reference if the kernel cannot be
    used.
    """

    if _can_use_metal_kernel(x, offset, weight, bias, mask):
        if _can_use_im2col_matmul_path(x, weight):
            return _deform_conv2d_forward_im2col_matmul(
                x,
                offset,
                weight,
                bias,
                stride=stride,
                padding=padding,
                dilation=dilation,
                mask=mask,
            )
        return _deform_conv2d_forward_metal(
            x,
            offset,
            weight,
            bias,
            stride=stride,
            padding=padding,
            dilation=dilation,
            mask=mask,
        )

    return deform_conv2d_forward_reference(
        x,
        offset,
        weight,
        bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=mask,
    )


def deform_conv2d_forward_reference(
    x: mx.array,
    offset: mx.array,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    mask: mx.array | None = None,
) -> mx.array:
    """Correctness-first reference path for validating the Metal kernel."""

    x_np = np.array(x, dtype=np.float32)
    offset_np = np.array(offset, dtype=np.float32)
    weight_np = np.array(weight, dtype=np.float32)
    bias_np = None if bias is None else np.array(bias, dtype=np.float32)
    mask_np = None if mask is None else np.array(mask, dtype=np.float32)

    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dilation_h, dilation_w = _pair(dilation)

    batch, in_channels, in_h, in_w = x_np.shape
    out_channels, weight_in_channels, kernel_h, kernel_w = weight_np.shape
    out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    groups = in_channels // weight_in_channels
    out_channels_per_group = out_channels // groups
    kernel_size = kernel_h * kernel_w
    deform_groups = offset_np.shape[1] // (2 * kernel_size)
    channels_per_deform_group = in_channels // deform_groups

    if mask_np is None:
        mask_np = np.ones((batch, deform_groups * kernel_size, out_h, out_w), dtype=np.float32)

    out = np.zeros((batch, out_channels, out_h, out_w), dtype=np.float32)
    for n in range(batch):
        for oc in range(out_channels):
            group = oc // out_channels_per_group
            in_start = group * weight_in_channels
            value_bias = 0.0 if bias_np is None else float(bias_np[oc])
            for oy in range(out_h):
                for ox in range(out_w):
                    acc = value_bias
                    for ic_in_group in range(weight_in_channels):
                        ic = in_start + ic_in_group
                        deform_group = ic // channels_per_deform_group
                        for ky in range(kernel_h):
                            for kx in range(kernel_w):
                                kernel_index = ky * kernel_w + kx
                                offset_base = deform_group * 2 * kernel_size + 2 * kernel_index
                                sample_y = (
                                    oy * stride_h
                                    - pad_h
                                    + ky * dilation_h
                                    + float(offset_np[n, offset_base, oy, ox])
                                )
                                sample_x = (
                                    ox * stride_w
                                    - pad_w
                                    + kx * dilation_w
                                    + float(offset_np[n, offset_base + 1, oy, ox])
                                )
                                sampled = _bilinear_sample(x_np[n, ic], sample_y, sample_x)
                                modulation = float(mask_np[n, deform_group * kernel_size + kernel_index, oy, ox])
                                acc += sampled * modulation * float(weight_np[oc, ic_in_group, ky, kx])
                    out[n, oc, oy, ox] = acc

    return mx.array(out)


_DEFORM_CONV2D_FORWARD_KERNEL = mx.fast.metal_kernel(
    name="mlx_dcnv2_forward_float32",
    input_names=["x", "offset", "weight", "bias", "mask", "params"],
    output_names=["out"],
    source="""
        uint elem = thread_position_in_grid.x;
        uint out_c = uint(params[6]);
        uint out_h = uint(params[7]);
        uint out_w = uint(params[8]);
        uint out_hw = out_h * out_w;
        uint total = uint(params[9]) * out_c * out_hw;
        if (elem >= total) {
            return;
        }

        uint ox = elem % out_w;
        uint oy = (elem / out_w) % out_h;
        uint oc = (elem / out_hw) % out_c;
        uint n = elem / (out_c * out_hw);

        uint in_c = x_shape[1];
        uint in_h = x_shape[2];
        uint in_w = x_shape[3];
        uint weight_in_c = weight_shape[1];
        uint kernel_h = weight_shape[2];
        uint kernel_w = weight_shape[3];
        uint kernel_size = kernel_h * kernel_w;

        int stride_h = int(params[0]);
        int stride_w = int(params[1]);
        int pad_h = int(params[2]);
        int pad_w = int(params[3]);
        int dilation_h = int(params[4]);
        int dilation_w = int(params[5]);

        uint groups = in_c / weight_in_c;
        uint out_channels_per_group = out_c / groups;
        uint deform_groups = offset_shape[1] / (2 * kernel_size);
        uint channels_per_deform_group = in_c / deform_groups;
        uint group = oc / out_channels_per_group;
        uint in_start = group * weight_in_c;

        float acc = float(bias[oc]);
        for (uint icg = 0; icg < weight_in_c; ++icg) {
            uint ic = in_start + icg;
            uint deform_group = ic / channels_per_deform_group;
            for (uint ky = 0; ky < kernel_h; ++ky) {
                for (uint kx = 0; kx < kernel_w; ++kx) {
                    uint kernel_index = ky * kernel_w + kx;
                    uint offset_base = deform_group * 2 * kernel_size + 2 * kernel_index;
                    uint offset_y_index = ((n * offset_shape[1] + offset_base) * out_h + oy) * out_w + ox;
                    uint offset_x_index = ((n * offset_shape[1] + offset_base + 1) * out_h + oy) * out_w + ox;
                    float sample_y = float(int(oy) * stride_h - pad_h + int(ky) * dilation_h) + float(offset[offset_y_index]);
                    float sample_x = float(int(ox) * stride_w - pad_w + int(kx) * dilation_w) + float(offset[offset_x_index]);

                    int y0 = int(floor(sample_y));
                    int x0 = int(floor(sample_x));
                    int y1 = y0 + 1;
                    int x1 = x0 + 1;
                    float wy1 = sample_y - float(y0);
                    float wx1 = sample_x - float(x0);
                    float wy0 = 1.0f - wy1;
                    float wx0 = 1.0f - wx1;

                    float sampled = 0.0f;
                    if (y0 >= 0 && y0 < int(in_h) && x0 >= 0 && x0 < int(in_w)) {
                        sampled += float(x[((n * in_c + ic) * in_h + uint(y0)) * in_w + uint(x0)]) * wy0 * wx0;
                    }
                    if (y0 >= 0 && y0 < int(in_h) && x1 >= 0 && x1 < int(in_w)) {
                        sampled += float(x[((n * in_c + ic) * in_h + uint(y0)) * in_w + uint(x1)]) * wy0 * wx1;
                    }
                    if (y1 >= 0 && y1 < int(in_h) && x0 >= 0 && x0 < int(in_w)) {
                        sampled += float(x[((n * in_c + ic) * in_h + uint(y1)) * in_w + uint(x0)]) * wy1 * wx0;
                    }
                    if (y1 >= 0 && y1 < int(in_h) && x1 >= 0 && x1 < int(in_w)) {
                        sampled += float(x[((n * in_c + ic) * in_h + uint(y1)) * in_w + uint(x1)]) * wy1 * wx1;
                    }

                    uint mask_index = ((n * mask_shape[1] + deform_group * kernel_size + kernel_index) * out_h + oy) * out_w + ox;
                    uint weight_index = ((oc * weight_in_c + icg) * kernel_h + ky) * kernel_w + kx;
                    acc += sampled * float(mask[mask_index]) * float(weight[weight_index]);
                }
            }
        }
        out[elem] = acc;
    """,
)


_DEFORM_IM2COL_KERNEL = mx.fast.metal_kernel(
    name="mlx_dcnv2_im2col_float32",
    input_names=["x", "offset", "mask", "params"],
    output_names=["cols"],
    source="""
        uint elem = thread_position_in_grid.x;
        uint cols_k = uint(params[6]);
        uint out_h = uint(params[7]);
        uint out_w = uint(params[8]);
        uint batch = uint(params[9]);
        uint out_hw = out_h * out_w;
        uint total = batch * out_hw * cols_k;
        if (elem >= total) {
            return;
        }

        uint ck = elem % cols_k;
        uint hw = (elem / cols_k) % out_hw;
        uint n = elem / (out_hw * cols_k);
        uint oy = hw / out_w;
        uint ox = hw % out_w;

        uint in_c = x_shape[1];
        uint in_h = x_shape[2];
        uint in_w = x_shape[3];
        uint kernel_h = uint(params[10]);
        uint kernel_w = uint(params[11]);
        uint kernel_size = kernel_h * kernel_w;
        uint ic = ck / kernel_size;
        uint kernel_index = ck % kernel_size;
        uint ky = kernel_index / kernel_w;
        uint kx = kernel_index % kernel_w;

        int stride_h = int(params[0]);
        int stride_w = int(params[1]);
        int pad_h = int(params[2]);
        int pad_w = int(params[3]);
        int dilation_h = int(params[4]);
        int dilation_w = int(params[5]);

        uint deform_groups = offset_shape[1] / (2 * kernel_size);
        uint channels_per_deform_group = in_c / deform_groups;
        uint deform_group = ic / channels_per_deform_group;
        uint offset_base = deform_group * 2 * kernel_size + 2 * kernel_index;
        uint offset_y_index = ((n * offset_shape[1] + offset_base) * out_h + oy) * out_w + ox;
        uint offset_x_index = ((n * offset_shape[1] + offset_base + 1) * out_h + oy) * out_w + ox;

        float sample_y = float(int(oy) * stride_h - pad_h + int(ky) * dilation_h) + float(offset[offset_y_index]);
        float sample_x = float(int(ox) * stride_w - pad_w + int(kx) * dilation_w) + float(offset[offset_x_index]);

        int y0 = int(floor(sample_y));
        int x0 = int(floor(sample_x));
        int y1 = y0 + 1;
        int x1 = x0 + 1;
        float wy1 = sample_y - float(y0);
        float wx1 = sample_x - float(x0);
        float wy0 = 1.0f - wy1;
        float wx0 = 1.0f - wx1;

        float sampled = 0.0f;
        if (y0 >= 0 && y0 < int(in_h) && x0 >= 0 && x0 < int(in_w)) {
            sampled += float(x[((n * in_c + ic) * in_h + uint(y0)) * in_w + uint(x0)]) * wy0 * wx0;
        }
        if (y0 >= 0 && y0 < int(in_h) && x1 >= 0 && x1 < int(in_w)) {
            sampled += float(x[((n * in_c + ic) * in_h + uint(y0)) * in_w + uint(x1)]) * wy0 * wx1;
        }
        if (y1 >= 0 && y1 < int(in_h) && x0 >= 0 && x0 < int(in_w)) {
            sampled += float(x[((n * in_c + ic) * in_h + uint(y1)) * in_w + uint(x0)]) * wy1 * wx0;
        }
        if (y1 >= 0 && y1 < int(in_h) && x1 >= 0 && x1 < int(in_w)) {
            sampled += float(x[((n * in_c + ic) * in_h + uint(y1)) * in_w + uint(x1)]) * wy1 * wx1;
        }

        uint mask_index = ((n * mask_shape[1] + deform_group * kernel_size + kernel_index) * out_h + oy) * out_w + ox;
        cols[elem] = sampled * float(mask[mask_index]);
    """,
)


def _deform_conv2d_forward_im2col_matmul(
    x: mx.array,
    offset: mx.array,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    mask: mx.array | None = None,
) -> mx.array:
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dilation_h, dilation_w = _pair(dilation)

    batch, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    kernel_size = kernel_h * kernel_w
    cols_k = in_channels * kernel_size

    if mask is None:
        deform_groups = offset.shape[1] // (2 * kernel_h * kernel_w)
        mask = mx.ones((batch, deform_groups * kernel_h * kernel_w, out_h, out_w), dtype=x.dtype)

    params = mx.array(
        [stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, cols_k, out_h, out_w, batch, kernel_h, kernel_w],
        dtype=mx.int32,
    )
    cols = _DEFORM_IM2COL_KERNEL(
        inputs=[x, offset, mask, params],
        grid=(batch * out_h * out_w * cols_k, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(batch, out_h * out_w, cols_k)],
        output_dtypes=[x.dtype],
    )[0]
    weight_flat = mx.reshape(weight, (out_channels, cols_k))
    out = mx.matmul(cols, mx.transpose(weight_flat))
    if bias is not None:
        out = out + mx.reshape(bias, (1, 1, out_channels))
    out = mx.reshape(out, (batch, out_h, out_w, out_channels))
    return mx.transpose(out, (0, 3, 1, 2))


def _deform_conv2d_forward_metal(
    x: mx.array,
    offset: mx.array,
    weight: mx.array,
    bias: mx.array | None = None,
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    mask: mx.array | None = None,
) -> mx.array:
    stride_h, stride_w = _pair(stride)
    pad_h, pad_w = _pair(padding)
    dilation_h, dilation_w = _pair(dilation)

    batch, in_channels, in_h, in_w = x.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    out_h = (in_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_w = (in_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

    if bias is None:
        bias = mx.zeros((out_channels,), dtype=x.dtype)
    if mask is None:
        deform_groups = offset.shape[1] // (2 * kernel_h * kernel_w)
        mask = mx.ones((batch, deform_groups * kernel_h * kernel_w, out_h, out_w), dtype=x.dtype)

    params = mx.array(
        [stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, out_channels, out_h, out_w, batch],
        dtype=mx.int32,
    )
    output_shape = (batch, out_channels, out_h, out_w)
    outputs = _DEFORM_CONV2D_FORWARD_KERNEL(
        inputs=[x, offset, weight, bias, mask, params],
        grid=(int(np.prod(output_shape)), 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[output_shape],
        output_dtypes=[x.dtype],
    )
    return outputs[0]


def _can_use_metal_kernel(
    x: mx.array,
    offset: mx.array,
    weight: mx.array,
    bias: mx.array | None,
    mask: mx.array | None,
) -> bool:
    arrays = [x, offset, weight]
    if bias is not None:
        arrays.append(bias)
    if mask is not None:
        arrays.append(mask)
    return all(array.dtype == mx.float32 for array in arrays)


def _can_use_im2col_matmul_path(x: mx.array, weight: mx.array) -> bool:
    # BasicVSR++ uses groups=1 for deformable alignment.  Matmul is much faster
    # than doing the whole input-channel reduction inside one Metal thread.
    return weight.shape[1] == x.shape[1]


def _pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"expected int or pair, got {value!r}")
    return int(value[0]), int(value[1])


def _bilinear_sample(image: np.ndarray, y: float, x: float) -> float:
    height, width = image.shape
    y0 = int(np.floor(y))
    x0 = int(np.floor(x))
    y1 = y0 + 1
    x1 = x0 + 1

    wy1 = y - y0
    wx1 = x - x0
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1

    return (
        _pixel_or_zero(image, y0, x0, height, width) * wy0 * wx0
        + _pixel_or_zero(image, y0, x1, height, width) * wy0 * wx1
        + _pixel_or_zero(image, y1, x0, height, width) * wy1 * wx0
        + _pixel_or_zero(image, y1, x1, height, width) * wy1 * wx1
    )


def _pixel_or_zero(image: np.ndarray, y: int, x: int, height: int, width: int) -> float:
    if y < 0 or y >= height or x < 0 or x >= width:
        return 0.0
    return float(image[y, x])
