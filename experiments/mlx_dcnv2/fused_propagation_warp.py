"""Fused MLX kernels for BasicVSR++ propagation warp setup."""

from __future__ import annotations

import mlx.core as mx


def fused_propagation_warp_cond(
    feat_current: mx.array,
    feat_prop: mx.array,
    feat_n2: mx.array,
    flow_n1: mx.array,
    flow_n2: mx.array,
) -> mx.array:
    """Build `[cond_n1, feat_current, cond_n2]` with one Metal kernel.

    Inputs use the BasicVSR++ layout:
    - features: NCHW
    - flows: N2HW, pixel offsets, not normalized
    """

    if feat_current.shape != feat_prop.shape or feat_current.shape != feat_n2.shape:
        raise ValueError("feature tensors must share NCHW shape")
    if flow_n1.shape != flow_n2.shape:
        raise ValueError("flow tensors must share N2HW shape")
    if flow_n1.shape[0] != feat_current.shape[0] or flow_n1.shape[1] != 2:
        raise ValueError("flow tensors must be N2HW with the same batch size as features")
    if flow_n1.shape[2] != feat_current.shape[2] or flow_n1.shape[3] != feat_current.shape[3]:
        raise ValueError("flow and feature tensors must share spatial size")

    output_shape = (feat_current.shape[0], feat_current.shape[1] * 3, feat_current.shape[2], feat_current.shape[3])
    (out,) = _FUSED_PROPAGATION_WARP_COND_KERNEL(
        inputs=[feat_current, feat_prop, feat_n2, flow_n1, flow_n2],
        output_shapes=[output_shape],
        output_dtypes=[feat_current.dtype],
        grid=(output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3], 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out


def two_stage_propagation_warp_cond(
    feat_current: mx.array,
    feat_prop: mx.array,
    feat_n2: mx.array,
    flow_n1: mx.array,
    flow_n2: mx.array,
) -> mx.array:
    """Build propagation warp condition with two fused kernels.

    This avoids the single-kernel variant's biggest inefficiency: recomputing
    the warped 2-channel second-order flow for every feature channel.
    """

    if feat_current.shape != feat_prop.shape or feat_current.shape != feat_n2.shape:
        raise ValueError("feature tensors must share NCHW shape")
    if flow_n1.shape != flow_n2.shape:
        raise ValueError("flow tensors must share N2HW shape")
    if flow_n1.shape[0] != feat_current.shape[0] or flow_n1.shape[1] != 2:
        raise ValueError("flow tensors must be N2HW with the same batch size as features")
    if flow_n1.shape[2] != feat_current.shape[2] or flow_n1.shape[3] != feat_current.shape[3]:
        raise ValueError("flow and feature tensors must share spatial size")

    flow_total = fused_second_order_flow(flow_n1, flow_n2)
    return fused_cond_from_flow_total(feat_current, feat_prop, feat_n2, flow_n1, flow_total)


def fused_second_order_flow(flow_n1: mx.array, flow_n2: mx.array) -> mx.array:
    """Compute `flow_n1 + flow_warp(flow_n2, flow_n1)` as N2HW."""

    if flow_n1.shape != flow_n2.shape or flow_n1.shape[1] != 2:
        raise ValueError("flow tensors must share N2HW shape")
    (out,) = _FUSED_SECOND_ORDER_FLOW_KERNEL(
        inputs=[flow_n1, flow_n2],
        output_shapes=[flow_n1.shape],
        output_dtypes=[flow_n1.dtype],
        grid=(flow_n1.size, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out


def fused_cond_from_flow_total(
    feat_current: mx.array,
    feat_prop: mx.array,
    feat_n2: mx.array,
    flow_n1: mx.array,
    flow_total: mx.array,
) -> mx.array:
    """Build `[flow_warp(feat_prop, flow_n1), feat_current, flow_warp(feat_n2, flow_total)]`."""

    if feat_current.shape != feat_prop.shape or feat_current.shape != feat_n2.shape:
        raise ValueError("feature tensors must share NCHW shape")
    if flow_n1.shape != flow_total.shape or flow_n1.shape[1] != 2:
        raise ValueError("flow tensors must share N2HW shape")
    output_shape = (feat_current.shape[0], feat_current.shape[1] * 3, feat_current.shape[2], feat_current.shape[3])
    (out,) = _FUSED_COND_FROM_FLOW_TOTAL_KERNEL(
        inputs=[feat_current, feat_prop, feat_n2, flow_n1, flow_total],
        output_shapes=[output_shape],
        output_dtypes=[feat_current.dtype],
        grid=(output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3], 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out


_FUSED_PROPAGATION_WARP_COND_KERNEL = mx.fast.metal_kernel(
    name="mlx_fused_propagation_warp_cond_float32",
    input_names=["feat_current", "feat_prop", "feat_n2", "flow_n1", "flow_n2"],
    output_names=["out"],
    header="""
        float sample_feature(
            const device float *x,
            uint n,
            uint c,
            float sample_x,
            float sample_y,
            uint channels,
            uint height,
            uint width
        ) {
            int x0 = int(floor(sample_x));
            int y0 = int(floor(sample_y));
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            float wx1 = sample_x - float(x0);
            float wy1 = sample_y - float(y0);
            float wx0 = 1.0f - wx1;
            float wy0 = 1.0f - wy1;
            float value = 0.0f;
            if (y0 >= 0 && y0 < int(height) && x0 >= 0 && x0 < int(width)) {
                value += float(x[((n * channels + c) * height + uint(y0)) * width + uint(x0)]) * wy0 * wx0;
            }
            if (y0 >= 0 && y0 < int(height) && x1 >= 0 && x1 < int(width)) {
                value += float(x[((n * channels + c) * height + uint(y0)) * width + uint(x1)]) * wy0 * wx1;
            }
            if (y1 >= 0 && y1 < int(height) && x0 >= 0 && x0 < int(width)) {
                value += float(x[((n * channels + c) * height + uint(y1)) * width + uint(x0)]) * wy1 * wx0;
            }
            if (y1 >= 0 && y1 < int(height) && x1 >= 0 && x1 < int(width)) {
                value += float(x[((n * channels + c) * height + uint(y1)) * width + uint(x1)]) * wy1 * wx1;
            }
            return value;
        }

        float sample_flow(
            const device float *flow,
            uint n,
            uint c,
            float sample_x,
            float sample_y,
            uint height,
            uint width
        ) {
            int x0 = int(floor(sample_x));
            int y0 = int(floor(sample_y));
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            float wx1 = sample_x - float(x0);
            float wy1 = sample_y - float(y0);
            float wx0 = 1.0f - wx1;
            float wy0 = 1.0f - wy1;
            float value = 0.0f;
            if (y0 >= 0 && y0 < int(height) && x0 >= 0 && x0 < int(width)) {
                value += float(flow[((n * 2 + c) * height + uint(y0)) * width + uint(x0)]) * wy0 * wx0;
            }
            if (y0 >= 0 && y0 < int(height) && x1 >= 0 && x1 < int(width)) {
                value += float(flow[((n * 2 + c) * height + uint(y0)) * width + uint(x1)]) * wy0 * wx1;
            }
            if (y1 >= 0 && y1 < int(height) && x0 >= 0 && x0 < int(width)) {
                value += float(flow[((n * 2 + c) * height + uint(y1)) * width + uint(x0)]) * wy1 * wx0;
            }
            if (y1 >= 0 && y1 < int(height) && x1 >= 0 && x1 < int(width)) {
                value += float(flow[((n * 2 + c) * height + uint(y1)) * width + uint(x1)]) * wy1 * wx1;
            }
            return value;
        }
    """,
    source="""
        uint elem = thread_position_in_grid.x;
        uint batch = feat_current_shape[0];
        uint channels = feat_current_shape[1];
        uint height = feat_current_shape[2];
        uint width = feat_current_shape[3];
        uint out_channels = channels * 3;
        uint total = batch * out_channels * height * width;
        if (elem >= total) {
            return;
        }

        uint ox = elem % width;
        uint oy = (elem / width) % height;
        uint out_c = (elem / (height * width)) % out_channels;
        uint n = elem / (out_channels * height * width);
        uint plane = out_c / channels;
        uint c = out_c % channels;

        if (plane == 1) {
            out[elem] = feat_current[((n * channels + c) * height + oy) * width + ox];
            return;
        }

        uint flow_base = ((n * 2) * height + oy) * width + ox;
        float f1x = float(flow_n1[flow_base]);
        float f1y = float(flow_n1[((n * 2 + 1) * height + oy) * width + ox]);

        if (plane == 0) {
            out[elem] = sample_feature(feat_prop, n, c, float(ox) + f1x, float(oy) + f1y, channels, height, width);
            return;
        }

        float warped_flow2_x = sample_flow(flow_n2, n, 0, float(ox) + f1x, float(oy) + f1y, height, width);
        float warped_flow2_y = sample_flow(flow_n2, n, 1, float(ox) + f1x, float(oy) + f1y, height, width);
        float total_x = f1x + warped_flow2_x;
        float total_y = f1y + warped_flow2_y;
        out[elem] = sample_feature(feat_n2, n, c, float(ox) + total_x, float(oy) + total_y, channels, height, width);
    """,
)


_FUSED_SECOND_ORDER_FLOW_KERNEL = mx.fast.metal_kernel(
    name="mlx_fused_second_order_flow_float32",
    input_names=["flow_n1", "flow_n2"],
    output_names=["out"],
    header="""
        float sample_flow(
            const device float *flow,
            uint n,
            uint c,
            float sample_x,
            float sample_y,
            uint height,
            uint width
        ) {
            int x0 = int(floor(sample_x));
            int y0 = int(floor(sample_y));
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            float wx1 = sample_x - float(x0);
            float wy1 = sample_y - float(y0);
            float wx0 = 1.0f - wx1;
            float wy0 = 1.0f - wy1;
            float value = 0.0f;
            if (y0 >= 0 && y0 < int(height) && x0 >= 0 && x0 < int(width)) {
                value += float(flow[((n * 2 + c) * height + uint(y0)) * width + uint(x0)]) * wy0 * wx0;
            }
            if (y0 >= 0 && y0 < int(height) && x1 >= 0 && x1 < int(width)) {
                value += float(flow[((n * 2 + c) * height + uint(y0)) * width + uint(x1)]) * wy0 * wx1;
            }
            if (y1 >= 0 && y1 < int(height) && x0 >= 0 && x0 < int(width)) {
                value += float(flow[((n * 2 + c) * height + uint(y1)) * width + uint(x0)]) * wy1 * wx0;
            }
            if (y1 >= 0 && y1 < int(height) && x1 >= 0 && x1 < int(width)) {
                value += float(flow[((n * 2 + c) * height + uint(y1)) * width + uint(x1)]) * wy1 * wx1;
            }
            return value;
        }
    """,
    source="""
        uint elem = thread_position_in_grid.x;
        uint height = flow_n1_shape[2];
        uint width = flow_n1_shape[3];
        uint total = flow_n1_shape[0] * 2 * height * width;
        if (elem >= total) {
            return;
        }

        uint ox = elem % width;
        uint oy = (elem / width) % height;
        uint c = (elem / (height * width)) % 2;
        uint n = elem / (2 * height * width);
        uint flow_base = ((n * 2) * height + oy) * width + ox;
        float f1x = float(flow_n1[flow_base]);
        float f1y = float(flow_n1[((n * 2 + 1) * height + oy) * width + ox]);
        float warped = sample_flow(flow_n2, n, c, float(ox) + f1x, float(oy) + f1y, height, width);
        out[elem] = flow_n1[elem] + warped;
    """,
)


_FUSED_COND_FROM_FLOW_TOTAL_KERNEL = mx.fast.metal_kernel(
    name="mlx_fused_cond_from_flow_total_float32",
    input_names=["feat_current", "feat_prop", "feat_n2", "flow_n1", "flow_total"],
    output_names=["out"],
    header="""
        float sample_feature(
            const device float *x,
            uint n,
            uint c,
            float sample_x,
            float sample_y,
            uint channels,
            uint height,
            uint width
        ) {
            int x0 = int(floor(sample_x));
            int y0 = int(floor(sample_y));
            int x1 = x0 + 1;
            int y1 = y0 + 1;
            float wx1 = sample_x - float(x0);
            float wy1 = sample_y - float(y0);
            float wx0 = 1.0f - wx1;
            float wy0 = 1.0f - wy1;
            float value = 0.0f;
            if (y0 >= 0 && y0 < int(height) && x0 >= 0 && x0 < int(width)) {
                value += float(x[((n * channels + c) * height + uint(y0)) * width + uint(x0)]) * wy0 * wx0;
            }
            if (y0 >= 0 && y0 < int(height) && x1 >= 0 && x1 < int(width)) {
                value += float(x[((n * channels + c) * height + uint(y0)) * width + uint(x1)]) * wy0 * wx1;
            }
            if (y1 >= 0 && y1 < int(height) && x0 >= 0 && x0 < int(width)) {
                value += float(x[((n * channels + c) * height + uint(y1)) * width + uint(x0)]) * wy1 * wx0;
            }
            if (y1 >= 0 && y1 < int(height) && x1 >= 0 && x1 < int(width)) {
                value += float(x[((n * channels + c) * height + uint(y1)) * width + uint(x1)]) * wy1 * wx1;
            }
            return value;
        }
    """,
    source="""
        uint elem = thread_position_in_grid.x;
        uint batch = feat_current_shape[0];
        uint channels = feat_current_shape[1];
        uint height = feat_current_shape[2];
        uint width = feat_current_shape[3];
        uint out_channels = channels * 3;
        uint total = batch * out_channels * height * width;
        if (elem >= total) {
            return;
        }

        uint ox = elem % width;
        uint oy = (elem / width) % height;
        uint out_c = (elem / (height * width)) % out_channels;
        uint n = elem / (out_channels * height * width);
        uint plane = out_c / channels;
        uint c = out_c % channels;

        if (plane == 1) {
            out[elem] = feat_current[((n * channels + c) * height + oy) * width + ox];
            return;
        }

        const device float *flow = plane == 0 ? flow_n1 : flow_total;
        uint flow_base = ((n * 2) * height + oy) * width + ox;
        float fx = float(flow[flow_base]);
        float fy = float(flow[((n * 2 + 1) * height + oy) * width + ox]);
        const device float *feat = plane == 0 ? feat_prop : feat_n2;
        out[elem] = sample_feature(feat, n, c, float(ox) + fx, float(oy) + fy, channels, height, width);
    """,
)
