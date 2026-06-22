"""MLX implementation of BasicVSR++ flow_warp."""

from __future__ import annotations

import mlx.core as mx


def flow_warp(
    x: mx.array,
    flow: mx.array,
    *,
    padding_mode: str = "zeros",
) -> mx.array:
    """Warp NCHW feature tensor using NHW2 flow offsets."""

    if padding_mode not in {"zeros", "border"}:
        raise ValueError(f"unsupported padding_mode: {padding_mode}")
    if x.shape[0] != flow.shape[0] or x.shape[2] != flow.shape[1] or x.shape[3] != flow.shape[2]:
        raise ValueError(f"x shape {x.shape} and flow shape {flow.shape} are not spatially compatible")

    params = mx.array([1 if padding_mode == "border" else 0], dtype=mx.int32)
    (out,) = _FLOW_WARP_KERNEL(
        inputs=[x, flow, params],
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
        grid=(x.size, 1, 1),
        threadgroup=(256, 1, 1),
    )
    return out


_FLOW_WARP_KERNEL = mx.fast.metal_kernel(
    name="mlx_flow_warp_float32",
    input_names=["x", "flow", "params"],
    output_names=["out"],
    source="""
        uint elem = thread_position_in_grid.x;
        uint batch = x_shape[0];
        uint channels = x_shape[1];
        uint height = x_shape[2];
        uint width = x_shape[3];
        uint total = batch * channels * height * width;
        if (elem >= total) {
            return;
        }

        uint ox = elem % width;
        uint oy = (elem / width) % height;
        uint c = (elem / (height * width)) % channels;
        uint n = elem / (channels * height * width);
        bool border = params[0] != 0;

        uint flow_index = ((n * height + oy) * width + ox) * 2;
        float sample_x = float(ox) + float(flow[flow_index]);
        float sample_y = float(oy) + float(flow[flow_index + 1]);

        if (border) {
            sample_x = min(max(sample_x, 0.0f), float(width - 1));
            sample_y = min(max(sample_y, 0.0f), float(height - 1));
        }

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

        out[elem] = value;
    """,
)
