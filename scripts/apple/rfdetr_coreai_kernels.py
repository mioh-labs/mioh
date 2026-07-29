# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Core AI Metal kernels used while exporting fixed-shape RF-DETR Seg models.

The currently supported RF-DETR variants use one feature level, 16 attention
heads and two sampling points per query.  The feature side is derived from the
fixed model resolution (32 for Seg Small 384, 48 for Jasna v6 Medium 576).
Folding ``level`` and ``point`` into one axis keeps every intermediate at rank
five or lower and maps cleanly to Core AI.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn.functional as F

FEATURE_SIDE = 32
ATTENTION_HEADS = 16
POINTS_PER_HEAD = 2


MS_DEFORM_ATTN_METAL_SOURCE = r"""
const uint output_channel = gid.x;
const uint query_index = gid.y;
const uint batch_index = gid.z;

const uint model_width = output.get_extent(0);
const uint query_count = output.get_extent(1);
const uint batch_size = output.get_extent(2);
if (output_channel >= model_width ||
    query_index >= query_count ||
    batch_index >= batch_size) {
    return;
}

constexpr uint feature_side = __FEATURE_SIDE__;
constexpr uint point_count = __POINT_COUNT__;
const uint head_dim = value.get_extent(1);
const uint head_index = output_channel / head_dim;
const uint channel_in_head = output_channel - head_index * head_dim;

float accumulated = 0.0f;
for (uint point_index = 0; point_index < point_count; ++point_index) {
    const float location_x = float(sampling_locations[
        0, point_index, head_index, query_index, batch_index
    ]);
    const float location_y = float(sampling_locations[
        1, point_index, head_index, query_index, batch_index
    ]);

    // RF-DETR calls grid_sample with align_corners=False after mapping
    // locations from [0, 1] to [-1, 1].
    const float source_x = location_x * float(feature_side) - 0.5f;
    const float source_y = location_y * float(feature_side) - 0.5f;
    const int x0 = int(floor(source_x));
    const int y0 = int(floor(source_y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;
    const float wx = source_x - float(x0);
    const float wy = source_y - float(y0);

    float sampled = 0.0f;
    if (x0 >= 0 && x0 < int(feature_side) &&
        y0 >= 0 && y0 < int(feature_side)) {
        sampled += float(value[
            channel_in_head,
            head_index,
            uint(y0) * feature_side + uint(x0),
            batch_index
        ]) * (1.0f - wx) * (1.0f - wy);
    }
    if (x1 >= 0 && x1 < int(feature_side) &&
        y0 >= 0 && y0 < int(feature_side)) {
        sampled += float(value[
            channel_in_head,
            head_index,
            uint(y0) * feature_side + uint(x1),
            batch_index
        ]) * wx * (1.0f - wy);
    }
    if (x0 >= 0 && x0 < int(feature_side) &&
        y1 >= 0 && y1 < int(feature_side)) {
        sampled += float(value[
            channel_in_head,
            head_index,
            uint(y1) * feature_side + uint(x0),
            batch_index
        ]) * (1.0f - wx) * wy;
    }
    if (x1 >= 0 && x1 < int(feature_side) &&
        y1 >= 0 && y1 < int(feature_side)) {
        sampled += float(value[
            channel_in_head,
            head_index,
            uint(y1) * feature_side + uint(x1),
            batch_index
        ]) * wx * wy;
    }

    accumulated += sampled * float(attention_weights[
        point_index, head_index, query_index, batch_index
    ]);
}

output[output_channel, query_index, batch_index] = TYPE(accumulated);
"""


def _ms_deform_attn_reference(
    value: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    *,
    feature_side: int,
    attention_heads: int,
    points_per_head: int,
) -> torch.Tensor:
    """Reference for one fixed-shape RF-DETR deformable attention kernel."""

    batch, spatial, heads, head_dim = value.shape
    if heads != attention_heads or spatial != feature_side * feature_side:
        raise ValueError(
            "RF-DETR Core AI kernel received an incompatible value tensor"
        )
    if sampling_locations.shape[-2:] != (points_per_head, 2):
        raise ValueError(
            "RF-DETR Core AI kernel received incompatible sampling locations"
        )

    query_count = sampling_locations.shape[1]
    value_image = (
        value.permute(0, 2, 3, 1)
        .contiguous()
        .reshape(
            batch * heads,
            head_dim,
            feature_side,
            feature_side,
        )
    )
    grid = (
        sampling_locations.transpose(1, 2)
        .reshape(batch * heads, query_count, points_per_head, 2)
        .mul(2)
        .sub(1)
    )
    # CPU does not implement fp16 grid_sample.  The cast is part of the
    # reference only; the Metal kernel evaluates directly in TYPE.
    sampled = F.grid_sample(
        value_image.float(),
        grid.float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).to(value.dtype)
    weights = attention_weights.transpose(1, 2).reshape(
        batch * heads,
        1,
        query_count,
        points_per_head,
    )
    return (
        sampled.mul(weights)
        .sum(-1)
        .reshape(batch, heads * head_dim, query_count)
        .transpose(1, 2)
        .contiguous()
    )


def make_ms_deform_attn_reference(
    *,
    feature_side: int = FEATURE_SIDE,
    attention_heads: int = ATTENTION_HEADS,
    points_per_head: int = POINTS_PER_HEAD,
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    def reference(
        value: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        return _ms_deform_attn_reference(
            value,
            sampling_locations,
            attention_weights,
            feature_side=feature_side,
            attention_heads=attention_heads,
            points_per_head=points_per_head,
        )

    return reference


ms_deform_attn_reference = make_ms_deform_attn_reference()


def build_ms_deform_attn_kernel(
    coreai_torch: Any,
    *,
    feature_side: int = FEATURE_SIDE,
    attention_heads: int = ATTENTION_HEADS,
    points_per_head: int = POINTS_PER_HEAD,
):
    source = (
        MS_DEFORM_ATTN_METAL_SOURCE
        .replace("__FEATURE_SIDE__", str(feature_side))
        .replace("__POINT_COUNT__", str(points_per_head))
    )
    return coreai_torch.TorchMetalKernel(
        f"rfdetr_ms_deform_attn_f{feature_side}_h{attention_heads}_p{points_per_head}",
        input_names=["value", "sampling_locations", "attention_weights"],
        result_names=["output"],
        src=source,
        torch_defn=make_ms_deform_attn_reference(
            feature_side=feature_side,
            attention_heads=attention_heads,
            points_per_head=points_per_head,
        ),
        metal_params=[
            coreai_torch.MetalParameter("gid", "uint3", "thread_position_in_grid")
        ],
        template_dtypes={"value": "TYPE"},
    )


def run_ms_deform_attn_kernel(
    kernel: Any,
    value: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    *,
    feature_side: int = FEATURE_SIDE,
    attention_heads: int = ATTENTION_HEADS,
    points_per_head: int = POINTS_PER_HEAD,
) -> torch.Tensor:
    batch, spatial, heads, head_dim = value.shape
    query_count = sampling_locations.shape[1]
    if heads != attention_heads or spatial != feature_side * feature_side:
        raise ValueError(
            "RF-DETR Core AI export received an incompatible feature tensor"
        )
    if sampling_locations.shape != (
        batch,
        query_count,
        heads,
        points_per_head,
        2,
    ):
        raise ValueError(
            "unexpected RF-DETR sampling location shape "
            f"{tuple(sampling_locations.shape)}"
        )
    return kernel(
        value,
        sampling_locations,
        attention_weights,
        threads_per_grid=(heads * head_dim, query_count, batch),
        threads_per_thread_group=(16, 4, 1),
        result_shapes=[[batch, query_count, heads * head_dim]],
    )
