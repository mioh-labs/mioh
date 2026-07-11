# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Core AI Metal kernels used only while exporting BasicVSR++."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
import torchvision

DEFORM_CONV_THREADS_PER_GROUP = 256
DEFORM_CONV_SIMDGROUPS = 8
DEFORM_CONV_MAX_REDUCTION = 1152
DEFORM_CONV_TENSOROPS_TILE_ROWS = 8

GRID_SAMPLE_METAL_SOURCE = r"""
const uint out_x = gid.x;
const uint out_y = gid.y;
const uint out_z = gid.z;

const uint out_width = warped.get_extent(0);
const uint out_height = warped.get_extent(1);
const uint channels = warped.get_extent(2);
const uint batch_size = warped.get_extent(3);
if (out_x >= out_width || out_y >= out_height || out_z >= batch_size * channels) {
    return;
}

const uint channel = out_z % channels;
const uint batch = out_z / channels;
const uint in_width = image.get_extent(0);
const uint in_height = image.get_extent(1);

const float grid_x = float(grid[0, out_x, out_y, batch]);
const float grid_y = float(grid[1, out_x, out_y, batch]);
float source_x = ((grid_x + 1.0f) * float(in_width - 1)) * 0.5f;
float source_y = ((grid_y + 1.0f) * float(in_height - 1)) * 0.5f;
if (border) {
    source_x = clamp(source_x, 0.0f, float(in_width - 1));
    source_y = clamp(source_y, 0.0f, float(in_height - 1));
}

const int x0 = int(floor(source_x));
const int y0 = int(floor(source_y));
const int x1 = x0 + 1;
const int y1 = y0 + 1;
const float wx = source_x - float(x0);
const float wy = source_y - float(y0);

float value = 0.0f;
if (x0 >= 0 && x0 < int(in_width) && y0 >= 0 && y0 < int(in_height)) {
    value += float(image[uint(x0), uint(y0), channel, batch]) *
        (1.0f - wx) * (1.0f - wy);
}
if (x1 >= 0 && x1 < int(in_width) && y0 >= 0 && y0 < int(in_height)) {
    value += float(image[uint(x1), uint(y0), channel, batch]) *
        wx * (1.0f - wy);
}
if (x0 >= 0 && x0 < int(in_width) && y1 >= 0 && y1 < int(in_height)) {
    value += float(image[uint(x0), uint(y1), channel, batch]) *
        (1.0f - wx) * wy;
}
if (x1 >= 0 && x1 < int(in_width) && y1 >= 0 && y1 < int(in_height)) {
    value += float(image[uint(x1), uint(y1), channel, batch]) * wx * wy;
}

warped[out_x, out_y, channel, batch] = TYPE(value);
"""


DEFORM_CONV_METAL_SOURCE = r"""
constexpr uint tile_rows = 8;
constexpr uint max_reduction = 1152;
const uint in_width = image.get_extent(0);
const uint in_height = image.get_extent(1);
const uint in_channels = image.get_extent(2);
const uint batch_size = image.get_extent(3);
const uint out_width = offset.get_extent(0);
const uint out_height = offset.get_extent(1);
const uint output_rows = batch_size * out_width * out_height;
const uint out_channels = weight_matrix.get_extent(0);
const uint reduction_size = weight_matrix.get_extent(1);
const uint kernel_size = reduction_size / in_channels;
const uint kernel_width = 3;
const uint deform_groups = offset.get_extent(2) / (2 * kernel_size);
const uint channels_per_deform_group = in_channels / deform_groups;
const uint tile_start = tgid.x * tile_rows;

// Eight deformably sampled pixels share one 18 KiB half-precision tile.
// The tile becomes the left operand of a Metal 4 TensorOps matrix multiply.
threadgroup TYPE sampled_values[tile_rows * max_reduction];
const uint tile_elements = tile_rows * reduction_size;
for (uint tile_index = thread_index;
     tile_index < tile_elements;
     tile_index += 256) {
    const uint tile_row = tile_index / reduction_size;
    const uint reduction_index = tile_index - tile_row * reduction_size;
    const uint output_index = tile_start + tile_row;
    float sampled = 0.0f;

    if (output_index < output_rows) {
        const uint spatial_per_batch = out_width * out_height;
        const uint batch = output_index / spatial_per_batch;
        const uint spatial_index = output_index - batch * spatial_per_batch;
        const uint out_x = spatial_index % out_width;
        const uint out_y = spatial_index / out_width;
        const uint in_channel = reduction_index / kernel_size;
        const uint kernel_index = reduction_index - in_channel * kernel_size;
        const uint kernel_y = kernel_index / kernel_width;
        const uint kernel_x = kernel_index - kernel_y * kernel_width;
        const uint deform_group = in_channel / channels_per_deform_group;
        const uint offset_index =
            deform_group * 2 * kernel_size + 2 * kernel_index;
        const float offset_y =
            float(offset[out_x, out_y, offset_index, batch]);
        const float offset_x =
            float(offset[out_x, out_y, offset_index + 1, batch]);
        const float source_x =
            float(int(out_x) - 1 + int(kernel_x)) + offset_x;
        const float source_y =
            float(int(out_y) - 1 + int(kernel_y)) + offset_y;
        const int x0 = int(floor(source_x));
        const int y0 = int(floor(source_y));
        const int x1 = x0 + 1;
        const int y1 = y0 + 1;
        const float wx = source_x - float(x0);
        const float wy = source_y - float(y0);

        if (x0 >= 0 && x0 < int(in_width) &&
            y0 >= 0 && y0 < int(in_height)) {
            sampled += float(image[uint(x0), uint(y0), in_channel, batch]) *
                (1.0f - wx) * (1.0f - wy);
        }
        if (x1 >= 0 && x1 < int(in_width) &&
            y0 >= 0 && y0 < int(in_height)) {
            sampled += float(image[uint(x1), uint(y0), in_channel, batch]) *
                wx * (1.0f - wy);
        }
        if (x0 >= 0 && x0 < int(in_width) &&
            y1 >= 0 && y1 < int(in_height)) {
            sampled += float(image[uint(x0), uint(y1), in_channel, batch]) *
                (1.0f - wx) * wy;
        }
        if (x1 >= 0 && x1 < int(in_width) &&
            y1 >= 0 && y1 < int(in_height)) {
            sampled += float(image[uint(x1), uint(y1), in_channel, batch]) *
                wx * wy;
        }

        sampled *= float(mask[
            out_x,
            out_y,
            deform_group * kernel_size + kernel_index,
            batch
        ]);
    }
    sampled_values[tile_index] = TYPE(sampled);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

constexpr auto descriptor = matmul2d_descriptor(
    8,
    64,
    static_cast<int>(dynamic_extent),
    false,
    false,
    false
);
matmul2d<descriptor, execution_simdgroups<8>> matmul_op;
auto sample_extents = dextents<int, 2>(int(reduction_size), int(tile_rows));
auto sample_tensor = tensor(sampled_values, sample_extents);
auto output_tile = aligned_matrix.slice(0, int(tile_start));
for (uint output_tile_index = thread_index;
     output_tile_index < tile_rows * out_channels;
     output_tile_index += 256) {
    const uint tile_row = output_tile_index / out_channels;
    const uint out_channel = output_tile_index - tile_row * out_channels;
    const uint output_index = tile_start + tile_row;
    if (output_index < output_rows) {
        aligned_matrix[out_channel, output_index] = TYPE(0.0f);
    }
}
threadgroup_barrier(mem_flags::mem_device);
matmul_op.run(sample_tensor, weight_matrix, output_tile);
"""


def grid_sample_reference(
    image: torch.Tensor,
    grid: torch.Tensor,
    border: bool,
) -> torch.Tensor:
    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border" if border else "zeros",
        align_corners=True,
    )


def build_grid_sample_kernel(coreai_torch: Any):
    return coreai_torch.TorchMetalKernel(
        "grid_sample_bilinear_align_corners",
        input_names=["image", "grid", "border"],
        result_names=["warped"],
        src=GRID_SAMPLE_METAL_SOURCE,
        torch_defn=grid_sample_reference,
        metal_params=[
            coreai_torch.MetalParameter("gid", "uint3", "thread_position_in_grid")
        ],
        template_dtypes={"image": "TYPE"},
    )


def deform_conv_reference(
    image: torch.Tensor,
    offset: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return torchvision.ops.deform_conv2d(
        image,
        offset,
        weight,
        bias,
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        mask=mask,
    )


def deform_conv_tensorops_reference(
    image: torch.Tensor,
    offset: torch.Tensor,
    weight_matrix: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    in_channels = image.shape[1]
    out_channels = weight_matrix.shape[1]
    kernel_size = 3
    weight = (
        weight_matrix.reshape(in_channels, kernel_size, kernel_size, out_channels)
        .permute(3, 0, 1, 2)
        .contiguous()
    )
    output = torchvision.ops.deform_conv2d(
        image,
        offset,
        weight,
        bias=None,
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        mask=mask,
    )
    return output.permute(0, 2, 3, 1).reshape(-1, out_channels)


def build_deform_conv_kernel(coreai_torch: Any):
    return coreai_torch.TorchMetalKernel(
        "modulated_deform_conv2d",
        input_names=["image", "offset", "weight_matrix", "mask"],
        result_names=["aligned_matrix"],
        src=DEFORM_CONV_METAL_SOURCE,
        torch_defn=deform_conv_tensorops_reference,
        metal_params=[
            coreai_torch.MetalParameter(
                "tgid", "uint3", "threadgroup_position_in_grid"
            ),
            coreai_torch.MetalParameter(
                "thread_index", "uint", "thread_index_in_threadgroup"
            ),
        ],
        template_dtypes={"image": "TYPE"},
    )


def run_grid_sample_kernel(
    kernel: Any,
    image: torch.Tensor,
    grid: torch.Tensor,
    padding_mode: str = "zeros",
) -> torch.Tensor:
    if padding_mode not in {"zeros", "border"}:
        raise ValueError(f"unsupported grid-sample padding mode: {padding_mode}")
    batch, channels, _, _ = image.shape
    _, out_height, out_width, _ = grid.shape
    return kernel(
        image,
        grid,
        padding_mode == "border",
        threads_per_grid=(out_width, out_height, batch * channels),
        threads_per_thread_group=(16, 16, 1),
        result_shapes=[[batch, channels, out_height, out_width]],
    )


def run_deform_conv_kernel(
    kernel: Any,
    image: torch.Tensor,
    offset: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if weight.shape[1] != image.shape[1]:
        raise ValueError("Core AI deform conv supports only convolution groups=1")
    batch = image.shape[0]
    out_channels = weight.shape[0]
    out_height, out_width = offset.shape[-2:]
    reduction_size = image.shape[1] * weight.shape[2] * weight.shape[3]
    if reduction_size > DEFORM_CONV_MAX_REDUCTION:
        raise ValueError(
            f"Core AI deform conv reduction size {reduction_size} exceeds "
            f"threadgroup tile capacity {DEFORM_CONV_MAX_REDUCTION}"
        )
    output_rows = batch * out_height * out_width
    threadgroups = (
        output_rows + DEFORM_CONV_TENSOROPS_TILE_ROWS - 1
    ) // DEFORM_CONV_TENSOROPS_TILE_ROWS
    weight_matrix = (
        weight.permute(1, 2, 3, 0)
        .reshape(reduction_size, out_channels)
        .contiguous()
    )
    aligned_matrix = kernel(
        image,
        offset,
        weight_matrix,
        mask,
        threads_per_grid=(
            threadgroups * DEFORM_CONV_THREADS_PER_GROUP,
            1,
            1,
        ),
        threads_per_thread_group=(DEFORM_CONV_THREADS_PER_GROUP, 1, 1),
        result_shapes=[[output_rows, out_channels]],
    )
    return (
        aligned_matrix.add(bias)
        .reshape(batch, out_height, out_width, out_channels)
        .permute(0, 3, 1, 2)
    )
