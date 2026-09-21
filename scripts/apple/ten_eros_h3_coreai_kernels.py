# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Experimental Core AI Metal kernels for the 10Eros-Max H3 DiT path."""

from __future__ import annotations

from typing import Any

import torch


CONVROT_GROUP_SIZE = 256
ROTATION_TILE_ROWS = 8
ROTATION_THREADS_PER_GROUP = 256
LINEAR_TILE_ROWS = 64
LINEAR_TILE_COLUMNS = 32
LINEAR_THREADS_PER_GROUP = 128


CONVROT_METAL_SOURCE = r"""
constexpr uint group_size = 256;
constexpr uint tile_rows = 8;

const uint input_features = hidden_states.get_extent(0);
const uint rows = hidden_states.get_extent(1);
const uint input_start = tgid.x * group_size;
const uint row_start = tgid.y * tile_rows;
if (input_start >= input_features || row_start >= rows) {
    return;
}

threadgroup float rotated_work[tile_rows * group_size];
for (uint index = thread_index;
     index < tile_rows * group_size;
     index += 256) {
    const uint local_row = index / group_size;
    const uint feature = index - local_row * group_size;
    const uint row = row_start + local_row;
    rotated_work[index] = row < rows
        ? float(hidden_states[input_start + feature, row])
        : 0.0f;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

// Fast normalized H4^\u22974 transform, matching the checkpoint's ConvRot.
for (uint stride = 1; stride < group_size; stride *= 4) {
    for (uint index = thread_index;
         index < tile_rows * (group_size / 4);
         index += 256) {
        const uint local_row = index / (group_size / 4);
        const uint butterfly = index - local_row * (group_size / 4);
        const uint quotient = butterfly / stride;
        const uint block = quotient * (4 * stride);
        const uint offset = butterfly - quotient * stride;
        const uint base = local_row * group_size + block + offset;
        const uint i0 = base;
        const uint i1 = base + stride;
        const uint i2 = base + 2 * stride;
        const uint i3 = base + 3 * stride;
        const float a = rotated_work[i0];
        const float b = rotated_work[i1];
        const float c = rotated_work[i2];
        const float d = rotated_work[i3];
        rotated_work[i0] = a + b + c - d;
        rotated_work[i1] = a + b - c + d;
        rotated_work[i2] = a - b + c + d;
        rotated_work[i3] = -a + b + c + d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint index = thread_index;
     index < tile_rows * group_size;
     index += 256) {
    const uint local_row = index / group_size;
    const uint feature = index - local_row * group_size;
    const uint row = row_start + local_row;
    if (row < rows) {
        rotated_hidden[input_start + feature, row] = bfloat(
            rotated_work[index] * 0.0625f
        );
    }
}
"""


INT8_LINEAR_METAL_SOURCE = r"""
constexpr uint group_size = 256;
constexpr uint tile_rows = 64;
constexpr uint tile_columns = 32;

const uint input_features = rotated_hidden.get_extent(0);
const uint rows = rotated_hidden.get_extent(1);
const uint output_features = quantized_weight.get_extent(0);
const uint output_start = tgid.x * tile_columns;
const uint row_start = tgid.y * tile_rows;
if (output_start >= output_features || row_start >= rows) {
    return;
}

constexpr auto descriptor = matmul2d_descriptor(
    tile_rows, tile_columns, group_size,
    false, false, false,
    matmul2d_descriptor::mode::multiply_accumulate
);
matmul2d<descriptor, execution_simdgroups<4>> matmul_op;

auto hidden_tile = rotated_hidden.slice(0, int(row_start));
auto weight_tile = quantized_weight.slice(int(output_start), 0);
auto accumulator = matmul_op
    .get_destination_cooperative_tensor<
        __tensor_ops_detail::__remove_addrspace_t<decltype(hidden_tile)>,
        __tensor_ops_detail::__remove_addrspace_t<decltype(weight_tile)>,
        float
    >();

#pragma unroll
for (uint16_t index = 0; index < accumulator.get_capacity(); ++index) {
    if (accumulator.is_valid_element(index)) {
        accumulator[index] = 0.0f;
    }
}

for (uint input_start = 0; input_start < input_features; input_start += group_size) {
    hidden_tile = rotated_hidden.slice(int(input_start), int(row_start));
    weight_tile = quantized_weight.slice(int(output_start), int(input_start));
    matmul_op.run(hidden_tile, weight_tile, accumulator);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#pragma unroll
for (uint16_t index = 0; index < accumulator.get_capacity(); ++index) {
    if (accumulator.is_valid_element(index)) {
        const auto local = accumulator.get_multidimensional_index(index);
        const uint output_feature = output_start + uint(local[0]);
        const uint output_row = row_start + uint(local[1]);
        if (output_feature < output_features && output_row < rows) {
            const float value = accumulator[index] * float(scale[output_feature])
                + float(bias[output_feature]);
            projected[output_feature, output_row] = bfloat(value);
        }
    }
}
"""


BF16_LINEAR_METAL_SOURCE = r"""
constexpr uint group_size = 256;
constexpr uint tile_rows = 64;
constexpr uint tile_columns = 32;

const uint input_features = rotated_hidden.get_extent(0);
const uint rows = rotated_hidden.get_extent(1);
const uint output_features = weight.get_extent(0);
const uint output_start = tgid.x * tile_columns;
const uint row_start = tgid.y * tile_rows;
if (output_start >= output_features || row_start >= rows) {
    return;
}

constexpr auto descriptor = matmul2d_descriptor(
    tile_rows, tile_columns, group_size,
    false, false, false,
    matmul2d_descriptor::mode::multiply_accumulate
);
matmul2d<descriptor, execution_simdgroups<4>> matmul_op;

auto hidden_tile = rotated_hidden.slice(0, int(row_start));
auto weight_tile = weight.slice(int(output_start), 0);
auto accumulator = matmul_op
    .get_destination_cooperative_tensor<
        __tensor_ops_detail::__remove_addrspace_t<decltype(hidden_tile)>,
        __tensor_ops_detail::__remove_addrspace_t<decltype(weight_tile)>,
        float
    >();

#pragma unroll
for (uint16_t index = 0; index < accumulator.get_capacity(); ++index) {
    if (accumulator.is_valid_element(index)) {
        accumulator[index] = 0.0f;
    }
}

for (uint input_start = 0; input_start < input_features; input_start += group_size) {
    hidden_tile = rotated_hidden.slice(int(input_start), int(row_start));
    weight_tile = weight.slice(int(output_start), int(input_start));
    matmul_op.run(hidden_tile, weight_tile, accumulator);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#pragma unroll
for (uint16_t index = 0; index < accumulator.get_capacity(); ++index) {
    if (accumulator.is_valid_element(index)) {
        const auto local = accumulator.get_multidimensional_index(index);
        const uint output_feature = output_start + uint(local[0]);
        const uint output_row = row_start + uint(local[1]);
        if (output_feature < output_features && output_row < rows) {
            const float value = accumulator[index] + float(bias[output_feature]);
            projected[output_feature, output_row] = bfloat(value);
        }
    }
}
"""


def normalized_hadamard_256(value: torch.Tensor) -> torch.Tensor:
    """Apply the exact normalized H4^\u22974 transform used by ConvRot."""

    if value.shape[-1] != CONVROT_GROUP_SIZE:
        raise ValueError(
            f"ConvRot group must end in {CONVROT_GROUP_SIZE}, got {value.shape}"
        )
    transformed = value
    stride = 1
    while stride < CONVROT_GROUP_SIZE:
        shape = transformed.shape
        transformed = transformed.reshape(*shape[:-1], -1, 4, stride)
        a, b, c, d = transformed.unbind(dim=-2)
        transformed = torch.stack(
            (a + b + c - d, a + b - c + d, a - b + c + d, -a + b + c + d),
            dim=-2,
        ).reshape(shape)
        stride *= 4
    return transformed * (1.0 / 16.0)


def convrot_reference(hidden_states: torch.Tensor) -> torch.Tensor:
    """PyTorch reference for the normalized ConvRot transform."""

    input_features = hidden_states.shape[-1]
    if input_features % CONVROT_GROUP_SIZE:
        raise ValueError(
            f"input features {input_features} are not divisible by ConvRot group size"
        )
    rows = hidden_states.reshape(-1, input_features)
    groups = input_features // CONVROT_GROUP_SIZE
    return normalized_hadamard_256(
        rows.reshape(-1, groups, CONVROT_GROUP_SIZE).float()
    ).to(hidden_states.dtype).reshape_as(rows)


def int8_linear_reference(
    rotated_hidden: torch.Tensor,
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference for the tiled INT8-dequantized matrix product."""

    raw = torch.matmul(rotated_hidden, quantized_weight.to(rotated_hidden.dtype))
    return raw.mul(scale.to(rotated_hidden.dtype)).add(
        bias.to(rotated_hidden.dtype)
    )


def int8_convrot_linear_reference(
    hidden_states: torch.Tensor,
    quantized_weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return int8_linear_reference(
        convrot_reference(hidden_states), quantized_weight, scale, bias
    )


def bf16_linear_reference(
    rotated_hidden: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """PyTorch reference for a pre-dequantized BF16 matrix product."""

    return torch.matmul(rotated_hidden, weight).add(bias)


def build_int8_convrot_linear_kernels(
    coreai_torch: Any, *, scalar_type: str = "bfloat16"
) -> tuple[Any, Any]:
    if scalar_type not in {"bfloat16", "float16"}:
        raise ValueError(f"unsupported ConvRot scalar type: {scalar_type}")
    metal_scalar = "bfloat" if scalar_type == "bfloat16" else "half"
    scalar_tag = "bf16" if scalar_type == "bfloat16" else "fp16"
    rotation_source = CONVROT_METAL_SOURCE.replace("bfloat(", f"{metal_scalar}(")
    linear_source = INT8_LINEAR_METAL_SOURCE.replace(
        "bfloat(", f"{metal_scalar}("
    )
    parameters = [
        coreai_torch.MetalParameter(
            "tgid", "uint3", "threadgroup_position_in_grid"
        ),
        coreai_torch.MetalParameter(
            "thread_index", "uint", "thread_index_in_threadgroup"
        ),
    ]
    rotation = coreai_torch.TorchMetalKernel(
        f"ten_eros_convrot_{scalar_tag}_v3",
        input_names=["hidden_states"],
        result_names=["rotated_hidden"],
        src=rotation_source,
        torch_defn=convrot_reference,
        metal_params=parameters,
    )
    linear = coreai_torch.TorchMetalKernel(
        f"ten_eros_int8_linear_{scalar_tag}_v4",
        input_names=["rotated_hidden", "quantized_weight", "scale", "bias"],
        result_names=["projected"],
        src=linear_source,
        torch_defn=int8_linear_reference,
        metal_params=parameters,
    )
    return rotation, linear


def build_bf16_convrot_linear_kernels(coreai_torch: Any) -> tuple[Any, Any]:
    parameters = [
        coreai_torch.MetalParameter(
            "tgid", "uint3", "threadgroup_position_in_grid"
        ),
        coreai_torch.MetalParameter(
            "thread_index", "uint", "thread_index_in_threadgroup"
        ),
    ]
    rotation = coreai_torch.TorchMetalKernel(
        "ten_eros_convrot_bf16_v3",
        input_names=["hidden_states"],
        result_names=["rotated_hidden"],
        src=CONVROT_METAL_SOURCE,
        torch_defn=convrot_reference,
        metal_params=parameters,
    )
    linear = coreai_torch.TorchMetalKernel(
        "ten_eros_linear_bf16_v1",
        input_names=["rotated_hidden", "weight", "bias"],
        result_names=["projected"],
        src=BF16_LINEAR_METAL_SOURCE,
        torch_defn=bf16_linear_reference,
        metal_params=parameters,
    )
    return rotation, linear


class MetalINT8ConvRotLinear(torch.nn.Module):
    """Opt-in two-kernel Metal implementation of an INT8 ConvRot Linear."""

    def __init__(
        self,
        kernels: tuple[Any, Any],
        quantized_weight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
        group_size: int,
    ) -> None:
        super().__init__()
        if group_size != CONVROT_GROUP_SIZE:
            raise ValueError(
                f"Metal ConvRot requires group size {CONVROT_GROUP_SIZE}, got {group_size}"
            )
        if quantized_weight.dtype != torch.int8 or quantized_weight.ndim != 2:
            raise TypeError("Metal ConvRot weight must be rank-2 INT8")
        output_features, input_features = quantized_weight.shape
        if input_features % group_size:
            raise ValueError("Metal ConvRot input features must align to group size")
        self.rotation_kernel, self.linear_kernel = kernels
        self.register_buffer(
            "quantized_weight", quantized_weight.transpose(0, 1).contiguous()
        )
        self.register_buffer(
            "scale", scale.reshape(output_features).to(torch.bfloat16).contiguous()
        )
        self.register_buffer(
            "bias",
            (
                torch.zeros(output_features, dtype=torch.bfloat16)
                if bias is None
                else bias.reshape(output_features).to(torch.bfloat16)
            ).contiguous(),
        )
        self.input_features = input_features
        self.output_features = output_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.input_features:
            raise ValueError(
                f"Metal ConvRot expected {self.input_features} features, "
                f"got {hidden_states.shape[-1]}"
            )
        rows = hidden_states.numel() // self.input_features
        matrix = hidden_states.reshape(rows, self.input_features)
        rotation_row_tiles = (
            rows + ROTATION_TILE_ROWS - 1
        ) // ROTATION_TILE_ROWS
        linear_row_tiles = (rows + LINEAR_TILE_ROWS - 1) // LINEAR_TILE_ROWS
        group_tiles = self.input_features // CONVROT_GROUP_SIZE
        output_tiles = (
            self.output_features + LINEAR_TILE_COLUMNS - 1
        ) // LINEAR_TILE_COLUMNS
        rotated = self.rotation_kernel(
            matrix,
            threads_per_grid=(
                group_tiles * ROTATION_THREADS_PER_GROUP,
                rotation_row_tiles,
                1,
            ),
            threads_per_thread_group=(ROTATION_THREADS_PER_GROUP, 1, 1),
            result_shapes=[[rows, self.input_features]],
        )
        projected = self.linear_kernel(
            rotated,
            self.quantized_weight,
            self.scale,
            self.bias,
            threads_per_grid=(
                output_tiles * LINEAR_THREADS_PER_GROUP,
                linear_row_tiles,
                1,
            ),
            threads_per_thread_group=(LINEAR_THREADS_PER_GROUP, 1, 1),
            result_shapes=[[rows, self.output_features]],
        )
        return projected.reshape(*hidden_states.shape[:-1], self.output_features)


class MetalBF16ConvRotLinear(torch.nn.Module):
    """Metal ConvRot Linear with conversion-time BF16 weight expansion."""

    def __init__(
        self,
        kernels: tuple[Any, Any],
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        group_size: int,
    ) -> None:
        super().__init__()
        if group_size != CONVROT_GROUP_SIZE:
            raise ValueError(
                f"Metal ConvRot requires group size {CONVROT_GROUP_SIZE}, got {group_size}"
            )
        if weight.dtype != torch.bfloat16 or weight.ndim != 2:
            raise TypeError("Metal ConvRot weight must be rank-2 BF16")
        output_features, input_features = weight.shape
        if input_features % group_size:
            raise ValueError("Metal ConvRot input features must align to group size")
        self.rotation_kernel, self.linear_kernel = kernels
        self.register_buffer("weight", weight.transpose(0, 1).contiguous())
        self.register_buffer(
            "bias",
            (
                torch.zeros(output_features, dtype=torch.bfloat16)
                if bias is None
                else bias.reshape(output_features).to(torch.bfloat16)
            ).contiguous(),
        )
        self.input_features = input_features
        self.output_features = output_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.input_features:
            raise ValueError(
                f"Metal ConvRot expected {self.input_features} features, "
                f"got {hidden_states.shape[-1]}"
            )
        rows = hidden_states.numel() // self.input_features
        matrix = hidden_states.reshape(rows, self.input_features)
        rotation_row_tiles = (
            rows + ROTATION_TILE_ROWS - 1
        ) // ROTATION_TILE_ROWS
        linear_row_tiles = (rows + LINEAR_TILE_ROWS - 1) // LINEAR_TILE_ROWS
        group_tiles = self.input_features // CONVROT_GROUP_SIZE
        output_tiles = (
            self.output_features + LINEAR_TILE_COLUMNS - 1
        ) // LINEAR_TILE_COLUMNS
        rotated = self.rotation_kernel(
            matrix,
            threads_per_grid=(
                group_tiles * ROTATION_THREADS_PER_GROUP,
                rotation_row_tiles,
                1,
            ),
            threads_per_thread_group=(ROTATION_THREADS_PER_GROUP, 1, 1),
            result_shapes=[[rows, self.input_features]],
        )
        projected = self.linear_kernel(
            rotated,
            self.weight,
            self.bias,
            threads_per_grid=(
                output_tiles * LINEAR_THREADS_PER_GROUP,
                linear_row_tiles,
                1,
            ),
            threads_per_thread_group=(LINEAR_THREADS_PER_GROUP, 1, 1),
            result_shapes=[[rows, self.output_features]],
        )
        return projected.reshape(*hidden_states.shape[:-1], self.output_features)
