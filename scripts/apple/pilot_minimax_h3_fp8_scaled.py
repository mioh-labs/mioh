#!/usr/bin/env python3
"""Validate a MiniMax H3 scaled-FP8 Linear through a Metal TensorOps kernel."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open


TILE_ROWS = 64
TILE_COLUMNS = 32
TILE_INNER = 256
THREADS = 128


FP8_SCALED_LINEAR_METAL_SOURCE = r"""
constexpr uint tile_rows = 64;
constexpr uint tile_columns = 32;
constexpr uint tile_inner = 256;

const uint input_features = hidden_states.get_extent(0);
const uint rows = hidden_states.get_extent(1);
const uint output_features = weight_bits.get_extent(0);
const uint output_start = tgid.x * tile_columns;
const uint row_start = tgid.y * tile_rows;
if (output_start >= output_features || row_start >= rows) {
    return;
}

constexpr auto descriptor = matmul2d_descriptor(
    tile_rows, tile_columns, tile_inner,
    false, false, false,
    matmul2d_descriptor::mode::multiply_accumulate
);
matmul2d<descriptor, execution_simdgroups<4>> matmul_op;

auto hidden_tile = hidden_states.slice(0, int(row_start));
threadgroup bfloat decoded_storage[tile_inner * tile_columns];
using decoded_tensor = tensor<
    threadgroup bfloat,
    dextents<int, 2>,
    tensor_inline
>;
decoded_tensor decoded_weight(
    decoded_storage,
    dextents<int, 2>(tile_columns, tile_inner),
    array<int, 2>({1, tile_columns})
);
auto accumulator = matmul_op
    .get_destination_cooperative_tensor<
        __tensor_ops_detail::__remove_addrspace_t<decltype(hidden_tile)>,
        decoded_tensor,
        float
    >();

#pragma unroll
for (uint16_t index = 0; index < accumulator.get_capacity(); ++index) {
    if (accumulator.is_valid_element(index)) {
        accumulator[index] = 0.0f;
    }
}

const bfloat tensor_scale = bfloat(weight_scale[0]);
for (uint input_start = 0; input_start < input_features; input_start += tile_inner) {
    for (uint index = thread_index;
         index < tile_inner * tile_columns;
         index += THREADS_PER_GROUP) {
        const uint inner = index / tile_columns;
        const uint column = index - inner * tile_columns;
        const uint output_feature = output_start + column;
        const uint input_feature = input_start + inner;
        float value = 0.0f;
        if (output_feature < output_features && input_feature < input_features) {
            const bfloat quantized_value = bfloat(decode_e4m3fn(
                weight_bits[output_feature, input_feature]
            ));
            value = float(bfloat(quantized_value * tensor_scale));
        }
        decoded_storage[index] = bfloat(value);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hidden_tile = hidden_states.slice(int(input_start), int(row_start));
    matmul_op.run(hidden_tile, decoded_weight, accumulator);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

#pragma unroll
for (uint16_t index = 0; index < accumulator.get_capacity(); ++index) {
    if (accumulator.is_valid_element(index)) {
        const auto local = accumulator.get_multidimensional_index(index);
        const uint output_feature = output_start + uint(local[0]);
        const uint output_row = row_start + uint(local[1]);
        if (output_feature < output_features && output_row < rows) {
            projected[output_feature, output_row] = bfloat(
                accumulator[index] + float(bias[output_feature])
            );
        }
    }
}
"""


FP8_HELPER_SOURCE = r"""
#define THREADS_PER_GROUP 128

inline float decode_e4m3fn(uint8_t bits) {
    const uint sign = uint(bits >> 7);
    const uint exponent = (uint(bits) >> 3) & 0x0f;
    const uint mantissa = uint(bits) & 0x07;
    float magnitude;
    if (exponent == 0) {
        magnitude = float(mantissa) * 0.001953125f;
    } else if (exponent == 15 && mantissa == 7) {
        magnitude = 0.0f;
    } else {
        magnitude = (1.0f + float(mantissa) * 0.125f)
            * exp2(float(int(exponent) - 7));
    }
    return sign ? -magnitude : magnitude;
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--tensor-prefix", default="blocks.0.attn.out_proj"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument(
        "--implementation", choices=("metal", "dense"), default="metal"
    )
    parser.add_argument("--synthetic-input-features", type=int, default=256)
    parser.add_argument("--synthetic-output-features", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def remove_existing(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def fp8_linear_reference(
    hidden_states: torch.Tensor,
    weight_bits: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    weight = (
        weight_bits.transpose(0, 1)
        .contiguous()
        .view(torch.float8_e4m3fn)
        .to(torch.bfloat16)
    )
    weight = weight * weight_scale.to(torch.bfloat16)
    return torch.nn.functional.linear(hidden_states, weight, bias)


def build_kernel(coreai_torch: Any) -> Any:
    parameters = [
        coreai_torch.MetalParameter(
            "tgid", "uint3", "threadgroup_position_in_grid"
        ),
        coreai_torch.MetalParameter(
            "thread_index", "uint", "thread_index_in_threadgroup"
        ),
    ]
    return coreai_torch.TorchMetalKernel(
        "minimax_h3_scaled_fp8_linear_bf16_v1",
        input_names=["hidden_states", "weight_bits", "weight_scale", "bias"],
        result_names=["projected"],
        src=FP8_SCALED_LINEAR_METAL_SOURCE,
        helper_src=FP8_HELPER_SOURCE,
        torch_defn=fp8_linear_reference,
        metal_params=parameters,
    )


def load_mapping(
    checkpoint: Path | None,
    prefix: str,
    synthetic_input_features: int,
    synthetic_output_features: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if checkpoint is None:
        values = torch.linspace(
            -2.0,
            2.0,
            synthetic_output_features * synthetic_input_features,
            dtype=torch.float32,
        ).reshape(synthetic_output_features, synthetic_input_features)
        weight = values.to(torch.float8_e4m3fn)
        return (
            weight.view(torch.uint8).transpose(0, 1).contiguous(),
            torch.tensor([0.125], dtype=torch.float32),
            torch.zeros(synthetic_output_features, dtype=torch.bfloat16),
        )

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(f"{prefix}.weight")
        scale = handle.get_tensor(f"{prefix}.weight_scale").float().reshape(1)
        bias = (
            handle.get_tensor(f"{prefix}.bias").to(torch.bfloat16)
            if f"{prefix}.bias" in handle.keys()
            else torch.zeros(weight.shape[0], dtype=torch.bfloat16)
        )
    if weight.dtype != torch.float8_e4m3fn or weight.ndim != 2:
        raise TypeError(f"{prefix}.weight must be rank-2 float8_e4m3fn")
    return weight.view(torch.uint8).transpose(0, 1).contiguous(), scale, bias


class FP8ScaledLinear(torch.nn.Module):
    def __init__(
        self,
        kernel: Any,
        weight_bits: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.register_buffer("weight_bits", weight_bits)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("bias", bias)
        self.input_features = weight_bits.shape[0]
        self.output_features = weight_bits.shape[1]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        rows = hidden_states.numel() // self.input_features
        matrix = hidden_states.reshape(rows, self.input_features)
        return self.kernel(
            matrix,
            self.weight_bits,
            self.weight_scale,
            self.bias,
            threads_per_grid=(
                ((self.output_features + TILE_COLUMNS - 1) // TILE_COLUMNS)
                * THREADS,
                (rows + TILE_ROWS - 1) // TILE_ROWS,
                1,
            ),
            threads_per_thread_group=(THREADS, 1, 1),
            result_shapes=[[rows, self.output_features]],
        )


class DenseFP8ScaledLinear(torch.nn.Module):
    """Comparison path that expands the FP8 checkpoint weight to BF16."""

    def __init__(
        self,
        weight_bits: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        super().__init__()
        weight = (
            weight_bits.transpose(0, 1)
            .contiguous()
            .view(torch.float8_e4m3fn)
            .to(torch.bfloat16)
        )
        self.register_buffer(
            "weight", (weight * weight_scale.to(torch.bfloat16)).contiguous()
        )
        self.register_buffer("bias", bias)
        self.input_features = weight.shape[1]
        self.output_features = weight.shape[0]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(hidden_states, self.weight, self.bias)


def export_coreai(
    model: torch.nn.Module,
    example: torch.Tensor,
    destination: Path,
    kernel: Any | None,
) -> None:
    import coreai_torch

    exported = torch.export.export(model, (example,))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    if kernel is not None:
        converter.register_custom_kernels([kernel])
    converter.add_exported_program(
        exported,
        input_names=["hidden_states"],
        output_names=["projected"],
    )
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(destination)


def main() -> int:
    args = parse_args()
    if args.checkpoint is not None and not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    remove_existing(args.output, args.overwrite)
    remove_existing(args.reference_directory, args.overwrite)
    args.reference_directory.mkdir(parents=True, exist_ok=True)

    import coreai_torch

    kernel = build_kernel(coreai_torch)
    weight_bits, scale, bias = load_mapping(
        args.checkpoint,
        args.tensor_prefix,
        args.synthetic_input_features,
        args.synthetic_output_features,
    )
    if args.implementation == "metal":
        model = FP8ScaledLinear(kernel, weight_bits, scale, bias).eval()
        export_kernel = kernel
    else:
        model = DenseFP8ScaledLinear(weight_bits, scale, bias).eval()
        export_kernel = None
    example = torch.sin(
        torch.arange(
            args.rows * model.input_features, dtype=torch.float32
        ) * 0.00390625
    ).reshape(args.rows, model.input_features).to(torch.bfloat16)
    with torch.no_grad():
        reference = fp8_linear_reference(
            example, weight_bits, scale, bias
        ).float()

    np.asarray(example.float().numpy(), dtype=np.float32).tofile(
        args.reference_directory / "input.f32"
    )
    np.asarray(reference.numpy(), dtype=np.float32).tofile(
        args.reference_directory / "reference.f32"
    )
    metadata = {
        "tensorPrefix": args.tensor_prefix,
        "implementation": args.implementation,
        "scalarType": "bfloat16",
        "inputShape": list(example.shape),
        "outputShape": list(reference.shape),
        "referenceMean": float(reference.mean()),
        "referenceRMS": float(reference.square().mean().sqrt()),
        "weightBytes": weight_bits.numel(),
        "scale": float(scale[0]),
    }
    (args.reference_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    export_coreai(model, example, args.output, export_kernel)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
