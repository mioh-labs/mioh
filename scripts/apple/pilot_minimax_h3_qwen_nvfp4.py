#!/usr/bin/env python3
"""Validate an exact MiniMax H3 Qwen NVFP4 -> Core AI linear mapping.

The checkpoint stores E2M1 values packed two per byte and one E4M3 block
scale per 16 logical weights.  E2M1 values are multiples of 0.5, so they can
be represented without requantization as INT8 ``2 * value`` with
``0.5 * block_scale``.  Core AI's blockwise shift/scale operation can then
materialize the original dense weight lazily at execution time.

This pilot intentionally exports one real checkpoint layer.  It also writes
the deterministic input and PyTorch reference output used by the Swift probe.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


E2M1_TIMES_TWO = torch.tensor(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=torch.int8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tensor-prefix",
        default="model.layers.0.self_attn.k_proj",
        help="linear tensor prefix without .weight",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
    parser.add_argument(
        "--representation",
        choices=("int8", "lut4"),
        default="lut4",
        help="Core AI constant representation; lut4 keeps the original four-bit density",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def from_blocked(
    blocked: torch.Tensor, num_rows: int, num_cols: int
) -> torch.Tensor:
    """Undo the cuBLAS 128x4 scale swizzle used by Comfy's NVFP4 files."""

    row_blocks = ceil_div(num_rows, 128)
    column_blocks = ceil_div(num_cols, 4)
    padded_rows = row_blocks * 128
    padded_columns = column_blocks * 4
    value = blocked.reshape(-1, 32, 16)
    value = value.reshape(-1, 32, 4, 4).transpose(1, 2)
    value = value.reshape(row_blocks, column_blocks, 4, 32, 4)
    value = value.reshape(row_blocks, column_blocks, 128, 4)
    value = value.permute(0, 2, 1, 3).reshape(padded_rows, padded_columns)
    return value[:num_rows, :num_cols].contiguous()


def remove_existing(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite")
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def load_exact_int8_mapping(
    checkpoint: Path, prefix: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    weight_key = f"{prefix}.weight"
    scale_key = f"{prefix}.weight_scale"
    tensor_scale_key = f"{prefix}.weight_scale_2"
    pre_quant_key = f"{prefix}.pre_quant_scale"
    bias_key = f"{prefix}.bias"
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = {weight_key, scale_key, tensor_scale_key} - keys
        if missing:
            raise KeyError(f"missing NVFP4 tensors: {sorted(missing)}")
        packed = handle.get_tensor(weight_key)
        blocked_scale = handle.get_tensor(scale_key)
        tensor_scale = handle.get_tensor(tensor_scale_key).float()
        pre_quant = (
            handle.get_tensor(pre_quant_key).float() if pre_quant_key in keys else None
        )
        bias = handle.get_tensor(bias_key).float() if bias_key in keys else None

    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise TypeError(f"{weight_key} must be a packed rank-2 uint8 tensor")
    output_features, packed_input_features = packed.shape
    input_features = packed_input_features * 2
    if input_features % 16:
        raise ValueError("logical input features must be divisible by 16")
    scale = from_blocked(
        blocked_scale,
        num_rows=output_features,
        num_cols=input_features // 16,
    ).float()
    total_scale = scale * tensor_scale

    high = packed >> 4
    low = packed & 0x0F
    codes = torch.stack((high, low), dim=-1).reshape(output_features, input_features)
    exact_int8 = E2M1_TIMES_TWO[codes.long()].contiguous()
    # q = 2 * E2M1 and s = 0.5 * original_scale preserves q*s exactly,
    # apart from the FP16 scale boundary used by the deployed Core AI model.
    coreai_scale = (total_scale * 0.5).to(torch.float16).contiguous()
    return exact_int8, coreai_scale, pre_quant, bias


def load_exact_palettized_mapping(
    checkpoint: Path, prefix: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Return unpacked four-bit LUT indices and the exact block scale."""

    weight_key = f"{prefix}.weight"
    scale_key = f"{prefix}.weight_scale"
    tensor_scale_key = f"{prefix}.weight_scale_2"
    pre_quant_key = f"{prefix}.pre_quant_scale"
    bias_key = f"{prefix}.bias"
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        missing = {weight_key, scale_key, tensor_scale_key} - keys
        if missing:
            raise KeyError(f"missing NVFP4 tensors: {sorted(missing)}")
        packed = handle.get_tensor(weight_key)
        blocked_scale = handle.get_tensor(scale_key)
        tensor_scale = handle.get_tensor(tensor_scale_key).float()
        pre_quant = (
            handle.get_tensor(pre_quant_key).float() if pre_quant_key in keys else None
        )
        bias = handle.get_tensor(bias_key).float() if bias_key in keys else None

    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise TypeError(f"{weight_key} must be a packed rank-2 uint8 tensor")
    output_features, packed_input_features = packed.shape
    input_features = packed_input_features * 2
    if input_features % 16:
        raise ValueError("logical input features must be divisible by 16")
    scale = from_blocked(
        blocked_scale,
        num_rows=output_features,
        num_cols=input_features // 16,
    ).float()
    total_scale = scale * tensor_scale
    high = packed >> 4
    low = packed & 0x0F
    indices = torch.stack((high, low), dim=-1).reshape(
        output_features, input_features
    ).contiguous()
    return (
        indices,
        (total_scale * 0.5).to(torch.float16).contiguous(),
        pre_quant,
        bias,
    )


class ExactNVFP4Linear(torch.nn.Module):
    def __init__(
        self,
        quantized_weight: torch.Tensor,
        block_scale: torch.Tensor,
        pre_quant_scale: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        from coreai_torch._compression.custom_layers import WeightDequantizeModule

        self.weight = WeightDequantizeModule(
            quantized_data=quantized_weight,
            scale=block_scale,
            output_dtype=torch.float16,
        )
        self.register_buffer(
            "pre_quant_scale",
            None if pre_quant_scale is None else pre_quant_scale.to(torch.float16),
        )
        self.register_buffer("bias", None if bias is None else bias.to(torch.float16))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pre_quant_scale is not None:
            hidden_states = hidden_states * self.pre_quant_scale
        return torch.nn.functional.linear(hidden_states, self.weight(), self.bias)


class ExactNVFP4PalettizedLinear(torch.nn.Module):
    def __init__(
        self,
        indices: torch.Tensor,
        block_scale: torch.Tensor,
        pre_quant_scale: torch.Tensor | None,
        bias: torch.Tensor | None,
    ) -> None:
        super().__init__()
        from coreai_torch._compression.custom_layers import ScaledPalettizeModule

        lut = E2M1_TIMES_TWO.to(torch.float16).reshape(1, 1, 16, 1)
        self.weight = ScaledPalettizeModule(
            indices=indices,
            lut=lut,
            scale=block_scale,
            output_dtype=torch.float16,
        )
        self.register_buffer(
            "pre_quant_scale",
            None if pre_quant_scale is None else pre_quant_scale.to(torch.float16),
        )
        self.register_buffer("bias", None if bias is None else bias.to(torch.float16))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.pre_quant_scale is not None:
            hidden_states = hidden_states * self.pre_quant_scale
        return torch.nn.functional.linear(hidden_states, self.weight(), self.bias)


def export_coreai(
    model: torch.nn.Module, example: torch.Tensor, destination: Path
) -> None:
    import coreai_torch

    exported = torch.export.export(model, (example,))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
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
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    remove_existing(args.output, args.overwrite)
    if args.reference_directory.exists() and args.overwrite:
        shutil.rmtree(args.reference_directory)
    args.reference_directory.mkdir(parents=True, exist_ok=True)

    if args.representation == "lut4":
        weight, scale, pre_quant, bias = load_exact_palettized_mapping(
            args.checkpoint, args.tensor_prefix
        )
        model = ExactNVFP4PalettizedLinear(weight, scale, pre_quant, bias).eval()
    else:
        weight, scale, pre_quant, bias = load_exact_int8_mapping(
            args.checkpoint, args.tensor_prefix
        )
        model = ExactNVFP4Linear(weight, scale, pre_quant, bias).eval()
    input_features = weight.shape[1]
    example = torch.sin(
        torch.arange(input_features, dtype=torch.float32) * 0.03125
    ).reshape(1, input_features).to(torch.float16)
    with torch.no_grad():
        reference = model(example).float()

    np.asarray(example.float().numpy(), dtype=np.float32).tofile(
        args.reference_directory / "input.f32"
    )
    np.asarray(reference.numpy(), dtype=np.float32).tofile(
        args.reference_directory / "reference.f32"
    )
    metadata = {
        "tensorPrefix": args.tensor_prefix,
        "representation": args.representation,
        "inputShape": list(example.shape),
        "outputShape": list(reference.shape),
        "referenceMean": float(reference.mean()),
        "referenceRMS": float(reference.square().mean().sqrt()),
        "weightBytes": weight.numel() * weight.element_size(),
        "scaleBytes": scale.numel() * scale.element_size(),
        "hasPreQuantScale": pre_quant is not None,
        "hasBias": bias is not None,
    }
    (args.reference_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    export_coreai(model, example, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
