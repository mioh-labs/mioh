#!/usr/bin/env python3
"""Validate an exact 10Eros-Max H3 INT8 ConvRot -> Core AI mapping."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--tensor-prefix", default="blocks.2.attn.out_proj"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path, required=True)
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


def hadamard(size: int, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    if size < 4 or size & (size - 1) or math.log(size, 4) % 1:
        raise ValueError(f"ConvRot size must be a power of four, got {size}")
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]],
        dtype=torch.float32,
    )
    value = h4
    while value.shape[0] < size:
        value = torch.kron(value, h4)
    return (value / math.sqrt(size)).to(dtype).contiguous()


def load_mapping(
    checkpoint: Path, prefix: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        quant_key = f"{prefix}.comfy_quant"
        if quant_key not in keys:
            raise KeyError(f"{prefix} is not an INT8 ConvRot layer")
        config = json.loads(bytes(handle.get_tensor(quant_key).tolist()))
        if config.get("format") != "int8_tensorwise" or not config.get("convrot"):
            raise ValueError(f"unexpected quantization config: {config}")
        weight = handle.get_tensor(f"{prefix}.weight")
        scale = handle.get_tensor(f"{prefix}.weight_scale")
        bias = (
            handle.get_tensor(f"{prefix}.bias")
            if f"{prefix}.bias" in keys
            else None
        )
    if weight.dtype != torch.int8 or weight.ndim != 2:
        raise TypeError("ConvRot weight must be rank-2 INT8")
    return (
        weight.contiguous(),
        scale.to(torch.float16).contiguous(),
        None if bias is None else bias.to(torch.float16).contiguous(),
        int(config.get("convrot_groupsize", 256)),
    )


class ExactINT8ConvRotLinear(torch.nn.Module):
    def __init__(
        self,
        quantized_weight: torch.Tensor,
        scale: torch.Tensor,
        bias: torch.Tensor | None,
        group_size: int,
        output_dtype: torch.dtype = torch.float16,
    ) -> None:
        super().__init__()
        from coreai_torch._compression.custom_layers import WeightDequantizeModule

        self.weight = WeightDequantizeModule(
            quantized_data=quantized_weight,
            scale=scale,
            output_dtype=output_dtype,
        )
        self.register_buffer("rotation", hadamard(group_size, output_dtype))
        self.register_buffer(
            "bias", None if bias is None else bias.to(output_dtype)
        )
        self.group_size = group_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = hidden_states.shape
        groups = shape[-1] // self.group_size
        rotated = torch.matmul(
            hidden_states.reshape(-1, groups, self.group_size), self.rotation
        ).reshape(shape)
        return torch.nn.functional.linear(rotated, self.weight(), self.bias)


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

    weight, scale, bias, group_size = load_mapping(
        args.checkpoint, args.tensor_prefix
    )
    model = ExactINT8ConvRotLinear(weight, scale, bias, group_size).eval()
    input_features = weight.shape[1]
    example = torch.sin(
        torch.arange(input_features, dtype=torch.float32) * 0.00390625
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
        "groupSize": group_size,
        "inputShape": list(example.shape),
        "outputShape": list(reference.shape),
        "referenceMean": float(reference.mean()),
        "referenceRMS": float(reference.square().mean().sqrt()),
        "quantizedWeightBytes": weight.numel() * weight.element_size(),
        "scaleBytes": scale.numel() * scale.element_size(),
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
