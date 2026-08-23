#!/usr/bin/env python3
"""Export the quantized Qwen3-VL-32B token embedding for MiniMax H3."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4152)
    parser.add_argument("--output", type=Path, required=True)
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


class QuantizedEmbedding(torch.nn.Module):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        from coreai_torch._compression.custom_layers import WeightDequantizeModule

        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            weight = handle.get_tensor("model.embed_tokens.weight").contiguous()
            scale = handle.get_tensor("model.embed_tokens.weight_scale").to(
                torch.float16
            ).contiguous()
        if weight.dtype != torch.int8 or weight.shape != (151936, 5120):
            raise TypeError(f"unexpected Qwen embedding tensor {weight.shape} {weight.dtype}")
        self.weight = WeightDequantizeModule(
            quantized_data=weight,
            scale=scale,
            output_dtype=torch.float16,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(input_ids, self.weight())


def export_coreai(
    model: torch.nn.Module, example: torch.Tensor, destination: Path
) -> None:
    import coreai_torch

    exported = torch.export.export(model, (example,))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported,
        input_names=["input_ids"],
        output_names=["token_embeddings"],
    )
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(destination)


def main() -> int:
    args = parse_args()
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    remove_existing(args.output, args.overwrite)
    model = QuantizedEmbedding(args.checkpoint).eval()
    example = torch.zeros((1, args.sequence_length), dtype=torch.int32)
    export_coreai(model, example, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
