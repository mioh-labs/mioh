#!/usr/bin/env python3
"""Combine MiniMax H3 LoRAs while preserving each adapter's effective scale."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--component",
        action="append",
        nargs=2,
        metavar=("LORA", "STRENGTH"),
        required=True,
        help="LoRA path and user strength; repeat for every adapter",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_component(path: Path, user_strength: float):
    tensors: dict[str, torch.Tensor] = {}
    scales: dict[str, float] = {}
    metadata: dict[str, str] = {}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata.update(handle.metadata() or {})
        keys = set(handle.keys())
        for key in keys:
            if key.endswith(".alpha"):
                continue
            tensors[key] = handle.get_tensor(key)
        for key, tensor in tensors.items():
            if not key.endswith(".lora_A.weight"):
                continue
            stem = key.removesuffix(".lora_A.weight")
            alpha_key = f"{stem}.alpha"
            alpha = (
                float(handle.get_tensor(alpha_key).item())
                if alpha_key in keys
                else float(tensor.shape[0])
            )
            scales[stem] = user_strength * alpha / tensor.shape[0]
    return tensors, scales, metadata


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")

    components = []
    for raw_path, raw_strength in args.component:
        path = Path(raw_path).expanduser().resolve()
        components.append((path, float(raw_strength), *load_component(path, float(raw_strength))))

    stems: set[str] = set()
    for _, _, tensors, _, _ in components:
        stems.update(
            key.removesuffix(".lora_A.weight")
            for key in tensors
            if key.endswith(".lora_A.weight")
        )

    output: dict[str, torch.Tensor] = {}
    for stem in sorted(stems):
        downs: list[torch.Tensor] = []
        ups: list[torch.Tensor] = []
        for path, _, tensors, scales, _ in components:
            down_key = f"{stem}.lora_A.weight"
            up_key = f"{stem}.lora_B.weight"
            if down_key not in tensors and up_key not in tensors:
                continue
            if down_key not in tensors or up_key not in tensors:
                raise ValueError(f"incomplete LoRA pair for {stem} in {path}")
            downs.append(tensors[down_key].to(torch.bfloat16))
            ups.append((tensors[up_key].float() * scales[stem]).to(torch.bfloat16))
        if not downs:
            continue
        input_dims = {tensor.shape[1] for tensor in downs}
        output_dims = {tensor.shape[0] for tensor in ups}
        if len(input_dims) != 1 or len(output_dims) != 1:
            raise ValueError(f"incompatible LoRA shapes for {stem}")
        output[f"{stem}.lora_A.weight"] = torch.cat(downs, dim=0).contiguous()
        output[f"{stem}.lora_B.weight"] = torch.cat(ups, dim=1).contiguous()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format": "pt",
        "base_model": "MiniMax-H3",
        "combination": ";".join(
            f"{path.name}@{strength:g}" for path, strength, *_ in components
        ),
        "scale_baked_into_lora_B": "true",
    }
    save_file(output, str(args.output), metadata=metadata)
    print(f"saved {len(output) // 2} LoRA pairs to {args.output}")


if __name__ == "__main__":
    main()
