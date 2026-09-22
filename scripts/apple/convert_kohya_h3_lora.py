#!/usr/bin/env python3
"""Rewrite a kohya-style MiniMax H3 LoRA into the layout this repo's exporter reads.

Community H3 LoRAs ship kohya naming:

    lora_unet_blocks_0_attn_qkv_proj.lora_down.weight
    lora_unet_blocks_0_attn_qkv_proj.lora_up.weight
    lora_unet_blocks_0_attn_qkv_proj.alpha

`export_10eros_max_h3_dit_block.py` looks the weights up as:

    diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight
    diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight
    diffusion_model.blocks.0.attn.qkv_proj.alpha

Only the names differ; the tensors are copied through untouched so the merge
stays bit-identical to what the LoRA author trained.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

# kohya flattens the module path with underscores, so the block index and the
# submodule have to be recovered positionally rather than by splitting on ".".
MODULE_PATTERN = re.compile(r"^lora_unet_blocks_(\d+)_(.+)$")

# Submodule tails the exporter knows about, longest first so that
# "adaln_proj_linear" is not shortened to "adaln_proj".
SUBMODULES = (
    ("adaln_proj_linear", "adaln.proj_linear"),
    ("attn_qkv_proj", "attn.qkv_proj"),
    ("attn_out_proj", "attn.out_proj"),
    ("mlp_fc1", "mlp.fc1"),
    ("mlp_fc2", "mlp.fc2"),
)

SUFFIXES = {
    ".lora_down.weight": ".lora_A.weight",
    ".lora_up.weight": ".lora_B.weight",
    ".alpha": ".alpha",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def convert_key(key: str) -> str | None:
    for kohya, internal in SUFFIXES.items():
        if key.endswith(kohya):
            stem, suffix = key[: -len(kohya)], internal
            break
    else:
        return None
    match = MODULE_PATTERN.match(stem)
    if match is None:
        return None
    index, tail = match.groups()
    for kohya_tail, internal_tail in SUBMODULES:
        if tail == kohya_tail:
            return f"diffusion_model.blocks.{index}.{internal_tail}{suffix}"
    return None


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    tensors: dict = {}
    skipped: list[str] = []
    with safe_open(str(args.input), framework="pt") as handle:
        metadata = handle.metadata() or {}
        for key in handle.keys():
            converted = convert_key(key)
            if converted is None:
                skipped.append(key)
                continue
            tensors[converted] = handle.get_tensor(key)
    if not tensors:
        raise ValueError(f"{args.input} contained no convertible LoRA keys")
    if skipped:
        raise ValueError(
            f"{len(skipped)} keys did not match the known module layout, "
            f"for example {skipped[:3]}"
        )
    kept = {k: v for k, v in metadata.items() if k.startswith("ss_")}
    kept["converted_from"] = args.input.name
    kept["key_layout"] = "diffusion_model.blocks.N.<module>.lora_A/lora_B"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata=kept)
    blocks = sorted({int(k.split(".")[2]) for k in tensors})
    print(f"wrote {args.output}")
    print(f"  tensors: {len(tensors)}")
    print(f"  blocks : {len(blocks)} ({blocks[0]}..{blocks[-1]})")
    modules = sorted({".".join(k.split(".")[3:-2]) for k in tensors})
    print(f"  modules: {modules}")


if __name__ == "__main__":
    main()
