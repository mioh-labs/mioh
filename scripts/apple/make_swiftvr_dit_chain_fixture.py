#!/usr/bin/env python3
"""Generate a realistic-condition PyTorch fixture for the native DiT chain.

This reads the external checkpoint but does not invoke a Python worker from the
app. The fixture lets the native 30-layer result be compared before release.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--scale", type=int, choices=(2, 4), required=True)
    parser.add_argument("--latent-frames", type=int, choices=(6, 7), required=True)
    parser.add_argument("--layers", type=int, choices=range(1, 31), default=30)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source))
    from swiftvr.models.transformer import (
        WanShiftWindow2DInferProcessor,
        WanTransformerBlock,
        set_attention_backend,
    )

    torch.set_num_threads(4)
    torch.manual_seed(301)
    set_attention_backend("sdpa")
    side = 16 if args.scale == 2 else 32
    latent_side = side * 2
    temporal = args.latent_frames
    token_count = temporal * side * side
    shape = [1, token_count, 3072]
    root = args.output_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)

    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as source:
        latent = torch.randn(1, 48, temporal, latent_side, latent_side) * 0.25
        with torch.no_grad():
            hidden = F.conv3d(
                latent,
                source.get_tensor("patch_embedding.weight"),
                source.get_tensor("patch_embedding.bias"),
                stride=(1, 2, 2),
            ).flatten(2).transpose(1, 2).contiguous()
        hidden.numpy().astype("<f4").tofile(root / "hidden.f32")

        context = np.fromfile(args.components / "context.f32", dtype="<f4")
        modulation = np.fromfile(args.components / "modulation.f32", dtype="<f4")
        context = context.reshape(1, 512, 3072)
        modulation = modulation.reshape(1, 6, 3072)
        context.tofile(root / "context.f32")
        modulation.tofile(root / "modulation.f32")
        rope_cos = np.fromfile(
            args.components / "rope-cosine.f32", dtype="<f4"
        ).reshape(1024, 128)
        rope_sin = np.fromfile(
            args.components / "rope-sine.f32", dtype="<f4"
        ).reshape(1024, 128)
        # WanRotaryPosEmbed(128) allocates 44 temporal + 42 height + 42 width.
        dimensions = (44, 42, 42)
        sections = np.cumsum((0,) + dimensions)

        def grid(table: np.ndarray) -> np.ndarray:
            t = np.broadcast_to(
                table[:temporal, sections[0]:sections[1]][:, None, None, :],
                (temporal, side, side, dimensions[0]),
            )
            h = np.broadcast_to(
                table[:side, sections[1]:sections[2]][None, :, None, :],
                (temporal, side, side, dimensions[1]),
            )
            w = np.broadcast_to(
                table[:side, sections[2]:sections[3]][None, None, :, :],
                (temporal, side, side, dimensions[2]),
            )
            return np.concatenate((t, h, w), axis=-1).reshape(
                1, token_count, 1, 128
            ).copy()

        cosine = grid(rope_cos)
        sine = grid(rope_sin)
        cosine.tofile(root / "cosine.f32")
        sine.tofile(root / "sine.f32")
        (root / "shapes.json").write_text(json.dumps({
            "hidden": shape,
            "context": [1, 512, 3072],
            "modulation": [1, 6, 3072],
            "cosine": [1, token_count, 1, 128],
            "sine": [1, token_count, 1, 128],
        }, indent=2) + "\n")

        context_t = torch.from_numpy(context)
        modulation_t = torch.from_numpy(modulation)
        rope = torch.from_numpy(cosine), torch.from_numpy(sine)
        for index in range(args.layers):
            with torch.device("meta"):
                block = WanTransformerBlock(
                    dim=3072, ffn_dim=14336, num_heads=24,
                    cross_attn_norm=True,
                )
            prefix = f"blocks.{index}."
            block.load_state_dict({
                key.removeprefix(prefix): source.get_tensor(key)
                for key in source.keys() if key.startswith(prefix)
            }, strict=True, assign=True)
            block.eval()
            block.attn1.set_processor(WanShiftWindow2DInferProcessor((16, 16)))
            block.attn1._thw = (temporal, side, side)
            block.attn1._do_shift = index % 2 == 1
            with torch.no_grad():
                hidden = block(hidden, context_t, modulation_t, rope)
            if not torch.isfinite(hidden).all():
                raise RuntimeError(f"Official block {index} produced nonfinite output")
            print(f"official block {index}: mean-abs={hidden.abs().mean().item():.6g}", flush=True)
            del block
        hidden.numpy().astype("<f4").tofile(root / "expected-chain.f32")
    print(f"Wrote fixture to {root}")


if __name__ == "__main__":
    main()
