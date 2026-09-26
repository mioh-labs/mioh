#!/usr/bin/env python3
"""Export SwiftVR's non-recurrent DiT components and fixed conditioning.

This is an offline model-conversion tool. It does not make a complete native
runtime; see docs/swiftvr-native-roi-integration.md for the release gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


class PatchEmbedding(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.conv = nn.Conv3d(48, 3072, (1, 2, 2), stride=(1, 2, 2))
        self.conv.weight = nn.Parameter(weight)
        self.conv.bias = nn.Parameter(bias)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.conv(latents).flatten(2).transpose(1, 2).contiguous()


class OutputProjection(nn.Module):
    def __init__(
        self, weight: torch.Tensor, bias: torch.Tensor,
        shift: torch.Tensor, scale: torch.Tensor,
        temporal: int, side: int,
    ):
        super().__init__()
        self.projection = nn.Linear(3072, 192)
        self.projection.weight = nn.Parameter(weight)
        self.projection.bias = nn.Parameter(bias)
        self.register_buffer("shift", shift.reshape(1, 1, 3072))
        self.register_buffer("scale", scale.reshape(1, 1, 3072))
        self.temporal = temporal
        self.side = side

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(hidden.float(), (3072,), eps=1e-6)
        projected = self.projection(normalized * (1 + self.scale) + self.shift)
        value = projected.reshape(
            1, self.temporal, self.side, self.side, 1, 2, 2, 48
        ).permute(0, 7, 1, 4, 2, 5, 3, 6)
        return value.flatten(6, 7).flatten(4, 5).flatten(2, 3)


def convert_and_check(model, example, name: str, output: Path, precision) -> dict:
    model.eval()
    with torch.no_grad():
        expected = model(example).numpy()
        traced = torch.jit.trace(model, example)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name=name, shape=example.shape)],
        outputs=[ct.TensorType(name="output")],
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
        compute_precision=precision,
        # Validate where mioh runs it. On the ANE the 2x FP16 patch embedding
        # computed wrong results (mean error 0.31) that went unnoticed.
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    converted.save(str(output))
    native = converted.predict({name: example.numpy()})["output"]
    error = np.abs(native - expected)
    return {
        "output_shape": list(native.shape),
        "max_abs_error": float(error.max()),
        "mean_abs_error": float(error.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt-embedding", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--scale", type=int, choices=(2, 4), required=True)
    parser.add_argument("--latent-frames", type=int, choices=(6, 7), required=True)
    parser.add_argument("--precision", choices=("float16", "float32"), default="float16")
    args = parser.parse_args()

    sys.path.insert(0, str(args.source))
    from swiftvr.models.transformer import (
        WanRotaryPosEmbed, WanTimeTextImageEmbedding,
    )

    root = args.output_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    side = 16 if args.scale == 2 else 32
    latent_side = side * 2
    precision = ct.precision.FLOAT16 if args.precision == "float16" else ct.precision.FLOAT32
    with safe_open(str(args.checkpoint), framework="pt", device="cpu") as weights:
        embedder = WanTimeTextImageEmbedding(
            dim=3072, time_freq_dim=256, time_proj_dim=3072 * 6,
            text_embed_dim=4096,
        ).eval()
        prefix = "condition_embedder."
        embedder.load_state_dict({
            key.removeprefix(prefix): weights.get_tensor(key)
            for key in weights.keys() if key.startswith(prefix)
        }, strict=True)
        with safe_open(str(args.prompt_embedding), framework="pt", device="cpu") as prompt:
            prompt_emb = prompt.get_tensor("prompt_emb").float()
        with torch.no_grad():
            temb, tp, context, image = embedder(
                torch.tensor([1000.0]), prompt_emb, None
            )
            if image is not None:
                raise RuntimeError("Unexpected image conditioning in SwiftVR")
            modulation = tp.unflatten(1, (6, 3072))
            mods = weights.get_tensor("scale_shift_table") + temb.unsqueeze(1)
            shift, scale = mods.chunk(2, dim=1)
        del embedder

        patch = PatchEmbedding(
            weights.get_tensor("patch_embedding.weight"),
            weights.get_tensor("patch_embedding.bias"),
        )
        head = OutputProjection(
            weights.get_tensor("proj_out.weight"),
            weights.get_tensor("proj_out.bias"),
            shift, scale, args.latent_frames, side,
        )

    context.detach().numpy().astype("<f4").tofile(root / "context.f32")
    modulation.detach().numpy().astype("<f4").tofile(root / "modulation.f32")
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024)
    rope.freqs_cos.detach().numpy().astype("<f4").tofile(root / "rope-cosine.f32")
    rope.freqs_sin.detach().numpy().astype("<f4").tofile(root / "rope-sine.f32")

    torch.manual_seed(1)
    latents = torch.randn(1, 48, args.latent_frames, latent_side, latent_side)
    hidden = torch.randn(1, args.latent_frames * side * side, 3072)
    patch_report = convert_and_check(
        patch, latents, "latents", root / "patch.mlpackage", precision
    )
    head_report = convert_and_check(
        head, hidden, "hidden", root / "head.mlpackage", precision
    )
    report = {
        "format": "swiftvr-components-coreml-probe-v1",
        "scale": args.scale,
        "latent_frames": args.latent_frames,
        "precision": args.precision,
        "context_shape": list(context.shape),
        "modulation_shape": list(modulation.shape),
        "rope_shape": list(rope.freqs_cos.shape),
        "patch": patch_report,
        "head": head_report,
    }
    (root / "components.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    # Healthy FP16 components stay near 1e-4; a large error used to be
    # recorded and silently shipped.
    for name, part in (("patch", patch_report), ("head", head_report)):
        if part["mean_abs_error"] > 0.01:
            raise SystemExit(
                f"{name} differs from PyTorch: mean {part['mean_abs_error']:.4g}, "
                f"max {part['max_abs_error']:.4g}"
            )


if __name__ == "__main__":
    main()
