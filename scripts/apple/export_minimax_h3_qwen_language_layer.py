#!/usr/bin/env python3
"""Export one truncated Qwen3-VL-32B language block for MiniMax H3.

The exported block keeps the source NVFP4 linears compressed through the
exact INT8/block-scale representation established by the single-linear pilot.
It accepts precomputed interleaved MRoPE cosine/sine tensors so the same block
asset remains independent of how Swift builds multimodal position IDs.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from pilot_minimax_h3_qwen_nvfp4 import (
    ExactNVFP4PalettizedLinear,
    load_exact_palettized_mapping,
)


HIDDEN_SIZE = 5120
NUM_HEADS = 64
NUM_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE_SIZE = 25600
ROPE_THETA = 5_000_000.0
RMS_EPSILON = 1e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-directory", type=Path)
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--inputs-only", action="store_true")
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


class RMSNorm(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weight", weight.to(torch.float16).contiguous())

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + RMS_EPSILON
        )
        return (normalized * self.weight.float()).to(value.dtype)


class QwenLanguageLayer(torch.nn.Module):
    def __init__(self, checkpoint: Path, layer: int) -> None:
        super().__init__()
        prefix = f"model.layers.{layer}"

        def linear(name: str) -> ExactNVFP4PalettizedLinear:
            weight, scale, pre_quant, bias = load_exact_palettized_mapping(
                checkpoint, f"{prefix}.{name}"
            )
            return ExactNVFP4PalettizedLinear(weight, scale, pre_quant, bias)

        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            self.input_layernorm = RMSNorm(
                handle.get_tensor(f"{prefix}.input_layernorm.weight")
            )
            self.post_attention_layernorm = RMSNorm(
                handle.get_tensor(f"{prefix}.post_attention_layernorm.weight")
            )
            self.q_norm = RMSNorm(
                handle.get_tensor(f"{prefix}.self_attn.q_norm.weight")
            )
            self.k_norm = RMSNorm(
                handle.get_tensor(f"{prefix}.self_attn.k_norm.weight")
            )

        self.q_proj = linear("self_attn.q_proj")
        self.k_proj = linear("self_attn.k_proj")
        self.v_proj = linear("self_attn.v_proj")
        self.o_proj = linear("self_attn.o_proj")
        self.gate_proj = linear("mlp.gate_proj")
        self.up_proj = linear("mlp.up_proj")
        self.down_proj = linear("mlp.down_proj")

    @staticmethod
    def apply_rope(
        value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
    ) -> torch.Tensor:
        first, second = value.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return value * cosine + rotated * sine

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cosine: torch.Tensor,
        rope_sine: torch.Tensor,
    ) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        query = self.q_proj(normalized).reshape(
            batch, sequence, NUM_HEADS, HEAD_DIM
        ).transpose(1, 2)
        key = self.k_proj(normalized).reshape(
            batch, sequence, NUM_KV_HEADS, HEAD_DIM
        ).transpose(1, 2)
        value = self.v_proj(normalized).reshape(
            batch, sequence, NUM_KV_HEADS, HEAD_DIM
        ).transpose(1, 2)
        query = self.apply_rope(self.q_norm(query), rope_cosine, rope_sine)
        key = self.apply_rope(self.k_norm(key), rope_cosine, rope_sine)
        key = torch.repeat_interleave(key, NUM_HEADS // NUM_KV_HEADS, dim=1)
        value = torch.repeat_interleave(value, NUM_HEADS // NUM_KV_HEADS, dim=1)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).reshape(batch, sequence, -1)
        hidden_states = residual + self.o_proj(attended)

        residual = hidden_states
        normalized = self.post_attention_layernorm(hidden_states)
        gated = torch.nn.functional.silu(self.gate_proj(normalized))
        hidden_states = self.down_proj(gated * self.up_proj(normalized))
        return residual + hidden_states


class CoreAIExportWrapper(torch.nn.Module):
    """Give each weight-distinct layer a unique Core AI function signature."""

    def __init__(self, layer: QwenLanguageLayer, salt_width: int) -> None:
        super().__init__()
        self.layer = layer
        self.salt_width = salt_width

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cosine: torch.Tensor,
        rope_sine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.layer(hidden_states, rope_cosine, rope_sine)
        return hidden_states, hidden_states[:, :1, : self.salt_width]


def sequential_rope(sequence_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    position = torch.arange(sequence_length, dtype=torch.float32)
    numerator = torch.arange(0, HEAD_DIM, 2, dtype=torch.float32)
    inverse_frequency = 1.0 / (ROPE_THETA ** (numerator / HEAD_DIM))
    frequency = torch.outer(position, inverse_frequency)
    embedding = torch.cat((frequency, frequency), dim=-1)
    return (
        embedding.cos().reshape(1, 1, sequence_length, HEAD_DIM).to(torch.float16),
        embedding.sin().reshape(1, 1, sequence_length, HEAD_DIM).to(torch.float16),
    )


def export_coreai(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    destination: Path,
) -> None:
    import coreai_torch

    exported = torch.export.export(model, (hidden_states, cosine, sine))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported,
        input_names=["hidden_states", "rope_cosine", "rope_sine"],
        output_names=["hidden_states_out", "cache_salt"],
    )
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(destination)


def main() -> int:
    args = parse_args()
    if not 0 <= args.layer < 50:
        raise ValueError("--layer must be in 0...49")
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.inputs_only and args.output is None:
        raise ValueError("--output is required unless --inputs-only is set")
    if args.inputs_only and args.reference_directory is None:
        raise ValueError("--inputs-only requires --reference-directory")
    if args.output is not None and not args.inputs_only:
        remove_existing(args.output, args.overwrite)
    if args.reference_directory is not None:
        if args.reference_directory.exists() and args.overwrite:
            shutil.rmtree(args.reference_directory)
        args.reference_directory.mkdir(parents=True, exist_ok=True)

    model = QwenLanguageLayer(args.checkpoint, args.layer).eval()
    elements = args.sequence_length * HIDDEN_SIZE
    hidden_states = torch.sin(
        torch.arange(elements, dtype=torch.float32) * 0.001953125
    ).reshape(1, args.sequence_length, HIDDEN_SIZE).to(torch.float16)
    cosine, sine = sequential_rope(args.sequence_length)
    metadata = {
        "layer": args.layer,
        "sequenceLength": args.sequence_length,
        "hiddenShape": list(hidden_states.shape),
        "ropeShape": list(cosine.shape),
        "outputShape": list(hidden_states.shape),
    }
    reference = None
    if not args.skip_reference:
        with torch.no_grad():
            reference = model(hidden_states, cosine, sine).float()
        metadata["referenceMean"] = float(reference.mean())
        metadata["referenceRMS"] = float(reference.square().mean().sqrt())
    print(json.dumps(metadata, indent=2), flush=True)
    if args.reference_directory is not None:
        arrays = {
            "hidden_states.f32": hidden_states.float().numpy(),
            "rope_cosine.f32": cosine.float().numpy(),
            "rope_sine.f32": sine.float().numpy(),
        }
        if reference is not None:
            arrays["reference.f32"] = reference.numpy()
        for name, value in arrays.items():
            np.asarray(value, dtype=np.float32).tofile(args.reference_directory / name)
        (args.reference_directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    if not args.inputs_only:
        assert args.output is not None
        wrapped = CoreAIExportWrapper(model, args.layer + 1).eval()
        export_coreai(wrapped, hidden_states, cosine, sine, args.output)
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
