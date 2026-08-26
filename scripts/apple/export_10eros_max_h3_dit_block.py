#!/usr/bin/env python3
"""Export one or more fused 10Eros-Max H3 DiT blocks as a Core AI asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from pilot_10eros_max_h3_int8_convrot import (
    ExactINT8ConvRotLinear,
    load_mapping as load_convrot_mapping,
)


HIDDEN = 5376
HEADS = 56
HEAD_DIM = 128
INNER = HEADS * HEAD_DIM
FFN = 14336
T_ROWS = 4
T_DIM = 8
MODALITIES = 3
MOD_ROWS = T_ROWS * MODALITIES
RMS_EPSILON = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--layer-count", type=int, default=1, choices=range(1, 5))
    parser.add_argument("--scalar-type", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--dynamic-max-tokens", type=int, default=131_072)
    parser.add_argument(
        "--fixed-shape",
        action="store_true",
        help="Export the exact --tokens shape instead of a dynamic token axis.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path)
    parser.add_argument(
        "--graph-identity",
        help=(
            "Weight-bound identity embedded into the Core AI entrypoint and "
            "the executed hidden-state graph. Defaults to a SHA-256 digest "
            "of the checkpoint plus the exported layer range."
        ),
    )
    parser.add_argument("--skip-reference", action="store_true")
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
    def __init__(
        self, weight: torch.Tensor, dtype: torch.dtype,
        epsilon: float = RMS_EPSILON,
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight.to(dtype).contiguous())
        self.epsilon = epsilon

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.dtype == torch.float16:
            # ANE cannot accept the Float32 region introduced by BF16-style
            # accumulation. Keep the experimental FP16 graph homogeneous.
            normalized = value * torch.rsqrt(
                value.square().mean(dim=-1, keepdim=True) + self.epsilon
            )
            return normalized * self.weight
        normalized = value.float() * torch.rsqrt(
            value.float().square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return (normalized * self.weight.float()).to(value.dtype)


class DenseLinear(torch.nn.Module):
    def __init__(
        self, weight: torch.Tensor, bias: torch.Tensor | None, dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.register_buffer("weight", weight.to(dtype).contiguous())
        self.register_buffer(
            "bias", None if bias is None else bias.to(dtype).contiguous()
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(value, self.weight, self.bias)


def load_linear(
    checkpoint: Path, prefix: str, dtype: torch.dtype
) -> torch.nn.Module:
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        quantized = f"{prefix}.comfy_quant" in keys
        if not quantized:
            weight = handle.get_tensor(f"{prefix}.weight")
            bias = (
                handle.get_tensor(f"{prefix}.bias")
                if f"{prefix}.bias" in keys
                else None
            )
    if quantized:
        weight, scale, bias, group_size = load_convrot_mapping(checkpoint, prefix)
        return ExactINT8ConvRotLinear(
            weight,
            scale.to(dtype),
            bias,
            group_size,
            output_dtype=dtype,
        )
    return DenseLinear(weight, bias, dtype)


class TenErosDiTBlock(torch.nn.Module):
    def __init__(self, checkpoint: Path, layer: int, dtype: torch.dtype) -> None:
        super().__init__()
        prefix = f"blocks.{layer}"
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            self.norm1 = RMSNorm(handle.get_tensor(f"{prefix}.norm1.weight"), dtype)
            self.norm2 = RMSNorm(handle.get_tensor(f"{prefix}.norm2.weight"), dtype)
            self.q_norm = RMSNorm(handle.get_tensor(f"{prefix}.attn.q_norm.weight"), dtype)
            self.k_norm = RMSNorm(handle.get_tensor(f"{prefix}.attn.k_norm.weight"), dtype)
        self.qkv = load_linear(checkpoint, f"{prefix}.attn.qkv_proj", dtype)
        self.out = load_linear(checkpoint, f"{prefix}.attn.out_proj", dtype)
        self.fc1 = load_linear(checkpoint, f"{prefix}.mlp.fc1", dtype)
        self.fc2 = load_linear(checkpoint, f"{prefix}.mlp.fc2", dtype)
        self.adaln = load_linear(checkpoint, f"{prefix}.adaln_proj.linear", dtype)

    @staticmethod
    def apply_rope(
        value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
    ) -> torch.Tensor:
        first = value[..., :48]
        second = value[..., 48:96]
        rotated = torch.cat(
            (first * cosine[:, None] - second * sine[:, None],
             first * sine[:, None] + second * cosine[:, None]),
            dim=-1,
        )
        return torch.cat((rotated, value[..., 96:]), dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_coordinates: torch.Tensor,
        modulation_weights: torch.Tensor,
        rope_cosine: torch.Tensor,
        rope_sine: torch.Tensor,
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        modulation = self.adaln(timestep_coordinates).reshape(
            T_ROWS * MODALITIES, 6, HIDDEN
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            modulation_weights @ modulation[:, index]
            for index in range(6)
        )

        normalized = self.norm1(hidden_states)
        normalized = normalized * (1 + scale_msa) + shift_msa
        query, key, value = self.qkv(normalized).split(INNER, dim=-1)
        query = self.q_norm(query.reshape(tokens, HEADS, HEAD_DIM))
        key = self.k_norm(key.reshape(tokens, HEADS, HEAD_DIM))
        value = value.reshape(tokens, HEADS, HEAD_DIM)
        query = self.apply_rope(query, rope_cosine, rope_sine)
        key = self.apply_rope(key, rope_cosine, rope_sine)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.squeeze(0).transpose(0, 1).reshape(tokens, INNER)
        hidden_states = hidden_states + self.out(attended) * gate_msa

        normalized = self.norm2(hidden_states)
        normalized = normalized * (1 + scale_mlp) + shift_mlp
        gate, up = self.fc1(normalized).split(FFN, dim=-1)
        branch = self.fc2(torch.nn.functional.silu(gate) * up)
        return hidden_states + branch * gate_mlp


class TenErosDiTBlockGroup(torch.nn.Module):
    """Fuses adjacent blocks so Core AI loads one asset for several layers."""

    def __init__(
        self, checkpoint: Path, first_layer: int, layer_count: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            TenErosDiTBlock(checkpoint, layer, dtype)
            for layer in range(first_layer, first_layer + layer_count)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_coordinates: torch.Tensor,
        modulation_weights: torch.Tensor,
        rope_cosine: torch.Tensor,
        rope_sine: torch.Tensor,
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                timestep_coordinates,
                modulation_weights,
                rope_cosine,
                rope_sine,
            )
        return hidden_states


class CoreAIExportWrapper(torch.nn.Module):
    """Makes each weight-distinct asset structurally unique on Core AI.

    macOS 27 beta 6 may reuse a loaded function when two assets have the same
    graph hash even though their external weights differ. The salt is consumed
    by the primary hidden-state result so it survives delegate partitioning.
    """

    def __init__(self, group: TenErosDiTBlockGroup, salt_width: int) -> None:
        super().__init__()
        self.group = group
        self.salt_width = salt_width

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep_coordinates: torch.Tensor,
        modulation_weights: torch.Tensor,
        rope_cosine: torch.Tensor,
        rope_sine: torch.Tensor,
        graph_identity_salt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.group(
            hidden_states,
            timestep_coordinates,
            modulation_weights,
            rope_cosine,
            rope_sine,
        )
        # This zero-valued input must participate in the primary output path.
        # A separate cache_salt output changes the outer Core AI asset hash but
        # is split away from the MPSGraph delegate, leaving all weight-distinct
        # DiT groups with the same internal graph hash. The reduction/addition
        # is numerically exact for an all-zero salt while its weight-bound
        # shape and input name make the executed delegate structurally unique.
        hidden_states = hidden_states + graph_identity_salt.sum()
        return hidden_states, hidden_states[:1, : self.salt_width]


def examples(
    tokens: int, dtype: torch.dtype, salt_width: int = 0
) -> tuple[torch.Tensor, ...]:
    hidden = torch.sin(
        torch.arange(tokens * HIDDEN, dtype=torch.float32) * 0.000244140625
    ).reshape(tokens, HIDDEN).to(dtype)
    timestep = torch.linspace(-0.2, 0.2, T_ROWS * T_DIM).reshape(
        T_ROWS, T_DIM
    ).to(dtype)
    modulation = torch.nn.functional.one_hot(
        torch.arange(tokens) % MOD_ROWS, num_classes=MOD_ROWS
    ).to(dtype)
    angle = torch.arange(tokens * 48, dtype=torch.float32).reshape(tokens, 48)
    angle = angle * 0.0009765625
    values = (
        hidden,
        timestep,
        modulation,
        angle.cos().to(dtype),
        angle.sin().to(dtype),
    )
    if salt_width > 0:
        return values + (torch.zeros(salt_width, dtype=dtype),)
    return values


def export_coreai(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    destination: Path,
    maximum_tokens: int,
    fixed_shape: bool,
    entrypoint_name: str,
    graph_salt_name: str,
) -> None:
    import coreai_torch

    dynamic_shapes = None
    if not fixed_shape:
        token_dimension = torch.export.Dim("tokens", min=1, max=maximum_tokens)
        dynamic_shapes = (
            {0: token_dimension},
            None,
            {0: token_dimension},
            {0: token_dimension},
            {0: token_dimension},
            None,
        )
    exported = torch.export.export(model, inputs, dynamic_shapes=dynamic_shapes)
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported,
        input_names=[
            "hidden_states",
            "timestep_coordinates",
            "modulation_weights",
            "rope_cosine",
            "rope_sine",
            graph_salt_name,
        ],
        output_names=["hidden_states_out", "cache_salt"],
        entrypoint_name=entrypoint_name,
    )
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(destination)


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if not 0 <= args.layer < 50:
        raise ValueError("--layer must be in 0...49")
    if args.layer + args.layer_count > 50:
        raise ValueError("--layer + --layer-count must not exceed 50")
    if args.tokens <= 0 or args.dynamic_max_tokens < args.tokens:
        raise ValueError("invalid token bounds")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    graph_identity = args.graph_identity or (
        f"w{checkpoint_sha256(args.checkpoint)[:16]}_"
        f"b{args.layer:02d}_{args.layer + args.layer_count - 1:02d}"
    )
    if not re.fullmatch(r"[A-Za-z0-9_]+", graph_identity):
        raise ValueError(
            "--graph-identity must contain only letters, numbers, and underscores"
        )
    entrypoint_name = f"main_{graph_identity}"
    graph_salt_name = f"graph_identity_salt_{graph_identity}"
    salt_width = args.layer + 1
    remove_existing(args.output, args.overwrite)
    if args.reference_directory is not None:
        if args.reference_directory.exists() and args.overwrite:
            shutil.rmtree(args.reference_directory)
        args.reference_directory.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.scalar_type == "bfloat16" else torch.float16
    group = TenErosDiTBlockGroup(
        args.checkpoint, args.layer, args.layer_count, dtype
    ).eval()
    inputs = examples(args.tokens, dtype, salt_width)
    reference = None
    if not args.skip_reference:
        with torch.no_grad():
            reference = group(*inputs[:5]).float()
    model = CoreAIExportWrapper(group, salt_width).eval()
    metadata = {
        "layer": args.layer,
        "layers": list(range(args.layer, args.layer + args.layer_count)),
        "logicalLayerCount": args.layer_count,
        "scalarType": args.scalar_type,
        "tokens": args.tokens,
        "dynamicMaximumTokens": args.dynamic_max_tokens,
        "fixedShape": args.fixed_shape,
        "graphIdentity": graph_identity,
        "entrypointName": entrypoint_name,
        "graphSaltInputName": graph_salt_name,
        "graphSaltShape": [salt_width],
        "inputShapes": [list(value.shape) for value in inputs],
        "outputShape": [args.tokens, HIDDEN],
    }
    if reference is not None:
        metadata["referenceMean"] = float(reference.mean())
        metadata["referenceRMS"] = float(reference.square().mean().sqrt())
    print(json.dumps(metadata, indent=2), flush=True)
    if args.reference_directory is not None:
        names = [
            "hidden_states.f32",
            "timestep_coordinates.f32",
            "modulation_weights.f32",
            "rope_cosine.f32",
            "rope_sine.f32",
            "graph_identity_salt.f32",
        ]
        for name, value in zip(names, inputs):
            np.asarray(value.float().numpy(), dtype=np.float32).tofile(
                args.reference_directory / name
            )
        if reference is not None:
            np.asarray(reference.numpy(), dtype=np.float32).tofile(
                args.reference_directory / "reference.f32"
            )
        (args.reference_directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    export_coreai(
        model,
        inputs,
        args.output,
        args.dynamic_max_tokens,
        args.fixed_shape,
        entrypoint_name,
        graph_salt_name,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
