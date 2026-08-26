#!/usr/bin/env python3
"""Export dynamic-token support stages around 10Eros-Max H3's 50 DiT blocks."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from export_10eros_max_h3_dit_block import (
    DenseLinear,
    HEAD_DIM,
    HEADS,
    HIDDEN,
    INNER,
    FFN,
    RMSNorm,
    load_linear,
)


TEXT_DIM = 5120
VIDEO_PATCH_DIM = 96
AUDIO_PATCH_DIM = 32
T_DIM = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "text-refiner",
            "video-projection",
            "audio-projection",
            "final-video",
            "final-audio",
        ),
        required=True,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--dynamic-max-tokens", type=int, default=131_072)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-directory", type=Path)
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


class RefinerBlock(torch.nn.Module):
    def __init__(self, checkpoint: Path, layer: int) -> None:
        super().__init__()
        prefix = f"token_refiner.blocks.{layer}"
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            self.norm1 = RMSNorm(
                handle.get_tensor(f"{prefix}.norm1.weight"), torch.bfloat16
            )
            self.norm2 = RMSNorm(
                handle.get_tensor(f"{prefix}.norm2.weight"), torch.bfloat16
            )
            self.q_norm = RMSNorm(
                handle.get_tensor(f"{prefix}.attn.q_norm.weight"), torch.bfloat16
            )
            self.k_norm = RMSNorm(
                handle.get_tensor(f"{prefix}.attn.k_norm.weight"), torch.bfloat16
            )
        self.qkv = load_linear(
            checkpoint, f"{prefix}.attn.qkv_proj", torch.bfloat16
        )
        self.out = load_linear(
            checkpoint, f"{prefix}.attn.out_proj", torch.bfloat16
        )
        self.fc1 = load_linear(
            checkpoint, f"{prefix}.mlp.fc1", torch.bfloat16
        )
        self.fc2 = load_linear(
            checkpoint, f"{prefix}.mlp.fc2", torch.bfloat16
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        tokens = hidden.shape[0]
        query, key, value = self.qkv(self.norm1(hidden)).split(INNER, dim=-1)
        query = self.q_norm(query.reshape(tokens, HEADS, HEAD_DIM))
        key = self.k_norm(key.reshape(tokens, HEADS, HEAD_DIM))
        value = value.reshape(tokens, HEADS, HEAD_DIM)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query.transpose(0, 1).unsqueeze(0),
            key.transpose(0, 1).unsqueeze(0),
            value.transpose(0, 1).unsqueeze(0),
            dropout_p=0,
            is_causal=False,
        )
        attended = attended.squeeze(0).transpose(0, 1).reshape(tokens, INNER)
        hidden = hidden + self.out(attended)
        gate, up = self.fc1(self.norm2(hidden)).split(FFN, dim=-1)
        return hidden + self.fc2(torch.nn.functional.silu(gate) * up)


class TextRefiner(torch.nn.Module):
    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        self.condition = load_linear(
            checkpoint, "condition_proj", torch.bfloat16
        )
        self.blocks = torch.nn.ModuleList(
            [RefinerBlock(checkpoint, 0), RefinerBlock(checkpoint, 1)]
        )
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            self.final_norm = RMSNorm(
                handle.get_tensor("token_refiner.final_norm.weight"),
                torch.bfloat16,
            )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        hidden = self.condition(context)
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_norm(hidden)


class Projection(torch.nn.Module):
    def __init__(self, checkpoint: Path, prefix: str) -> None:
        super().__init__()
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            self.register_buffer(
                "weight", handle.get_tensor(f"{prefix}.weight").float().contiguous()
            )
            self.register_buffer(
                "bias", handle.get_tensor(f"{prefix}.bias").float().contiguous()
            )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(rows, self.weight, self.bias).to(
            torch.bfloat16
        )


class FinalHead(torch.nn.Module):
    def __init__(self, checkpoint: Path, kind: str) -> None:
        super().__init__()
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            self.norm = RMSNorm(
                handle.get_tensor("final_layer.norm.weight"), torch.bfloat16
            )
        self.adaln = load_linear(
            checkpoint, "final_layer.adaln_proj.linear", torch.bfloat16
        )
        self.output = load_linear(
            checkpoint, f"final_layer.{kind}_out", torch.bfloat16
        )

    def forward(
        self, hidden: torch.Tensor, timestep_coordinate: torch.Tensor
    ) -> torch.Tensor:
        shift, scale = self.adaln(timestep_coordinate).split(HIDDEN, dim=-1)
        normalized = self.norm(hidden)
        normalized = normalized * (1 + scale) + shift
        return self.output(normalized)


def model_and_inputs(
    stage: str, checkpoint: Path, tokens: int
) -> tuple[torch.nn.Module, tuple[torch.Tensor, ...], list[str], list[str]]:
    if stage == "text-refiner":
        model = TextRefiner(checkpoint).eval()
        input_value = torch.sin(
            torch.arange(tokens * TEXT_DIM, dtype=torch.float32) * 0.000244140625
        ).reshape(tokens, TEXT_DIM).to(torch.bfloat16)
        return model, (input_value,), ["context"], ["text_hidden"]
    if stage == "video-projection":
        model = Projection(checkpoint, "video_patch_proj").eval()
        rows = torch.sin(
            torch.arange(tokens * VIDEO_PATCH_DIM, dtype=torch.float32) * 0.00390625
        ).reshape(tokens, VIDEO_PATCH_DIM).to(torch.float32)
        return model, (rows,), ["video_rows"], ["video_hidden"]
    if stage == "audio-projection":
        model = Projection(checkpoint, "audio_patch_proj").eval()
        rows = torch.sin(
            torch.arange(tokens * AUDIO_PATCH_DIM, dtype=torch.float32) * 0.0078125
        ).reshape(tokens, AUDIO_PATCH_DIM).to(torch.float32)
        return model, (rows,), ["audio_rows"], ["audio_hidden"]
    kind = "video" if stage == "final-video" else "audio"
    model = FinalHead(checkpoint, kind).eval()
    hidden = torch.sin(
        torch.arange(tokens * HIDDEN, dtype=torch.float32) * 0.000244140625
    ).reshape(tokens, HIDDEN).to(torch.bfloat16)
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        timestep = handle.get_tensor("adaln_t_table")[512:513].to(torch.bfloat16)
    return (
        model,
        (hidden, timestep),
        ["hidden_states", "timestep_coordinate"],
        [f"{kind}_rows"],
    )


def export_coreai(
    model: torch.nn.Module,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
    output_names: list[str],
    destination: Path,
    maximum_tokens: int,
) -> None:
    import coreai_torch

    token_dimension = torch.export.Dim("tokens", min=1, max=maximum_tokens)
    dynamic_shapes = ({0: token_dimension},) + (None,) * (len(inputs) - 1)
    exported = torch.export.export(
        model, inputs, dynamic_shapes=dynamic_shapes
    )
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported,
        input_names=input_names,
        output_names=output_names,
    )
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(destination)


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.tokens <= 0 or args.dynamic_max_tokens < args.tokens:
        raise ValueError("invalid token bounds")
    remove_existing(args.output, args.overwrite)
    if args.reference_directory is not None:
        if args.reference_directory.exists() and args.overwrite:
            shutil.rmtree(args.reference_directory)
        args.reference_directory.mkdir(parents=True, exist_ok=True)
    model, inputs, input_names, output_names = model_and_inputs(
        args.stage, args.checkpoint, args.tokens
    )
    reference = None
    if not args.skip_reference:
        with torch.no_grad():
            reference = model(*inputs).float()
    metadata = {
        "stage": args.stage,
        "inputNames": input_names,
        "outputNames": output_names,
        "inputShapes": [list(value.shape) for value in inputs],
        "inputScalarTypes": [str(value.dtype).removeprefix("torch.") for value in inputs],
        "outputShape": list(reference.shape) if reference is not None else [],
    }
    if reference is not None:
        metadata["referenceMean"] = float(reference.mean())
        metadata["referenceRMS"] = float(reference.square().mean().sqrt())
    print(json.dumps(metadata, indent=2), flush=True)
    if args.reference_directory is not None:
        for name, value in zip(input_names, inputs):
            np.asarray(value.float().numpy(), dtype=np.float32).tofile(
                args.reference_directory / f"{name}.f32"
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
        input_names,
        output_names,
        args.output,
        args.dynamic_max_tokens,
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
