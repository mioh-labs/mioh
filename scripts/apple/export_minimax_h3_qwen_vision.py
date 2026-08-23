#!/usr/bin/env python3
"""Export fixed-canvas Qwen3-VL vision stages for MiniMax H3.

The Mioh H3 profile is fixed to 864x480 and ten two-frame reference blocks.
At Qwen patch size 16 that is a 30x54 grid (1620 patches, 405 merged
tokens) per block.  Splitting patch embedding, 27 transformer blocks, and
the four mergers lets Swift execute the tower sequentially without loading a
monolithic multi-gigabyte graph.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


HIDDEN_SIZE = 1152
NUM_HEADS = 16
HEAD_DIM = 72
INTERMEDIATE_SIZE = 4304
MERGE_SIZE = 2
MERGE_DIM = HIDDEN_SIZE * MERGE_SIZE * MERGE_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("patch", "block", "merger", "deepstack"), required=True
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--block", type=int)
    parser.add_argument("--deepstack", type=int)
    parser.add_argument("--batch", type=int, default=10)
    parser.add_argument("--grid-height", type=int, default=30)
    parser.add_argument("--grid-width", type=int, default=54)
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


def load_tensor(handle, key: str) -> torch.Tensor:
    return handle.get_tensor(key).to(torch.float16).contiguous()


def vision_coordinates(grid_height: int, grid_width: int) -> torch.Tensor:
    if grid_height % MERGE_SIZE or grid_width % MERGE_SIZE:
        raise ValueError("vision grid dimensions must be divisible by two")
    values: list[tuple[int, int]] = []
    for merged_y in range(grid_height // MERGE_SIZE):
        for merged_x in range(grid_width // MERGE_SIZE):
            for patch_y in range(MERGE_SIZE):
                for patch_x in range(MERGE_SIZE):
                    values.append(
                        (
                            merged_y * MERGE_SIZE + patch_y,
                            merged_x * MERGE_SIZE + patch_x,
                        )
                    )
    return torch.tensor(values, dtype=torch.long)


def fixed_rotary(grid_height: int, grid_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    coordinates = vision_coordinates(grid_height, grid_width)
    inverse_frequency = 1.0 / (
        10_000.0 ** (torch.arange(0, HEAD_DIM // 2, 2).float() / (HEAD_DIM // 2))
    )
    table = torch.outer(
        torch.arange(max(grid_height, grid_width), dtype=torch.float32),
        inverse_frequency,
    )
    frequency = table[coordinates].flatten(1)
    embedding = torch.cat((frequency, frequency), dim=-1)
    return (
        embedding.cos().reshape(1, 1, -1, HEAD_DIM).to(torch.float16),
        embedding.sin().reshape(1, 1, -1, HEAD_DIM).to(torch.float16),
    )


def fixed_position_embedding(
    weight: torch.Tensor, grid_height: int, grid_width: int
) -> torch.Tensor:
    side = 48
    height_indices = torch.linspace(0, side - 1, grid_height)
    width_indices = torch.linspace(0, side - 1, grid_width)
    height_floor = height_indices.int()
    width_floor = width_indices.int()
    height_ceil = (height_floor + 1).clamp(max=side - 1)
    width_ceil = (width_floor + 1).clamp(max=side - 1)
    dh = height_indices - height_floor
    dw = width_indices - width_floor
    position = weight.reshape(side, side, HIDDEN_SIZE).float()
    rows: list[torch.Tensor] = []
    for y in range(grid_height):
        upper = (
            position[height_floor[y], width_floor] * (1 - dw).unsqueeze(1)
            + position[height_floor[y], width_ceil] * dw.unsqueeze(1)
        )
        lower = (
            position[height_ceil[y], width_floor] * (1 - dw).unsqueeze(1)
            + position[height_ceil[y], width_ceil] * dw.unsqueeze(1)
        )
        rows.append(upper * (1 - dh[y]) + lower * dh[y])
    raster = torch.stack(rows).reshape(grid_height, grid_width, HIDDEN_SIZE)
    grouped = raster.reshape(
        grid_height // MERGE_SIZE,
        MERGE_SIZE,
        grid_width // MERGE_SIZE,
        MERGE_SIZE,
        HIDDEN_SIZE,
    ).permute(0, 2, 1, 3, 4)
    return grouped.reshape(1, grid_height * grid_width, HIDDEN_SIZE).to(torch.float16)


class VisionPatch(torch.nn.Module):
    def __init__(
        self, checkpoint: Path, grid_height: int, grid_width: int
    ) -> None:
        super().__init__()
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            convolution = load_tensor(handle, "visual.patch_embed.proj.weight")
            self.register_buffer("weight", convolution.reshape(HIDDEN_SIZE, -1))
            self.register_buffer(
                "bias", load_tensor(handle, "visual.patch_embed.proj.bias")
            )
            self.register_buffer(
                "position",
                fixed_position_embedding(
                    handle.get_tensor("visual.pos_embed.weight"),
                    grid_height,
                    grid_width,
                ),
            )

    def forward(self, pixel_patches: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(
            pixel_patches, self.weight, self.bias
        ) + self.position


class VisionBlock(torch.nn.Module):
    def __init__(
        self, checkpoint: Path, block: int, grid_height: int, grid_width: int
    ) -> None:
        super().__init__()
        prefix = f"visual.blocks.{block}"
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            for name in ("norm1", "norm2"):
                norm = torch.nn.LayerNorm(HIDDEN_SIZE, eps=1e-6)
                norm.weight.data.copy_(load_tensor(handle, f"{prefix}.{name}.weight"))
                norm.bias.data.copy_(load_tensor(handle, f"{prefix}.{name}.bias"))
                setattr(self, name, norm.to(torch.float16))
            self.qkv = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE * 3, bias=True).to(
                torch.float16
            )
            self.qkv.weight.data.copy_(load_tensor(handle, f"{prefix}.attn.qkv.weight"))
            self.qkv.bias.data.copy_(load_tensor(handle, f"{prefix}.attn.qkv.bias"))
            self.proj = torch.nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE, bias=True).to(
                torch.float16
            )
            self.proj.weight.data.copy_(load_tensor(handle, f"{prefix}.attn.proj.weight"))
            self.proj.bias.data.copy_(load_tensor(handle, f"{prefix}.attn.proj.bias"))
            self.fc1 = torch.nn.Linear(HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=True).to(
                torch.float16
            )
            self.fc1.weight.data.copy_(load_tensor(handle, f"{prefix}.mlp.linear_fc1.weight"))
            self.fc1.bias.data.copy_(load_tensor(handle, f"{prefix}.mlp.linear_fc1.bias"))
            self.fc2 = torch.nn.Linear(INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=True).to(
                torch.float16
            )
            self.fc2.weight.data.copy_(load_tensor(handle, f"{prefix}.mlp.linear_fc2.weight"))
            self.fc2.bias.data.copy_(load_tensor(handle, f"{prefix}.mlp.linear_fc2.bias"))
        cosine, sine = fixed_rotary(grid_height, grid_width)
        self.register_buffer("rope_cosine", cosine)
        self.register_buffer("rope_sine", sine)

    def apply_rope(self, value: torch.Tensor) -> torch.Tensor:
        first, second = value.chunk(2, dim=-1)
        rotated = torch.cat((-second, first), dim=-1)
        return value * self.rope_cosine + rotated * self.rope_sine

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        normalized = self.norm1(hidden_states)
        qkv = self.qkv(normalized).reshape(
            batch, sequence, 3, NUM_HEADS, HEAD_DIM
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        query = self.apply_rope(query)
        key = self.apply_rope(key)
        attended = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        attended = attended.transpose(1, 2).reshape(batch, sequence, HIDDEN_SIZE)
        hidden_states = hidden_states + self.proj(attended)
        normalized = self.norm2(hidden_states)
        mlp = self.fc2(torch.nn.functional.gelu(self.fc1(normalized), approximate="tanh"))
        return hidden_states + mlp


class VisionMerger(torch.nn.Module):
    def __init__(self, checkpoint: Path, deepstack: int | None) -> None:
        super().__init__()
        prefix = (
            "visual.merger"
            if deepstack is None
            else f"visual.deepstack_merger_list.{deepstack}"
        )
        norm_size = HIDDEN_SIZE if deepstack is None else MERGE_DIM
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            self.norm = torch.nn.LayerNorm(norm_size, eps=1e-6).to(torch.float16)
            self.norm.weight.data.copy_(load_tensor(handle, f"{prefix}.norm.weight"))
            self.norm.bias.data.copy_(load_tensor(handle, f"{prefix}.norm.bias"))
            self.fc1 = torch.nn.Linear(MERGE_DIM, MERGE_DIM, bias=True).to(
                torch.float16
            )
            self.fc1.weight.data.copy_(load_tensor(handle, f"{prefix}.linear_fc1.weight"))
            self.fc1.bias.data.copy_(load_tensor(handle, f"{prefix}.linear_fc1.bias"))
            self.fc2 = torch.nn.Linear(MERGE_DIM, 5120, bias=True).to(torch.float16)
            self.fc2.weight.data.copy_(load_tensor(handle, f"{prefix}.linear_fc2.weight"))
            self.fc2.bias.data.copy_(load_tensor(handle, f"{prefix}.linear_fc2.bias"))
        self.deepstack = deepstack is not None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        if self.deepstack:
            merged = hidden_states.reshape(batch, sequence // 4, MERGE_DIM)
            merged = self.norm(merged)
        else:
            merged = self.norm(hidden_states).reshape(batch, sequence // 4, MERGE_DIM)
        return self.fc2(torch.nn.functional.gelu(self.fc1(merged)))


class CoreAIBlockExportWrapper(torch.nn.Module):
    """Give each weight-distinct vision block a unique function signature."""

    def __init__(self, block: torch.nn.Module, salt_width: int) -> None:
        super().__init__()
        self.block = block
        self.salt_width = salt_width

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.block(hidden_states)
        return hidden_states, hidden_states[:, :1, : self.salt_width]


def model_and_example(args: argparse.Namespace) -> tuple[torch.nn.Module, torch.Tensor, str, str]:
    sequence = args.grid_height * args.grid_width
    if args.stage == "patch":
        return (
            VisionPatch(args.checkpoint, args.grid_height, args.grid_width).eval(),
            torch.zeros((args.batch, sequence, 1536), dtype=torch.float16),
            "pixel_patches",
            "vision_hidden",
        )
    if args.stage == "block":
        if args.block is None or not 0 <= args.block < 27:
            raise ValueError("--block 0...26 is required for block stage")
        return (
            VisionBlock(
                args.checkpoint, args.block, args.grid_height, args.grid_width
            ).eval(),
            torch.zeros((args.batch, sequence, HIDDEN_SIZE), dtype=torch.float16),
            "vision_hidden",
            "vision_hidden_out",
        )
    if args.stage == "deepstack":
        if args.deepstack is None or not 0 <= args.deepstack < 3:
            raise ValueError("--deepstack 0...2 is required")
        model = VisionMerger(args.checkpoint, args.deepstack).eval()
        output_name = f"deepstack_{args.deepstack}"
    else:
        model = VisionMerger(args.checkpoint, None).eval()
        output_name = "vision_merged"
    return (
        model,
        torch.zeros((args.batch, sequence, HIDDEN_SIZE), dtype=torch.float16),
        "vision_hidden",
        output_name,
    )


def export_coreai(
    model: torch.nn.Module,
    example: torch.Tensor,
    input_name: str,
    output_name: str,
    destination: Path,
    extra_output: bool = False,
) -> None:
    import coreai_torch

    exported = torch.export.export(model, (example,))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    output_names = [output_name, "cache_salt"] if extra_output else [output_name]
    converter.add_exported_program(
        exported, input_names=[input_name], output_names=output_names
    )
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(destination)


def main() -> int:
    args = parse_args()
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    remove_existing(args.output, args.overwrite)
    model, example, input_name, output_name = model_and_example(args)
    reference = None
    if not args.skip_reference:
        probe = torch.sin(
            torch.arange(example.numel(), dtype=torch.float32) * 0.0009765625
        ).reshape(example.shape).to(torch.float16)
        with torch.no_grad():
            reference = model(probe).float()
        example = probe
    metadata = {
        "stage": args.stage,
        "block": args.block,
        "deepstack": args.deepstack,
        "inputName": input_name,
        "outputName": output_name,
        "inputShape": list(example.shape),
        "outputShape": (
            list(reference.shape)
            if reference is not None
            else (
                [args.batch, args.grid_height * args.grid_width // 4, 5120]
                if args.stage in ("merger", "deepstack")
                else list(example.shape[:-1]) + [HIDDEN_SIZE]
            )
        ),
    }
    if reference is not None:
        metadata["referenceMean"] = float(reference.mean())
        metadata["referenceRMS"] = float(reference.square().mean().sqrt())
    print(json.dumps(metadata, indent=2), flush=True)
    if args.reference_directory is not None and reference is not None:
        args.reference_directory.mkdir(parents=True, exist_ok=True)
        np.asarray(example.float().numpy(), dtype=np.float32).tofile(
            args.reference_directory / "input.f32"
        )
        np.asarray(reference.numpy(), dtype=np.float32).tofile(
            args.reference_directory / "reference.f32"
        )
        (args.reference_directory / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
    if args.stage == "block":
        assert args.block is not None
        model = CoreAIBlockExportWrapper(model, args.block + 1).eval()
        export_coreai(
            model, example, input_name, output_name, args.output,
            extra_output=True,
        )
    else:
        export_coreai(model, example, input_name, output_name, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
