#!/usr/bin/env python3
"""Run all 10Eros H3 DiT blocks on captured inputs for Core AI parity tests."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from export_10eros_max_h3_dit_block import TenErosDiTBlockGroup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def load_tensor(directory: Path, name: str, shape: list[int]) -> torch.Tensor:
    values = np.fromfile(directory / f"{name}.f32", dtype=np.float32)
    if values.size != int(np.prod(shape)):
        raise ValueError(f"{name} has {values.size} values, expected {shape}")
    return torch.from_numpy(values.copy()).reshape(shape).to(torch.bfloat16)


def main() -> int:
    args = parse_args()
    metadata = json.loads((args.input_directory / "metadata.json").read_text())
    shapes = metadata["inputShapes"]
    names = [
        "hidden_states", "timestep_coordinates", "modulation_weights",
        "rope_cosine", "rope_sine",
    ]
    tensors = [
        load_tensor(args.input_directory, name, shape)
        for name, shape in zip(names, shapes)
    ]
    hidden, shared = tensors[0], tensors[1:]
    args.output_directory.mkdir(parents=True, exist_ok=True)
    checkpoints: list[dict[str, float | int]] = []
    for first in range(0, 50, 4):
        count = min(4, 50 - first)
        group = TenErosDiTBlockGroup(
            args.checkpoint, first, count, torch.bfloat16
        ).eval()
        with torch.no_grad():
            hidden = group(hidden, *shared).to(torch.bfloat16)
        output = hidden.float().numpy()
        output.tofile(args.output_directory / f"blocks00-{first + count - 1:02d}.f32")
        rms = float(np.sqrt(np.mean(output * output)))
        checkpoints.append({"lastLayer": first + count - 1, "rms": rms})
        print(json.dumps(checkpoints[-1]), flush=True)
        del group
        gc.collect()
    metadata["checkpoints"] = checkpoints
    (args.output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
