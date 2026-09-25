#!/usr/bin/env python3
"""Probe whether SwiftVR's ReAE encoder can be exported for a Swift runtime.

This is an offline conversion check, not an inference dependency of mioh.
It deliberately exports only the first four-frame encoder chunk; continuation
state and the 30-layer DiT must be validated before enabling an app option.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F


def load_reae(source: Path):
    module_path = source / "swiftvr" / "models" / "reae.py"
    for name, path in (
        ("swiftvr", source / "swiftvr"),
        ("swiftvr.models", source / "swiftvr" / "models"),
        ("swiftvr.streaming", source / "swiftvr" / "streaming"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    spec = importlib.util.spec_from_file_location("swiftvr.models.reae", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_official_encoder(source: Path):
    from swiftvr.streaming.chunk import ChunkSpec, ChunkType
    from swiftvr.streaming.tae import StreamingTAE

    return StreamingTAE, ChunkSpec(ChunkType.FIRST, 0, 4, 0, 0, True)


class FirstEncoderChunk(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, module, size: int):
        super().__init__()
        self.encoder = model.encoder
        self.mem_block = module.MemBlock
        self.temporal_pool = module.TPool
        self.size = size

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # The first chunk has no prior state. Later chunks need explicit state
        # inputs/outputs, and are intentionally outside this feasibility probe.
        x = F.pixel_unshuffle(frames.reshape(4, 3, self.size, self.size), 2)
        for layer in self.encoder:
            if isinstance(layer, self.mem_block):
                indices = torch.tensor([max(0, i - 1) for i in range(x.shape[0])])
                mask = torch.tensor(
                    [0.0] + [1.0] * (x.shape[0] - 1), dtype=x.dtype
                ).view(-1, 1, 1, 1)
                previous = torch.index_select(x, 0, indices) * mask
                x = layer(x, previous)
            elif isinstance(layer, self.temporal_pool):
                x = layer(x)
            else:
                x = layer(x)
        return x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("float16", "float32"), default="float16")
    parser.add_argument("--size", type=int, choices=(256, 512, 1024), default=256)
    args = parser.parse_args()

    module = load_reae(args.source)
    official_encoder, first_chunk = load_official_encoder(args.source)
    model = module.ReAE(str(args.checkpoint)).eval()
    wrapper = FirstEncoderChunk(model, module, args.size).eval()
    sample = torch.linspace(-1.0, 1.0, 4 * 3 * args.size * args.size).reshape(
        1, 4, 3, args.size, args.size
    )
    with torch.no_grad():
        expected = wrapper(sample).numpy()
        official = official_encoder(model).encode_chunk_fixed(sample, first_chunk)
        official_difference = np.abs(official.numpy().reshape(expected.shape) - expected)
        if official_difference.max() > 1e-6:
            raise RuntimeError(
                "Probe wrapper differs from official SwiftVR encoder: "
                f"max abs error={official_difference.max():.6g}"
            )
        traced = torch.jit.trace(wrapper, sample)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="frames", shape=sample.shape)],
        outputs=[ct.TensorType(name="latents")],
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
        compute_precision=(
            ct.precision.FLOAT16
            if args.precision == "float16"
            else ct.precision.FLOAT32
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))
    result = converted.predict({"frames": sample.numpy()})["latents"]
    difference = np.abs(result - expected)
    print(
        f"ReAE first encoder chunk: {tuple(result.shape)}, "
        f"reference std={expected.std():.6g}, "
        f"max abs error={difference.max():.6g}, "
        f"mean abs error={difference.mean():.6g}"
    )


if __name__ == "__main__":
    main()
