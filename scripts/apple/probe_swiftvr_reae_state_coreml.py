#!/usr/bin/env python3
"""Validate a stateful four-frame SwiftVR ReAE encoder Core ML export.

This conversion runs offline. The app must not expose SwiftVR until the
matching decoder and 30-layer DiT also pass native parity tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F

from probe_swiftvr_reae_coreml import load_official_encoder, load_reae


class StatefulEncoderChunk(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, module, size: int, frames: int):
        super().__init__()
        self.encoder = model.encoder
        self.mem_block = module.MemBlock
        self.temporal_pool = module.TPool
        self.size = size
        self.frames = frames

    def forward(self, frames: torch.Tensor, *states: torch.Tensor):
        x = F.pixel_unshuffle(
            frames.reshape(self.frames, 3, self.size, self.size), 2
        )
        next_states = []
        state_index = 0
        for layer in self.encoder:
            if isinstance(layer, self.mem_block):
                indices = torch.tensor([max(0, i - 1) for i in range(x.shape[0])])
                first = torch.tensor(
                    [1.0] + [0.0] * (x.shape[0] - 1), dtype=x.dtype
                ).view(-1, 1, 1, 1)
                previous = torch.index_select(x, 0, indices) * (1 - first)
                previous = previous + states[state_index] * first
                next_states.append(
                    torch.index_select(x, 0, torch.tensor([x.shape[0] - 1]))
                )
                state_index += 1
                x = layer(x, previous)
            elif isinstance(layer, self.temporal_pool):
                x = layer(x)
            else:
                x = layer(x)
        return (x, *next_states)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(256, 512, 1024), default=256)
    parser.add_argument("--frames", type=int, choices=(4, 24, 28), default=4)
    args = parser.parse_args()

    module = load_reae(args.source)
    official_type, _ = load_official_encoder(args.source)
    from swiftvr.streaming.chunk import ChunkSpec, ChunkType

    first_chunk = ChunkSpec(ChunkType.FIRST, 0, args.frames, 0, 0, True)
    model = module.ReAE(str(args.checkpoint)).eval()
    wrapper = StatefulEncoderChunk(model, module, args.size, args.frames).eval()
    first = torch.linspace(-1.0, 1.0, args.frames * 3 * args.size * args.size).reshape(
        1, args.frames, 3, args.size, args.size
    )
    second = first.flip(1).contiguous()
    state_shapes = (
        ((64, args.size // 4, args.size // 4),) * 3
        + ((64, args.size // 8, args.size // 8),) * 3
        + ((64, args.size // 16, args.size // 16),) * 3
    )
    zeros = tuple(torch.zeros((1, *shape)) for shape in state_shapes)
    with torch.no_grad():
        official = official_type(model)
        official_first = official.encode_chunk_fixed(first, first_chunk)
        official_second = official.encode_chunk_fixed(second, first_chunk)
        first_result = wrapper(first, *zeros)
        second_result = wrapper(second, *first_result[1:])
        for name, expected, actual in (
            ("first", official_first, first_result[0]),
            ("continued", official_second, second_result[0]),
        ):
            difference = (expected.flatten() - actual.flatten()).abs().max().item()
            if difference > 1e-6:
                raise RuntimeError(f"{name} chunk differs from official path: {difference}")
        traced = torch.jit.trace(wrapper, (first, *zeros))

    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="frames", shape=first.shape)]
        + [
            ct.TensorType(name=f"state_{i}", shape=value.shape)
            for i, value in enumerate(zeros)
        ],
        outputs=[ct.TensorType(name="latents")]
        + [ct.TensorType(name=f"next_state_{i}") for i in range(9)],
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))
    def predict(frames, states):
        inputs = {"frames": frames.numpy()}
        inputs.update({f"state_{i}": x.numpy() for i, x in enumerate(states)})
        return converted.predict(inputs)

    native_first = predict(first, zeros)
    native_second = predict(
        second,
        tuple(torch.from_numpy(native_first[f"next_state_{i}"]) for i in range(9)),
    )
    for name, actual, expected in (
        ("first", native_first["latents"], first_result[0].numpy()),
        ("continued", native_second["latents"], second_result[0].numpy()),
    ):
        difference = np.abs(actual - expected)
        print(
            f"{name}: max abs error={difference.max():.6g}, "
            f"mean abs error={difference.mean():.6g}"
        )


if __name__ == "__main__":
    main()
