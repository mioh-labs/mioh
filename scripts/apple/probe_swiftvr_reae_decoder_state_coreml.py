#!/usr/bin/env python3
"""Validate cross-chunk state for a native SwiftVR ReAE decoder export."""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F

from probe_swiftvr_reae_coreml import load_official_encoder, load_reae


class StatefulDecoderChunk(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, module, size: int, latent_frames: int):
        super().__init__()
        self.decoder = model.decoder
        self.mem_block = module.MemBlock
        self.temporal_grow = module.TGrow
        self.size = size
        self.latent_frames = latent_frames

    def forward(self, latents: torch.Tensor, *states: torch.Tensor):
        x = latents.reshape(
            self.latent_frames, 48, self.size // 16, self.size // 16
        )
        next_states = []
        state_index = 0
        for layer in self.decoder:
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
            elif isinstance(layer, self.temporal_grow) and layer.stride == 2:
                count, channels, height, width = x.shape
                grown = layer.conv3d(x.unsqueeze(2).repeat_interleave(2, dim=2))
                x = grown.permute(0, 2, 1, 3, 4).reshape(
                    count * 2, channels, height, width
                )
            else:
                x = layer(x)
        frames = F.pixel_shuffle(torch.clamp(x, 0, 1), 2).reshape(
            1, self.latent_frames * 4, 3, self.size, self.size
        )
        return (frames, *next_states)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(256, 512, 1024), default=256)
    parser.add_argument("--latent-frames", type=int, choices=(1, 6, 7), default=1)
    args = parser.parse_args()

    module = load_reae(args.source)
    official_type, _ = load_official_encoder(args.source)
    from swiftvr.streaming.chunk import ChunkSpec, ChunkType

    first_spec = ChunkSpec(
        ChunkType.FIRST, 0, args.latent_frames * 4, 0, 0, True
    )
    second_spec = ChunkSpec(
        ChunkType.MIDDLE, args.latent_frames * 4,
        args.latent_frames * 4, 0, 1, False
    )
    model = module.ReAE(str(args.checkpoint)).eval()
    wrapper = StatefulDecoderChunk(model, module, args.size, args.latent_frames).eval()
    latent_size = args.size // 16
    first = torch.linspace(
        -1.0, 1.0, args.latent_frames * 48 * latent_size * latent_size
    ).reshape(
        1, args.latent_frames, 48, latent_size, latent_size
    )
    second = first.flip(-1).contiguous()
    state_shapes = (
        ((512, args.size // 16, args.size // 16),) * 3
        + ((256, args.size // 8, args.size // 8),) * 3
        + ((128, args.size // 4, args.size // 4),) * 3
    )
    zeros = tuple(torch.zeros((1, *shape)) for shape in state_shapes)
    with torch.no_grad():
        official = official_type(model)
        official_first = official.decode_chunk_fixed(first, first_spec)
        official_second = official.decode_chunk_fixed(second, second_spec)
        first_result = wrapper(first, *zeros)
        second_result = wrapper(second, *first_result[1:])
        for name, expected, actual in (
            ("first", official_first, first_result[0][:, 3:]),
            ("continued", official_second, second_result[0]),
        ):
            difference = (expected - actual).abs().max().item()
            if difference > 1e-6:
                raise RuntimeError(f"{name} decoder differs from official path: {difference}")
        traced = torch.jit.trace(wrapper, (first, *zeros))

    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latents", shape=first.shape)]
        + [ct.TensorType(name=f"state_{i}", shape=x.shape) for i, x in enumerate(zeros)],
        outputs=[ct.TensorType(name="frames")]
        + [ct.TensorType(name=f"next_state_{i}") for i in range(9)],
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))

    def predict(latents, states):
        inputs = {"latents": latents.numpy()}
        inputs.update({f"state_{i}": x.numpy() for i, x in enumerate(states)})
        return converted.predict(inputs)

    native_first = predict(first, zeros)
    native_second = predict(
        second,
        tuple(torch.from_numpy(native_first[f"next_state_{i}"]) for i in range(9)),
    )
    for name, actual, expected in (
        ("first", native_first["frames"][:, 3:], first_result[0][:, 3:].numpy()),
        ("continued", native_second["frames"], second_result[0].numpy()),
    ):
        error = np.abs(actual - expected)
        print(
            f"{name}: max abs error={error.max():.6g}, "
            f"mean abs error={error.mean():.6g}"
        )


if __name__ == "__main__":
    main()
