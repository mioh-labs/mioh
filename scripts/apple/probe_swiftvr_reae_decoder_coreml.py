#!/usr/bin/env python3
"""Check SwiftVR ReAE first-chunk decoder export and native parity.

This is a conversion-time probe. It exports the four raw decoder frames;
the official first-chunk protocol trims the first three afterwards.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import numpy as np
import torch
import torch.nn.functional as F

from probe_swiftvr_reae_coreml import load_official_encoder, load_reae


class FirstDecoderChunk(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, module, size: int):
        super().__init__()
        self.decoder = model.decoder
        self.mem_block = module.MemBlock
        self.temporal_grow = module.TGrow
        self.size = size

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = latents.reshape(1, 48, self.size // 16, self.size // 16)
        for layer in self.decoder:
            if isinstance(layer, self.mem_block):
                indices = torch.tensor([max(0, i - 1) for i in range(x.shape[0])])
                mask = torch.tensor(
                    [0.0] + [1.0] * (x.shape[0] - 1), dtype=x.dtype
                ).view(-1, 1, 1, 1)
                previous = torch.index_select(x, 0, indices) * mask
                x = layer(x, previous)
            elif isinstance(layer, self.temporal_grow) and layer.stride == 2:
                count, channels, height, width = x.shape
                expanded = x.unsqueeze(2).repeat_interleave(2, dim=2)
                grown = layer.conv3d(expanded)
                x = grown.permute(0, 2, 1, 3, 4).reshape(
                    count * 2, channels, height, width
                )
            else:
                x = layer(x)
        x = torch.clamp(x, 0, 1)
        return F.pixel_shuffle(x, 2).reshape(1, 4, 3, self.size, self.size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(256, 512, 1024), default=256)
    args = parser.parse_args()

    module = load_reae(args.source)
    official_type, first_chunk = load_official_encoder(args.source)
    model = module.ReAE(str(args.checkpoint)).eval()
    wrapper = FirstDecoderChunk(model, module, args.size).eval()
    latent_size = args.size // 16
    latents = torch.linspace(-1.0, 1.0, 48 * latent_size * latent_size).reshape(
        1, 1, 48, latent_size, latent_size
    )
    with torch.no_grad():
        expected = wrapper(latents)
        official = official_type(model).decode_chunk_fixed(latents, first_chunk)
        difference = (expected[:, 3:] - official).abs().max().item()
        if difference > 1e-6:
            raise RuntimeError(f"Decoder differs from official path: {difference}")
        traced = torch.jit.trace(wrapper, latents)

    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="latents", shape=latents.shape)],
        outputs=[ct.TensorType(name="frames")],
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    converted.save(str(args.output))
    native = converted.predict({"latents": latents.numpy()})["frames"]
    error = np.abs(native - expected.numpy())
    print(
        f"ReAE decoder first chunk: {tuple(native.shape)}, "
        f"max abs error={error.max():.6g}, "
        f"mean abs error={error.mean():.6g}"
    )


if __name__ == "__main__":
    main()
