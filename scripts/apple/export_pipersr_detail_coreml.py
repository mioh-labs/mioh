#!/usr/bin/env python3
"""Export a locally evaluated PiperSR detail checkpoint as a 256px Core ML model.

This is intentionally separate from the bundled PiperSR model. Checkpoint
licensing depends on the training images and must be reviewed before shipping.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))
from finetune_pipersr_detail import PiperSR  # noqa: E402


class PixelOutput(torch.nn.Module):
    def __init__(self, model: PiperSR) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x) * 255


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")

    import coremltools as ct

    model = PiperSR().eval()
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    wrapped = PixelOutput(model).eval()
    sample = torch.rand(1, 3, 256, 256)
    traced = torch.jit.trace(wrapped, sample)
    converted = ct.convert(
        traced,
        source="pytorch",
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        inputs=[ct.ImageType(name="input_image", shape=sample.shape, scale=1 / 255)],
        outputs=[ct.ImageType(name="output_image", color_layout=ct.colorlayout.RGB)],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    converted.author = "Mioh experimental fine-tune of ModelPiper PiperSR"
    converted.short_description = "Research-only PiperSR 2x detail variant"
    converted.save(str(args.output))

    pixels = np.random.default_rng(33).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    image = Image.fromarray(pixels, "RGB")
    with torch.inference_mode():
        expected = wrapped(
            torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].float() / 255
        )[0].permute(1, 2, 0).numpy()
    actual = np.asarray(converted.predict({"input_image": image})["output_image"].convert("RGB"))
    difference = np.abs(expected - actual)
    print(f"exported {args.output}; parity mean={difference.mean():.3f}/255 max={difference.max():.3f}/255")
    if difference.mean() > 2.0 or difference.max() > 20:
        raise ValueError("exported Core ML output differs too much from PyTorch")


if __name__ == "__main__":
    main()
