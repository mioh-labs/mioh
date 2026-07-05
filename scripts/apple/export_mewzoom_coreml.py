# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export a MewZoom super-resolution model to Core ML for the ROI enhancer.

MewZoom (https://huggingface.co/andrewdalpino, Apache-2.0) is a UNet
super-resolution family trained to remove blur, noise and compression
artifacts — the same role as jasna's proprietary unet-4x secondary
restoration. The export takes a fixed 256x256 RGB image (LADA's
restoration ROI size) and emits the upscaled RGB image in FP16 so Core ML
can schedule it on the Neural Engine.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO = "andrewdalpino/MewZoom-V1-4X-Unet"
DEFAULT_OUTPUT_DIR = Path("model_weights")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MewZoom to Core ML")
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO, help="HuggingFace repo id of the MewZoom model")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def export_model(repo: str, output_dir: Path, imgsz: int, allow_overwrite: bool = False) -> Path:
    import coremltools as ct
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{repo.rsplit('/', 1)[-1]}_{imgsz}.mlpackage"
    if output_path.exists() and not allow_overwrite:
        return output_path

    from mewzoom.model import MewZoom, SubpixelConv2d

    model = MewZoom.from_pretrained(repo)
    model.eval()

    class ChunkedSubpixelConv2d(torch.nn.Module):
        """
        Bit-identical replacement for SubpixelConv2d that the ANE compiler
        accepts. ANECCompile crashes on conv -> pixel_shuffle when the conv
        emits more than ~1024 channels; PixelShuffle consumes channels in
        contiguous groups of r^2, so shuffling channel chunks separately and
        concatenating gives exactly the same output.
        """

        def __init__(self, subpixel: SubpixelConv2d, chunks: int):
            super().__init__()
            self.conv = subpixel.conv
            self.shuffle = subpixel.shuffle
            self.chunks = chunks

        def forward(self, x):
            y = self.conv(x)
            return torch.cat([self.shuffle(c) for c in y.chunk(self.chunks, dim=1)], dim=1)

    def make_ane_friendly(module: torch.nn.Module):
        for name, child in module.named_children():
            if isinstance(child, SubpixelConv2d) and child.conv.out_channels > 1024:
                upscale_groups = child.shuffle.upscale_factor ** 2
                chunks = 2
                assert (child.conv.out_channels // chunks) % upscale_groups == 0
                setattr(module, name, ChunkedSubpixelConv2d(child, chunks))
            else:
                make_ane_friendly(child)

    make_ane_friendly(model)

    class ImageWrapper(torch.nn.Module):
        """0..1 float in, 0..255 image out; drops the auxiliary QA output."""

        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            y = self.net(x)
            if isinstance(y, tuple):
                y = y[0]
            return y.clamp(0.0, 1.0) * 255.0

    wrapper = ImageWrapper(model)
    wrapper.eval()
    example = torch.rand(1, 3, imgsz, imgsz)
    with torch.no_grad():
        out = wrapper(example)
    scale = out.shape[-1] // imgsz
    traced = torch.jit.trace(wrapper, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, scale=1.0 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.ImageType(name="enhanced", color_layout=ct.colorlayout.RGB)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.user_defined_metadata["lada.enhancer"] = "mewzoom"
    mlmodel.user_defined_metadata["lada.scale"] = str(scale)
    mlmodel.user_defined_metadata["lada.imgsz"] = str(imgsz)
    mlmodel.user_defined_metadata["lada.source"] = repo
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    mlmodel.save(str(output_path))
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exported = export_model(args.repo, args.output_dir, args.imgsz, args.allow_overwrite)
    print(exported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
