# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export a Real-ESRGAN RRDBNet model to Core ML for the ROI enhancer.

The exported model takes a fixed 256x256 RGB image (matching LADA's
restoration ROI size) and emits the upscaled RGB image. Weights and
activations are FP16 so Core ML can schedule the network on the Neural
Engine, keeping the GPU free for BasicVSR++.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_MODEL = Path("model_weights/RealESRGAN_x4plus.pth")
DEFAULT_OUTPUT_DIR = Path("model_weights")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Real-ESRGAN RRDBNet to Core ML")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--scale", type=int, default=4, choices=(2, 4))
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def build_rrdbnet(scale: int):
    import torch
    from export_realesrgan_coreai import build_rrdbnet as build_vendored_rrdbnet

    class ImageWrapper(torch.nn.Module):
        """Maps the 0..1 float output to 0..255 for a Core ML image output."""

        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x).clamp(0.0, 1.0) * 255.0

    # Keep the architecture local to mioh. The Universal application does not
    # ship basicsr, and its torchvision import compatibility has repeatedly
    # broken otherwise-correct offline conversions.
    net = build_vendored_rrdbnet(scale)
    return net, ImageWrapper(net)


def export_model(model_path: Path, output_dir: Path, imgsz: int, scale: int, allow_overwrite: bool = False) -> Path:
    import coremltools as ct
    import torch

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_path.stem}_{imgsz}.mlpackage"
    if output_path.exists() and not allow_overwrite:
        return output_path

    net, wrapper = build_rrdbnet(scale)
    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    net.load_state_dict(
        state.get("params_ema") or state.get("params") or state,
        strict=True,
    )
    wrapper.eval()

    example = torch.rand(1, 3, imgsz, imgsz)
    traced = torch.jit.trace(wrapper, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, scale=1.0 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.ImageType(name="enhanced", color_layout=ct.colorlayout.RGB)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        # RRDBNet x2 starts with pixel_unshuffle, which was added to the MIL
        # iOS16/macOS13 opset. Without an explicit target coremltools defaults
        # to iOS15 and fails after tracing the entire network.
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.user_defined_metadata["lada.enhancer"] = "realesrgan"
    mlmodel.user_defined_metadata["lada.scale"] = str(scale)
    mlmodel.user_defined_metadata["lada.imgsz"] = str(imgsz)
    if output_path.exists():
        import shutil
        shutil.rmtree(output_path)
    mlmodel.save(str(output_path))
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exported = export_model(args.model, args.output_dir, args.imgsz, args.scale, args.allow_overwrite)
    print(exported)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
