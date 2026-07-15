# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export a Spandrel-compatible image model to fixed-shape Core ML."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_MODEL = Path("model_weights/4xNomosWebPhoto_RealPLKSR.safetensors")
DEFAULT_OUTPUT_DIR = Path("model_weights")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def load_model(model_path: Path):
    import torch
    from spandrel import ImageModelDescriptor, ModelLoader

    descriptor = ModelLoader().load_from_file(model_path)
    if not isinstance(descriptor, ImageModelDescriptor):
        raise ValueError(f"{model_path} is not a Spandrel image model")

    class ImageWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, image):
            return self.model(image).clamp(0.0, 1.0) * 255.0

    return descriptor, ImageWrapper(descriptor.model.float().eval()).eval()


def export_model(
    model_path: Path,
    output_dir: Path,
    imgsz: int,
    allow_overwrite: bool = False,
) -> Path:
    import coremltools as ct
    import torch

    if imgsz <= 0:
        raise ValueError("imgsz must be positive")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    descriptor, wrapper = load_model(model_path)
    output_path = output_dir / f"{model_path.stem}_{imgsz}.mlpackage"
    if output_path.exists() and not allow_overwrite:
        raise FileExistsError(f"{output_path} exists; pass --allow-overwrite")

    example = torch.rand(1, descriptor.input_channels, imgsz, imgsz)
    with torch.inference_mode():
        output = wrapper(example)
    expected = (1, descriptor.output_channels, imgsz * descriptor.scale, imgsz * descriptor.scale)
    if tuple(output.shape) != expected:
        raise ValueError(f"unexpected output shape {tuple(output.shape)}; expected {expected}")
    traced = torch.jit.trace(wrapper, example)
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(name="image", shape=example.shape, scale=1.0 / 255.0, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.ImageType(name="enhanced", color_layout=ct.colorlayout.RGB)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.user_defined_metadata["lada.enhancer"] = "spandrel"
    mlmodel.user_defined_metadata["lada.scale"] = str(descriptor.scale)
    mlmodel.user_defined_metadata["lada.imgsz"] = str(imgsz)
    mlmodel.user_defined_metadata["lada.prefer_pre_resize"] = "1"
    mlmodel.user_defined_metadata["lada.source"] = model_path.name
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        shutil.rmtree(output_path)
    mlmodel.save(str(output_path))
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(export_model(args.model, args.output_dir, args.imgsz, args.allow_overwrite))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
