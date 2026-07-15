# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export a Spandrel-compatible image model to fixed-shape FP16 Core AI."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch


DEFAULT_MODEL = Path("model_weights/4xNomosWebPhoto_RealPLKSR.safetensors")
DEFAULT_OUTPUT = Path("model_weights/4xNomosWebPhoto_RealPLKSR-256-fp16.aimodel")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


class CoreAIImageWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image).clamp(0.0, 1.0)


def load_model(model_path: Path):
    from spandrel import ImageModelDescriptor, ModelLoader

    descriptor = ModelLoader().load_from_file(model_path)
    if not isinstance(descriptor, ImageModelDescriptor):
        raise ValueError(f"{model_path} is not a Spandrel image model")
    wrapper = CoreAIImageWrapper(descriptor.model.half().eval()).eval()
    return descriptor, wrapper


def export_model(args: argparse.Namespace) -> Path:
    if args.imgsz <= 0:
        raise ValueError("imgsz must be positive")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if args.output.exists() and not args.allow_overwrite:
        raise FileExistsError(f"{args.output} exists; pass --allow-overwrite")

    import coreai_torch

    descriptor, model = load_model(args.model)
    example = torch.zeros(
        (1, descriptor.input_channels, args.imgsz, args.imgsz),
        dtype=torch.float16,
    )
    exported = torch.export.export(model, (example,))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported,
        input_names=["image"],
        output_names=["enhanced"],
    )
    program = converter.to_coreai()
    program.optimize()

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(args.output)
    if args.verbose:
        print(
            f"Core AI Spandrel asset: {args.output} "
            f"({args.imgsz}px, {descriptor.scale}x)"
        )
    return args.output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(export_model(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
