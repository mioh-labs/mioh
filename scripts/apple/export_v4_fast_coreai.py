# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export the v4-fast YOLO segmentation model to fixed FP16 Core AI."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch

DEFAULT_MODEL = Path("model_weights/lada_mosaic_detection_model_v4_fast.pt")
DEFAULT_OUTPUT = Path(
    "model_weights/lada_mosaic_detection_model_v4_fast-fp16.aimodel"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export LADA v4-fast detection to Core AI"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


class RawSegmentationOutputs(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(image)[0]
        return output[0], output[1]


def load_model(model_path: Path) -> RawSegmentationOutputs:
    from ultralytics import YOLO

    yolo = YOLO(str(model_path), task="segment")
    if yolo.task != "segment":
        raise ValueError(f"Expected segment model, got {yolo.task!r}")
    return RawSegmentationOutputs(yolo.model.eval().half()).eval()


def convert_exported_program(exported: torch.export.ExportedProgram, coreai_torch: Any):
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported,
        input_names=["image"],
        output_names=["candidates", "prototypes"],
    )
    return converter.to_coreai()


def export_model(args: argparse.Namespace) -> Path:
    if args.imgsz != 640:
        raise ValueError("v4-fast Core AI export requires imgsz=640")
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if args.output.exists() and not args.allow_overwrite:
        raise FileExistsError(f"{args.output} exists; pass --allow-overwrite")

    import coreai_torch

    model = load_model(args.model)
    example = torch.zeros((1, 3, args.imgsz, args.imgsz), dtype=torch.float16)
    exported = torch.export.export(model, (example,))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    program = convert_exported_program(exported, coreai_torch)
    program.optimize()

    if args.output.exists():
        shutil.rmtree(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    program.save_asset(args.output)
    if args.verbose:
        print(f"Core AI detection asset: {args.output}")
    return args.output


def main(argv: list[str] | None = None) -> int:
    export_model(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
