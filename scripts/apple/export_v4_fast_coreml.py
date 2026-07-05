# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export the LADA v4-fast YOLO segmentation model to Core ML.

Exports with ``nms=False`` on purpose: the raw YOLO segmentation outputs are
preserved so a native (Swift/Core ML) postprocess can replicate LADA's
Python postprocess deterministically instead of depending on
exporter-specific postprocessing. See docs/apple/v4-fast-coreml-postprocess.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_MODEL = Path("model_weights/lada_mosaic_detection_model_v4_fast.pt")
DEFAULT_OUTPUT_DIR = Path("build/apple/coreml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LADA v4-fast YOLO segmentation model to Core ML")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def build_export_options(imgsz: int) -> dict[str, Any]:
    return {
        "format": "coreml",
        "imgsz": imgsz,
        "half": False,
        "nms": False,
        "simplify": True,
    }


def export_model(model_path: Path, output_dir: Path, imgsz: int, allow_overwrite: bool = False) -> Path:
    from ultralytics import YOLO

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    expected_output = output_dir / f"{model_path.stem}.mlpackage"
    if expected_output.exists() and not allow_overwrite:
        return expected_output

    model = YOLO(str(model_path))
    if model.task != "segment":
        raise ValueError(f"Expected segment model, got {model.task!r}")

    exported = Path(model.export(**build_export_options(imgsz)))
    final_path = output_dir / exported.name
    if exported.resolve() != final_path.resolve():
        if final_path.exists() and allow_overwrite:
            import shutil
            shutil.rmtree(final_path)
        exported.replace(final_path)
    return final_path


def describe_coreml_model(model_path: Path) -> dict[str, Any]:
    import coremltools as ct

    mlmodel = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.ALL)
    spec = mlmodel.get_spec()
    return {
        "type": spec.WhichOneof("Type"),
        "inputs": [feature.name for feature in spec.description.input],
        "outputs": [
            {
                "name": feature.name,
                "shape": list(feature.type.multiArrayType.shape),
                "dataType": feature.type.multiArrayType.dataType,
            }
            for feature in spec.description.output
        ],
        "metadata": dict(spec.description.metadata.userDefined),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exported = export_model(args.model, args.output_dir, args.imgsz, args.allow_overwrite)
    print(exported)
    print(json.dumps(describe_coreml_model(exported), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
