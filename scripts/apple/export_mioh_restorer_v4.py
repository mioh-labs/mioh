#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export a trained MiohRestorerV4-Q checkpoint to Apple model assets."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from lada.models.mioh_restorer.model_v4 import (
    MiohRestorerV4ExportWrapper,
    MiohRestorerV4Q,
    parameter_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--skip-coreml", action="store_true")
    parser.add_argument("--skip-coreai", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def load_model(path: Path, *, raw: bool) -> tuple[MiohRestorerV4Q, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config", {})
    if int(config.get("version", 0)) != 4:
        raise ValueError(f"checkpoint is not MiohRestorerV4: {path}")
    revision = int(config.get("architecture_revision", 0))
    if revision not in (1, 2):
        raise ValueError("unsupported MiohRestorerV4 architecture revision")
    model = MiohRestorerV4Q(
        alignment_variant=str(config["alignment_variant"]),
        execution_mode=str(config["execution_mode"]),
        quarter_channels=int(config["quarter_channels"]),
        eighth_channels=int(config["eighth_channels"]),
        fusion_eighth_channels=int(config["fusion_eighth_channels"]),
        fusion_quarter_channels=int(config["fusion_quarter_channels"]),
        eighth_blocks=int(config["eighth_blocks"]),
        quarter_blocks=int(config["quarter_blocks"]),
        high_resolution_detail=bool(config.get("high_resolution_detail", False)),
        detail_full_channels=int(config.get("detail_full_channels", 32)),
        detail_half_channels=int(config.get("detail_half_channels", 48)),
        detail_fusion_channels=int(config.get("detail_fusion_channels", 64)),
    )
    key = "state_dict" if raw else "ema_state_dict"
    if key not in payload:
        raise ValueError(f"checkpoint is missing {key}")
    model.load_state_dict(payload[key], strict=True)
    model.enable_gradient_checkpointing(False)
    return model.eval(), payload


def remove_existing(path: Path, allow: bool) -> None:
    if not path.exists():
        return
    if not allow:
        raise FileExistsError(f"{path} exists; pass --allow-overwrite")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def export_coreml(
    model: MiohRestorerV4Q,
    image_size: int,
    output: Path,
    allow_overwrite: bool,
) -> None:
    import coremltools as ct

    remove_existing(output, allow_overwrite)
    wrapper = MiohRestorerV4ExportWrapper(copy.deepcopy(model).float()).eval()
    example = torch.zeros(1, 36, image_size, image_size)
    traced = torch.jit.trace(wrapper, example, check_trace=False)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="frames", shape=example.shape, dtype=np.float16)],
        outputs=[
            ct.TensorType(name="rgb", dtype=np.float16),
            ct.TensorType(name="confidence", dtype=np.float16),
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    converted.user_defined_metadata.update(
        {
            "mioh.restorer": "v4q",
            "mioh.input_frames": "9",
            "mioh.output_frames": "5",
            "mioh.stride": "4",
            "mioh.imgsz": str(image_size),
        }
    )
    converted.save(str(output))


def export_coreai(
    model: MiohRestorerV4Q,
    image_size: int,
    output: Path,
    allow_overwrite: bool,
) -> None:
    import coreai_torch

    remove_existing(output, allow_overwrite)
    wrapper = MiohRestorerV4ExportWrapper(copy.deepcopy(model).half()).eval()
    example = torch.zeros(1, 36, image_size, image_size, dtype=torch.float16)
    exported = torch.export.export(wrapper, (example,))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported,
        input_names=["frames"],
        output_names=["rgb", "confidence"],
    )
    program = converter.to_coreai()
    program.optimize()
    program.save_asset(output)


def main() -> int:
    args = parse_args()
    if args.skip_coreml and args.skip_coreai:
        raise ValueError("at least one Apple output must remain enabled")
    model, payload = load_model(args.checkpoint, raw=args.raw_weights)
    config = payload["config"]
    image_size = int(config["image_size"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    version_label = "v41q" if model.high_resolution_detail else "v4q"
    stem = f"mioh-restorer-{version_label}-t9-o5-s{image_size}-fp16"
    coreml_path = args.output_dir / f"{stem}.mlpackage"
    coreai_path = args.output_dir / f"{stem}.aimodel"
    selected_path = args.output_dir / f"mioh-restorer-{version_label}.pth"
    report_path = args.output_dir / "export-report-v4.json"
    remove_existing(selected_path, args.allow_overwrite)
    selected_key = "state_dict" if args.raw_weights else "ema_state_dict"
    torch.save(
        {
            "state_dict": payload[selected_key],
            "config": config,
            "step": int(payload.get("step", 0)),
            "weights": "raw" if args.raw_weights else "ema",
        },
        selected_path,
    )
    report: dict[str, object] = {
        "step": int(payload.get("step", 0)),
        "weights": "raw" if args.raw_weights else "ema",
        "parameters": parameter_count(model),
        "contract": {
            "frames": [1, 36, image_size, image_size],
            "rgb": [1, 15, image_size, image_size],
            "confidence": [1, 5, image_size, image_size],
            "input_frames": 9,
            "output_frames": 5,
            "stride": 4,
        },
        "checkpoint": str(selected_path),
        "coreml": None,
        "coreai": None,
        "stage_seconds": {},
    }
    if not args.skip_coreml:
        started = time.perf_counter()
        export_coreml(model, image_size, coreml_path, args.allow_overwrite)
        report["coreml"] = str(coreml_path)
        report["stage_seconds"]["coreml"] = time.perf_counter() - started
    if not args.skip_coreai:
        started = time.perf_counter()
        export_coreai(model, image_size, coreai_path, args.allow_overwrite)
        report["coreai"] = str(coreai_path)
        report["stage_seconds"]["coreai"] = time.perf_counter() - started
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
