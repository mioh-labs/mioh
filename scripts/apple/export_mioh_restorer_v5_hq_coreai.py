#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export the quality-first V5-HQ graph with Core AI Metal kernels."""

from __future__ import annotations

import argparse
import json
import shutil
import traceback
from collections import Counter
from pathlib import Path

import torch

from lada.models.mioh_restorer.model_v5_hq import (
    MiohRestorerV5HQ,
    MiohRestorerV5HQExportWrapper,
)
if __package__:
    from .basicvsrpp_coreai_kernels import (
        build_deform_conv_kernel,
        build_grid_sample_kernel,
    )
    from .export_basicvsrpp_coreai import (
        import_coreai,
        use_deform_conv_metal_kernel,
        use_grid_sample_metal_kernel,
    )
else:
    from basicvsrpp_coreai_kernels import (  # type: ignore[import-not-found]
        build_deform_conv_kernel,
        build_grid_sample_kernel,
    )
    from export_basicvsrpp_coreai import (  # type: ignore[import-not-found]
        import_coreai,
        use_deform_conv_metal_kernel,
        use_grid_sample_metal_kernel,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--basicvsrpp-checkpoint",
        type=Path,
        default=Path("model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="optional V5-HQ training checkpoint; EMA is selected by default",
    )
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--skip-reference-inference", action="store_true")
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> MiohRestorerV5HQExportWrapper:
    model = MiohRestorerV5HQ()
    model.load_basicvsrpp_checkpoint(args.basicvsrpp_checkpoint)
    if args.checkpoint is not None:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        expected = "hq"
        if payload.get("variant") != expected:
            raise ValueError(f"checkpoint is not V5-{expected.upper()}")
        state = payload["state_dict"] if args.raw else payload.get(
            "ema_state_dict", payload["state_dict"]
        )
        model.load_state_dict(state, strict=True)
    return MiohRestorerV5HQExportWrapper(model.half().eval()).eval()


def example_input(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.rand(1, 9, 5, 256, 256, generator=generator, dtype=torch.float16)
    values[:, :, 3:4] = (values[:, :, 3:4] > 0.75).to(values.dtype)
    values[:, :, 4:5] = 1
    return values


def remove_output(path: Path) -> None:
    if path.is_file() or path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def main() -> int:
    args = parse_args()
    report_path = args.report or args.output.with_suffix(".report.json")
    report: dict[str, object] = {
        "format": "mioh-restorer-v5-hq-coreai-v1",
        "success": False,
        "input": [1, 9, 5, 256, 256],
        "outputs": [[1, 15, 256, 256], [1, 5, 256, 256]],
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "basicvsrpp_checkpoint": str(args.basicvsrpp_checkpoint),
        "custom_kernels": [
            "grid_sample_bilinear_align_corners",
            "modulated_deform_conv2d",
        ],
    }
    stage = "preflight"
    try:
        if not args.basicvsrpp_checkpoint.is_file():
            raise FileNotFoundError(args.basicvsrpp_checkpoint)
        if args.checkpoint is not None and not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        if args.output.exists() and not args.allow_overwrite:
            raise FileExistsError(args.output)
        _coreai, coreai_torch = import_coreai()
        kernels = [
            build_grid_sample_kernel(coreai_torch),
            build_deform_conv_kernel(coreai_torch),
        ]
        grid_kernel, deform_kernel = kernels
        stage = "load_model"
        wrapper = load_model(args)
        values = example_input(args.seed)
        if not args.skip_reference_inference:
            stage = "reference_inference"
            with torch.inference_mode():
                rgb, confidence = wrapper(values)
            report["reference"] = {
                "rgb_finite": bool(torch.isfinite(rgb).all()),
                "confidence_finite": bool(torch.isfinite(confidence).all()),
                "rgb_shape": list(rgb.shape),
                "confidence_shape": list(confidence.shape),
            }
        stage = "torch_export"
        with (
            use_grid_sample_metal_kernel(grid_kernel),
            use_deform_conv_metal_kernel(deform_kernel),
        ):
            exported = torch.export.export(wrapper, (values,))
        stage = "decompose"
        exported = exported.run_decompositions(coreai_torch.get_decomp_table())
        report["operators"] = dict(
            sorted(
                Counter(
                    str(node.target)
                    for node in exported.graph.nodes
                    if node.op == "call_function"
                ).items()
            )
        )
        stage = "coreai_convert"
        converter = coreai_torch.TorchConverter()
        converter.register_custom_kernels(kernels)
        converter.add_exported_program(
            exported,
            input_names=["frames"],
            output_names=["rgb", "confidence"],
        )
        program = converter.to_coreai()
        stage = "optimize"
        program.optimize()
        stage = "save"
        remove_output(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        program.save_asset(args.output)
        report["success"] = True
        report["output"] = str(args.output)
        return_code = 0
    except Exception as error:
        report["failed_stage"] = stage
        report["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        }
        return_code = 1
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
