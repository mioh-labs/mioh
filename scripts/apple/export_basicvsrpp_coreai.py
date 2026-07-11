# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Probe fixed-shape FP16 BasicVSR++ conversion to a Core AI asset."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import platform
import re
import shutil
import time
import traceback
from collections import Counter
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

import torch

if __package__:
    from .basicvsrpp_coreai_kernels import (
        build_deform_conv_kernel,
        build_grid_sample_kernel,
        run_deform_conv_kernel,
        run_grid_sample_kernel,
    )
else:
    from basicvsrpp_coreai_kernels import (  # type: ignore[import-not-found]
        build_deform_conv_kernel,
        build_grid_sample_kernel,
        run_deform_conv_kernel,
        run_grid_sample_kernel,
    )

DEFAULT_MODEL = Path("model_weights/lada_mosaic_restoration_model_generic_v1.2.pth")
DEFAULT_OUTPUT = Path("model_weights/basicvsrpp-v1.2-t18-fp16.aimodel")
FIXED_FRAMES = 18
SUPPORTED_FRAME_COUNTS = (18, 36)
FIXED_IMAGE_SIZE = 256

_T = TypeVar("_T")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe fixed-shape FP16 BasicVSR++ export to Core AI"
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--frames",
        type=int,
        choices=SUPPORTED_FRAME_COUNTS,
        default=FIXED_FRAMES,
    )
    parser.add_argument("--imgsz", type=int, default=FIXED_IMAGE_SIZE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument(
        "--skip-reference-inference",
        action="store_true",
        help="Skip the slow CPU reference pass and probe conversion directly",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = Path(
            f"model_weights/basicvsrpp-v1.2-t{args.frames}-fp16.aimodel"
        )
    return args


def derive_report_path(output_path: Path) -> Path:
    name = output_path.name
    if name.endswith(".aimodel"):
        name = name.removesuffix(".aimodel")
    return output_path.with_name(f"{name}.report.json")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def checkpoint_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def new_report(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "report_version": 1,
        "success": False,
        "failed_stage": None,
        "error": None,
        "contract": {
            "shape": [1, args.frames, 3, args.imgsz, args.imgsz],
            "dtype": "float16",
            "channel_order": "lada-native",
        },
        "checkpoint": None,
        "output": str(args.output),
        "versions": {
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "coreai": _package_version("coreai"),
            "coreai-torch": _package_version("coreai-torch"),
        },
        "stage_seconds": {},
    }


def record_failure(report: dict[str, Any], stage: str, exc: BaseException) -> None:
    operators = sorted(
        set(
            re.findall(
                r"\b(?:aten|torchvision|coreai)(?:::|\.)[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*",
                str(exc),
            )
        )
    )
    report["success"] = False
    report["failed_stage"] = stage
    report["error"] = {
        "type": type(exc).__name__,
        "message": str(exc),
        "operators": operators,
        "traceback": "".join(traceback.format_exception(exc)),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class BasicVSRPPExportWrapper(torch.nn.Module):
    def __init__(self, generator: torch.nn.Module):
        super().__init__()
        self.generator = generator

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.generator(frames)


def make_example_input(frames: int, imgsz: int, seed: int) -> torch.Tensor:
    if frames <= 0 or imgsz <= 0:
        raise ValueError("frames and imgsz must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(
        (1, frames, 3, imgsz, imgsz),
        generator=generator,
        dtype=torch.float16,
    )


def validate_output(output: torch.Tensor, example: torch.Tensor) -> dict[str, Any]:
    if output.shape != example.shape:
        raise ValueError(
            f"output shape {tuple(output.shape)} does not match input {tuple(example.shape)}"
        )
    if output.dtype != torch.float16:
        raise ValueError(f"output dtype {output.dtype} is not float16")
    if not torch.isfinite(output).all().item():
        raise ValueError("reference output contains non-finite values")

    output_float = output.float()
    return {
        "shape": list(output.shape),
        "dtype": "float16",
        "min": float(output_float.amin().item()),
        "max": float(output_float.amax().item()),
        "mean": float(output_float.mean().item()),
    }


def summarize_exported_operators(
    exported: torch.export.ExportedProgram,
) -> dict[str, int]:
    counts = Counter(
        str(node.target)
        for node in exported.graph.nodes
        if node.op == "call_function"
    )
    return dict(sorted(counts.items()))


def load_generator(model_path: Path) -> BasicVSRPPExportWrapper:
    from lada.models.basicvsrpp.inference import load_model

    model = load_model(None, str(model_path), device="cpu", fp16=True)
    return BasicVSRPPExportWrapper(model.generator.eval()).eval()


def import_coreai() -> tuple[Any, Any]:
    try:
        import coreai
        import coreai_torch
    except ImportError as exc:
        raise RuntimeError(
            "Core AI export requires an isolated environment with coreai-torch installed"
        ) from exc
    return coreai, coreai_torch


def _run_stage(
    report: dict[str, Any],
    name: str,
    operation: Callable[[], _T],
    *,
    verbose: bool = False,
) -> _T:
    if verbose:
        print(f"[{name}] start", flush=True)
    started = time.perf_counter()
    try:
        return operation()
    finally:
        elapsed = round(time.perf_counter() - started, 6)
        report["stage_seconds"][name] = elapsed
        if verbose:
            print(f"[{name}] {elapsed:.3f}s", flush=True)


def _remove_existing_output(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def save_program_asset(program: Any, path: Path) -> None:
    program.save_asset(path)


def convert_exported_program(
    exported: torch.export.ExportedProgram,
    coreai_torch: Any,
    custom_kernels: list[Any],
):
    converter = coreai_torch.TorchConverter()
    converter.register_custom_kernels(custom_kernels)
    converter.add_exported_program(
        exported,
        input_names=["frames"],
        output_names=["restored"],
    )
    return converter.to_coreai()


@contextmanager
def use_grid_sample_metal_kernel(kernel: Any):
    flow_warp_module = importlib.import_module(
        "lada.models.basicvsrpp.mmagic.flow_warp"
    )
    original = flow_warp_module.safe_mps_grid_sample

    def grid_sample(image, grid, *, mode, padding_mode, align_corners):
        if (
            mode != "bilinear"
            or padding_mode not in {"zeros", "border"}
            or not align_corners
        ):
            raise ValueError(
                "Core AI flow warp supports only bilinear, zeros/border, align_corners=True"
            )
        return run_grid_sample_kernel(kernel, image, grid, padding_mode)

    flow_warp_module.safe_mps_grid_sample = grid_sample
    try:
        yield
    finally:
        flow_warp_module.safe_mps_grid_sample = original


@contextmanager
def use_deform_conv_metal_kernel(kernel: Any):
    basicvsrpp_module = importlib.import_module(
        "lada.models.basicvsrpp.mmagic.basicvsr_plusplus_net"
    )
    original = basicvsrpp_module.dispatch_deform_conv2d

    def deform_conv(image, offset, weight, bias, stride, padding, dilation, mask):
        def pair(value):
            return value if isinstance(value, tuple) else (value, value)

        if (
            pair(stride) != (1, 1)
            or pair(padding) != (1, 1)
            or pair(dilation) != (1, 1)
        ):
            raise ValueError(
                "Core AI deform conv supports only stride=1, padding=1, dilation=1"
            )
        if bias is None or mask is None:
            raise ValueError("Core AI deform conv requires bias and modulation mask")
        return run_deform_conv_kernel(kernel, image, offset, weight, bias, mask)

    basicvsrpp_module.dispatch_deform_conv2d = deform_conv
    try:
        yield
    finally:
        basicvsrpp_module.dispatch_deform_conv2d = original


def run_probe(args: argparse.Namespace) -> int:
    report_path = args.report or derive_report_path(args.output)
    report = new_report(args)
    stage = "preflight"

    try:
        def preflight() -> tuple[Any, Any]:
            if args.frames not in SUPPORTED_FRAME_COUNTS or args.imgsz != FIXED_IMAGE_SIZE:
                raise ValueError(
                    "this exporter supports fixed T18/T36 FP16 inputs at 256x256"
                )
            if not args.model.is_file():
                raise FileNotFoundError(args.model)
            if args.output.exists() and not args.allow_overwrite:
                raise FileExistsError(f"{args.output} exists; pass --allow-overwrite")
            report["checkpoint"] = checkpoint_identity(args.model)
            return import_coreai()

        coreai, coreai_torch = _run_stage(
            report, stage, preflight, verbose=args.verbose
        )
        del coreai

        stage = "build_custom_kernels"
        custom_kernels = _run_stage(
            report,
            stage,
            lambda: [
                build_grid_sample_kernel(coreai_torch),
                build_deform_conv_kernel(coreai_torch),
            ],
            verbose=args.verbose,
        )
        grid_sample_kernel, deform_conv_kernel = custom_kernels
        report["custom_kernels"] = [str(kernel.name) for kernel in custom_kernels]

        stage = "load_model"
        wrapper = _run_stage(
            report,
            stage,
            lambda: load_generator(args.model),
            verbose=args.verbose,
        )
        example = make_example_input(args.frames, args.imgsz, args.seed)

        if args.skip_reference_inference:
            report["reference_inference"] = "skipped"
        else:
            stage = "reference_inference"
            with torch.inference_mode():
                output = _run_stage(
                    report,
                    stage,
                    lambda: wrapper(example),
                    verbose=args.verbose,
                )
            report["reference_output"] = validate_output(output, example)
            del output

        stage = "torch_export"

        def export_model():
            with (
                use_grid_sample_metal_kernel(grid_sample_kernel),
                use_deform_conv_metal_kernel(deform_conv_kernel),
            ):
                return torch.export.export(wrapper, (example,))

        exported = _run_stage(
            report,
            stage,
            export_model,
            verbose=args.verbose,
        )

        stage = "decompose"
        exported = _run_stage(
            report,
            stage,
            lambda: exported.run_decompositions(coreai_torch.get_decomp_table()),
            verbose=args.verbose,
        )
        report["exported_operators"] = summarize_exported_operators(exported)

        stage = "coreai_convert"
        program = _run_stage(
            report,
            stage,
            lambda: convert_exported_program(
                exported,
                coreai_torch,
                custom_kernels,
            ),
            verbose=args.verbose,
        )

        stage = "optimize"
        _run_stage(report, stage, program.optimize, verbose=args.verbose)

        stage = "save_asset"
        _remove_existing_output(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        _run_stage(
            report,
            stage,
            lambda: save_program_asset(program, args.output),
            verbose=args.verbose,
        )

        report["success"] = True
        report["failed_stage"] = None
        report["error"] = None
        write_report(report_path, report)
        print(f"Core AI asset: {args.output}")
        print(f"Report: {report_path}")
        return 0
    except Exception as exc:
        record_failure(report, stage, exc)
        if args.verbose:
            traceback.print_exc()
        write_report(report_path, report)
        print(f"Core AI export failed at {stage}: {exc}")
        print(f"Report: {report_path}")
        return 1


def main(argv: list[str] | None = None) -> int:
    return run_probe(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
