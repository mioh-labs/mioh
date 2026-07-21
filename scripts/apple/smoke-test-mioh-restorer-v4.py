#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Convert untrained V4 variants before spending time on training.

The smoke test verifies graph conversion, numerical agreement, Apple compute
placement and warm latency.  It can also build and execute a Core AI asset on
the current Mac when ``--coreai`` is supplied.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from lada.coreai.compiled_runtime import CompiledCoreAIRuntime, TensorSpec
from lada.models.mioh_restorer.model_v4 import (
    MiohRestorerV4ExportWrapper,
    MiohRestorerV4Q,
    parameter_count,
)


CONFIGURATIONS = (
    ("hier27", "batch"),
    ("hier27", "serial"),
    ("hier27", "center1"),
    ("full121", "batch"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/v4-smoke"))
    parser.add_argument("--only", help="variant:mode, for example hier27:center1")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--skip-coreml", action="store_true")
    parser.add_argument("--coreai", action="store_true")
    parser.add_argument("--coreai-architecture", default="h17s")
    parser.add_argument("--coreai-preferred-compute", default="gpu")
    parser.add_argument(
        "--coreai-runner",
        type=Path,
        default=Path("build/macos-standalone/mioh.app/Contents/Resources/bin/lada-coreai-runner"),
    )
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args()


def configurations(value: str | None) -> tuple[tuple[str, str], ...]:
    if value is None:
        return CONFIGURATIONS
    variant, separator, mode = value.partition(":")
    if not separator or (variant, mode) not in CONFIGURATIONS:
        raise ValueError(f"unsupported smoke configuration: {value}")
    return ((variant, mode),)


def remove_existing(path: Path, allow: bool) -> None:
    if not path.exists():
        return
    if not allow:
        raise FileExistsError(f"{path} exists; pass --allow-overwrite")
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def build(
    variant: str,
    mode: str,
    size: int,
) -> tuple[MiohRestorerV4Q, MiohRestorerV4ExportWrapper, torch.Tensor, tuple[np.ndarray, np.ndarray]]:
    torch.manual_seed(0)
    model = MiohRestorerV4Q(
        alignment_variant=variant, execution_mode=mode
    ).eval()
    wrapper = MiohRestorerV4ExportWrapper(model).eval()
    flat = torch.rand(1, 36, size, size)
    with torch.no_grad():
        reference = tuple(value.numpy() for value in wrapper(flat))
    return model, wrapper, flat, reference


def convert_coreml(
    wrapper: MiohRestorerV4ExportWrapper,
    flat: torch.Tensor,
    output: Path,
    allow_overwrite: bool,
):
    import coremltools as ct

    remove_existing(output, allow_overwrite)
    traced = torch.jit.trace(wrapper, flat, check_trace=False)
    converted = ct.convert(
        traced,
        inputs=[
            ct.TensorType(
                name="frames", shape=tuple(flat.shape), dtype=np.float16
            )
        ],
        outputs=[
            ct.TensorType(name="rgb", dtype=np.float16),
            ct.TensorType(name="confidence", dtype=np.float16),
        ],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.macOS15,
    )
    converted.user_defined_metadata["mioh.restorer"] = "v4q"
    converted.user_defined_metadata["mioh.imgsz"] = str(flat.shape[-1])
    converted.save(str(output))
    return converted


def all_operations(block):
    for operation in block.operations:
        yield operation
        for child in getattr(operation, "blocks", ()):
            yield from all_operations(child)


def compute_plan(path: Path) -> dict[str, int]:
    import coremltools as ct

    compiled_path = path.with_suffix(".mlmodelc")
    if compiled_path.exists():
        shutil.rmtree(compiled_path)
    ct.models.utils.compile_model(str(path), destination_path=str(compiled_path))
    plan = ct.models.compute_plan.MLComputePlan.load_from_path(
        str(compiled_path), compute_units=ct.ComputeUnit.ALL
    )
    function = plan.model_structure.program.functions["main"]
    counts: dict[str, int] = {}
    for operation in all_operations(function.block):
        usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
        if usage is None:
            continue
        name = type(usage.preferred_compute_device).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


def benchmark_coreml(model, value: np.ndarray, runs: int) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    for _ in range(3):
        model.predict({"frames": value})
    durations = []
    result = None
    for _ in range(runs):
        started = time.perf_counter()
        result = model.predict({"frames": value})
        durations.append((time.perf_counter() - started) * 1_000)
    assert result is not None
    values = np.asarray(durations)
    return result, {
        "median_ms": float(np.median(values)),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
    }


def numeric_report(
    result: dict[str, np.ndarray], reference: tuple[np.ndarray, np.ndarray]
) -> dict[str, dict[str, float]]:
    report = {}
    for name, expected in zip(("rgb", "confidence"), reference, strict=True):
        difference = np.abs(np.asarray(result[name], dtype=np.float32) - expected)
        report[name] = {
            "maximum_absolute": float(difference.max()),
            "mean_absolute": float(difference.mean()),
            "psnr": float(
                -10 * np.log10(max(float(np.mean(difference**2)), 1e-12))
            ),
        }
    return report


def convert_coreai(
    wrapper: MiohRestorerV4ExportWrapper,
    flat: torch.Tensor,
    output: Path,
    allow_overwrite: bool,
) -> None:
    import coreai_torch

    remove_existing(output, allow_overwrite)
    fp16_wrapper = copy.deepcopy(wrapper).half().eval()
    example = flat.half()
    exported = torch.export.export(fp16_wrapper, (example,))
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


def compile_coreai(args: argparse.Namespace, source: Path, output_dir: Path) -> Path:
    compiled = output_dir / f"{source.stem}.{args.coreai_architecture}.aimodelc"
    remove_existing(compiled, args.allow_overwrite)
    subprocess.run(
        [
            "xcrun",
            "coreai-build",
            "compile",
            str(source),
            "--output",
            str(output_dir),
            "--platform",
            "macOS",
            "--min-deployment-version",
            "27.0",
            "--preferred-compute",
            args.coreai_preferred_compute,
            "--architecture",
            args.coreai_architecture,
        ],
        check=True,
    )
    generated = output_dir / f"{source.stem}.aimodelc"
    if generated.exists() and generated != compiled:
        generated.replace(compiled)
    if not compiled.is_dir():
        candidates = list(output_dir.glob(f"{source.stem}*.aimodelc"))
        if len(candidates) != 1:
            raise FileNotFoundError(f"compiled Core AI model not found for {source}")
        candidates[0].replace(compiled)
    return compiled


def benchmark_coreai(
    compiled: Path,
    runner: Path,
    value: np.ndarray,
    output_frames: int,
    runs: int,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    size = value.shape[-1]
    runtime = CompiledCoreAIRuntime(
        compiled,
        (TensorSpec("frames", tuple(value.shape)),),
        (
            TensorSpec("rgb", (1, output_frames * 3, size, size)),
            TensorSpec("confidence", (1, output_frames, size, size)),
        ),
        runner_path=str(runner),
    )
    durations: list[float] = []
    result = None
    try:
        # Core AI's first few calls include pipeline and device warmup.  Five
        # warm calls were needed for stable 384px measurements on M5 Pro.
        for index in range(runs + 5):
            started = time.perf_counter()
            result = runtime.infer({"frames": np.ascontiguousarray(value)})
            if index >= 5:
                durations.append((time.perf_counter() - started) * 1_000)
    finally:
        runtime.close()
    assert result is not None
    values = np.asarray(durations)
    return result, {
        "median_ms": float(np.median(values)),
        "p10_ms": float(np.percentile(values, 10)),
        "p90_ms": float(np.percentile(values, 90)),
    }


def main() -> int:
    args = parse_args()
    if args.image_size <= 0 or args.image_size % 8:
        raise ValueError("image-size must be positive and divisible by eight")
    if args.runs <= 0:
        raise ValueError("runs must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for variant, mode in configurations(args.only):
        tag = f"{variant}-{mode}"
        print(f"=== {tag} ===", flush=True)
        model, wrapper, flat, reference = build(variant, mode, args.image_size)
        output_frames = 1 if mode == "center1" else 5
        report: dict[str, object] = {
            "configuration": tag,
            "parameters": parameter_count(model),
            "input_shape": list(flat.shape),
            "output_frames": output_frames,
        }
        input_value = flat.numpy().astype(np.float16)
        if not args.skip_coreml:
            coreml_path = args.output_dir / f"v4q-{tag}.mlpackage"
            coreml = convert_coreml(wrapper, flat, coreml_path, args.allow_overwrite)
            result, latency = benchmark_coreml(coreml, input_value, args.runs)
            report["coreml"] = {
                "path": str(coreml_path),
                "compute_plan": compute_plan(coreml_path),
                "latency": latency,
                "numeric": numeric_report(result, reference),
            }
        if args.coreai:
            coreai_path = args.output_dir / f"v4q-{tag}.aimodel"
            convert_coreai(wrapper, flat, coreai_path, args.allow_overwrite)
            compiled = compile_coreai(args, coreai_path, args.output_dir)
            result, latency = benchmark_coreai(
                compiled,
                args.coreai_runner,
                input_value,
                output_frames,
                args.runs,
            )
            report["coreai"] = {
                "path": str(coreai_path),
                "compiled": str(compiled),
                "latency": latency,
                "numeric": numeric_report(result, reference),
            }
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)
    report_path = args.output_dir / "smoke-report.json"
    report_path.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
    print(f"saved: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
