# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Validate and benchmark a fixed-shape RF-DETR Seg Core AI asset."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lada.coreai.compiled_runtime import CompiledCoreAIRuntime, TensorSpec

if __package__:
    from .export_rfdetr_seg_coreai import download_or_load_model, make_example
else:
    from export_rfdetr_seg_coreai import (  # type: ignore[import-not-found]
        download_or_load_model,
        make_example,
    )


DEFAULT_WEIGHTS = Path("model_weights/3rd_party/rf-detr-seg-small.pt")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RF-DETR Seg Core AI against PyTorch/MPS"
    )
    parser.add_argument(
        "model",
        type=Path,
        help="portable .aimodel or FP16 compiled .aimodelc directory",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variant",
        choices=("small", "medium", "large"),
        default="small",
    )
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--logit-classes", type=int, default=91)
    parser.add_argument(
        "--image",
        type=Path,
        help="optional real image; resized and ImageNet-normalized",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="validate the experimental FP16 model instead of FP32",
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def error_metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, float]:
    difference = np.abs(
        actual.astype(np.float32) - expected.astype(np.float32)
    )
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
    }


def load_image(
    path: Path,
    *,
    fp16: bool = False,
    resolution: int = 384,
) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize(
        (resolution, resolution),
        Image.Resampling.BICUBIC,
    )
    value = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
    value = value.div(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    value = value.sub(mean).div(std).unsqueeze(0)
    return value.half() if fp16 else value


def infer_compiled(
    args: argparse.Namespace,
    value: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[float]]:
    if not args.fp16:
        raise ValueError("the shared-memory compiled runner currently supports FP16")
    if args.runner is None:
        raise ValueError("--runner is required for a compiled .aimodelc")
    runtime = CompiledCoreAIRuntime(
        args.model,
        (TensorSpec("image", (1, 3, args.resolution, args.resolution)),),
        (
            TensorSpec("boxes", (1, args.queries, 4)),
            TensorSpec("logits", (1, args.queries, args.logit_classes)),
            TensorSpec(
                "masks",
                (1, args.queries, args.resolution // 4, args.resolution // 4),
            ),
        ),
        runner_path=str(args.runner),
    )
    durations: list[float] = []
    result: dict[str, np.ndarray] | None = None
    try:
        for index in range(args.warmup + args.runs):
            started = time.perf_counter()
            result = runtime.infer({"image": value})
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if index >= args.warmup:
                durations.append(elapsed_ms)
    finally:
        runtime.close()
    assert result is not None
    return result, durations


def infer_source(
    args: argparse.Namespace,
    value: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[float]]:
    from coreai.runtime import AIModel, NDArray

    runner = asyncio.Runner()
    try:
        model = runner.run(AIModel.load(args.model))
        function = model.load_function("main")

        async def invoke() -> dict[str, np.ndarray]:
            outputs = await function({"image": NDArray(value)})
            return {
                name: outputs[name].numpy().copy()
                for name in ("boxes", "logits", "masks")
            }

        durations: list[float] = []
        result: dict[str, np.ndarray] | None = None
        for index in range(args.warmup + args.runs):
            started = time.perf_counter()
            result = runner.run(invoke())
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if index >= args.warmup:
                durations.append(elapsed_ms)
    finally:
        runner.close()
    assert result is not None
    return result, durations


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runs < 1 or args.warmup < 0:
        raise ValueError("runs must be positive and warmup must not be negative")
    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is required for the reference run")

    value = (
        load_image(
            args.image,
            fp16=args.fp16,
            resolution=args.resolution,
        )
        if args.image
        else make_example(
            args.seed,
            fp16=args.fp16,
            resolution=args.resolution,
        )
    )
    model = download_or_load_model(
        args.weights,
        fp16=args.fp16,
        variant=args.variant,
        resolution=args.resolution,
    )
    with torch.inference_mode():
        reference = tuple(item.numpy() for item in model(value))

        model = model.to("mps")
        mps_value = value.to("mps")
        # Warm once so graph compilation is not included in the timing.
        model(mps_value)
        torch.mps.synchronize()
        started = time.perf_counter()
        mps_output = tuple(item.cpu().numpy() for item in model(mps_value))
        torch.mps.synchronize()
        mps_duration_ms = (time.perf_counter() - started) * 1000.0

    input_value = np.ascontiguousarray(value.numpy())
    if args.model.name.endswith(".aimodelc"):
        result, durations = infer_compiled(args, input_value)
    else:
        result, durations = infer_source(args, input_value)
    output_names = ("boxes", "logits", "masks")
    errors_vs_cpu = {
        name: error_metrics(result[name], expected)
        for name, expected in zip(output_names, reference, strict=True)
    }
    mps_errors_vs_cpu = {
        name: error_metrics(actual, expected)
        for name, actual, expected in zip(
            output_names,
            mps_output,
            reference,
            strict=True,
        )
    }
    errors_vs_mps = {
        name: error_metrics(result[name], actual)
        for name, actual in zip(output_names, mps_output, strict=True)
    }
    report = {
        "model": str(args.model),
        "weights": str(args.weights),
        "variant": args.variant,
        "resolution": args.resolution,
        "dtype": "float16" if args.fp16 else "float32",
        "seed": args.seed,
        "image": str(args.image) if args.image else None,
        "mps_reference_ms": mps_duration_ms,
        "coreai": {
            "runs": args.runs,
            "warmup": args.warmup,
            "median_ms": statistics.median(durations),
            "mean_ms": statistics.fmean(durations),
            "min_ms": min(durations),
            "max_ms": max(durations),
        },
        "errors_vs_cpu": errors_vs_cpu,
        "errors_vs_mps": errors_vs_mps,
        "mps_errors_vs_cpu": mps_errors_vs_cpu,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
