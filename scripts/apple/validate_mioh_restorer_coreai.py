#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Run a compiled MiohRestorer model and report real Core AI throughput."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from lada.coreai.compiled_runtime import CompiledCoreAIRuntime, TensorSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--warm-runs", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.frames <= 1 or args.size <= 0 or args.size % 4:
        raise ValueError("frames must exceed one and size must be divisible by four")
    if args.warm_runs <= 0:
        raise ValueError("warm-runs must be positive")
    frame_shape = (1, args.frames, 3, args.size, args.size)
    mask_shape = (1, args.frames, 1, args.size, args.size)
    frames = np.full(frame_shape, 0.25, dtype=np.float16)
    masks = np.ones(mask_shape, dtype=np.float16)
    runtime = CompiledCoreAIRuntime(
        args.model,
        (TensorSpec("frames", frame_shape), TensorSpec("masks", mask_shape)),
        (TensorSpec("restored", frame_shape),),
        runner_path=str(args.runner) if args.runner else None,
    )
    durations: list[float] = []
    outputs: list[np.ndarray] = []
    try:
        for _ in range(args.warm_runs + 1):
            started = time.perf_counter()
            outputs.append(
                runtime.infer({"frames": frames, "masks": masks})["restored"]
            )
            durations.append(time.perf_counter() - started)
    finally:
        runtime.close()
    output = outputs[-1]
    if not np.isfinite(output).all():
        raise RuntimeError("compiled Core AI output contains NaN or infinity")
    warm_seconds = sum(durations[1:]) / len(durations[1:])
    report = {
        "model": str(args.model.resolve()),
        "shape": list(output.shape),
        "finite": True,
        "output": {
            "minimum": float(output.min()),
            "maximum": float(output.max()),
            "mean": float(output.mean()),
        },
        "first_seconds": durations[0],
        "warm_seconds": warm_seconds,
        "fps": args.frames / warm_seconds,
        "repeat_max_abs": float(
            np.max(
                np.abs(
                    outputs[-1].astype(np.float32)
                    - outputs[-2].astype(np.float32)
                )
            )
        ),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
