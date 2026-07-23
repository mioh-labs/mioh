#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Measure V5 model-only training cost before scheduling full stages."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from lada.models.mioh_restorer.losses_v5 import MiohRestorerV5Loss
from lada.models.mioh_restorer.model_v5 import (
    FRAME_CHANNELS,
    NUM_INPUT_FRAMES,
    MiohRestorerV5,
    MiohRestorerV5Config,
    V5_BUCKETS,
    parameter_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("q", "s"), default="q")
    parser.add_argument("--sizes", default="128,192,256")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"), default="mps")
    parser.add_argument("--outputs", type=int, choices=(1, 3, 5), default=1)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item) for item in value.split(",") if item)
    if not sizes or any(size not in V5_BUCKETS for size in sizes):
        raise ValueError(f"sizes must be selected from {V5_BUCKETS}")
    return sizes


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def memory_report(device: torch.device) -> dict[str, int]:
    if device.type == "mps":
        return {
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
        }
    if device.type == "cuda":
        return {
            "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    return {}


def profile_size(
    variant: str,
    size: int,
    *,
    steps: int,
    warmup: int,
    device: torch.device,
    outputs: int,
) -> dict[str, object]:
    if variant == "q":
        indices = {1: (4,), 3: (3, 4, 5), 5: (2, 3, 4, 5, 6)}[outputs]
        model = MiohRestorerV5(
            replace(MiohRestorerV5Config.quality(), output_indices=indices)
        )
    else:
        if outputs != 1:
            raise ValueError("V5-S has one output")
        model = MiohRestorerV5.shipping()
    model = model.to(device).train()
    # Stage 3 exercises the complete model graph without downloading VGG16.
    # Stage 4 adds perceptual-loss cost and is measured by the real trainer.
    loss_function = MiohRestorerV5Loss(stage=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    durations: list[float] = []
    for step in range(warmup + steps):
        values = torch.rand(
            1, NUM_INPUT_FRAMES, FRAME_CHANNELS, size, size, device=device
        )
        values[:, :, 3:4] = (values[:, :, 3:4] > 0.65).to(values.dtype)
        values[:, :, 4:5] = 1
        output_indices = model.config.output_indices
        source = torch.stack([values[:, index, :3] for index in output_indices], dim=1)
        mask = torch.stack([values[:, index, 3:4] for index in output_indices], dim=1)
        target = torch.clamp(source + mask * torch.randn_like(source) * 0.05, 0, 1)
        started = time.perf_counter()
        restored, confidence, base, texture = model.forward_components(values)
        loss, _stats = loss_function(
            restored, confidence, base, texture, target, source, mask
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        synchronize(device)
        if step >= warmup:
            durations.append(time.perf_counter() - started)
    values = np.asarray(durations)
    result = {
        "steps": steps,
        "seconds_per_step_median": float(np.median(values)),
        "seconds_per_step_p90": float(np.percentile(values, 90)),
        "projected_hours_78000_steps": float(np.median(values) * 78_000 / 3600),
        "memory": memory_report(device),
    }
    del optimizer, loss_function, model
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> int:
    args = parse_args()
    if args.steps <= 0 or args.warmup < 0:
        raise ValueError("step counts are invalid")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    report: dict[str, object] = {
        "variant": args.variant,
        "device": str(device),
        "note": "model-only synthetic estimate; teacher, codec and data I/O are excluded",
        "sizes": {},
    }
    for size in parse_sizes(args.sizes):
        print(f"profiling V5-{args.variant.upper()} {size}", flush=True)
        result = profile_size(
            args.variant,
            size,
            steps=args.steps,
            warmup=args.warmup,
            device=device,
            outputs=args.outputs,
        )
        report["sizes"][str(size)] = result
        print(json.dumps(result, indent=2), flush=True)
    model = MiohRestorerV5.quality() if args.variant == "q" else MiohRestorerV5.shipping()
    report["parameters"] = parameter_count(model)
    report["training_outputs"] = args.outputs
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
