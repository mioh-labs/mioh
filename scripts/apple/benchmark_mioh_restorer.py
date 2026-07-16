# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Benchmark the MiohRestorerV1 prototype across Apple runtimes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from lada.models.mioh_restorer import MiohRestorerV1
from lada.restorationpipeline.mioh_restorer import (
    CoreAIMiohRestorerRuntime,
    CoreMLMiohRestorerRuntime,
    TorchMiohRestorerRuntime,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MiohRestorerV1")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--coreml-model", type=Path)
    parser.add_argument("--coreai-model", type=Path)
    parser.add_argument("--coreai-runner", type=str)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def load_model(checkpoint_path: Path) -> tuple[MiohRestorerV1, dict]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = payload["config"]
    model = MiohRestorerV1(
        chunk_frames=config["chunk_frames"],
        channels=config["channels"],
        num_blocks=config["num_blocks"],
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    config = dict(config)
    config["trained"] = bool(payload.get("trained", False)) or int(
        payload.get("step", 0)
    ) > 0
    return model.eval(), config


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def benchmark_runtime(
    name: str,
    runtime,
    frames: torch.Tensor,
    masks: torch.Tensor,
    state: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> dict:
    current_state = state
    for _ in range(warmup):
        _restored, current_state = runtime(frames, masks, current_state)
    device = getattr(runtime, "device", torch.device("cpu"))
    synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        _restored, current_state = runtime(frames, masks, current_state)
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "runtime": name,
        "iterations": iterations,
        "elapsed_seconds": round(elapsed, 6),
        "milliseconds_per_chunk": round(elapsed * 1000 / iterations, 3),
        "frames_per_second": round(iterations * runtime.chunk_frames / elapsed, 3),
        "realtime_factor_29_97fps": round(
            iterations * runtime.chunk_frames / elapsed / (30000 / 1001), 3
        ),
    }


def compare_runtime(
    runtime,
    reference: tuple[torch.Tensor, torch.Tensor],
    frames: torch.Tensor,
    masks: torch.Tensor,
    state: torch.Tensor,
) -> dict:
    restored, next_state = runtime(frames, masks, state)
    reference_restored, reference_state = reference
    return {
        "restored_max_abs_error": float(
            (restored.float().cpu() - reference_restored.float().cpu()).abs().max()
        ),
        "state_max_abs_error": float(
            (next_state.float().cpu() - reference_state.float().cpu()).abs().max()
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.iterations <= 0 or args.warmup < 0:
        raise ValueError("iterations must be positive and warmup must not be negative")
    model, config = load_model(args.checkpoint)
    image_size = config["image_size"]
    generator = torch.Generator().manual_seed(7)
    frames = torch.rand(
        (1, model.chunk_frames, 3, image_size, image_size),
        generator=generator,
        dtype=torch.float16,
    )
    masks = torch.rand(
        (1, model.chunk_frames, 1, image_size, image_size),
        generator=generator,
        dtype=torch.float16,
    )
    state = torch.zeros(
        model.state_shape(image_height=image_size, image_width=image_size),
        dtype=torch.float16,
    )

    cpu_reference_model = model.half().eval()
    with torch.inference_mode():
        reference = cpu_reference_model(frames, masks, state)
    report = {
        "prototype": not config["trained"],
        "trained": config["trained"],
        "image_size": image_size,
        "chunk_frames": model.chunk_frames,
        "results": [],
    }

    if torch.backends.mps.is_available():
        mps_runtime = TorchMiohRestorerRuntime(model.half(), "mps")
        result = benchmark_runtime(
            "pytorch-mps",
            mps_runtime,
            frames,
            masks,
            state,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        result.update(compare_runtime(mps_runtime, reference, frames, masks, state))
        report["results"].append(result)

    if args.coreml_model is not None:
        coreml_runtime = CoreMLMiohRestorerRuntime(
            args.coreml_model,
            chunk_frames=model.chunk_frames,
            channels=model.channels,
            image_size=image_size,
        )
        result = benchmark_runtime(
            "coreml",
            coreml_runtime,
            frames,
            masks,
            state,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        result.update(compare_runtime(coreml_runtime, reference, frames, masks, state))
        report["results"].append(result)

    if args.coreai_model is not None:
        coreai_runtime = CoreAIMiohRestorerRuntime(
            args.coreai_model,
            chunk_frames=model.chunk_frames,
            channels=model.channels,
            image_size=image_size,
            runner_path=args.coreai_runner,
        )
        try:
            result = benchmark_runtime(
                "coreai",
                coreai_runtime,
                frames,
                masks,
                state,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            result.update(compare_runtime(coreai_runtime, reference, frames, masks, state))
            report["results"].append(result)
        finally:
            coreai_runtime.close()

    text = json.dumps(report, indent=2) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
