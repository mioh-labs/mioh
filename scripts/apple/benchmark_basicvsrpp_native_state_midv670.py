#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Run an end-to-end native-state A/B on the MIDV-670 mosaic ROI clip."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.apple.benchmark_basicvsrpp_native_state import (
    compile_asset,
    export_stateful,
)
from scripts.apple.benchmark_basicvsrpp_variable_coreai import BRANCHES
from scripts.apple.export_basicvsrpp_variable_chunk6 import export_assets

VARIABLE_RUNNER_SOURCE = (
    ROOT / "packaging/macOS/standalone/VariableBasicVSRPPChunk6Runner.swift"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT
        / "output/evaluations/"
        "basicvsrpp-large-roi-step21000-vs-step27000-midv670-20260818/"
        "MIDV-670-2h18m00s-source-10s.mp4",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT
        / "model_weights/basicvsrpp-v1.2-detail-recovery-30000-ema.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/mioh-basicvsrpp-native-state-midv670"),
    )
    parser.add_argument("--architecture", default="h17s")
    parser.add_argument(
        "--preview-executable",
        type=Path,
        default=ROOT
        / "build/macos-standalone/mioh.app/Contents/Resources/bin/"
        "mioh-native-coreai-preview",
    )
    parser.add_argument(
        "--reuse-assets",
        type=Path,
        help="Reuse control/native compiled assets and the experimental runner",
    )
    parser.add_argument("--runs-per-mode", type=int, default=2)
    parser.add_argument("--temporal-frames", type=int, default=36)
    parser.add_argument("--temporal-overlap", type=int, default=8)
    parser.add_argument("--ring-capacity", type=int)
    return parser.parse_args()


def compile_runner(destination: Path) -> None:
    subprocess.run(
        [
            "xcrun",
            "swiftc",
            "-O",
            "-parse-as-library",
            "-target",
            "arm64-apple-macosx27.0",
            "-framework",
            "CoreAI",
            "-framework",
            "Metal",
            str(VARIABLE_RUNNER_SOURCE),
            "-o",
            str(destination),
        ],
        check=True,
    )


def link_control_assets(control: Path, candidate: Path) -> None:
    candidate.mkdir(parents=True)
    for asset in sorted(control.iterdir()):
        if asset.name.endswith(".aimodelc"):
            (candidate / asset.name).symlink_to(asset.resolve(), target_is_directory=True)


def build_assets(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.reuse_assets is not None:
        root = args.reuse_assets.resolve()
        control = root / "control-compiled"
        candidate = root / "native-state-compiled"
        runner = root / "lada-basicvsrpp-variable-native-state-runner"
        for path in (control, candidate, runner):
            if not path.exists():
                raise FileNotFoundError(path)
        return control, candidate, runner
    source_assets = args.output_dir / "control-source"
    control = args.output_dir / "control-compiled"
    candidate = args.output_dir / "native-state-compiled"
    runner = args.output_dir / "lada-basicvsrpp-variable-native-state-runner"
    export_assets(
        args.checkpoint,
        source_assets,
        overwrite=False,
        optimize=True,
    )
    control.mkdir()
    for source in sorted(source_assets.glob("*.aimodel")):
        compile_asset(source, control, args.architecture)
    link_control_assets(control, candidate)
    for branch in BRANCHES:
        name = f"basicvsrpp-variable-{branch}_continue6"
        linked = candidate / f"{name}.{args.architecture}.aimodelc"
        linked.unlink()
        source = args.output_dir / f"{name}-stateful.aimodel"
        export_stateful(args.checkpoint, source, branch=branch)
        compiled = compile_asset(source, candidate, args.architecture)
        expected = candidate / f"{name}.{args.architecture}.aimodelc"
        if compiled != expected:
            compiled.rename(expected)
    compile_runner(runner)
    return control, candidate, runner


def configuration(
    *,
    args: argparse.Namespace,
    models: Path,
    runner: Path,
    run_dir: Path,
) -> dict[str, object]:
    resources = ROOT / "build/macos-standalone/mioh.app/Contents/Resources"
    return {
        "mode": "export",
        "input": str(args.source.resolve()),
        "outputDirectory": str(run_dir),
        "ffmpegTemporaryDirectory": str(run_dir / "ffmpeg"),
        "miohTemporaryDirectory": str(run_dir / "mioh"),
        "outputFile": str(run_dir / "output.mp4"),
        "ffmpeg": str(resources / "bin/ffmpeg"),
        "detectionModel": str(
            resources
            / "models/lada_mosaic_detection_model_v4_fast-fp16.h17s.aimodelc"
        ),
        "detectionBackend": "yolo",
        "detectionInputSize": 640,
        "detectionCandidateChannels": 38,
        "detectionMaxDet": 16,
        "detectionComputeUnits": "cpuAndGPU",
        "restorationModels": str(models.resolve()),
        "restorationRunner": str(runner.resolve()),
        "startNanoseconds": 0,
        "decodeEndNanoseconds": 10_000_000_000,
        "workerMode": False,
        "generation": 0,
        "splitMode": "none",
        "segmentCount": 1,
        "segmentSeconds": 60.0,
        "bufferLimitSeconds": 1.0,
        "temporalBatchFrames": args.temporal_frames,
        "temporalOverlap": args.temporal_overlap,
        "ringCapacity": args.ring_capacity
        or max(args.temporal_frames + args.temporal_overlap + 8, 64),
        "nativeParallelWorkers": 1,
        "confidenceThreshold": 0.25,
        "iouThreshold": 0.7,
        "contextFraction": 0.30,
        "blendFeather": 1.0,
        "sharpenStrength": 0.0,
        "detailBoost": 0.0,
        "textureMix": 0.0,
        "smoothStrength": 0.0,
        "effectUpscale": 1,
        "roiEnhancerStrength": 0.0,
        "roiEnhancerScale": 4,
        "detectionEmptyLookahead": 10,
        "detectionMaskReuseSkipFrames": 0,
        "detectFaceMosaics": False,
        "crossfade": True,
        "targetFPS": 30000,
        "targetFPSDenominator": 1001,
        "preFPSConversion": True,
        "videoCodec": "h264",
        "bitrateMultiplier": 1.5,
        "mp4FastStart": True,
    }


def run_pipeline(
    args: argparse.Namespace,
    mode: str,
    index: int,
    models: Path,
    runner: Path,
) -> dict[str, object]:
    run_dir = args.output_dir / f"{mode}-{index}"
    run_dir.mkdir(parents=True)
    (run_dir / "ffmpeg").mkdir()
    (run_dir / "mioh").mkdir()
    config = configuration(args=args, models=models, runner=runner, run_dir=run_dir)
    config_path = run_dir / "configuration.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    log_path = run_dir / "runner.log"
    executable = args.preview_executable.resolve()
    started = time.perf_counter()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [str(executable), str(config_path)],
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            returncode = process.wait()
        finally:
            if process.stdin is not None:
                process.stdin.close()
    elapsed = time.perf_counter() - started
    if returncode:
        tail = "\n".join(log_path.read_text(errors="replace").splitlines()[-40:])
        raise RuntimeError(f"{mode} run failed ({returncode}):\n{tail}")
    output = run_dir / "output.mp4"
    if not output.is_file():
        raise FileNotFoundError(output)
    events = [
        json.loads(line)
        for line in log_path.read_text(errors="replace").splitlines()
        if line.startswith("{")
    ]
    native_stats = next(
        (event for event in reversed(events) if event.get("kind") == "native_stats"),
        None,
    )
    return {
        "mode": mode,
        "index": index,
        "seconds": elapsed,
        "output": str(output.resolve()),
        "log": str(log_path.resolve()),
        "native_stats": native_stats,
    }


def compare_videos(source: Path, control: Path, candidate: Path) -> dict[str, object]:
    captures = [cv2.VideoCapture(str(path)) for path in (source, control, candidate)]
    if not all(capture.isOpened() for capture in captures):
        raise RuntimeError("could not open one of the A/B videos")
    frame_count = 0
    pixel_count = 0
    active_count = 0
    difference_sum = 0.0
    difference_square_sum = 0.0
    active_difference_sum = 0.0
    maximum = 0.0
    temporal_difference_sum = 0.0
    temporal_pixel_count = 0
    previous_control: np.ndarray | None = None
    previous_candidate: np.ndarray | None = None
    try:
        while True:
            decoded = [capture.read() for capture in captures]
            if not all(ok for ok, _frame in decoded):
                if any(ok for ok, _frame in decoded):
                    raise RuntimeError("A/B frame counts differ")
                break
            source_frame, control_frame, candidate_frame = [
                frame.astype(np.float32) for _ok, frame in decoded
            ]
            if source_frame.shape != control_frame.shape or control_frame.shape != candidate_frame.shape:
                raise RuntimeError("A/B frame shapes differ")
            difference = np.abs(candidate_frame - control_frame)
            maximum = max(maximum, float(difference.max()))
            difference_sum += float(difference.sum())
            difference_square_sum += float(np.square(difference).sum())
            pixel_count += int(difference.size)
            # The actual ROI is where the control restoration materially
            # differs from the decoded mosaic source. The threshold suppresses
            # ordinary H.264 round-trip noise outside the composited ROI.
            active = np.mean(np.abs(control_frame - source_frame), axis=2) >= 2.5
            active_count += int(active.sum()) * 3
            active_difference_sum += float(difference[active].sum())
            if previous_control is not None and previous_candidate is not None:
                control_delta = control_frame - previous_control
                candidate_delta = candidate_frame - previous_candidate
                temporal_difference_sum += float(
                    np.abs(candidate_delta - control_delta).sum()
                )
                temporal_pixel_count += int(difference.size)
            previous_control = control_frame
            previous_candidate = candidate_frame
            frame_count += 1
    finally:
        for capture in captures:
            capture.release()
    mse = difference_square_sum / max(1, pixel_count)
    return {
        "decoded_frames": frame_count,
        "candidate_vs_control_mean_absolute_0_255": difference_sum
        / max(1, pixel_count),
        "candidate_vs_control_rmse_0_255": math.sqrt(mse),
        "candidate_vs_control_psnr_db": (
            float("inf") if mse == 0 else 20 * math.log10(255 / math.sqrt(mse))
        ),
        "candidate_vs_control_max_absolute_0_255": maximum,
        "active_roi_pixel_fraction": active_count / max(1, pixel_count),
        "active_roi_candidate_vs_control_mean_absolute_0_255": (
            active_difference_sum / max(1, active_count)
        ),
        "temporal_delta_candidate_vs_control_mean_absolute_0_255": (
            temporal_difference_sum / max(1, temporal_pixel_count)
        ),
    }


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; choose a fresh path: {args.output_dir}"
        )
    for path in (args.source, args.checkpoint, args.preview_executable):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.runs_per_mode < 1:
        raise ValueError("--runs-per-mode must be positive")
    args.output_dir.mkdir(parents=True)
    control, candidate, runner = build_assets(args)
    order = []
    for _ in range(args.runs_per_mode):
        order.extend(("control", "native"))
    if args.runs_per_mode >= 2:
        order = ["control", "native", "native", "control"] + order[4:]
    counts = {"control": 0, "native": 0}
    runs: list[dict[str, object]] = []
    for mode in order:
        counts[mode] += 1
        runs.append(
            run_pipeline(
                args,
                mode,
                counts[mode],
                control if mode == "control" else candidate,
                runner,
            )
        )
    control_times = [float(run["seconds"]) for run in runs if run["mode"] == "control"]
    native_times = [float(run["seconds"]) for run in runs if run["mode"] == "native"]
    control_restoration_times = [
        float(run["native_stats"]["restoration_seconds"])
        for run in runs
        if run["mode"] == "control" and run["native_stats"] is not None
    ]
    native_restoration_times = [
        float(run["native_stats"]["restoration_seconds"])
        for run in runs
        if run["mode"] == "native" and run["native_stats"] is not None
    ]
    control_output = Path(
        next(str(run["output"]) for run in runs if run["mode"] == "control")
    )
    native_output = Path(
        next(str(run["output"]) for run in runs if run["mode"] == "native")
    )
    control_median = statistics.median(control_times)
    native_median = statistics.median(native_times)
    control_restoration_median = statistics.median(control_restoration_times)
    native_restoration_median = statistics.median(native_restoration_times)
    report = {
        "source": str(args.source.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "temporal_frames": args.temporal_frames,
        "temporal_overlap": args.temporal_overlap,
        "runs": runs,
        "control_median_seconds": control_median,
        "native_state_median_seconds": native_median,
        "native_state_speedup": control_median / native_median,
        "native_state_end_to_end_time_reduction_fraction": (
            control_median - native_median
        )
        / control_median,
        "control_restoration_median_seconds": control_restoration_median,
        "native_state_restoration_median_seconds": native_restoration_median,
        "native_state_restoration_speedup": control_restoration_median
        / native_restoration_median,
        "native_state_restoration_time_reduction_fraction": (
            control_restoration_median - native_restoration_median
        )
        / control_restoration_median,
        "video_comparison": compare_videos(
            args.source, control_output, native_output
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(json.dumps(report, indent=2, allow_nan=True))
    print(f"report={report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
