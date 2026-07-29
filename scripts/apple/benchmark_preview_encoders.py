#!/usr/bin/env python3
"""Benchmark mioh's PyAV and CVPixelBufferPool preview encoders."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "packaging" / "macOS" / "standalone" / "mioh_preview_worker.py"


def load_worker():
    spec = importlib.util.spec_from_file_location("mioh_preview_worker", WORKER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {WORKER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_frames(width: int, height: int) -> list[np.ndarray]:
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    base = np.empty((height, width, 3), dtype=np.uint8)
    base[..., 0] = x
    base[..., 1] = y
    base[..., 2] = ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(
        np.uint8
    )
    return [np.roll(base, index * 8, axis=1) for index in range(4)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("pyav", "swift", "swift-async"),
        required=True,
    )
    parser.add_argument("--runner")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument(
        "--producer-delay-ms",
        type=float,
        default=0.0,
        help="Synthetic per-frame restoration/detection work to overlap",
    )
    args = parser.parse_args()
    worker = load_worker()
    frames = build_frames(args.width, args.height)

    with tempfile.TemporaryDirectory(prefix=f"mioh-{args.backend}-") as directory:
        kwargs = dict(
            output_dir=Path(directory),
            width=args.width,
            height=args.height,
            fps=Fraction(args.fps, 1),
            generation=1,
            preferred_codec="h264_videotoolbox",
            segment_seconds=2.0,
        )
        if args.backend in {"swift", "swift-async"}:
            if not args.runner:
                parser.error("--runner is required for the swift backend")
            encoder = worker.SwiftVideoToolboxSegmentEncoder(
                **kwargs,
                runner_path=args.runner,
            )
            if args.backend == "swift-async":
                encoder = worker.AsyncSegmentEncoder(
                    encoder,
                    max_pending_frames=2,
                )
        else:
            encoder = worker.SegmentEncoder(**kwargs)

        started = time.perf_counter()
        events = []
        for index in range(args.frames):
            if args.producer_delay_ms > 0:
                time.sleep(args.producer_delay_ms / 1000.0)
            events.extend(
                encoder.add_frame(
                    frames[index % len(frames)],
                    index * 1_000_000_000 // args.fps,
                )
            )
        events.extend(encoder.finish())
        elapsed = time.perf_counter() - started
        output_bytes = sum(Path(event["path"]).stat().st_size for event in events)
        with av.open(events[0]["path"]) as container:
            decoded = next(container.decode(video=0)).to_ndarray(format="bgr24")
        difference = decoded.astype(np.float32) - frames[0].astype(np.float32)
        mse = float(np.mean(difference * difference))
        psnr = float("inf") if mse == 0 else 10 * np.log10((255.0 ** 2) / mse)
        channel_mse = np.mean(difference * difference, axis=(0, 1))
        channel_psnr = [
            float("inf") if value == 0 else 10 * np.log10((255.0 ** 2) / value)
            for value in channel_mse
        ]

    print(
        json.dumps(
            {
                "backend": args.backend,
                "width": args.width,
                "height": args.height,
                "frames": args.frames,
                "producer_delay_ms": args.producer_delay_ms,
                "elapsed_seconds": round(elapsed, 4),
                "encoder_fps": round(args.frames / elapsed, 2),
                "output_bytes": output_bytes,
                "segments": len(events),
                "first_frame_psnr_db": round(float(psnr), 3),
                "channel_psnr_bgr_db": [
                    round(float(value), 3) for value in channel_psnr
                ],
                "mean_error_bgr": [
                    round(float(value), 3)
                    for value in difference.mean(axis=(0, 1))
                ],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
