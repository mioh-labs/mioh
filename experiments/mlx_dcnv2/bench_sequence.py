"""Smoke benchmark for the MLX LADA BasicVSR++ sequence path."""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from experiments.mlx_dcnv2.bundle import load_lada_mlx_bundle
from experiments.mlx_dcnv2.sequence import lada_sequence_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()

    row = run_case(
        manifest_path=args.manifest,
        height=args.height,
        width=args.width,
        frame_count=args.frames,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(f"input=1x{args.frames}x3x{args.height}x{args.width}")
    print(f"output_shape={row['output_shape']}")
    print(f"mlx_sequence={row['mlx_sequence_ms']:.3f} ms/iter")
    print(f"effective_fps={row['effective_fps']:.3f}")


def run_case(
    *,
    manifest_path: str,
    height: int,
    width: int,
    frame_count: int,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    rng = np.random.default_rng(1301)
    bundle = load_lada_mlx_bundle(manifest_path)
    frames = mx.array(rng.normal(size=(1, frame_count, 3, height, width)).astype(np.float32))

    for _ in range(warmup):
        y = lada_sequence_forward(frames, bundle)
        mx.eval(y)
    start = time.perf_counter()
    for _ in range(iters):
        y = lada_sequence_forward(frames, bundle)
        mx.eval(y)
    seconds = (time.perf_counter() - start) / iters
    return {
        "output_shape": str(y.shape),
        "mlx_sequence_ms": seconds * 1000,
        "effective_fps": frame_count / seconds,
    }


if __name__ == "__main__":
    main()
