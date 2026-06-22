"""Smoke benchmark for masked ROI restore using the MLX LADA sequence path."""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from experiments.mlx_dcnv2.bundle import load_lada_mlx_bundle
from experiments.mlx_dcnv2.roi_restore import restore_masked_roi_sequence_with_lada


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--roi-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()

    row = run_case(
        manifest_path=args.manifest,
        height=args.height,
        width=args.width,
        frame_count=args.frames,
        roi_size=args.roi_size,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(f"input={args.frames}x3x{args.height}x{args.width} roi={args.roi_size}x{args.roi_size}")
    print(f"output_shape={row['output_shape']}")
    print(f"mlx_masked_roi_restore={row['mlx_masked_roi_restore_ms']:.3f} ms/iter")
    print(f"effective_fps={row['effective_fps']:.3f}")


def run_case(
    *,
    manifest_path: str,
    height: int,
    width: int,
    frame_count: int,
    roi_size: int,
    warmup: int,
    iters: int,
) -> dict[str, object]:
    rng = np.random.default_rng(1401)
    bundle = load_lada_mlx_bundle(manifest_path)
    frames = mx.array(rng.normal(size=(frame_count, 3, height, width)).astype(np.float32))
    masks_np = np.zeros((frame_count, height, width), dtype=np.float32)
    y0 = max(0, (height - roi_size) // 2)
    x0 = max(0, (width - roi_size) // 2)
    masks_np[:, y0:y0 + roi_size, x0:x0 + roi_size] = 1.0
    masks = mx.array(masks_np)

    for _ in range(warmup):
        y = restore_masked_roi_sequence_with_lada(frames, masks, bundle)
        mx.eval(y)
    start = time.perf_counter()
    for _ in range(iters):
        y = restore_masked_roi_sequence_with_lada(frames, masks, bundle)
        mx.eval(y)
    seconds = (time.perf_counter() - start) / iters
    return {
        "output_shape": str(y.shape),
        "mlx_masked_roi_restore_ms": seconds * 1000,
        "effective_fps": frame_count / seconds,
    }


if __name__ == "__main__":
    main()
