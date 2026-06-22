"""Benchmark one MLX BasicVSR++ propagation step."""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np

from experiments.mlx_dcnv2.propagation import propagation_step_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--alignment-npz", required=True)
    parser.add_argument("--backbone-npz", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    row = run_case(
        height=args.height,
        width=args.width,
        alignment_npz=args.alignment_npz,
        backbone_npz=args.backbone_npz,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(f"shape=1x{int(row['mid_channels'])}x{args.height}x{args.width} blocks={int(row['num_blocks'])}")
    print(f"mlx_propagation_step={row['mlx_propagation_step_ms']:.3f} ms/iter")


def run_case(
    *,
    height: int,
    width: int,
    alignment_npz: str,
    backbone_npz: str,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    rng = np.random.default_rng(888)
    alignment_tensors = {name: mx.array(value) for name, value in _load_npz_tensors(alignment_npz).items()}
    backbone_raw = _load_npz_tensors(backbone_npz)
    backbone_tensors = {name: mx.array(value) for name, value in backbone_raw.items()}
    mid_channels = int(alignment_tensors["weight"].shape[0])
    num_blocks = _infer_num_blocks(backbone_raw)

    inputs = (
        mx.array(rng.normal(size=(1, mid_channels, height, width)).astype(np.float32)),
        mx.array(rng.normal(size=(1, mid_channels, height, width)).astype(np.float32)),
        mx.zeros((1, mid_channels, height, width), dtype=mx.float32),
        mx.array((rng.normal(size=(1, 2, height, width)) * 0.2).astype(np.float32)),
        mx.zeros((1, 2, height, width), dtype=mx.float32),
        alignment_tensors,
        backbone_tensors,
    )

    for _ in range(warmup):
        y = propagation_step_forward(*inputs, num_backbone_blocks=num_blocks)
        mx.eval(y)
    start = time.perf_counter()
    for _ in range(iters):
        y = propagation_step_forward(*inputs, num_backbone_blocks=num_blocks)
        mx.eval(y)
    seconds = (time.perf_counter() - start) / iters

    return {
        "mid_channels": float(mid_channels),
        "num_blocks": float(num_blocks),
        "mlx_propagation_step_ms": seconds * 1000,
    }


def _load_npz_tensors(path: str) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {name: data[name].astype(np.float32, copy=False) for name in data.files}


def _infer_num_blocks(tensors: dict[str, np.ndarray]) -> int:
    block_indices = {
        int(name.split(".")[2])
        for name in tensors
        if name.startswith("main.2.") and name.endswith(".conv1.weight")
    }
    return max(block_indices) + 1 if block_indices else 0


if __name__ == "__main__":
    main()
