"""Benchmark MLX LADA BasicVSR++ feature extraction."""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from experiments.mlx_dcnv2.feature_extract import feature_extract_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    row = run_case(height=args.height, width=args.width, npz_path=args.npz, warmup=args.warmup, iters=args.iters)
    print(f"input=1x3x{args.height}x{args.width} output=1x{int(row['mid_channels'])}x{args.height // 4}x{args.width // 4}")
    print(f"max_abs_error={row['max_abs_error']:.6g}")
    print(f"mlx_feature_extract={row['mlx_feature_extract_ms']:.3f} ms/iter")
    print(f"torch_feature_extract_cpu={row['torch_feature_extract_cpu_ms']:.3f} ms/iter")
    print(f"speedup_vs_torch_cpu={row['speedup_vs_torch_cpu']:.2f}x")


def run_case(*, height: int, width: int, npz_path: str, warmup: int, iters: int) -> dict[str, float]:
    rng = np.random.default_rng(902)
    tensors = _load_npz_tensors(npz_path)
    mid_channels = int(tensors["0.weight"].shape[0])
    num_blocks = _infer_num_blocks(tensors)
    x = rng.normal(size=(1, 3, height, width)).astype(np.float32)

    expected = _torch_forward(torch.from_numpy(x), {k: torch.from_numpy(v) for k, v in tensors.items()}, num_blocks).numpy()
    mlx_tensors = {k: mx.array(v) for k, v in tensors.items()}
    actual = np.array(feature_extract_forward(mx.array(x), mlx_tensors, num_blocks=num_blocks))
    max_abs_error = float(np.max(np.abs(expected - actual)))

    mlx_inputs = (mx.array(x), mlx_tensors)
    for _ in range(warmup):
        y = feature_extract_forward(*mlx_inputs, num_blocks=num_blocks)
        mx.eval(y)
    start = time.perf_counter()
    for _ in range(iters):
        y = feature_extract_forward(*mlx_inputs, num_blocks=num_blocks)
        mx.eval(y)
    mlx_seconds = (time.perf_counter() - start) / iters

    torch_inputs = (torch.from_numpy(x), {k: torch.from_numpy(v) for k, v in tensors.items()})
    for _ in range(warmup):
        _torch_forward(*torch_inputs, num_blocks)
    start = time.perf_counter()
    for _ in range(iters):
        _torch_forward(*torch_inputs, num_blocks)
    torch_seconds = (time.perf_counter() - start) / iters

    return {
        "max_abs_error": max_abs_error,
        "mid_channels": float(mid_channels),
        "mlx_feature_extract_ms": mlx_seconds * 1000,
        "torch_feature_extract_cpu_ms": torch_seconds * 1000,
        "speedup_vs_torch_cpu": torch_seconds / mlx_seconds,
    }


def _torch_forward(x, tensors, num_blocks):
    out = F.conv2d(x, tensors["0.weight"], tensors["0.bias"], stride=2, padding=1)
    out = F.leaky_relu(out, negative_slope=0.1)
    out = F.conv2d(out, tensors["2.weight"], tensors["2.bias"], stride=2, padding=1)
    out = F.leaky_relu(out, negative_slope=0.1)
    out = F.conv2d(out, tensors["4.main.0.weight"], tensors["4.main.0.bias"], padding=1)
    out = F.leaky_relu(out, negative_slope=0.1)
    for block_index in range(num_blocks):
        identity = out
        out = F.conv2d(out, tensors[f"4.main.2.{block_index}.conv1.weight"], tensors[f"4.main.2.{block_index}.conv1.bias"], padding=1)
        out = F.relu(out)
        out = F.conv2d(out, tensors[f"4.main.2.{block_index}.conv2.weight"], tensors[f"4.main.2.{block_index}.conv2.bias"], padding=1)
        out = identity + out
    return out


def _load_npz_tensors(path: str) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {name: data[name].astype(np.float32, copy=False) for name in data.files}


def _infer_num_blocks(tensors: dict[str, np.ndarray]) -> int:
    block_indices = {
        int(name.split(".")[3])
        for name in tensors
        if name.startswith("4.main.2.") and name.endswith(".conv1.weight")
    }
    return max(block_indices) + 1 if block_indices else 0


if __name__ == "__main__":
    main()
