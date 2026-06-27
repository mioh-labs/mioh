"""Benchmark MLX BasicVSR++ residual backbone blocks."""

from __future__ import annotations

import argparse
import json
import time

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F

from experiments.mlx_dcnv2.backbone import (
    prepare_backbone_tensors_nhwc,
    residual_blocks_with_input_conv_forward,
    residual_blocks_with_input_conv_forward_nhwc,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--npz", required=True, help="Exported BasicVSR++ backbone NPZ")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--torch-device", default="cpu", choices=["cpu", "mps"])
    parser.add_argument("--jsonl-output")
    args = parser.parse_args()

    row = run_case(
        height=args.height,
        width=args.width,
        npz_path=args.npz,
        warmup=args.warmup,
        iters=args.iters,
        torch_device=args.torch_device,
    )
    print(
        f"shape=1x{int(row['input_channels'])}x{args.height}x{args.width} "
        f"mid={int(row['mid_channels'])} blocks={int(row['num_blocks'])}"
    )
    print(f"max_abs_error={row['max_abs_error']:.6g}")
    print(f"max_abs_error_nhwc={row['max_abs_error_nhwc']:.6g}")
    print(f"mlx_backbone_nchw={row['mlx_backbone_nchw_ms']:.3f} ms/iter")
    print(f"mlx_backbone_nhwc_prepared={row['mlx_backbone_nhwc_ms']:.3f} ms/iter")
    print(f"torch_backbone_{args.torch_device}={row['torch_backbone_ms']:.3f} ms/iter")
    print(f"speedup_nchw_vs_torch_{args.torch_device}={row['speedup_nchw_vs_torch']:.2f}x")
    print(f"speedup_nhwc_vs_torch_{args.torch_device}={row['speedup_nhwc_vs_torch']:.2f}x")
    if args.jsonl_output:
        with open(args.jsonl_output, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def run_case(
    *,
    height: int,
    width: int,
    npz_path: str,
    warmup: int,
    iters: int,
    torch_device: str,
) -> dict[str, float]:
    rng = np.random.default_rng(777)
    tensors = _load_npz_tensors(npz_path)
    input_channels = int(tensors["main.0.weight"].shape[1])
    mid_channels = int(tensors["main.0.weight"].shape[0])
    num_blocks = _infer_num_blocks(tensors)
    x = rng.normal(size=(1, input_channels, height, width)).astype(np.float32)

    expected = _torch_forward(
        torch.from_numpy(x),
        {name: torch.from_numpy(value) for name, value in tensors.items()},
        num_blocks,
    ).numpy()
    mlx_tensors = {name: mx.array(value) for name, value in tensors.items()}
    actual_nchw = np.array(
        residual_blocks_with_input_conv_forward(
            mx.array(x),
            mlx_tensors,
            num_blocks=num_blocks,
        )
    )
    max_abs_error = float(np.max(np.abs(expected - actual_nchw)))

    x_nhwc = np.transpose(x, (0, 2, 3, 1))
    mlx_tensors_nhwc = prepare_backbone_tensors_nhwc(mlx_tensors)
    actual_nhwc = np.array(
        residual_blocks_with_input_conv_forward_nhwc(
            mx.array(x_nhwc),
            mlx_tensors_nhwc,
            num_blocks=num_blocks,
        )
    )
    actual_nhwc_nchw = np.transpose(actual_nhwc, (0, 3, 1, 2))
    max_abs_error_nhwc = float(np.max(np.abs(expected - actual_nhwc_nchw)))

    mlx_inputs = (mx.array(x), mlx_tensors)
    for _ in range(warmup):
        y = residual_blocks_with_input_conv_forward(*mlx_inputs, num_blocks=num_blocks)
        mx.eval(y)
    start = time.perf_counter()
    for _ in range(iters):
        y = residual_blocks_with_input_conv_forward(*mlx_inputs, num_blocks=num_blocks)
        mx.eval(y)
    mlx_nchw_seconds = (time.perf_counter() - start) / iters

    mlx_nhwc_inputs = (mx.array(x_nhwc), mlx_tensors_nhwc)
    for _ in range(warmup):
        y = residual_blocks_with_input_conv_forward_nhwc(*mlx_nhwc_inputs, num_blocks=num_blocks)
        mx.eval(y)
    start = time.perf_counter()
    for _ in range(iters):
        y = residual_blocks_with_input_conv_forward_nhwc(*mlx_nhwc_inputs, num_blocks=num_blocks)
        mx.eval(y)
    mlx_nhwc_seconds = (time.perf_counter() - start) / iters

    device = torch.device(torch_device)
    torch_inputs = (
        torch.from_numpy(x).to(device),
        {name: torch.from_numpy(value).to(device) for name, value in tensors.items()},
    )
    for _ in range(warmup):
        y = _torch_forward(*torch_inputs, num_blocks)
        _sync_torch_device(device)
    start = time.perf_counter()
    for _ in range(iters):
        y = _torch_forward(*torch_inputs, num_blocks)
        _sync_torch_device(device)
    torch_seconds = (time.perf_counter() - start) / iters

    return {
        "height": float(height),
        "width": float(width),
        "max_abs_error": max_abs_error,
        "max_abs_error_nhwc": max_abs_error_nhwc,
        "input_channels": float(input_channels),
        "mid_channels": float(mid_channels),
        "num_blocks": float(num_blocks),
        "mlx_backbone_nchw_ms": mlx_nchw_seconds * 1000,
        "mlx_backbone_nhwc_ms": mlx_nhwc_seconds * 1000,
        "torch_backbone_ms": torch_seconds * 1000,
        "torch_device": torch_device,
        "speedup_nchw_vs_torch": torch_seconds / mlx_nchw_seconds,
        "speedup_nhwc_vs_torch": torch_seconds / mlx_nhwc_seconds,
    }


def _torch_forward(x, tensors, num_blocks):
    out = F.conv2d(x, tensors["main.0.weight"], tensors["main.0.bias"], padding=1)
    out = F.leaky_relu(out, negative_slope=0.1)
    for block_index in range(num_blocks):
        identity = out
        out = F.conv2d(
            out,
            tensors[f"main.2.{block_index}.conv1.weight"],
            tensors[f"main.2.{block_index}.conv1.bias"],
            padding=1,
        )
        out = F.relu(out)
        out = F.conv2d(
            out,
            tensors[f"main.2.{block_index}.conv2.weight"],
            tensors[f"main.2.{block_index}.conv2.bias"],
            padding=1,
        )
        out = identity + out
    return out


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


def _sync_torch_device(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


if __name__ == "__main__":
    main()
