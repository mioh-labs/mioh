"""Small benchmark for the experimental MLX DCNv2 forward kernel."""

from __future__ import annotations

import argparse
import csv
import json
import time

import mlx.core as mx
import numpy as np
import torch
import torchvision

from experiments.mlx_dcnv2 import deform_conv2d_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--channels", type=int, default=8)
    parser.add_argument("--out-channels", type=int, default=8)
    parser.add_argument("--deform-groups", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--matrix", action="store_true", help="Run BasicVSR++ ROI-like shape matrix")
    parser.add_argument("--json-output", type=str)
    parser.add_argument("--csv-output", type=str)
    args = parser.parse_args()

    if args.matrix:
        rows = [
            run_case(height=size, width=size, channels=128, out_channels=64, deform_groups=16, warmup=args.warmup, iters=args.iters)
            for size in (16, 32, 64, 96, 128)
        ]
        _print_rows(rows)
        if args.json_output:
            with open(args.json_output, "w", encoding="utf-8") as f:
                json.dump(rows, f, indent=2)
        if args.csv_output:
            with open(args.csv_output, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        return

    row = run_case(
        height=args.height,
        width=args.width,
        channels=args.channels,
        out_channels=args.out_channels,
        deform_groups=args.deform_groups,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(
        f"shape=1x{row['channels']}x{row['height']}x{row['width']} "
        f"out={row['out_channels']} deform_groups={row['deform_groups']}"
    )
    print(f"max_abs_error={row['max_abs_error']:.6g}")
    print(f"mlx_metal={row['mlx_metal_ms']:.3f} ms/iter")
    print(f"torchvision_cpu={row['torchvision_cpu_ms']:.3f} ms/iter")
    print(f"speedup_vs_torchvision_cpu={row['speedup_vs_torchvision_cpu']:.2f}x")


def run_case(
    *,
    height: int,
    width: int,
    channels: int,
    out_channels: int,
    deform_groups: int,
    warmup: int,
    iters: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(1, channels, height, width)).astype(np.float32)
    weight = rng.normal(size=(out_channels, channels, 3, 3)).astype(np.float32)
    bias = rng.normal(size=(out_channels,)).astype(np.float32)
    offset = rng.uniform(
        -0.25,
        0.25,
        size=(1, deform_groups * 2 * 3 * 3, height, width),
    ).astype(np.float32)
    mask = rng.uniform(
        0.2,
        1.0,
        size=(1, deform_groups * 3 * 3, height, width),
    ).astype(np.float32)

    expected = torchvision.ops.deform_conv2d(
        torch.from_numpy(x),
        torch.from_numpy(offset),
        torch.from_numpy(weight),
        torch.from_numpy(bias),
        stride=(1, 1),
        padding=(1, 1),
        dilation=(1, 1),
        mask=torch.from_numpy(mask),
    ).numpy()
    actual = np.array(
        deform_conv2d_forward(
            mx.array(x),
            mx.array(offset),
            mx.array(weight),
            mx.array(bias),
            padding=(1, 1),
            mask=mx.array(mask),
        )
    )
    max_abs_error = float(np.max(np.abs(expected - actual)))

    mlx_inputs = (mx.array(x), mx.array(offset), mx.array(weight), mx.array(bias), mx.array(mask))
    for _ in range(warmup):
        y = deform_conv2d_forward(mlx_inputs[0], mlx_inputs[1], mlx_inputs[2], mlx_inputs[3], padding=1, mask=mlx_inputs[4])
        mx.eval(y)
    start = time.perf_counter()
    for _ in range(iters):
        y = deform_conv2d_forward(mlx_inputs[0], mlx_inputs[1], mlx_inputs[2], mlx_inputs[3], padding=1, mask=mlx_inputs[4])
        mx.eval(y)
    mlx_seconds = (time.perf_counter() - start) / iters

    torch_inputs = tuple(torch.from_numpy(v) for v in (x, offset, weight, bias, mask))
    for _ in range(warmup):
        torchvision.ops.deform_conv2d(
            torch_inputs[0], torch_inputs[1], torch_inputs[2], torch_inputs[3], padding=(1, 1), mask=torch_inputs[4]
        )
    start = time.perf_counter()
    for _ in range(iters):
        torchvision.ops.deform_conv2d(
            torch_inputs[0], torch_inputs[1], torch_inputs[2], torch_inputs[3], padding=(1, 1), mask=torch_inputs[4]
        )
    torch_seconds = (time.perf_counter() - start) / iters

    return {
        "height": height,
        "width": width,
        "channels": channels,
        "out_channels": out_channels,
        "deform_groups": deform_groups,
        "max_abs_error": max_abs_error,
        "mlx_metal_ms": mlx_seconds * 1000,
        "torchvision_cpu_ms": torch_seconds * 1000,
        "speedup_vs_torchvision_cpu": torch_seconds / mlx_seconds,
    }


def _print_rows(rows: list[dict[str, float | int]]) -> None:
    print("size,mlx_metal_ms,torchvision_cpu_ms,speedup,max_abs_error")
    for row in rows:
        print(
            f"{row['height']}x{row['width']},"
            f"{float(row['mlx_metal_ms']):.3f},"
            f"{float(row['torchvision_cpu_ms']):.3f},"
            f"{float(row['speedup_vs_torchvision_cpu']):.2f}x,"
            f"{float(row['max_abs_error']):.6g}"
        )


if __name__ == "__main__":
    main()
