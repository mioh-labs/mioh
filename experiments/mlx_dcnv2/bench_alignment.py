"""Benchmark MLX BasicVSR++ second-order deformable alignment."""

from __future__ import annotations

import argparse
import time

import mlx.core as mx
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from experiments.mlx_dcnv2.alignment import second_order_deformable_alignment_forward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--mid-channels", type=int, default=64)
    parser.add_argument("--deform-groups", type=int, default=16)
    parser.add_argument("--npz", type=str, help="Exported deform alignment NPZ to benchmark with real weights")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    row = run_case(
        height=args.height,
        width=args.width,
        mid_channels=args.mid_channels,
        deform_groups=args.deform_groups,
        npz_path=args.npz,
        warmup=args.warmup,
        iters=args.iters,
    )
    print(
        f"shape=1x{int(row['input_channels'])}x{args.height}x{args.width} "
        f"mid={int(row['mid_channels'])} deform_groups={int(row['deform_groups'])}"
    )
    print(f"max_abs_error={row['max_abs_error']:.6g}")
    print(f"mlx_alignment={row['mlx_alignment_ms']:.3f} ms/iter")
    print(f"torch_alignment_cpu={row['torch_alignment_cpu_ms']:.3f} ms/iter")
    print(f"speedup_vs_torch_cpu={row['speedup_vs_torch_cpu']:.2f}x")


def run_case(
    *,
    height: int,
    width: int,
    mid_channels: int,
    deform_groups: int,
    npz_path: str | None,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    rng = np.random.default_rng(99)
    if npz_path:
        tensors = _load_npz_tensors(npz_path)
        mid_channels = int(tensors["weight"].shape[0])
        deform_groups = int(tensors["conv_offset.6.weight"].shape[0] // 27)
    else:
        tensors = _random_tensors(rng, mid_channels, deform_groups)
    x = rng.normal(size=(1, 2 * mid_channels, height, width)).astype(np.float32)
    extra = rng.normal(size=(1, 3 * mid_channels, height, width)).astype(np.float32)
    flow1 = rng.normal(size=(1, 2, height, width)).astype(np.float32)
    flow2 = rng.normal(size=(1, 2, height, width)).astype(np.float32)

    expected = _torch_forward(x, extra, flow1, flow2, tensors)
    mlx_tensors = {name: mx.array(value) for name, value in tensors.items()}
    actual = np.array(
        second_order_deformable_alignment_forward(
            mx.array(x),
            mx.array(extra),
            mx.array(flow1),
            mx.array(flow2),
            mlx_tensors,
        )
    )
    max_abs_error = float(np.max(np.abs(expected - actual)))

    mlx_inputs = (mx.array(x), mx.array(extra), mx.array(flow1), mx.array(flow2), mlx_tensors)
    for _ in range(warmup):
        y = second_order_deformable_alignment_forward(*mlx_inputs)
        mx.eval(y)
    start = time.perf_counter()
    for _ in range(iters):
        y = second_order_deformable_alignment_forward(*mlx_inputs)
        mx.eval(y)
    mlx_seconds = (time.perf_counter() - start) / iters

    torch_inputs = (
        torch.from_numpy(x),
        torch.from_numpy(extra),
        torch.from_numpy(flow1),
        torch.from_numpy(flow2),
        {name: torch.from_numpy(value) for name, value in tensors.items()},
    )
    for _ in range(warmup):
        _torch_forward_tensors(*torch_inputs)
    start = time.perf_counter()
    for _ in range(iters):
        _torch_forward_tensors(*torch_inputs)
    torch_seconds = (time.perf_counter() - start) / iters

    return {
        "max_abs_error": max_abs_error,
        "input_channels": float(2 * mid_channels),
        "mid_channels": float(mid_channels),
        "deform_groups": float(deform_groups),
        "mlx_alignment_ms": mlx_seconds * 1000,
        "torch_alignment_cpu_ms": torch_seconds * 1000,
        "speedup_vs_torch_cpu": torch_seconds / mlx_seconds,
    }


def _torch_forward(x, extra, flow1, flow2, tensors):
    return _torch_forward_tensors(
        torch.from_numpy(x),
        torch.from_numpy(extra),
        torch.from_numpy(flow1),
        torch.from_numpy(flow2),
        {name: torch.from_numpy(value) for name, value in tensors.items()},
    ).numpy()


def _torch_forward_tensors(x, extra, flow1, flow2, tensors):
    feat = torch.cat([extra, flow1, flow2], dim=1)
    for layer in (0, 2, 4):
        feat = F.conv2d(feat, tensors[f"conv_offset.{layer}.weight"], tensors[f"conv_offset.{layer}.bias"], padding=1)
        feat = F.leaky_relu(feat, negative_slope=0.1)
    feat = F.conv2d(feat, tensors["conv_offset.6.weight"], tensors["conv_offset.6.bias"], padding=1)
    o1, o2, mask = torch.chunk(feat, 3, dim=1)
    offset = 10 * torch.tanh(torch.cat((o1, o2), dim=1))
    offset1, offset2 = torch.chunk(offset, 2, dim=1)
    offset1 = offset1 + flow1.flip(1).repeat(1, offset1.size(1) // 2, 1, 1)
    offset2 = offset2 + flow2.flip(1).repeat(1, offset2.size(1) // 2, 1, 1)
    offset = torch.cat([offset1, offset2], dim=1)
    return torchvision.ops.deform_conv2d(
        x,
        offset,
        tensors["weight"],
        tensors["bias"],
        padding=(1, 1),
        mask=torch.sigmoid(mask),
    )


def _random_tensors(rng, mid_channels, deform_groups):
    scale = 0.02
    return {
        "weight": (rng.normal(size=(mid_channels, 2 * mid_channels, 3, 3)) * scale).astype(np.float32),
        "bias": (rng.normal(size=(mid_channels,)) * scale).astype(np.float32),
        "conv_offset.0.weight": (rng.normal(size=(mid_channels, 3 * mid_channels + 4, 3, 3)) * scale).astype(np.float32),
        "conv_offset.0.bias": (rng.normal(size=(mid_channels,)) * scale).astype(np.float32),
        "conv_offset.2.weight": (rng.normal(size=(mid_channels, mid_channels, 3, 3)) * scale).astype(np.float32),
        "conv_offset.2.bias": (rng.normal(size=(mid_channels,)) * scale).astype(np.float32),
        "conv_offset.4.weight": (rng.normal(size=(mid_channels, mid_channels, 3, 3)) * scale).astype(np.float32),
        "conv_offset.4.bias": (rng.normal(size=(mid_channels,)) * scale).astype(np.float32),
        "conv_offset.6.weight": (rng.normal(size=(27 * deform_groups, mid_channels, 3, 3)) * scale).astype(np.float32),
        "conv_offset.6.bias": (rng.normal(size=(27 * deform_groups,)) * scale).astype(np.float32),
    }


def _load_npz_tensors(path: str) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {name: data[name].astype(np.float32, copy=False) for name in data.files}


if __name__ == "__main__":
    main()
