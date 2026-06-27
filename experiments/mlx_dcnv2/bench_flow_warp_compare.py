"""Compare Torch and MLX flow-warp kernels on BasicVSR++-like shapes."""

from __future__ import annotations

import argparse
import json
import time
from typing import Callable

import mlx.core as mx
import numpy as np
import torch

from experiments.mlx_dcnv2.flow_warp import flow_warp as mlx_flow_warp
from experiments.mlx_dcnv2.fused_propagation_warp import fused_propagation_warp_cond
from experiments.mlx_dcnv2.fused_propagation_warp import two_stage_propagation_warp_cond
from lada.models.basicvsrpp.mmagic.flow_warp import flow_warp as torch_flow_warp


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark Torch MPS/CPU flow_warp against the experimental MLX "
            "kernel using identical synthetic BasicVSR++ tensors."
        )
    )
    parser.add_argument(
        "--case",
        choices=[
            "flow-warp",
            "propagation-warp",
            "fused-propagation-warp",
            "two-stage-propagation-warp",
            "bridge-flow-warp",
            "bridge-propagation-warp",
            "bridge-fused-propagation-warp",
            "bridge-two-stage-propagation-warp",
            "window-bridge-propagation-warp",
            "both",
            "all",
        ],
        default="both",
    )
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20, help="Propagation-warp steps for window bridge benchmarks")
    parser.add_argument("--padding-mode", choices=["zeros", "border"], default="zeros")
    parser.add_argument("--torch-device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260627)
    parser.add_argument("--json", action="store_true", help="Print JSON lines instead of a compact table")
    args = parser.parse_args()

    torch_device = _resolve_torch_device(args.torch_device)
    rows: list[dict[str, float | int | str]] = []
    if args.case in {"flow-warp", "both", "all"}:
        rows.append(
            run_flow_warp_case(
                channels=args.channels,
                height=args.height,
                width=args.width,
                padding_mode=args.padding_mode,
                torch_device=torch_device,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed,
            )
        )
    if args.case in {"propagation-warp", "both", "all"}:
        rows.append(
            run_propagation_warp_case(
                channels=args.channels,
                height=args.height,
                width=args.width,
                padding_mode=args.padding_mode,
                torch_device=torch_device,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed + 1,
            )
        )
    if args.case in {"bridge-flow-warp", "all"}:
        rows.append(
            run_bridge_flow_warp_case(
                channels=args.channels,
                height=args.height,
                width=args.width,
                padding_mode=args.padding_mode,
                torch_device=torch_device,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed + 2,
            )
        )
    if args.case in {"bridge-propagation-warp", "all"}:
        rows.append(
            run_bridge_propagation_warp_case(
                channels=args.channels,
                height=args.height,
                width=args.width,
                padding_mode=args.padding_mode,
                torch_device=torch_device,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed + 3,
            )
        )
    if args.case in {"fused-propagation-warp", "all"}:
        rows.append(
            run_fused_propagation_warp_case(
                channels=args.channels,
                height=args.height,
                width=args.width,
                padding_mode=args.padding_mode,
                torch_device=torch_device,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed + 4,
            )
        )
    if args.case in {"bridge-fused-propagation-warp", "all"}:
        rows.append(
            run_bridge_fused_propagation_warp_case(
                channels=args.channels,
                height=args.height,
                width=args.width,
                padding_mode=args.padding_mode,
                torch_device=torch_device,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed + 5,
            )
        )
    if args.case in {"two-stage-propagation-warp", "all"}:
        rows.append(
            run_two_stage_propagation_warp_case(
                channels=args.channels,
                height=args.height,
                width=args.width,
                padding_mode=args.padding_mode,
                torch_device=torch_device,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed + 6,
            )
        )
    if args.case in {"bridge-two-stage-propagation-warp", "all"}:
        rows.append(
            run_bridge_two_stage_propagation_warp_case(
                channels=args.channels,
                height=args.height,
                width=args.width,
                padding_mode=args.padding_mode,
                torch_device=torch_device,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed + 7,
            )
        )
    if args.case in {"window-bridge-propagation-warp", "all"}:
        rows.append(
            run_window_bridge_propagation_warp_case(
                channels=args.channels,
                height=args.height,
                width=args.width,
                steps=args.steps,
                padding_mode=args.padding_mode,
                torch_device=torch_device,
                warmup=args.warmup,
                iters=args.iters,
                seed=args.seed + 8,
            )
        )

    for row in rows:
        if args.json:
            print(json.dumps(row, sort_keys=True))
        else:
            _print_row(row)


def run_flow_warp_case(
    *,
    channels: int,
    height: int,
    width: int,
    padding_mode: str,
    torch_device: str,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    x, flow = _make_feature_and_flow(rng, channels=channels, height=height, width=width)

    torch_output = _torch_flow_warp_once(x, flow, padding_mode=padding_mode, device=torch_device)
    mlx_output = _mlx_flow_warp_once(x, flow, padding_mode=padding_mode)
    max_abs_error = float(np.max(np.abs(torch_output - mlx_output)))

    torch_ms = _time_torch(
        lambda: _torch_flow_warp_tensor(x, flow, padding_mode=padding_mode, device=torch_device),
        device=torch_device,
        warmup=warmup,
        iters=iters,
    )
    mlx_ms = _time_mlx(
        lambda: _mlx_flow_warp_array(x, flow, padding_mode=padding_mode),
        warmup=warmup,
        iters=iters,
    )

    return _make_row(
        case="flow_warp",
        shape=f"1x{channels}x{height}x{width}",
        torch_device=torch_device,
        padding_mode=padding_mode,
        warmup=warmup,
        iters=iters,
        torch_ms=torch_ms,
        mlx_ms=mlx_ms,
        max_abs_error=max_abs_error,
        warp_calls=1,
    )


def run_propagation_warp_case(
    *,
    channels: int,
    height: int,
    width: int,
    padding_mode: str,
    torch_device: str,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    feat_current, flow_n1 = _make_feature_and_flow(rng, channels=channels, height=height, width=width)
    feat_prop, flow_n2 = _make_feature_and_flow(rng, channels=channels, height=height, width=width)
    feat_n2 = rng.normal(size=(1, channels, height, width)).astype(np.float32)
    flow_n1_nchw = np.transpose(flow_n1, (0, 3, 1, 2)).astype(np.float32, copy=False)
    flow_n2_nchw = np.transpose(flow_n2, (0, 3, 1, 2)).astype(np.float32, copy=False)

    torch_output = _torch_propagation_warp_once(
        feat_current,
        feat_prop,
        feat_n2,
        flow_n1_nchw,
        flow_n2_nchw,
        padding_mode=padding_mode,
        device=torch_device,
    )
    mlx_output = _mlx_propagation_warp_once(
        feat_current,
        feat_prop,
        feat_n2,
        flow_n1_nchw,
        flow_n2_nchw,
        padding_mode=padding_mode,
    )
    max_abs_error = float(np.max(np.abs(torch_output - mlx_output)))

    torch_ms = _time_torch(
        lambda: _torch_propagation_warp_tensor(
            feat_current,
            feat_prop,
            feat_n2,
            flow_n1_nchw,
            flow_n2_nchw,
            padding_mode=padding_mode,
            device=torch_device,
        ),
        device=torch_device,
        warmup=warmup,
        iters=iters,
    )
    mlx_ms = _time_mlx(
        lambda: _mlx_propagation_warp_array(
            feat_current,
            feat_prop,
            feat_n2,
            flow_n1_nchw,
            flow_n2_nchw,
            padding_mode=padding_mode,
        ),
        warmup=warmup,
        iters=iters,
    )

    return _make_row(
        case="propagation_warp",
        shape=f"1x{channels}x{height}x{width}",
        torch_device=torch_device,
        padding_mode=padding_mode,
        warmup=warmup,
        iters=iters,
        torch_ms=torch_ms,
        mlx_ms=mlx_ms,
        max_abs_error=max_abs_error,
        warp_calls=3,
    )


def run_bridge_flow_warp_case(
    *,
    channels: int,
    height: int,
    width: int,
    padding_mode: str,
    torch_device: str,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    x, flow = _make_feature_and_flow(rng, channels=channels, height=height, width=width)
    x_t = torch.from_numpy(x).to(torch_device)
    flow_t = torch.from_numpy(flow).to(torch_device)

    torch_output = _torch_flow_warp_tensor_resident(x_t, flow_t, padding_mode=padding_mode)
    _sync_torch(torch_device)
    mlx_output = _bridge_flow_warp_tensor_resident(x_t, flow_t, padding_mode=padding_mode, device=torch_device)
    _sync_torch(torch_device)
    max_abs_error = float(np.max(np.abs(torch_output.detach().cpu().numpy() - mlx_output.detach().cpu().numpy())))

    torch_ms = _time_torch(
        lambda: _torch_flow_warp_tensor_resident(x_t, flow_t, padding_mode=padding_mode),
        device=torch_device,
        warmup=warmup,
        iters=iters,
    )
    mlx_ms = _time_torch(
        lambda: _bridge_flow_warp_tensor_resident(x_t, flow_t, padding_mode=padding_mode, device=torch_device),
        device=torch_device,
        warmup=warmup,
        iters=iters,
    )

    return _make_row(
        case="flow_warp_bridge",
        shape=f"1x{channels}x{height}x{width}",
        torch_device=torch_device,
        padding_mode=padding_mode,
        warmup=warmup,
        iters=iters,
        torch_ms=torch_ms,
        mlx_ms=mlx_ms,
        max_abs_error=max_abs_error,
        warp_calls=1,
        candidate="mlx_bridge",
    )


def run_bridge_propagation_warp_case(
    *,
    channels: int,
    height: int,
    width: int,
    padding_mode: str,
    torch_device: str,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, float | int | str]:
    rng = np.random.default_rng(seed)
    feat_current, flow_n1 = _make_feature_and_flow(rng, channels=channels, height=height, width=width)
    feat_prop, flow_n2 = _make_feature_and_flow(rng, channels=channels, height=height, width=width)
    feat_n2 = rng.normal(size=(1, channels, height, width)).astype(np.float32)
    flow_n1_nchw = np.transpose(flow_n1, (0, 3, 1, 2)).astype(np.float32, copy=False)
    flow_n2_nchw = np.transpose(flow_n2, (0, 3, 1, 2)).astype(np.float32, copy=False)
    tensors = tuple(
        torch.from_numpy(value).to(torch_device)
        for value in (feat_current, feat_prop, feat_n2, flow_n1_nchw, flow_n2_nchw)
    )

    torch_output = _torch_propagation_warp_tensor_resident(*tensors, padding_mode=padding_mode)
    _sync_torch(torch_device)
    mlx_output = _bridge_propagation_warp_tensor_resident(
        *tensors,
        padding_mode=padding_mode,
        device=torch_device,
    )
    _sync_torch(torch_device)
    max_abs_error = float(np.max(np.abs(torch_output.detach().cpu().numpy() - mlx_output.detach().cpu().numpy())))

    torch_ms = _time_torch(
        lambda: _torch_propagation_warp_tensor_resident(*tensors, padding_mode=padding_mode),
        device=torch_device,
        warmup=warmup,
        iters=iters,
    )
    mlx_ms = _time_torch(
        lambda: _bridge_propagation_warp_tensor_resident(*tensors, padding_mode=padding_mode, device=torch_device),
        device=torch_device,
        warmup=warmup,
        iters=iters,
    )

    return _make_row(
        case="propagation_warp_bridge",
        shape=f"1x{channels}x{height}x{width}",
        torch_device=torch_device,
        padding_mode=padding_mode,
        warmup=warmup,
        iters=iters,
        torch_ms=torch_ms,
        mlx_ms=mlx_ms,
        max_abs_error=max_abs_error,
        warp_calls=3,
        candidate="mlx_bridge",
    )


def run_fused_propagation_warp_case(
    *,
    channels: int,
    height: int,
    width: int,
    padding_mode: str,
    torch_device: str,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, float | int | str]:
    if padding_mode != "zeros":
        raise ValueError("fused propagation warp currently supports zeros padding only")
    rng = np.random.default_rng(seed)
    feat_current, flow_n1 = _make_feature_and_flow(rng, channels=channels, height=height, width=width)
    feat_prop, flow_n2 = _make_feature_and_flow(rng, channels=channels, height=height, width=width)
    feat_n2 = rng.normal(size=(1, channels, height, width)).astype(np.float32)
    flow_n1_nchw = np.transpose(flow_n1, (0, 3, 1, 2)).astype(np.float32, copy=False)
    flow_n2_nchw = np.transpose(flow_n2, (0, 3, 1, 2)).astype(np.float32, copy=False)

    torch_output = _torch_propagation_warp_once(
        feat_current,
        feat_prop,
        feat_n2,
        flow_n1_nchw,
        flow_n2_nchw,
        padding_mode=padding_mode,
        device=torch_device,
    )
    mlx_output = _mlx_fused_propagation_warp_once(
        feat_current,
        feat_prop,
        feat_n2,
        flow_n1_nchw,
        flow_n2_nchw,
    )
    max_abs_error = float(np.max(np.abs(torch_output - mlx_output)))

    torch_ms = _time_torch(
        lambda: _torch_propagation_warp_tensor(
            feat_current,
            feat_prop,
            feat_n2,
            flow_n1_nchw,
            flow_n2_nchw,
            padding_mode=padding_mode,
            device=torch_device,
        ),
        device=torch_device,
        warmup=warmup,
        iters=iters,
    )
    mlx_ms = _time_mlx(
        lambda: _mlx_fused_propagation_warp_array(
            feat_current,
            feat_prop,
            feat_n2,
            flow_n1_nchw,
            flow_n2_nchw,
        ),
        warmup=warmup,
        iters=iters,
    )

    return _make_row(
        case="propagation_warp_fused",
        shape=f"1x{channels}x{height}x{width}",
        torch_device=torch_device,
        padding_mode=padding_mode,
        warmup=warmup,
        iters=iters,
        torch_ms=torch_ms,
        mlx_ms=mlx_ms,
        max_abs_error=max_abs_error,
        warp_calls=3,
        candidate="mlx_fused_resident",
    )


def run_bridge_fused_propagation_warp_case(
    *,
    channels: int,
    height: int,
    width: int,
    padding_mode: str,
    torch_device: str,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, float | int | str]:
    if padding_mode != "zeros":
        raise ValueError("fused propagation warp currently supports zeros padding only")
    rng = np.random.default_rng(seed)
    feat_current, flow_n1 = _make_feature_and_flow(rng, channels=channels, height=height, width=width)
    feat_prop, flow_n2 = _make_feature_and_flow(rng, channels=channels, height=height, width=width)
    feat_n2 = rng.normal(size=(1, channels, height, width)).astype(np.float32)
    flow_n1_nchw = np.transpose(flow_n1, (0, 3, 1, 2)).astype(np.float32, copy=False)
    flow_n2_nchw = np.transpose(flow_n2, (0, 3, 1, 2)).astype(np.float32, copy=False)
    tensors = tuple(
        torch.from_numpy(value).to(torch_device)
        for value in (feat_current, feat_prop, feat_n2, flow_n1_nchw, flow_n2_nchw)
    )

    torch_output = _torch_propagation_warp_tensor_resident(*tensors, padding_mode=padding_mode)
    _sync_torch(torch_device)
    mlx_output = _bridge_fused_propagation_warp_tensor_resident(*tensors, device=torch_device)
    _sync_torch(torch_device)
    max_abs_error = float(np.max(np.abs(torch_output.detach().cpu().numpy() - mlx_output.detach().cpu().numpy())))

    torch_ms = _time_torch(
        lambda: _torch_propagation_warp_tensor_resident(*tensors, padding_mode=padding_mode),
        device=torch_device,
        warmup=warmup,
        iters=iters,
    )
    mlx_ms = _time_torch(
        lambda: _bridge_fused_propagation_warp_tensor_resident(*tensors, device=torch_device),
        device=torch_device,
        warmup=warmup,
        iters=iters,
    )

    return _make_row(
        case="propagation_warp_fused_bridge",
        shape=f"1x{channels}x{height}x{width}",
        torch_device=torch_device,
        padding_mode=padding_mode,
        warmup=warmup,
        iters=iters,
        torch_ms=torch_ms,
        mlx_ms=mlx_ms,
        max_abs_error=max_abs_error,
        warp_calls=3,
        candidate="mlx_fused_bridge",
    )


def run_two_stage_propagation_warp_case(
    *,
    channels: int,
    height: int,
    width: int,
    padding_mode: str,
    torch_device: str,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, float | int | str]:
    if padding_mode != "zeros":
        raise ValueError("two-stage propagation warp currently supports zeros padding only")
    rng = np.random.default_rng(seed)
    feat_current, flow_n1 = _make_feature_and_flow(rng, channels=channels, height=height, width=width)
    feat_prop, flow_n2 = _make_feature_and_flow(rng, channels=channels, height=height, width=width)
    feat_n2 = rng.normal(size=(1, channels, height, width)).astype(np.float32)
    flow_n1_nchw = np.transpose(flow_n1, (0, 3, 1, 2)).astype(np.float32, copy=False)
    flow_n2_nchw = np.transpose(flow_n2, (0, 3, 1, 2)).astype(np.float32, copy=False)

    torch_output = _torch_propagation_warp_once(
        feat_current,
        feat_prop,
        feat_n2,
        flow_n1_nchw,
        flow_n2_nchw,
        padding_mode=padding_mode,
        device=torch_device,
    )
    mlx_output = _mlx_two_stage_propagation_warp_once(
        feat_current,
        feat_prop,
        feat_n2,
        flow_n1_nchw,
        flow_n2_nchw,
    )
    max_abs_error = float(np.max(np.abs(torch_output - mlx_output)))

    torch_ms = _time_torch(
        lambda: _torch_propagation_warp_tensor(
            feat_current,
            feat_prop,
            feat_n2,
            flow_n1_nchw,
            flow_n2_nchw,
            padding_mode=padding_mode,
            device=torch_device,
        ),
        device=torch_device,
        warmup=warmup,
        iters=iters,
    )
    mlx_ms = _time_mlx(
        lambda: _mlx_two_stage_propagation_warp_array(
            feat_current,
            feat_prop,
            feat_n2,
            flow_n1_nchw,
            flow_n2_nchw,
        ),
        warmup=warmup,
        iters=iters,
    )

    return _make_row(
        case="propagation_warp_two_stage",
        shape=f"1x{channels}x{height}x{width}",
        torch_device=torch_device,
        padding_mode=padding_mode,
        warmup=warmup,
        iters=iters,
        torch_ms=torch_ms,
        mlx_ms=mlx_ms,
        max_abs_error=max_abs_error,
        warp_calls=3,
        candidate="mlx_two_stage_resident",
    )


def run_bridge_two_stage_propagation_warp_case(
    *,
    channels: int,
    height: int,
    width: int,
    padding_mode: str,
    torch_device: str,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, float | int | str]:
    if padding_mode != "zeros":
        raise ValueError("two-stage propagation warp currently supports zeros padding only")
    rng = np.random.default_rng(seed)
    feat_current, flow_n1 = _make_feature_and_flow(rng, channels=channels, height=height, width=width)
    feat_prop, flow_n2 = _make_feature_and_flow(rng, channels=channels, height=height, width=width)
    feat_n2 = rng.normal(size=(1, channels, height, width)).astype(np.float32)
    flow_n1_nchw = np.transpose(flow_n1, (0, 3, 1, 2)).astype(np.float32, copy=False)
    flow_n2_nchw = np.transpose(flow_n2, (0, 3, 1, 2)).astype(np.float32, copy=False)
    tensors = tuple(
        torch.from_numpy(value).to(torch_device)
        for value in (feat_current, feat_prop, feat_n2, flow_n1_nchw, flow_n2_nchw)
    )

    torch_output = _torch_propagation_warp_tensor_resident(*tensors, padding_mode=padding_mode)
    _sync_torch(torch_device)
    mlx_output = _bridge_two_stage_propagation_warp_tensor_resident(*tensors, device=torch_device)
    _sync_torch(torch_device)
    max_abs_error = float(np.max(np.abs(torch_output.detach().cpu().numpy() - mlx_output.detach().cpu().numpy())))

    torch_ms = _time_torch(
        lambda: _torch_propagation_warp_tensor_resident(*tensors, padding_mode=padding_mode),
        device=torch_device,
        warmup=warmup,
        iters=iters,
    )
    mlx_ms = _time_torch(
        lambda: _bridge_two_stage_propagation_warp_tensor_resident(*tensors, device=torch_device),
        device=torch_device,
        warmup=warmup,
        iters=iters,
    )

    return _make_row(
        case="propagation_warp_two_stage_bridge",
        shape=f"1x{channels}x{height}x{width}",
        torch_device=torch_device,
        padding_mode=padding_mode,
        warmup=warmup,
        iters=iters,
        torch_ms=torch_ms,
        mlx_ms=mlx_ms,
        max_abs_error=max_abs_error,
        warp_calls=3,
        candidate="mlx_two_stage_bridge",
    )


def run_window_bridge_propagation_warp_case(
    *,
    channels: int,
    height: int,
    width: int,
    steps: int,
    padding_mode: str,
    torch_device: str,
    warmup: int,
    iters: int,
    seed: int,
) -> dict[str, float | int | str]:
    if padding_mode != "zeros":
        raise ValueError("window bridge propagation warp currently supports zeros padding only")
    rng = np.random.default_rng(seed)
    arrays = _make_window_propagation_inputs(
        rng,
        steps=steps,
        channels=channels,
        height=height,
        width=width,
    )
    tensors = tuple(torch.from_numpy(value).to(torch_device) for value in arrays)

    torch_output = _torch_window_propagation_warp_tensor_resident(*tensors, padding_mode=padding_mode)
    _sync_torch(torch_device)
    mlx_output = _bridge_window_two_stage_propagation_warp_tensor_resident(*tensors, device=torch_device)
    _sync_torch(torch_device)
    max_abs_error = float(np.max(np.abs(torch_output.detach().cpu().numpy() - mlx_output.detach().cpu().numpy())))

    torch_ms = _time_torch(
        lambda: _torch_window_propagation_warp_tensor_resident(*tensors, padding_mode=padding_mode),
        device=torch_device,
        warmup=warmup,
        iters=iters,
    )
    mlx_ms = _time_torch(
        lambda: _bridge_window_two_stage_propagation_warp_tensor_resident(*tensors, device=torch_device),
        device=torch_device,
        warmup=warmup,
        iters=iters,
    )

    row = _make_row(
        case="propagation_warp_window_bridge",
        shape=f"{steps}x{channels}x{height}x{width}",
        torch_device=torch_device,
        padding_mode=padding_mode,
        warmup=warmup,
        iters=iters,
        torch_ms=torch_ms,
        mlx_ms=mlx_ms,
        max_abs_error=max_abs_error,
        warp_calls=steps * 3,
        candidate="mlx_two_stage_window_bridge",
    )
    row["steps"] = steps
    row["ms_per_step_torch"] = torch_ms / max(steps, 1)
    row["ms_per_step_mlx"] = mlx_ms / max(steps, 1)
    return row


def _make_feature_and_flow(
    rng: np.random.Generator,
    *,
    channels: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.normal(size=(1, channels, height, width)).astype(np.float32)
    # Small offsets match BasicVSR++ flow-warp microbench use better than wild grids.
    flow = (rng.normal(size=(1, height, width, 2)) * 0.2).astype(np.float32)
    return x, flow


def _make_window_propagation_inputs(
    rng: np.random.Generator,
    *,
    steps: int,
    channels: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feat_current = rng.normal(size=(steps, channels, height, width)).astype(np.float32)
    feat_prop = rng.normal(size=(steps, channels, height, width)).astype(np.float32)
    feat_n2 = rng.normal(size=(steps, channels, height, width)).astype(np.float32)
    flow_n1 = (rng.normal(size=(steps, 2, height, width)) * 0.2).astype(np.float32)
    flow_n2 = (rng.normal(size=(steps, 2, height, width)) * 0.2).astype(np.float32)
    return feat_current, feat_prop, feat_n2, flow_n1, flow_n2


def _torch_flow_warp_once(x: np.ndarray, flow: np.ndarray, *, padding_mode: str, device: str) -> np.ndarray:
    y = _torch_flow_warp_tensor(x, flow, padding_mode=padding_mode, device=device)
    _sync_torch(device)
    return y.detach().cpu().numpy()


def _torch_flow_warp_tensor(x: np.ndarray, flow: np.ndarray, *, padding_mode: str, device: str) -> torch.Tensor:
    x_t = torch.from_numpy(x).to(device)
    flow_t = torch.from_numpy(flow).to(device)
    return _torch_flow_warp_tensor_resident(x_t, flow_t, padding_mode=padding_mode)


def _torch_flow_warp_tensor_resident(x_t: torch.Tensor, flow_t: torch.Tensor, *, padding_mode: str) -> torch.Tensor:
    return torch_flow_warp(x_t, flow_t, padding_mode=padding_mode)


def _mlx_flow_warp_once(x: np.ndarray, flow: np.ndarray, *, padding_mode: str) -> np.ndarray:
    y = _mlx_flow_warp_array(x, flow, padding_mode=padding_mode)
    mx.eval(y)
    return np.array(y)


def _mlx_flow_warp_array(x: np.ndarray, flow: np.ndarray, *, padding_mode: str) -> mx.array:
    return mlx_flow_warp(mx.array(x), mx.array(flow), padding_mode=padding_mode)


def _bridge_flow_warp_tensor_resident(
    x_t: torch.Tensor,
    flow_t: torch.Tensor,
    *,
    padding_mode: str,
    device: str,
) -> torch.Tensor:
    x_np = x_t.detach().cpu().numpy()
    flow_np = flow_t.detach().cpu().numpy()
    y = _mlx_flow_warp_array(x_np, flow_np, padding_mode=padding_mode)
    mx.eval(y)
    return torch.from_numpy(np.array(y)).to(device)


def _torch_propagation_warp_once(
    feat_current: np.ndarray,
    feat_prop: np.ndarray,
    feat_n2: np.ndarray,
    flow_n1: np.ndarray,
    flow_n2: np.ndarray,
    *,
    padding_mode: str,
    device: str,
) -> np.ndarray:
    y = _torch_propagation_warp_tensor(
        feat_current,
        feat_prop,
        feat_n2,
        flow_n1,
        flow_n2,
        padding_mode=padding_mode,
        device=device,
    )
    _sync_torch(device)
    return y.detach().cpu().numpy()


def _torch_propagation_warp_tensor(
    feat_current: np.ndarray,
    feat_prop: np.ndarray,
    feat_n2: np.ndarray,
    flow_n1: np.ndarray,
    flow_n2: np.ndarray,
    *,
    padding_mode: str,
    device: str,
) -> torch.Tensor:
    feat_current_t = torch.from_numpy(feat_current).to(device)
    feat_prop_t = torch.from_numpy(feat_prop).to(device)
    feat_n2_t = torch.from_numpy(feat_n2).to(device)
    flow_n1_t = torch.from_numpy(flow_n1).to(device)
    flow_n2_t = torch.from_numpy(flow_n2).to(device)
    return _torch_propagation_warp_tensor_resident(
        feat_current_t,
        feat_prop_t,
        feat_n2_t,
        flow_n1_t,
        flow_n2_t,
        padding_mode=padding_mode,
    )


def _torch_propagation_warp_tensor_resident(
    feat_current_t: torch.Tensor,
    feat_prop_t: torch.Tensor,
    feat_n2_t: torch.Tensor,
    flow_n1_t: torch.Tensor,
    flow_n2_t: torch.Tensor,
    *,
    padding_mode: str,
) -> torch.Tensor:
    flow_n1_grid = flow_n1_t.permute(0, 2, 3, 1).contiguous()
    cond_n1 = torch_flow_warp(feat_prop_t, flow_n1_grid, padding_mode=padding_mode)
    flow_n2_total = flow_n1_t + torch_flow_warp(flow_n2_t, flow_n1_grid, padding_mode=padding_mode)
    cond_n2 = torch_flow_warp(feat_n2_t, flow_n2_total.permute(0, 2, 3, 1).contiguous(), padding_mode=padding_mode)
    return torch.cat([cond_n1, feat_current_t, cond_n2], dim=1)


def _torch_window_propagation_warp_tensor_resident(
    feat_current_t: torch.Tensor,
    feat_prop_t: torch.Tensor,
    feat_n2_t: torch.Tensor,
    flow_n1_t: torch.Tensor,
    flow_n2_t: torch.Tensor,
    *,
    padding_mode: str,
) -> torch.Tensor:
    outputs = []
    for idx in range(feat_current_t.shape[0]):
        outputs.append(
            _torch_propagation_warp_tensor_resident(
                feat_current_t[idx : idx + 1],
                feat_prop_t[idx : idx + 1],
                feat_n2_t[idx : idx + 1],
                flow_n1_t[idx : idx + 1],
                flow_n2_t[idx : idx + 1],
                padding_mode=padding_mode,
            )
        )
    return torch.cat(outputs, dim=0)


def _bridge_propagation_warp_tensor_resident(
    feat_current_t: torch.Tensor,
    feat_prop_t: torch.Tensor,
    feat_n2_t: torch.Tensor,
    flow_n1_t: torch.Tensor,
    flow_n2_t: torch.Tensor,
    *,
    padding_mode: str,
    device: str,
) -> torch.Tensor:
    y = _mlx_propagation_warp_array(
        feat_current_t.detach().cpu().numpy(),
        feat_prop_t.detach().cpu().numpy(),
        feat_n2_t.detach().cpu().numpy(),
        flow_n1_t.detach().cpu().numpy(),
        flow_n2_t.detach().cpu().numpy(),
        padding_mode=padding_mode,
    )
    mx.eval(y)
    return torch.from_numpy(np.array(y)).to(device)


def _mlx_propagation_warp_once(
    feat_current: np.ndarray,
    feat_prop: np.ndarray,
    feat_n2: np.ndarray,
    flow_n1: np.ndarray,
    flow_n2: np.ndarray,
    *,
    padding_mode: str,
) -> np.ndarray:
    y = _mlx_propagation_warp_array(
        feat_current,
        feat_prop,
        feat_n2,
        flow_n1,
        flow_n2,
        padding_mode=padding_mode,
    )
    mx.eval(y)
    return np.array(y)


def _mlx_propagation_warp_array(
    feat_current: np.ndarray,
    feat_prop: np.ndarray,
    feat_n2: np.ndarray,
    flow_n1: np.ndarray,
    flow_n2: np.ndarray,
    *,
    padding_mode: str,
) -> mx.array:
    feat_current_m = mx.array(feat_current)
    feat_prop_m = mx.array(feat_prop)
    feat_n2_m = mx.array(feat_n2)
    flow_n1_m = mx.array(flow_n1)
    flow_n2_m = mx.array(flow_n2)
    flow_n1_grid = mx.transpose(flow_n1_m, (0, 2, 3, 1))
    cond_n1 = mlx_flow_warp(feat_prop_m, flow_n1_grid, padding_mode=padding_mode)
    flow_n2_total = flow_n1_m + mlx_flow_warp(flow_n2_m, flow_n1_grid, padding_mode=padding_mode)
    cond_n2 = mlx_flow_warp(feat_n2_m, mx.transpose(flow_n2_total, (0, 2, 3, 1)), padding_mode=padding_mode)
    return mx.concatenate([cond_n1, feat_current_m, cond_n2], axis=1)


def _mlx_fused_propagation_warp_once(
    feat_current: np.ndarray,
    feat_prop: np.ndarray,
    feat_n2: np.ndarray,
    flow_n1: np.ndarray,
    flow_n2: np.ndarray,
) -> np.ndarray:
    y = _mlx_fused_propagation_warp_array(feat_current, feat_prop, feat_n2, flow_n1, flow_n2)
    mx.eval(y)
    return np.array(y)


def _mlx_fused_propagation_warp_array(
    feat_current: np.ndarray,
    feat_prop: np.ndarray,
    feat_n2: np.ndarray,
    flow_n1: np.ndarray,
    flow_n2: np.ndarray,
) -> mx.array:
    return fused_propagation_warp_cond(
        mx.array(feat_current),
        mx.array(feat_prop),
        mx.array(feat_n2),
        mx.array(flow_n1),
        mx.array(flow_n2),
    )


def _bridge_fused_propagation_warp_tensor_resident(
    feat_current_t: torch.Tensor,
    feat_prop_t: torch.Tensor,
    feat_n2_t: torch.Tensor,
    flow_n1_t: torch.Tensor,
    flow_n2_t: torch.Tensor,
    *,
    device: str,
) -> torch.Tensor:
    y = _mlx_fused_propagation_warp_array(
        feat_current_t.detach().cpu().numpy(),
        feat_prop_t.detach().cpu().numpy(),
        feat_n2_t.detach().cpu().numpy(),
        flow_n1_t.detach().cpu().numpy(),
        flow_n2_t.detach().cpu().numpy(),
    )
    mx.eval(y)
    return torch.from_numpy(np.array(y)).to(device)


def _mlx_two_stage_propagation_warp_once(
    feat_current: np.ndarray,
    feat_prop: np.ndarray,
    feat_n2: np.ndarray,
    flow_n1: np.ndarray,
    flow_n2: np.ndarray,
) -> np.ndarray:
    y = _mlx_two_stage_propagation_warp_array(feat_current, feat_prop, feat_n2, flow_n1, flow_n2)
    mx.eval(y)
    return np.array(y)


def _mlx_two_stage_propagation_warp_array(
    feat_current: np.ndarray,
    feat_prop: np.ndarray,
    feat_n2: np.ndarray,
    flow_n1: np.ndarray,
    flow_n2: np.ndarray,
) -> mx.array:
    return two_stage_propagation_warp_cond(
        mx.array(feat_current),
        mx.array(feat_prop),
        mx.array(feat_n2),
        mx.array(flow_n1),
        mx.array(flow_n2),
    )


def _bridge_two_stage_propagation_warp_tensor_resident(
    feat_current_t: torch.Tensor,
    feat_prop_t: torch.Tensor,
    feat_n2_t: torch.Tensor,
    flow_n1_t: torch.Tensor,
    flow_n2_t: torch.Tensor,
    *,
    device: str,
) -> torch.Tensor:
    y = _mlx_two_stage_propagation_warp_array(
        feat_current_t.detach().cpu().numpy(),
        feat_prop_t.detach().cpu().numpy(),
        feat_n2_t.detach().cpu().numpy(),
        flow_n1_t.detach().cpu().numpy(),
        flow_n2_t.detach().cpu().numpy(),
    )
    mx.eval(y)
    return torch.from_numpy(np.array(y)).to(device)


def _bridge_window_two_stage_propagation_warp_tensor_resident(
    feat_current_t: torch.Tensor,
    feat_prop_t: torch.Tensor,
    feat_n2_t: torch.Tensor,
    flow_n1_t: torch.Tensor,
    flow_n2_t: torch.Tensor,
    *,
    device: str,
) -> torch.Tensor:
    y = two_stage_propagation_warp_cond(
        mx.array(feat_current_t.detach().cpu().numpy()),
        mx.array(feat_prop_t.detach().cpu().numpy()),
        mx.array(feat_n2_t.detach().cpu().numpy()),
        mx.array(flow_n1_t.detach().cpu().numpy()),
        mx.array(flow_n2_t.detach().cpu().numpy()),
    )
    mx.eval(y)
    return torch.from_numpy(np.array(y)).to(device)


def _time_torch(fn: Callable[[], torch.Tensor], *, device: str, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        y = fn()
        _sync_torch(device)
        del y
    start = time.perf_counter()
    for _ in range(iters):
        y = fn()
        _sync_torch(device)
        del y
    return (time.perf_counter() - start) * 1000 / max(iters, 1)


def _time_mlx(fn: Callable[[], mx.array], *, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        y = fn()
        mx.eval(y)
    start = time.perf_counter()
    for _ in range(iters):
        y = fn()
        mx.eval(y)
    return (time.perf_counter() - start) * 1000 / max(iters, 1)


def _sync_torch(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _resolve_torch_device(requested: str) -> str:
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("torch MPS is not available")
    return requested


def _make_row(
    *,
    case: str,
    shape: str,
    torch_device: str,
    padding_mode: str,
    warmup: int,
    iters: int,
    torch_ms: float,
    mlx_ms: float,
    max_abs_error: float,
    warp_calls: int,
    candidate: str = "mlx_resident",
) -> dict[str, float | int | str]:
    return {
        "case": case,
        "candidate": candidate,
        "shape": shape,
        "torch_device": torch_device,
        "padding_mode": padding_mode,
        "warmup": warmup,
        "iters": iters,
        "warp_calls": warp_calls,
        "torch_ms": torch_ms,
        "mlx_ms": mlx_ms,
        "speedup_mlx_vs_torch": torch_ms / mlx_ms if mlx_ms > 0 else float("inf"),
        "max_abs_error": max_abs_error,
    }


def _print_row(row: dict[str, float | int | str]) -> None:
    print(
        f"{row['case']} candidate={row['candidate']} shape={row['shape']} padding={row['padding_mode']} "
        f"torch_device={row['torch_device']} warmup={row['warmup']} iters={row['iters']}"
    )
    print(
        f"  torch={float(row['torch_ms']):.3f} ms/iter "
        f"mlx={float(row['mlx_ms']):.3f} ms/iter "
        f"speedup_mlx_vs_torch={float(row['speedup_mlx_vs_torch']):.2f}x "
        f"max_abs_error={float(row['max_abs_error']):.6g} "
        f"warp_calls={row['warp_calls']}"
    )


if __name__ == "__main__":
    main()
