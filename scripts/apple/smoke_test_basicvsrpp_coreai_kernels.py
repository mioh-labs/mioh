#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Compile and run the BasicVSR++ Core AI Metal kernels on the local Mac.

This is an execution test, not only an export test.  It verifies the actual
compiled Metal 4 kernels against PyTorch and records their warm latency.  Run
it from the isolated ``.venv-coreai`` environment.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from lada.coreai.compiled_runtime import CompiledCoreAIRuntime, TensorSpec

if __package__:
    from .basicvsrpp_coreai_kernels import (
        build_deform_conv_kernel,
        build_flow_warp_kernel,
        build_grid_sample_kernel,
        run_deform_conv_kernel,
        run_flow_warp_kernel,
        run_grid_sample_kernel,
    )
else:
    from basicvsrpp_coreai_kernels import (  # type: ignore[import-not-found]
        build_deform_conv_kernel,
        build_flow_warp_kernel,
        build_grid_sample_kernel,
        run_deform_conv_kernel,
        run_flow_warp_kernel,
        run_grid_sample_kernel,
    )


MAX_ACCEPTABLE_ABS_ERROR = 2.0e-3


class GridSampleWrapper(torch.nn.Module):
    def __init__(self, kernel: Any, padding_mode: str):
        super().__init__()
        self.kernel = kernel
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        return run_grid_sample_kernel(
            self.kernel,
            image,
            grid,
            self.padding_mode,
        )


class DeformConvWrapper(torch.nn.Module):
    def __init__(self, kernel: Any):
        super().__init__()
        self.kernel = kernel

    def forward(
        self,
        image: torch.Tensor,
        offset: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        return run_deform_conv_kernel(
            self.kernel,
            image,
            offset,
            weight,
            bias,
            mask,
        )


class FlowWarpWrapper(torch.nn.Module):
    def __init__(self, kernel: Any, padding_mode: str):
        super().__init__()
        self.kernel = kernel
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        return run_flow_warp_kernel(
            self.kernel,
            image,
            flow,
            self.padding_mode,
        )


class LegacyFlowWarpWrapper(torch.nn.Module):
    """Current Core AI path: build a normalized grid, then call grid sample."""

    def __init__(self, kernel: Any):
        super().__init__()
        self.kernel = kernel

    def forward(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[-2:]
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=flow.device, dtype=image.dtype),
            torch.arange(width, device=flow.device, dtype=image.dtype),
            indexing="ij",
        )
        flow_nhwc = flow.permute(0, 2, 3, 1)
        grid_flow = torch.stack((grid_x, grid_y), dim=-1) + flow_nhwc
        grid = torch.stack(
            (
                2.0 * grid_flow[..., 0] / max(width - 1, 1) - 1.0,
                2.0 * grid_flow[..., 1] / max(height - 1, 1) - 1.0,
            ),
            dim=-1,
        )
        return run_grid_sample_kernel(self.kernel, image, grid)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operation",
        choices=("grid", "flow", "flow-legacy", "dcn", "all"),
        default="all",
    )
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--architecture", default="h17s")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup-runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-side", type=int, default=64)
    parser.add_argument("--dcn-side", type=int, default=16)
    parser.add_argument(
        "--flow-pattern",
        choices=("random", "zero", "x1"),
        default="random",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dump-output", type=Path)
    return parser.parse_args(argv)


def _convert_compile(
    *,
    wrapper: torch.nn.Module,
    examples: tuple[torch.Tensor, ...],
    input_names: tuple[str, ...],
    output_name: str,
    kernels: list[Any],
    root: Path,
    tag: str,
    architecture: str,
) -> Path:
    import coreai_torch

    exported = torch.export.export(wrapper.eval(), examples, strict=False)
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.register_custom_kernels(kernels)
    converter.add_exported_program(
        exported,
        input_names=list(input_names),
        output_names=[output_name],
    )
    program = converter.to_coreai()
    program.optimize()
    source = root / f"{tag}.aimodel"
    program.save_asset(source)
    subprocess.run(
        [
            "xcrun",
            "coreai-build",
            "compile",
            str(source),
            "--output",
            str(root),
            "--platform",
            "macOS",
            "--min-deployment-version",
            "27.0",
            "--preferred-compute",
            "gpu",
            "--architecture",
            architecture,
        ],
        check=True,
    )
    compiled = root / f"{tag}.{architecture}.aimodelc"
    if not compiled.is_dir():
        raise FileNotFoundError(compiled)
    return compiled


def _run_compiled(
    *,
    model: Path,
    runner: Path,
    inputs: dict[str, np.ndarray],
    output_name: str,
    output_shape: tuple[int, ...],
    runs: int,
    warmup_runs: int,
) -> tuple[np.ndarray, list[float]]:
    runtime = CompiledCoreAIRuntime(
        model,
        tuple(TensorSpec(name, tuple(value.shape)) for name, value in inputs.items()),
        (TensorSpec(output_name, output_shape),),
        runner_path=str(runner),
    )
    durations: list[float] = []
    result: np.ndarray | None = None
    try:
        for index in range(warmup_runs + runs):
            started = time.perf_counter()
            result = runtime.infer(inputs)[output_name]
            if index >= warmup_runs:
                durations.append((time.perf_counter() - started) * 1_000.0)
    finally:
        runtime.close()
    if result is None:
        raise RuntimeError("Core AI kernel did not return an output")
    return result, durations


def _comparison(
    actual: np.ndarray,
    expected: np.ndarray,
    durations: list[float],
) -> dict[str, float]:
    difference = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
    mse = float(np.mean(np.square(difference), dtype=np.float64))
    psnr = math.inf if mse == 0 else 10.0 * math.log10(1.0 / mse)
    return {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "psnr_db": psnr,
        "median_ms": statistics.median(durations),
        "mean_ms": statistics.fmean(durations),
    }


def _grid_probe(args: argparse.Namespace, root: Path) -> dict[str, object]:
    import coreai_torch

    generator = torch.Generator().manual_seed(args.seed)
    side = args.grid_side
    image = torch.rand(
        (1, 64, side, side), generator=generator, dtype=torch.float16
    )
    # Include coordinates outside [-1, 1] so zero padding is exercised.
    grid = (
        torch.rand((1, side, side, 2), generator=generator, dtype=torch.float16)
        * 2.4
        - 1.2
    )
    expected = F.grid_sample(
        image.float(),
        grid.float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).half().numpy()
    kernel = build_grid_sample_kernel(coreai_torch)
    wrapper = GridSampleWrapper(kernel, "zeros")
    compiled = _convert_compile(
        wrapper=wrapper,
        examples=(image, grid),
        input_names=("image", "grid"),
        output_name="warped",
        kernels=[kernel],
        root=root,
        tag="grid-sample",
        architecture=args.architecture,
    )
    actual, durations = _run_compiled(
        model=compiled,
        runner=args.runner,
        inputs={
            "image": np.ascontiguousarray(image.numpy()),
            "grid": np.ascontiguousarray(grid.numpy()),
        },
        output_name="warped",
        output_shape=tuple(expected.shape),
        runs=args.runs,
        warmup_runs=args.warmup_runs,
    )
    return {
        "shape": list(expected.shape),
        **_comparison(actual, expected, durations),
    }


def _dcn_probe(args: argparse.Namespace, root: Path) -> dict[str, object]:
    import coreai_torch

    generator = torch.Generator().manual_seed(args.seed + 1)
    side = args.dcn_side
    # These are the real BasicVSR++ DCNv2 channel and deform-group dimensions.
    image = torch.rand(
        (1, 128, side, side), generator=generator, dtype=torch.float16
    )
    offset = (
        torch.rand(
            (1, 16 * 2 * 9, side, side),
            generator=generator,
            dtype=torch.float16,
        )
        - 0.5
    )
    weight = (
        torch.rand((64, 128, 3, 3), generator=generator, dtype=torch.float16)
        - 0.5
    ) * 0.05
    bias = (
        torch.rand((64,), generator=generator, dtype=torch.float16) - 0.5
    ) * 0.05
    mask = torch.rand(
        (1, 16 * 9, side, side), generator=generator, dtype=torch.float16
    )
    expected = torchvision.ops.deform_conv2d(
        image.float(),
        offset.float(),
        weight.float(),
        bias.float(),
        padding=(1, 1),
        mask=mask.float(),
    ).half().numpy()
    kernel = build_deform_conv_kernel(coreai_torch)
    wrapper = DeformConvWrapper(kernel)
    compiled = _convert_compile(
        wrapper=wrapper,
        examples=(image, offset, weight, bias, mask),
        input_names=("image", "offset", "weight", "bias", "mask"),
        output_name="aligned",
        kernels=[kernel],
        root=root,
        tag="deform-conv",
        architecture=args.architecture,
    )
    inputs = {
        "image": np.ascontiguousarray(image.numpy()),
        "offset": np.ascontiguousarray(offset.numpy()),
        "weight": np.ascontiguousarray(weight.numpy()),
        "bias": np.ascontiguousarray(bias.numpy()),
        "mask": np.ascontiguousarray(mask.numpy()),
    }
    actual, durations = _run_compiled(
        model=compiled,
        runner=args.runner,
        inputs=inputs,
        output_name="aligned",
        output_shape=tuple(expected.shape),
        runs=args.runs,
        warmup_runs=args.warmup_runs,
    )
    return {
        "shape": list(expected.shape),
        **_comparison(actual, expected, durations),
    }


def _flow_probe(args: argparse.Namespace, root: Path) -> dict[str, object]:
    import coreai_torch

    generator = torch.Generator().manual_seed(args.seed + 2)
    side = args.grid_side
    image = torch.rand(
        (1, 64, side, side), generator=generator, dtype=torch.float16
    )
    flow = _make_flow(args, generator, side)
    expected = _high_precision_flow_warp_reference(image, flow)
    kernel = build_flow_warp_kernel(coreai_torch)
    wrapper = FlowWarpWrapper(kernel, "zeros")
    compiled = _convert_compile(
        wrapper=wrapper,
        examples=(image, flow),
        input_names=("image", "flow"),
        output_name="warped",
        kernels=[kernel],
        root=root,
        tag="flow-warp",
        architecture=args.architecture,
    )
    actual, durations = _run_compiled(
        model=compiled,
        runner=args.runner,
        inputs={
            "image": np.ascontiguousarray(image.numpy()),
            "flow": np.ascontiguousarray(flow.numpy()),
        },
        output_name="warped",
        output_shape=tuple(expected.shape),
        runs=args.runs,
        warmup_runs=args.warmup_runs,
    )
    if args.dump_output:
        args.dump_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.dump_output,
            image=image.numpy(),
            flow=flow.numpy(),
            expected=expected,
            actual=actual,
        )
    return {
        "shape": list(expected.shape),
        **_comparison(actual, expected, durations),
    }


def _legacy_flow_probe(args: argparse.Namespace, root: Path) -> dict[str, object]:
    import coreai_torch

    generator = torch.Generator().manual_seed(args.seed + 2)
    side = args.grid_side
    image = torch.rand(
        (1, 64, side, side), generator=generator, dtype=torch.float16
    )
    flow = _make_flow(args, generator, side)
    expected = _legacy_flow_warp_reference(image, flow)
    kernel = build_grid_sample_kernel(coreai_torch)
    wrapper = LegacyFlowWarpWrapper(kernel)
    compiled = _convert_compile(
        wrapper=wrapper,
        examples=(image, flow),
        input_names=("image", "flow"),
        output_name="warped",
        kernels=[kernel],
        root=root,
        tag="flow-warp-legacy",
        architecture=args.architecture,
    )
    actual, durations = _run_compiled(
        model=compiled,
        runner=args.runner,
        inputs={
            "image": np.ascontiguousarray(image.numpy()),
            "flow": np.ascontiguousarray(flow.numpy()),
        },
        output_name="warped",
        output_shape=tuple(expected.shape),
        runs=args.runs,
        warmup_runs=args.warmup_runs,
    )
    return {
        "shape": list(expected.shape),
        **_comparison(actual, expected, durations),
    }


def _make_flow(
    args: argparse.Namespace,
    generator: torch.Generator,
    side: int,
) -> torch.Tensor:
    if args.flow_pattern == "zero":
        return torch.zeros((1, 2, side, side), dtype=torch.float16)
    if args.flow_pattern == "x1":
        flow = torch.zeros((1, 2, side, side), dtype=torch.float16)
        flow[:, 0] = 1.0
        return flow
    return (
        torch.rand(
            (1, 2, side, side), generator=generator, dtype=torch.float16
        )
        * 4.0
        - 2.0
    )


def _legacy_flow_warp_reference(
    image: torch.Tensor, flow: torch.Tensor
) -> np.ndarray:
    height, width = image.shape[-2:]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=image.dtype),
        torch.arange(width, dtype=image.dtype),
        indexing="ij",
    )
    grid_flow = (
        torch.stack((grid_x, grid_y), dim=-1)
        + flow.permute(0, 2, 3, 1)
    )
    grid = torch.stack(
        (
            2.0 * grid_flow[..., 0] / max(width - 1, 1) - 1.0,
            2.0 * grid_flow[..., 1] / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    )
    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).numpy()


def _high_precision_flow_warp_reference(
    image: torch.Tensor, flow: torch.Tensor
) -> np.ndarray:
    height, width = image.shape[-2:]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    grid_flow = (
        torch.stack((grid_x, grid_y), dim=-1)
        + flow.permute(0, 2, 3, 1).float()
    )
    grid = torch.stack(
        (
            2.0 * grid_flow[..., 0] / max(width - 1, 1) - 1.0,
            2.0 * grid_flow[..., 1] / max(height - 1, 1) - 1.0,
        ),
        dim=-1,
    )
    return F.grid_sample(
        image.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).half().numpy()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.runs < 1 or args.warmup_runs < 0:
        raise ValueError("runs must be positive and warmup-runs cannot be negative")
    if not args.runner.is_file():
        raise FileNotFoundError(args.runner)

    with tempfile.TemporaryDirectory(prefix="basicvsrpp-coreai-kernels-") as directory:
        root = Path(directory)
        report: dict[str, object] = {
            "architecture": args.architecture,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
        }
        if args.operation in {"grid", "all"}:
            report["grid_sample"] = _grid_probe(args, root)
        if args.operation in {"flow", "all"}:
            report["flow_warp"] = _flow_probe(args, root)
        if args.operation == "flow-legacy":
            report["flow_warp_legacy"] = _legacy_flow_probe(args, root)
        if args.operation in {"dcn", "all"}:
            report["deform_conv2d"] = _dcn_probe(args, root)

    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=True)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    failed: list[str] = []
    for name in (
        "grid_sample",
        "flow_warp",
        "flow_warp_legacy",
        "deform_conv2d",
    ):
        result = report.get(name)
        if (
            isinstance(result, dict)
            and float(result["max_abs"]) > MAX_ACCEPTABLE_ABS_ERROR
        ):
            failed.append(name)
    if failed:
        raise RuntimeError(
            "Core AI kernel accuracy gate failed for: " + ", ".join(failed)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
