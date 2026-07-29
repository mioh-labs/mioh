# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Compile and numerically verify the RF-DETR Core AI Metal kernel."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from lada.coreai.compiled_runtime import CompiledCoreAIRuntime, TensorSpec

if __package__:
    from .rfdetr_coreai_kernels import (
        ATTENTION_HEADS,
        FEATURE_SIDE,
        POINTS_PER_HEAD,
        build_ms_deform_attn_kernel,
        run_ms_deform_attn_kernel,
    )
else:
    from rfdetr_coreai_kernels import (  # type: ignore[import-not-found]
        ATTENTION_HEADS,
        FEATURE_SIDE,
        POINTS_PER_HEAD,
        build_ms_deform_attn_kernel,
        run_ms_deform_attn_kernel,
    )


class KernelWrapper(torch.nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, value, sampling_locations, attention_weights):
        return run_ms_deform_attn_kernel(
            self.kernel,
            value,
            sampling_locations,
            attention_weights,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--architecture", default="h17s")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import coreai_torch

    generator = torch.Generator().manual_seed(args.seed)
    value = torch.randn(
        1,
        FEATURE_SIDE * FEATURE_SIDE,
        ATTENTION_HEADS,
        16,
        generator=generator,
        dtype=torch.float16,
    )
    locations = torch.rand(
        1,
        100,
        ATTENTION_HEADS,
        POINTS_PER_HEAD,
        2,
        generator=generator,
        dtype=torch.float16,
    )
    weights = torch.randn(
        1,
        100,
        ATTENTION_HEADS,
        POINTS_PER_HEAD,
        generator=generator,
        dtype=torch.float16,
    ).softmax(-1)

    kernel = build_ms_deform_attn_kernel(coreai_torch)
    wrapper = KernelWrapper(kernel).eval()
    with torch.inference_mode():
        expected = wrapper(value, locations, weights).numpy()
    exported = torch.export.export(
        wrapper,
        (value, locations, weights),
        strict=False,
    ).run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.register_custom_kernels([kernel])
    converter.add_exported_program(
        exported,
        input_names=["value", "sampling_locations", "attention_weights"],
        output_names=["output"],
    )
    program = converter.to_coreai()
    program.optimize()

    with tempfile.TemporaryDirectory(prefix="rfdetr-coreai-kernel-") as directory:
        root = Path(directory)
        source = root / "kernel.aimodel"
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
                args.architecture,
            ],
            check=True,
        )
        compiled = root / f"kernel.{args.architecture}.aimodelc"
        runtime = CompiledCoreAIRuntime(
            compiled,
            (
                TensorSpec("value", tuple(value.shape)),
                TensorSpec("sampling_locations", tuple(locations.shape)),
                TensorSpec("attention_weights", tuple(weights.shape)),
            ),
            (TensorSpec("output", tuple(expected.shape)),),
            runner_path=str(args.runner),
        )
        durations: list[float] = []
        result = None
        inputs = {
            "value": np.ascontiguousarray(value.numpy()),
            "sampling_locations": np.ascontiguousarray(locations.numpy()),
            "attention_weights": np.ascontiguousarray(weights.numpy()),
        }
        try:
            for index in range(args.runs + 5):
                started = time.perf_counter()
                result = runtime.infer(inputs)["output"]
                if index >= 5:
                    durations.append((time.perf_counter() - started) * 1000)
        finally:
            runtime.close()

    assert result is not None
    difference = np.abs(result.astype(np.float32) - expected.astype(np.float32))
    report = {
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "median_ms": statistics.median(durations),
        "mean_ms": statistics.fmean(durations),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
