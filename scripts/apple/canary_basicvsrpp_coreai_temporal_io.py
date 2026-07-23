#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Core AI canary for temporal tensor packing and state lowering.

This deliberately exports equivalent BasicVSR++ propagation graphs twice:
one receives a contiguous ``[K,C,H,W]`` context tensor, while the other
receives K separately named context inputs.  The packed graph is the shipping
contract. The separate graph is retained as a beta regression control and as
the starting point for reducing the original full-pipeline failure.

Run this after each macOS/Xcode beta update with the repository's Core AI
environment:

    .venv-coreai/bin/python \
      scripts/apple/canary_basicvsrpp_coreai_temporal_io.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import platform
import shutil
import subprocess
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from coreai.runtime import AIModel, NDArray
from lada.coreai.compiled_runtime import CompiledCoreAIRuntime, TensorSpec
from scripts.apple.basicvsrpp_coreai_kernels import (
    build_deform_conv_kernel,
    build_grid_sample_kernel,
)
from scripts.apple.benchmark_basicvsrpp_variable_coreai import (
    MID_CHANNELS,
    PropagationFirst,
    PropagationInit,
    PropagationLater,
)
from scripts.apple.export_basicvsrpp_coreai import (
    import_coreai,
    load_generator,
    save_program_asset,
    use_deform_conv_metal_kernel,
    use_grid_sample_metal_kernel,
)


class PackedPropagation(torch.nn.Module):
    """One temporal input, used by production exports."""

    def __init__(
        self,
        generator: torch.nn.Module,
        length: int,
        branch: str = "forward_2",
    ):
        super().__init__()
        self.length = length
        self.initial = PropagationInit(generator, branch)
        self.first = PropagationFirst(generator, branch)
        self.later = PropagationLater(generator, branch)

    def forward(self, contexts: torch.Tensor, flows: torch.Tensor) -> torch.Tensor:
        outputs = [self.initial(contexts[0:1])]
        outputs.append(self.first(contexts[1:2], outputs[-1], flows[0:1]))
        for index in range(2, self.length):
            outputs.append(
                self.later(
                    contexts[index : index + 1],
                    outputs[-1],
                    outputs[-2],
                    flows[index - 1 : index],
                    flows[index - 2 : index - 1],
                )
            )
        return torch.cat(outputs, dim=0)


class SeparateComponentPropagation(torch.nn.Module):
    """Old contract: every frame/component pair is a separate named input."""

    def __init__(
        self,
        generator: torch.nn.Module,
        length: int,
        components: int = 4,
        branch: str = "forward_2",
    ):
        super().__init__()
        self.length = length
        self.components = components
        self.initial = PropagationInit(generator, branch)
        self.first = PropagationFirst(generator, branch)
        self.later = PropagationLater(generator, branch)

    def forward(self, *values: torch.Tensor) -> torch.Tensor:
        feature_count = self.length * self.components
        features = values[:feature_count]
        flows = values[feature_count:]
        contexts = [
            torch.cat(
                features[
                    index * self.components : (index + 1) * self.components
                ],
                dim=1,
            )
            for index in range(self.length)
        ]
        outputs = [self.initial(contexts[0])]
        outputs.append(self.first(contexts[1], outputs[-1], flows[0]))
        for index in range(2, self.length):
            outputs.append(
                self.later(
                    contexts[index],
                    outputs[-1],
                    outputs[-2],
                    flows[index - 1],
                    flows[index - 2],
                )
            )
        return torch.cat(outputs, dim=0)


async def infer(path: Path, values: dict[str, np.ndarray]) -> np.ndarray:
    model = await AIModel.load(path)
    function = model.load_function("main")
    result = await function(
        {
            name: NDArray(np.ascontiguousarray(value))
            for name, value in values.items()
        }
    )
    return result["features"].numpy().copy()


def metrics(reference: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    difference = reference.astype(np.float32) - actual.astype(np.float32)
    absolute = np.abs(difference)
    rmse = float(np.sqrt(np.mean(np.square(difference))))
    peak = max(1.0, float(np.max(np.abs(reference))))
    psnr = float("inf") if rmse == 0 else 20.0 * math.log10(peak / rmse)
    return {
        "max_absolute_error": float(absolute.max()),
        "mean_absolute_error": float(absolute.mean()),
        "rmse": rmse,
        "relative_psnr_db": psnr,
    }


def export_and_run(
    *,
    module: torch.nn.Module,
    arrays: tuple[np.ndarray, ...],
    input_names: tuple[str, ...],
    destination: Path,
    coreai_torch,
    grid_kernel,
    deform_kernel,
    architecture: str | None,
    runner_path: Path,
) -> dict[str, dict[str, float]]:
    module = module.half().eval()
    examples = tuple(torch.from_numpy(value) for value in arrays)
    with torch.inference_mode():
        reference = module(*examples).numpy().copy()
    with torch.no_grad(), ExitStack() as stack:
        stack.enter_context(use_grid_sample_metal_kernel(grid_kernel))
        stack.enter_context(use_deform_conv_metal_kernel(deform_kernel))
        exported = torch.export.export(module, examples)
        exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.register_custom_kernels([grid_kernel, deform_kernel])
    converter.add_exported_program(
        exported,
        input_names=list(input_names),
        output_names=["features"],
    )
    if destination.exists():
        shutil.rmtree(destination)
    save_program_asset(converter.to_coreai(), destination)
    actual = asyncio.run(
        infer(destination, dict(zip(input_names, arrays, strict=True)))
    )
    result = {"source": metrics(reference, actual)}
    if architecture is not None:
        compiled_root = destination.parent / "compiled"
        compiled_root.mkdir(parents=True, exist_ok=True)
        compiled_path = (
            compiled_root / f"{destination.stem}.{architecture}.aimodelc"
        )
        if compiled_path.exists():
            shutil.rmtree(compiled_path)
        subprocess.run(
            [
                "xcrun",
                "coreai-build",
                "compile",
                str(destination),
                "--output",
                str(compiled_root),
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
        compiled_runtime = CompiledCoreAIRuntime(
            compiled_path,
            tuple(
                TensorSpec(name, tuple(value.shape))
                for name, value in zip(input_names, arrays, strict=True)
            ),
            (TensorSpec("features", tuple(reference.shape)),),
            runner_path=str(runner_path),
        )
        try:
            compiled_result = compiled_runtime.infer(
                dict(zip(input_names, arrays, strict=True))
            )
        finally:
            compiled_runtime.close()
        result["compiled"] = metrics(reference, compiled_result["features"])
    return result


def inspect_state_asset(path: Path) -> dict[str, object]:
    command = ["xcrun", "coreai-build", "inspect", str(path), "--json"]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    payload = completed.stdout.strip() or completed.stderr.strip()
    lowered = None
    state_mentions: list[str] = []
    try:
        parsed = json.loads(payload)

        def visit(value, location: str = "") -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    child_location = f"{location}.{key}" if location else key
                    if "state" in key.lower():
                        state_mentions.append(child_location)
                    visit(child, child_location)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f"{location}[{index}]")

        visit(parsed)
        lowered = not state_mentions
    except json.JSONDecodeError:
        parsed = {"raw": payload}
    return {
        "path": str(path),
        "exit_code": completed.returncode,
        "state_fields": state_mentions,
        "appears_lowered_to_regular_io": lowered,
        "inspect": parsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("model_weights/lada_mosaic_restoration_model_generic_v1.2.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/mioh-coreai-temporal-io-canary"),
    )
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--minimum-packed-psnr",
        type=float,
        default=45.0,
        help="fail if a contiguous-input export falls below this value",
    )
    parser.add_argument(
        "--architecture",
        default="h17s",
        help="Core AI specialization to compile and run; use 'none' for source only",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(
            "build/macos-standalone/mioh.app/Contents/Resources/bin/"
            "lada-coreai-runner"
        ),
        help="Swift runner used for compiled-specialization inference",
    )
    parser.add_argument(
        "--state-asset",
        type=Path,
        action="append",
        default=[],
        help="optional .aimodel/.aimodelc to inspect for native state I/O",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    architecture = None if args.architecture.lower() == "none" else args.architecture
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if architecture is not None and not args.runner.is_file():
        raise FileNotFoundError(f"compiled Core AI runner not found: {args.runner}")
    _, coreai_torch = import_coreai()
    generator = load_generator(args.checkpoint).generator
    grid_kernel = build_grid_sample_kernel(coreai_torch)
    deform_kernel = build_deform_conv_kernel(coreai_torch)
    rng = np.random.default_rng(args.seed)
    components = tuple(
        rng.random((1, MID_CHANNELS, 64, 64), dtype=np.float32).astype(
            np.float16
        )
        for _ in range(6 * 4)
    )
    flows = (rng.random((5, 2, 64, 64), dtype=np.float32) - 0.5).astype(
        np.float16
    )
    report: dict[str, object] = {
        "platform": platform.platform(),
        "torch": torch.__version__,
        "checkpoint": str(args.checkpoint),
        "results": {},
    }
    results = report["results"]
    assert isinstance(results, dict)
    failed = False

    for length in (2, 6):
        packed_contexts = np.concatenate(
            [
                np.concatenate(
                    components[index * 4 : (index + 1) * 4],
                    axis=1,
                )
                for index in range(length)
            ],
            axis=0,
        )
        packed = PackedPropagation(generator, length)
        packed_metrics = export_and_run(
            module=packed,
            arrays=(packed_contexts, flows[: length - 1]),
            input_names=("contexts", "flows"),
            destination=args.output_dir / f"packed-k{length}.aimodel",
            coreai_torch=coreai_torch,
            grid_kernel=grid_kernel,
            deform_kernel=deform_kernel,
            architecture=architecture,
            runner_path=args.runner,
        )
        results[f"packed_k{length}"] = packed_metrics
        packed_gate = packed_metrics.get("compiled", packed_metrics["source"])
        if packed_gate["relative_psnr_db"] < args.minimum_packed_psnr:
            failed = True

        separate_module = SeparateComponentPropagation(generator, length)
        separate_arrays = tuple(components[: length * 4])
        separate_arrays += tuple(flows[index : index + 1] for index in range(length - 1))
        separate_names = tuple(
            f"frame{frame}_feature{component}"
            for frame in range(length)
            for component in range(4)
        )
        separate_names += tuple(f"flow{index}" for index in range(length - 1))
        results[f"separate_k{length}"] = export_and_run(
            module=separate_module,
            arrays=separate_arrays,
            input_names=separate_names,
            destination=args.output_dir / f"separate-k{length}.aimodel",
            coreai_torch=coreai_torch,
            grid_kernel=grid_kernel,
            deform_kernel=deform_kernel,
            architecture=architecture,
            runner_path=args.runner,
        )

    if args.state_asset:
        report["state_assets"] = [
            inspect_state_asset(path) for path in args.state_asset
        ]
    packed_k6 = results["packed_k6"]
    separate_k6 = results["separate_k6"]
    assert isinstance(packed_k6, dict) and isinstance(separate_k6, dict)
    packed_gate = packed_k6.get("compiled", packed_k6["source"])
    separate_gate = separate_k6.get("compiled", separate_k6["source"])
    assert isinstance(packed_gate, dict) and isinstance(separate_gate, dict)
    psnr_gap = float(packed_gate["relative_psnr_db"]) - float(
        separate_gate["relative_psnr_db"]
    )
    report["comparison"] = {
        "packed_minus_separate_psnr_db": psnr_gap,
        "separate_input_regression_observed": psnr_gap > 1.0,
        "apple_feedback_reproduction_ready": psnr_gap > 1.0,
        "note": (
            "This isolated canary is a submission-ready reproduction only when "
            "the separate-input graph diverges. The production contiguous "
            "contract remains mandatory because it is the end-to-end validated "
            "path."
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {report_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
