#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""A/B BasicVSR++ explicit recurrent I/O against Core AI native state.

The shipping variable runner remains unchanged. This probe exports one
production-shape forward_2 continuation block with three mutable state tensors,
compiles it for the selected Mac architecture, and compares it with the current
explicit-boundary asset through the same asynchronous Metal-buffer API.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.apple.basicvsrpp_coreai_kernels import (
    build_deform_conv_kernel,
    build_grid_sample_kernel,
)
from scripts.apple.benchmark_basicvsrpp_variable_coreai import (
    BRANCHES,
    FEATURE_SIZE,
    MID_CHANNELS,
    PropagationLater,
)
from scripts.apple.export_basicvsrpp_coreai import (
    import_coreai,
    load_generator,
    use_deform_conv_metal_kernel,
    use_grid_sample_metal_kernel,
)
from scripts.apple.export_basicvsrpp_variable_chunk6 import PropagationContinue6


SWIFT_SOURCE = (
    REPOSITORY_ROOT / "tests/swift/BasicVSRPPNativeStateBenchmark.swift"
)
CHUNK_SIZE = 6
BRANCH = "forward_2"


class StatefulPropagationContinue6(torch.nn.Module):
    def __init__(self, generator: torch.nn.Module, branch: str = BRANCH):
        super().__init__()
        self.later = PropagationLater(generator, branch)
        self.register_buffer(
            "state_n1",
            torch.zeros(1, MID_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
        )
        self.register_buffer(
            "state_n2",
            torch.zeros(1, MID_CHANNELS, FEATURE_SIZE, FEATURE_SIZE),
        )
        self.register_buffer(
            "flow_previous",
            torch.zeros(1, 2, FEATURE_SIZE, FEATURE_SIZE),
        )

    def forward(
        self, contexts: torch.Tensor, flows: torch.Tensor
    ) -> torch.Tensor:
        outputs = []
        previous = self.state_n1
        older = self.state_n2
        previous_flow = self.flow_previous
        for index in range(CHUNK_SIZE):
            flow = flows[index : index + 1]
            result = self.later(
                contexts[index : index + 1],
                previous,
                older,
                flow,
                previous_flow,
            )
            outputs.append(result)
            older, previous, previous_flow = previous, result, flow
        self.state_n1.copy_(previous)
        self.state_n2.copy_(older)
        self.flow_previous.copy_(previous_flow)
        return torch.cat(outputs, dim=0)


def graph_signature_summary(
    exported: torch.export.ExportedProgram,
) -> dict[str, object]:
    signature = exported.graph_signature
    return {
        "user_inputs": [str(value) for value in signature.user_inputs],
        "user_outputs": [str(value) for value in signature.user_outputs],
        "inputs_to_buffers": dict(signature.inputs_to_buffers),
        "buffers_to_mutate": dict(signature.buffers_to_mutate),
    }


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def export_stateful(
    checkpoint: Path, destination: Path, branch: str = BRANCH
) -> dict[str, object]:
    _, coreai_torch = import_coreai()
    generator = load_generator(checkpoint).generator
    module = StatefulPropagationContinue6(generator, branch).half().eval()
    rng = np.random.default_rng(20260825)
    if branch not in BRANCHES:
        raise ValueError(f"unsupported propagation branch: {branch}")
    context_channels = MID_CHANNELS * (BRANCHES.index(branch) + 1)
    contexts = torch.from_numpy(
        rng.normal(
            0,
            0.1,
            (CHUNK_SIZE, context_channels, FEATURE_SIZE, FEATURE_SIZE),
        ).astype(np.float16)
    )
    flows = torch.from_numpy(
        rng.normal(
            0, 0.04, (CHUNK_SIZE, 2, FEATURE_SIZE, FEATURE_SIZE)
        ).astype(np.float16)
    )
    grid_kernel = build_grid_sample_kernel(coreai_torch)
    deform_kernel = build_deform_conv_kernel(coreai_torch)
    started = time.perf_counter()
    with torch.no_grad(), ExitStack() as stack:
        stack.enter_context(use_grid_sample_metal_kernel(grid_kernel))
        stack.enter_context(use_deform_conv_metal_kernel(deform_kernel))
        exported = torch.export.export(module, (contexts, flows))
        exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.register_custom_kernels([grid_kernel, deform_kernel])
    converter.add_exported_program(
        exported,
        state_names=["state_n1", "state_n2", "flow_previous"],
        input_names=["contexts", "flows"],
        output_names=["features"],
    )
    program = converter.to_coreai()
    program.optimize()
    remove_existing(destination)
    program.save_asset(destination)
    return {
        "export_seconds": time.perf_counter() - started,
        "graph_signature": graph_signature_summary(exported),
    }


def export_control(checkpoint: Path, destination: Path) -> dict[str, object]:
    """Export the existing explicit-I/O graph with the candidate toolchain."""
    _, coreai_torch = import_coreai()
    generator = load_generator(checkpoint).generator
    module = PropagationContinue6(generator, BRANCH).half().eval()
    rng = np.random.default_rng(20260825)
    contexts = torch.from_numpy(
        rng.normal(
            0,
            0.1,
            (CHUNK_SIZE, MID_CHANNELS * 4, FEATURE_SIZE, FEATURE_SIZE),
        ).astype(np.float16)
    )
    state_n1 = torch.from_numpy(
        rng.normal(
            0, 0.1, (1, MID_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
        ).astype(np.float16)
    )
    state_n2 = torch.from_numpy(
        rng.normal(
            0, 0.1, (1, MID_CHANNELS, FEATURE_SIZE, FEATURE_SIZE)
        ).astype(np.float16)
    )
    flows = torch.from_numpy(
        rng.normal(
            0, 0.04, (CHUNK_SIZE, 2, FEATURE_SIZE, FEATURE_SIZE)
        ).astype(np.float16)
    )
    flow_previous = flows[0:1].clone()
    grid_kernel = build_grid_sample_kernel(coreai_torch)
    deform_kernel = build_deform_conv_kernel(coreai_torch)
    started = time.perf_counter()
    with torch.no_grad(), ExitStack() as stack:
        stack.enter_context(use_grid_sample_metal_kernel(grid_kernel))
        stack.enter_context(use_deform_conv_metal_kernel(deform_kernel))
        exported = torch.export.export(
            module,
            (contexts, state_n1, state_n2, flows, flow_previous),
        )
        exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.register_custom_kernels([grid_kernel, deform_kernel])
    converter.add_exported_program(
        exported,
        input_names=[
            "contexts",
            "state_n1",
            "state_n2",
            "flows",
            "flow_previous",
        ],
        output_names=["features"],
    )
    program = converter.to_coreai()
    program.optimize()
    remove_existing(destination)
    program.save_asset(destination)
    return {
        "export_seconds": time.perf_counter() - started,
        "graph_signature": graph_signature_summary(exported),
    }


def compile_asset(
    source: Path, output_dir: Path, architecture: str
) -> Path:
    destination = output_dir / f"{source.stem}.{architecture}.aimodelc"
    remove_existing(destination)
    subprocess.run(
        [
            "xcrun",
            "coreai-build",
            "compile",
            str(source),
            "--output",
            str(output_dir),
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
    if not destination.is_dir():
        raise FileNotFoundError(f"compiled candidate not found: {destination}")
    return destination


def compile_runner(destination: Path) -> None:
    subprocess.run(
        [
            "xcrun",
            "swiftc",
            "-parse-as-library",
            "-O",
            "-framework",
            "CoreAI",
            "-framework",
            "Metal",
            str(SWIFT_SOURCE),
            "-o",
            str(destination),
        ],
        check=True,
    )


def inspect_asset(path: Path) -> dict[str, object]:
    completed = subprocess.run(
        ["xcrun", "coreai-build", "inspect", str(path), "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def run_runner_json(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    return json.loads(completed.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "model_weights/basicvsrpp-v1.2-detail-recovery-30000-ema.pth"
        ),
    )
    parser.add_argument(
        "--explicit-asset",
        type=Path,
        default=Path(
            "model_weights/mioh-dedicated-h17s/"
            "basicvsrpp-v1.2-standard-variable-coreai.h17s.aimodelc/"
            "basicvsrpp-variable-forward_2_continue6.h17s.aimodelc"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/mioh-basicvsrpp-native-state-ab"),
    )
    parser.add_argument("--architecture", default="h17s")
    parser.add_argument("--maximum-error", type=float, default=0.002)
    parser.add_argument("--minimum-speedup", type=float, default=1.03)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not args.explicit_asset.is_dir():
        raise FileNotFoundError(args.explicit_asset)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "basicvsrpp-forward_2-continue6-stateful.aimodel"
    control_source = (
        args.output_dir / "basicvsrpp-forward_2-continue6-control.aimodel"
    )
    runner = args.output_dir / "basicvsrpp-native-state-benchmark"

    control_export = export_control(args.checkpoint, control_source)
    export = export_stateful(args.checkpoint, source)
    control_source_inspection = inspect_asset(control_source)
    source_inspection = inspect_asset(source)
    control_compiled = compile_asset(
        control_source, args.output_dir, args.architecture
    )
    compiled = compile_asset(source, args.output_dir, args.architecture)
    control_compiled_inspection = inspect_asset(control_compiled)
    compiled_inspection = inspect_asset(compiled)
    compile_runner(runner)
    co_resident = run_runner_json(
        [
            str(runner),
            str(args.explicit_asset),
            str(control_compiled),
            str(compiled),
        ]
    )
    isolated_runs: list[dict[str, object]] = []
    for mode in ("control", "stateful", "stateful", "control") * 3:
        asset = control_compiled if mode == "control" else compiled
        isolated_runs.append(
            run_runner_json(
                [str(runner), "--timing", mode, str(asset)]
            )
        )
    control_medians = [
        float(run["medianMilliseconds"])
        for run in isolated_runs
        if run["mode"] == "control"
    ]
    stateful_medians = [
        float(run["medianMilliseconds"])
        for run in isolated_runs
        if run["mode"] == "stateful"
    ]
    control_median = statistics.median(control_medians)
    stateful_median = statistics.median(stateful_medians)
    speedup = control_median / stateful_median
    benchmark: dict[str, object] = {
        "co_resident_correctness": co_resident,
        "isolated_process_runs": isolated_runs,
        "isolated_control_median_milliseconds": control_median,
        "isolated_stateful_median_milliseconds": stateful_median,
        "stateful_speedup": speedup,
    }
    maximum_error = max(
        float(co_resident["controlVsStateful"][name]["maxAbsoluteError"])
        for name in ("output", "stateN1", "stateN2", "flowPrevious")
    )
    quality_pass = maximum_error <= args.maximum_error
    performance_pass = speedup >= args.minimum_speedup
    report: dict[str, object] = {
        "quality_pass": quality_pass,
        "performance_pass": performance_pass,
        "adoption_recommended": quality_pass and performance_pass,
        "maximum_allowed_error": args.maximum_error,
        "minimum_required_speedup": args.minimum_speedup,
        "checkpoint": str(args.checkpoint.resolve()),
        "explicit_asset": str(args.explicit_asset.resolve()),
        "control_source_asset": str(control_source.resolve()),
        "control_compiled_asset": str(control_compiled.resolve()),
        "stateful_source_asset": str(source.resolve()),
        "stateful_compiled_asset": str(compiled.resolve()),
        "control_export": control_export,
        "export": export,
        "control_source_inspection": control_source_inspection,
        "control_compiled_inspection": control_compiled_inspection,
        "source_inspection": source_inspection,
        "compiled_inspection": compiled_inspection,
        "benchmark": benchmark,
        "decision": (
            "keep-explicit-boundary-io"
            if not (quality_pass and performance_pass)
            else "eligible-for-full-runner-ab"
        ),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {report_path.resolve()}")
    return 0 if quality_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
