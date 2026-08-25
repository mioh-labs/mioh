#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Export and execute a native mutable-state Core AI regression canary.

This is intentionally independent of Mioh's shipping BasicVSR++ graph. It
verifies all three layers needed before recurrent boundary tensors can be
reconsidered: Core AI Torch lowering, ahead-of-time specialization, and the
Swift Core AI state runtime.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SWIFT_SOURCE = REPOSITORY_ROOT / "tests/swift/CoreAINativeStateCanary.swift"


class Accumulator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("acc", torch.zeros(1, 4, dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        self.acc.add_(value)
        return self.acc + 1


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def remove_existing(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def run_json(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command, check=True, capture_output=True, text=True
    )
    return json.loads(completed.stdout)


def export_asset(destination: Path) -> None:
    import coreai_torch

    module = Accumulator().eval()
    example = torch.ones(1, 4, dtype=torch.float32)
    exported = torch.export.export(module, (example,))
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported,
        state_names=["acc"],
        input_names=["x"],
        output_names=["result"],
    )
    program = converter.to_coreai()
    # MutableBuffers annotations become runtime handles during optimization.
    # Saving the unoptimized program would expose the state as ordinary I/O.
    program.optimize()
    remove_existing(destination)
    program.save_asset(destination)


def compile_swift_runner(destination: Path) -> None:
    subprocess.run(
        [
            "xcrun",
            "swiftc",
            "-parse-as-library",
            "-O",
            "-framework",
            "CoreAI",
            str(SWIFT_SOURCE),
            "-o",
            str(destination),
        ],
        check=True,
    )


def compile_asset(source: Path, output_dir: Path, architecture: str) -> Path:
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
        raise FileNotFoundError(f"compiled Core AI asset not found: {destination}")
    return destination


def validate_inspection(document: dict[str, object]) -> None:
    summary = document.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError("Core AI inspection has no summary")
    functions = summary.get("functions")
    if not isinstance(functions, list) or len(functions) != 1:
        raise RuntimeError("Core AI inspection does not contain one function")
    function = functions[0]
    if not isinstance(function, dict):
        raise RuntimeError("Core AI function inspection is invalid")
    states = function.get("states")
    if not isinstance(states, list) or [state.get("name") for state in states] != [
        "acc"
    ]:
        raise RuntimeError("mutable acc was lowered to regular I/O")


def validate_runtime(document: dict[str, object], label: str) -> None:
    expected = {
        "stateNames": ["acc"],
        "calls": [[2.0] * 4, [3.0] * 4],
        "stateAfter": [2.0] * 4,
    }
    if document != expected:
        raise RuntimeError(
            f"{label} native-state result differs: "
            f"expected={expected!r}, actual={document!r}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/tmp/mioh-coreai-native-state-canary"),
    )
    parser.add_argument("--architecture", default="h17s")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "state-accumulator.aimodel"
    runner = args.output_dir / "coreai-native-state-canary"

    export_asset(source)
    compile_swift_runner(runner)
    source_inspection = run_json(
        ["xcrun", "coreai-build", "inspect", str(source), "--json"]
    )
    validate_inspection(source_inspection)
    source_runtime = run_json([str(runner), str(source)])
    validate_runtime(source_runtime, "source")

    compiled = compile_asset(source, args.output_dir, args.architecture)
    compiled_inspection = run_json(
        ["xcrun", "coreai-build", "inspect", str(compiled), "--json"]
    )
    validate_inspection(compiled_inspection)
    compiled_runtime = run_json([str(runner), str(compiled)])
    validate_runtime(compiled_runtime, "compiled")

    report: dict[str, object] = {
        "success": True,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "coreai_torch": package_version("coreai-torch"),
        "coreai_core": package_version("coreai-core"),
        "architecture": args.architecture,
        "source_asset": str(source.resolve()),
        "compiled_asset": str(compiled.resolve()),
        "source_inspection": source_inspection,
        "compiled_inspection": compiled_inspection,
        "source_runtime": source_runtime,
        "compiled_runtime": compiled_runtime,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {report_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
