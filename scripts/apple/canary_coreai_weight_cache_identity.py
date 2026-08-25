#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Detect Core AI cache aliasing for graph-identical, weight-distinct assets."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SWIFT_SOURCE = REPOSITORY_ROOT / "tests/swift/CoreAIWeightCacheCanary.swift"


class WeightedScale(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor([scale], dtype=torch.float32))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.scale


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


def export_asset(destination: Path, scale: float) -> None:
    import coreai_torch

    exported = torch.export.export(
        WeightedScale(scale).eval(), (torch.ones(1, dtype=torch.float32),)
    )
    exported = exported.run_decompositions(coreai_torch.get_decomp_table())
    converter = coreai_torch.TorchConverter()
    converter.add_exported_program(
        exported, input_names=["x"], output_names=["y"]
    )
    program = converter.to_coreai()
    program.optimize()
    remove_existing(destination)
    program.save_asset(destination)


def compile_runner(destination: Path) -> None:
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
        raise FileNotFoundError(destination)
    return destination


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_json(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def comparison(document: dict[str, object], order: str) -> dict[str, object]:
    calls = document.get("calls")
    if not isinstance(calls, list):
        return {"passed": False, "expected": [], "actual": [], "document": document}
    expected = [6.0 if name == "A" else 15.0 for name in order]
    actual = [entry.get("value") for entry in calls if isinstance(entry, dict)]
    return {
        "passed": actual == expected,
        "expected": expected,
        "actual": actual,
        "document": document,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/mioh-coreai-weight-cache")
    )
    parser.add_argument("--architecture", default="h17s")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_a = args.output_dir / "weight-a.aimodel"
    source_b = args.output_dir / "weight-b.aimodel"
    runner = args.output_dir / "coreai-weight-cache-canary"
    export_asset(source_a, 2.0)
    export_asset(source_b, 5.0)
    compile_runner(runner)

    graph_hashes = {
        "A": sha256(source_a / "main.hash"),
        "B": sha256(source_b / "main.hash"),
    }
    source_runs = {}
    for order in ("ABAB", "BABA", "AABB", "BBAA"):
        result = run_json([str(runner), str(source_a), str(source_b), order])
        source_runs[order] = comparison(result, order)

    compiled_a = compile_asset(source_a, args.output_dir, args.architecture)
    compiled_b = compile_asset(source_b, args.output_dir, args.architecture)
    compiled_runs = {}
    for order in ("ABAB", "BABA", "AABB", "BBAA"):
        result = run_json([str(runner), str(compiled_a), str(compiled_b), order])
        compiled_runs[order] = comparison(result, order)

    source_alias = any(not result["passed"] for result in source_runs.values())
    compiled_alias = any(not result["passed"] for result in compiled_runs.values())

    report: dict[str, object] = {
        "success": not source_alias and not compiled_alias,
        "cacheAliasDetected": source_alias or compiled_alias,
        "sourceCacheAliasDetected": source_alias,
        "compiledCacheAliasDetected": compiled_alias,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "coreai_torch": package_version("coreai-torch"),
        "coreai_core": package_version("coreai-core"),
        "architecture": args.architecture,
        "sourceAssets": [str(source_a.resolve()), str(source_b.resolve())],
        "compiledAssets": [str(compiled_a.resolve()), str(compiled_b.resolve())],
        "graphHashFileSHA256": graph_hashes,
        "sourceGraphHashFilesEqual": (source_a / "main.hash").read_bytes()
        == (source_b / "main.hash").read_bytes(),
        "compiledGraphHashFilesEqual": (compiled_a / "main.hash").read_bytes()
        == (compiled_b / "main.hash").read_bytes(),
        "compiledProgramFilesEqual": (
            compiled_a / f"main-{args.architecture}.mlirb"
        ).read_bytes()
        == (compiled_b / f"main-{args.architecture}.mlirb").read_bytes(),
        "sourceRuns": source_runs,
        "compiledRuns": compiled_runs,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {report_path.resolve()}")
    return 0 if not source_alias and not compiled_alias else 2


if __name__ == "__main__":
    raise SystemExit(main())
