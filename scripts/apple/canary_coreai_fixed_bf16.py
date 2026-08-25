#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Reproduce the fixed-shape BF16 coreai-build type-contract regression."""

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
SWIFT_SOURCE = REPOSITORY_ROOT / "tests/swift/CoreAIFixedBF16Canary.swift"


class FixedBF16(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 2 + 1


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
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def export_asset(destination: Path) -> None:
    import coreai_torch

    example = torch.ones(4, dtype=torch.bfloat16)
    exported = torch.export.export(FixedBF16().eval(), (example,))
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


def compile_asset(
    source: Path, output_dir: Path, architecture: str
) -> tuple[Path | None, dict[str, object]]:
    destination = output_dir / f"{source.stem}.{architecture}.aimodelc"
    remove_existing(destination)
    completed = subprocess.run(
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
        capture_output=True,
        text=True,
    )
    result: dict[str, object] = {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        return None, result
    if not destination.is_dir():
        result["missingOutput"] = str(destination)
        return None, result
    return destination, result


def validate_runtime(result: dict[str, object], label: str) -> None:
    expected = [3.0, -3.0, 2.0, 9.0]
    if result.get("values") != expected:
        raise RuntimeError(f"{label} result mismatch: {result!r}")
    if result.get("inputScalarType") != "bfloat16":
        raise RuntimeError(f"{label} input was not BF16: {result!r}")
    if result.get("outputScalarType") != "bfloat16":
        raise RuntimeError(f"{label} output was not BF16: {result!r}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/mioh-coreai-fixed-bf16")
    )
    parser.add_argument("--architecture", default="h17s")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = args.output_dir / "fixed-bf16.aimodel"
    runner = args.output_dir / "coreai-fixed-bf16-canary"
    export_asset(source)
    compile_runner(runner)
    source_inspection = run_json(
        ["xcrun", "coreai-build", "inspect", str(source), "--json"]
    )
    source_runtime = run_json([str(runner), str(source)])
    validate_runtime(source_runtime, "source")
    compiled, compilation = compile_asset(source, args.output_dir, args.architecture)
    compiled_inspection: dict[str, object] | None = None
    compiled_runtime: dict[str, object] | None = None
    if compiled is not None:
        compiled_inspection = run_json(
            ["xcrun", "coreai-build", "inspect", str(compiled), "--json"]
        )
        compiled_runtime = run_json([str(runner), str(compiled)])
        validate_runtime(compiled_runtime, "compiled")

    failure_text = f"{compilation.get('stdout', '')}\n{compilation.get('stderr', '')}"
    regression_reproduced = (
        compiled is None
        and "expected element type bf16 but received f16" in failure_text
    )

    report: dict[str, object] = {
        "success": compiled is not None,
        "regressionReproduced": regression_reproduced,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "coreai_torch": package_version("coreai-torch"),
        "coreai_core": package_version("coreai-core"),
        "architecture": args.architecture,
        "source_asset": str(source.resolve()),
        "compiled_asset": str(compiled.resolve()) if compiled is not None else None,
        "compilation": compilation,
        "source_inspection": source_inspection,
        "compiled_inspection": compiled_inspection,
        "source_runtime": source_runtime,
        "compiled_runtime": compiled_runtime,
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {report_path.resolve()}")
    return 0 if compiled is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
