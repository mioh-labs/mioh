#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Fail closed unless a V5 Stage 0 report preserves the quality contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lada.models.mioh_restorer.model_v5 import V5_BUCKETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "--sizes",
        default=",".join(str(value) for value in V5_BUCKETS),
        help="required bucket sizes",
    )
    parser.add_argument("--minimum-rgb-psnr", type=float, default=70.0)
    parser.add_argument("--minimum-confidence-psnr", type=float, default=70.0)
    parser.add_argument(
        "--execution",
        choices=("monolithic", "split", "stateful"),
        help="required execution path; defaults to monolithic for Q and split for S",
    )
    return parser.parse_args()


def parse_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item) for item in value.split(",") if item)
    if not sizes or any(size not in V5_BUCKETS for size in sizes):
        raise ValueError(f"sizes must be selected from {V5_BUCKETS}")
    return sizes


def check_numeric(
    failures: list[str],
    numeric: dict[str, object],
    *,
    prefix: str,
    rgb_threshold: float,
    confidence_threshold: float,
) -> None:
    for name, threshold in (("rgb", rgb_threshold), ("confidence", confidence_threshold)):
        try:
            psnr = float(numeric[name]["psnr"])  # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            failures.append(f"{prefix}: missing {name} numeric PSNR")
            continue
        if psnr < threshold:
            failures.append(f"{prefix}: {name} PSNR {psnr:.2f} dB < {threshold:.2f} dB")


def main() -> int:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    variant = report.get("variant")
    if variant not in ("q", "s"):
        raise ValueError("report variant must be q or s")
    execution = args.execution or ("monolithic" if variant == "q" else "split")
    expected_context = 7 if variant == "q" else 9
    expected_outputs = 5 if variant == "q" else 1
    failures: list[str] = []
    for size in parse_sizes(args.sizes):
        size_report = report.get("sizes", {}).get(str(size))
        prefix = f"V5-{variant.upper()} {size} {execution}"
        if not isinstance(size_report, dict):
            failures.append(f"{prefix}: bucket is missing")
            continue
        if size_report.get("context_frames") != expected_context:
            failures.append(f"{prefix}: quality contract requires {expected_context} context frames")
        if size_report.get("outputs") != expected_outputs:
            failures.append(f"{prefix}: quality contract requires {expected_outputs} output frame(s)")
        if size_report.get("coarse_mode") not in (None, "full49"):
            failures.append(f"{prefix}: factorized coarse search is experimental")
        execution_report = size_report.get(execution)
        if not isinstance(execution_report, dict):
            failures.append(f"{prefix}: execution result is missing")
            continue
        numeric = execution_report.get("numeric")
        if not isinstance(numeric, dict):
            failures.append(f"{prefix}: numeric comparison is missing")
        else:
            check_numeric(
                failures,
                numeric,
                prefix=prefix,
                rgb_threshold=args.minimum_rgb_psnr,
                confidence_threshold=args.minimum_confidence_psnr,
            )
        if execution == "split":
            end_to_end = execution_report.get("end_to_end_numeric")
            if not isinstance(end_to_end, dict):
                failures.append(f"{prefix}: end-to-end encoder/decoder comparison is missing")
            else:
                check_numeric(
                    failures,
                    end_to_end,
                    prefix=f"{prefix} end-to-end",
                    rgb_threshold=args.minimum_rgb_psnr,
                    confidence_threshold=args.minimum_confidence_psnr,
                )
    result = {
        "passed": not failures,
        "report": str(args.report),
        "variant": variant,
        "execution": execution,
        "failures": failures,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
