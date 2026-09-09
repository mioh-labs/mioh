#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Create balanced, high-quality 29.97-fps excerpts from source folders.

The input files are never modified.  Duplicate FC2 work IDs prefer HHD800
copies; multipart HHD800 files are retained.  Four short excerpts per selected
file cover the timeline without allowing a long work to dominate training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


FC2_ID = re.compile(r"FC2(?:-PPV|PPV)?[- _]*(\d{6,8})", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=24.0)
    parser.add_argument("--crf", type=int, default=10)
    parser.add_argument(
        "--position-step-percent",
        type=int,
        default=20,
        help="timeline sampling interval in percent (for example 5 = 5..95)",
    )
    parser.add_argument(
        "--exclude-work-id",
        action="append",
        default=[],
        help="FC2 work ID reserved for validation/holdout; repeatable",
    )
    args = parser.parse_args()
    if not 1 <= args.position_step_percent <= 50:
        parser.error("--position-step-percent must be in [1, 50]")
    return args


def _probe(path: Path) -> dict[str, Any]:
    return json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration:stream=width,height,avg_frame_rate,field_order",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )


def _work_id(path: Path) -> str | None:
    match = FC2_ID.search(path.name)
    return match.group(1) if match else None


def _part_number(path: Path, work_id: str) -> int | None:
    match = re.search(
        rf"{re.escape(work_id)}[-_](\d+)(?:\D|$)",
        path.name,
        re.IGNORECASE,
    )
    return int(match.group(1)) if match else None


def _select_sources(paths: list[Path]) -> tuple[list[Path], list[dict[str, str]]]:
    by_work: dict[str, list[Path]] = {}
    without_work: list[Path] = []
    for path in paths:
        work_id = _work_id(path)
        if work_id:
            by_work.setdefault(work_id, []).append(path)
        else:
            without_work.append(path)

    selected = list(without_work)
    excluded: list[dict[str, str]] = []
    for work_id, candidates in sorted(by_work.items()):
        hhd = [
            path
            for path in candidates
            if path.name.lower().startswith("hhd800.com@")
        ]
        part_numbers = [_part_number(path, work_id) for path in candidates]
        distinct_parts = {part for part in part_numbers if part is not None}
        if hhd:
            chosen = sorted(hhd)
        elif len(distinct_parts) == len(candidates) and len(candidates) > 1:
            chosen = sorted(candidates)
        else:
            chosen = [max(candidates, key=lambda path: path.stat().st_size)]
        selected.extend(chosen)
        chosen_set = set(chosen)
        for path in candidates:
            if path not in chosen_set:
                excluded.append(
                    {
                        "path": str(path),
                        "reason": f"lower-quality duplicate of FC2 work {work_id}",
                    }
                )
    return sorted(selected), excluded


def _output_name(source: Path, position: float) -> str:
    work_id = _work_id(source) or re.sub(r"[^A-Za-z0-9]+", "-", source.stem)[:36]
    part = ""
    match = re.search(r"_(\d+)$", source.stem)
    if match:
        part = f"-part{match.group(1)}"
    digest = hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:8]
    return f"{work_id}{part}-{digest}-p{round(position * 100):02d}.mp4"


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"output already exists; refusing to overwrite: {output_root}"
        )
    output_root.mkdir(parents=True)

    candidates: list[Path] = []
    for input_path in args.input:
        resolved = input_path.expanduser().resolve(strict=True)
        if resolved.is_dir():
            candidates.extend(
                path
                for path in resolved.rglob("*.mp4")
                if path.is_file() and not path.name.startswith("._")
            )
        else:
            candidates.append(resolved)
    selected, excluded = _select_sources(sorted(set(candidates)))
    excluded_work_ids = {str(value) for value in args.exclude_work_id}
    if excluded_work_ids:
        retained: list[Path] = []
        for source in selected:
            work_id = _work_id(source)
            if work_id in excluded_work_ids:
                excluded.append(
                    {
                        "path": str(source),
                        "reason": f"FC2 work {work_id} reserved for holdout",
                    }
                )
            else:
                retained.append(source)
        selected = retained
    positions = tuple(
        percent / 100.0
        for percent in range(
            args.position_step_percent,
            100,
            args.position_step_percent,
        )
    )

    report: dict[str, Any] = {
        "version": 1,
        "inputs": [str(path.expanduser().resolve()) for path in args.input],
        "duration_seconds": args.duration,
        "positions": positions,
        "excluded_work_ids": sorted(excluded_work_ids),
        "selected_sources": [str(path) for path in selected],
        "excluded_sources": excluded,
        "excerpts": [],
    }
    for source_index, source in enumerate(selected, start=1):
        probe = _probe(source)
        stream = probe["streams"][0]
        duration = float(probe["format"]["duration"])
        if duration <= args.duration:
            starts = (0.0,)
        else:
            starts = tuple(
                min(duration - args.duration, max(0.0, duration * position))
                for position in positions
            )
        for position, start in zip(positions, starts):
            output = output_root / _output_name(source, position)
            print(
                f"[{source_index}/{len(selected)}] {source.name}: "
                f"{start:.1f}s -> {output.name}",
                flush=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start:.6f}",
                    "-i",
                    str(source),
                    "-t",
                    f"{args.duration:.6f}",
                    "-map",
                    "0:v:0",
                    "-vf",
                    "fps=30000/1001,setfield=prog",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    str(args.crf),
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output),
                ],
                check=True,
            )
            output_probe = _probe(output)
            report["excerpts"].append(
                {
                    "source": str(source),
                    "source_width": stream["width"],
                    "source_height": stream["height"],
                    "source_fps": stream["avg_frame_rate"],
                    "source_field_order": stream.get("field_order"),
                    "position": position,
                    "start_seconds": start,
                    "output": str(output),
                    "output_duration": float(output_probe["format"]["duration"]),
                    "output_fps": output_probe["streams"][0]["avg_frame_rate"],
                }
            )

    report["source_count"] = len(selected)
    report["excerpt_count"] = len(report["excerpts"])
    (output_root / "excerpts-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared {report['excerpt_count']} excerpts from "
        f"{report['source_count']} selected sources: {output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
