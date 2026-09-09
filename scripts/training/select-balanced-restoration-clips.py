#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Select a bounded number of restoration clips per source excerpt.

The generated dataset is a non-destructive symlink view.  Longer clips and
larger native crops are preferred, while the per-source cap prevents one
performer or scene from dominating the training distribution.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


PATH_FIELDS = (
    "relative_nsfw_video_path",
    "relative_mask_video_path",
    "relative_mosaic_nsfw_video_path",
    "relative_mosaic_mask_video_path",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-per-source", type=int, default=2)
    parser.add_argument("--min-frames", type=int, default=26)
    return parser.parse_args()


def _link(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    destination.symlink_to(source)


def _destination_for_relative(
    metadata_dir: Path,
    output_root: Path,
    relative_value: str,
) -> Path:
    destination = Path(os.path.normpath(str(metadata_dir / relative_value)))
    destination.relative_to(output_root)
    return destination


def _score(metadata: dict[str, Any]) -> tuple[int, int, str]:
    frames = int(metadata.get("frames_count") or 0)
    shape = metadata.get("scene_shape") or (0, 0)
    area = int(shape[0]) * int(shape[1]) if len(shape) >= 2 else 0
    return frames, area, str(metadata.get("name") or "")


def main() -> None:
    args = parse_args()
    if args.max_per_source < 1:
        raise ValueError("--max-per-source must be at least 1")
    source_root = args.source_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(
            f"output already exists; refusing to overwrite: {output_root}"
        )

    source_metadata_dir = source_root / "crop_unscaled_meta"
    if not source_metadata_dir.is_dir():
        raise FileNotFoundError(source_metadata_dir)
    destination_metadata_dir = output_root / "crop_unscaled_meta"
    destination_metadata_dir.mkdir(parents=True)

    groups: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    rejected_short: list[str] = []
    for metadata_path in sorted(source_metadata_dir.glob("*.json")):
        if metadata_path.name.startswith("._"):
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(metadata.get("frames_count") or 0) < args.min_frames:
            rejected_short.append(str(metadata_path))
            continue
        source_name = str(metadata.get("name") or metadata_path.stem)
        groups[source_name].append((metadata_path, metadata))

    selected: list[tuple[Path, dict[str, Any]]] = []
    per_source: dict[str, dict[str, Any]] = {}
    for source_name, candidates in sorted(groups.items()):
        ranked = sorted(
            candidates,
            key=lambda item: (_score(item[1]), item[0].name),
            reverse=True,
        )
        chosen = ranked[: args.max_per_source]
        selected.extend(chosen)
        per_source[source_name] = {
            "available": len(candidates),
            "selected": len(chosen),
            "selected_metadata": [item[0].name for item in chosen],
        }

    for metadata_path, metadata in selected:
        _link(metadata_path, destination_metadata_dir / metadata_path.name)
        for field in PATH_FIELDS:
            relative_value = metadata.get(field)
            if not relative_value:
                continue
            source_asset = metadata_path.parent / str(relative_value)
            destination_asset = _destination_for_relative(
                destination_metadata_dir,
                output_root,
                str(relative_value),
            )
            _link(source_asset, destination_asset)

    report = {
        "version": 1,
        "source_root": str(source_root),
        "output_root": str(output_root),
        "max_per_source": args.max_per_source,
        "min_frames": args.min_frames,
        "source_count": len(groups),
        "candidate_count": sum(len(items) for items in groups.values()),
        "selected_count": len(selected),
        "rejected_short": rejected_short,
        "per_source": per_source,
    }
    (output_root / "selection-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Selected {len(selected)} clips from {len(groups)} source excerpts: "
        f"{output_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
