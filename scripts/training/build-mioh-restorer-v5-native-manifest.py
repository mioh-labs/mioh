#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Build native-resolution V5 windows from Lada restoration metadata.

The command decodes masks only.  It records even-pixel crop origins and one of
the fixed V5 buckets; it never decodes, resizes, or rewrites clean RGB frames.
Windows whose ROI plus context does not fit 512x512 are rejected rather than
silently clipped.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from json import JSONDecodeError
from pathlib import Path

import numpy as np

from lada.datasetcreation.restoration_dataset_metadata import RestorationDatasetMetadataV2
from lada.models.mioh_restorer.model_v5 import NUM_INPUT_FRAMES, V5_BUCKETS
from lada.models.mioh_restorer.native_dataset_v5 import decode_native_frames
from lada.models.mioh_restorer.runner_v5 import (
    required_v5_crop_size,
    native_tile_offsets,
    round_to_even,
    select_v5_bucket,
    smooth_even_centers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--context-fraction", type=float, default=0.30)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def mask_box(frame: np.ndarray) -> tuple[tuple[float, float], float, float] | None:
    if frame.ndim == 3:
        frame = frame[..., 0]
    vertical, horizontal = np.nonzero(frame > 0)
    if not len(horizontal):
        return None
    left = int(horizontal.min())
    right = int(horizontal.max()) + 1
    top = int(vertical.min())
    bottom = int(vertical.max()) + 1
    return ((left + right - 1) / 2.0, (top + bottom - 1) / 2.0), right - left, bottom - top


def relative_or_absolute(path: Path, output: Path) -> str:
    try:
        return str(path.relative_to(output.parent))
    except ValueError:
        return str(path)


def entries_for_metadata(
    metadata_path: Path,
    *,
    stride: int,
    context_fraction: float,
    tile_overlap: int,
):
    metadata = RestorationDatasetMetadataV2.from_json_file(metadata_path)
    target = (metadata_path.parent / metadata.relative_nsfw_video_path).resolve()
    mask_path = (metadata_path.parent / metadata.relative_mask_video_path).resolve()
    masks = decode_native_frames(mask_path, pixel_format="gray")
    frame_count = min(metadata.frames_count, len(masks))
    if frame_count < NUM_INPUT_FRAMES:
        return
    boxes = [mask_box(frame) for frame in masks[:frame_count]]
    centres = smooth_even_centers([value[0] if value else None for value in boxes])
    for start in range(0, frame_count - NUM_INPUT_FRAMES + 1, stride):
        window_boxes = boxes[start : start + NUM_INPUT_FRAMES]
        detected = [value for value in window_boxes if value is not None]
        if not detected:
            continue
        maximum_width = max(value[1] for value in detected)
        maximum_height = max(value[2] for value in detected)
        required = required_v5_crop_size(
            maximum_width,
            maximum_height,
            context_fraction=context_fraction,
        )
        bucket = (
            select_v5_bucket(
                maximum_width,
                maximum_height,
                context_fraction=context_fraction,
            )
            if required <= V5_BUCKETS[-1]
            else V5_BUCKETS[-1]
        )
        window_centres = centres[start : start + NUM_INPUT_FRAMES]
        offsets = native_tile_offsets(
            maximum_width,
            maximum_height,
            bucket=bucket,
            context_fraction=context_fraction,
            overlap=tile_overlap,
        )
        for tile_index, (offset_x, offset_y) in enumerate(offsets):
            origins = [
                [round_to_even(horizontal + offset_x), round_to_even(vertical + offset_y)]
                for horizontal, vertical in window_centres
            ]
            contains_mask = False
            for frame, (origin_x, origin_y) in zip(
                masks[start : start + NUM_INPUT_FRAMES], origins, strict=True
            ):
                height, width = frame.shape[:2]
                x0, y0 = max(origin_x, 0), max(origin_y, 0)
                x1, y1 = min(origin_x + bucket, width), min(origin_y + bucket, height)
                if x1 > x0 and y1 > y0 and np.any(frame[y0:y1, x0:x1] > 0):
                    contains_mask = True
                    break
            if not contains_mask:
                continue
            yield {
                "name": f"{metadata.name}:{start:06d}:tile-{tile_index:02d}",
                "target_video": target,
                "mask_video": mask_path,
                "start_frame": start,
                "bucket": bucket,
                "origins": origins,
                "mask_reliability": [1.0 if value is not None else 0.5 for value in window_boxes],
                "mosaic_block_size": metadata.base_mosaic_block_size.mosaic_size_v1_normal,
                "source_video_id": metadata.name,
            }


def main() -> int:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("stride must be positive")
    if args.context_fraction < 0:
        raise ValueError("context fraction must be non-negative")
    if args.tile_overlap < 0 or args.tile_overlap >= V5_BUCKETS[-1]:
        raise ValueError("tile overlap must be in [0, 512)")
    metadata_paths: list[Path] = []
    for root in args.metadata_root:
        if not root.is_dir():
            raise FileNotFoundError(root)
        metadata_paths.extend(
            path for path in sorted(root.glob("*.json")) if not path.name.startswith("._")
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    counts: Counter[int] = Counter()
    written = 0
    skipped_metadata = 0
    with temporary.open("w", encoding="utf-8") as destination:
        for metadata_path in metadata_paths:
            try:
                for entry in entries_for_metadata(
                    metadata_path,
                    stride=args.stride,
                    context_fraction=args.context_fraction,
                    tile_overlap=args.tile_overlap,
                ):
                    serializable = dict(entry)
                    serializable["target_video"] = relative_or_absolute(entry["target_video"], args.output)
                    serializable["mask_video"] = relative_or_absolute(entry["mask_video"], args.output)
                    destination.write(json.dumps(serializable, ensure_ascii=False) + "\n")
                    counts[int(entry["bucket"])] += 1
                    written += 1
                    if args.limit is not None and written >= args.limit:
                        break
            except (OSError, JSONDecodeError, KeyError, TypeError, ValueError):
                skipped_metadata += 1
            if args.limit is not None and written >= args.limit:
                break
    if not written:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("no native V5 windows were written")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "windows": written,
                "buckets": {str(key): counts[key] for key in V5_BUCKETS},
                "skipped_metadata": skipped_metadata,
                "resized_frames": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
