#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Expand native 512 ROI samples into overlapping, unscaled 256 tiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lada.models.mioh_restorer.native_dataset_v5 import (
    V5NativeManifestEntry,
    crop_native_frame,
    decode_native_frames,
    read_v5_native_manifest,
)


SOURCE_SIZE = 512
TILE_SIZE = 256
NUM_FRAMES = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic native-scale tile manifest for large ROIs."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--minimum-roi-extent", type=int, default=257)
    parser.add_argument("--minimum-tile-mask-fraction", type=float, default=0.15)
    parser.add_argument(
        "--minimum-frame-tile-mask-fraction",
        type=float,
        default=0.01,
        help="Required mask coverage in every one of the nine frames.",
    )
    parser.add_argument(
        "--max-tiles-per-entry",
        type=int,
        default=0,
        help="0 keeps every qualifying tile; positive values select diverse tiles.",
    )
    parser.add_argument(
        "--random-anchor-copies",
        type=int,
        default=0,
        help="Append unchanged entries to retain the original random-anchor curriculum.",
    )
    return parser.parse_args()


def axis_offsets(*, stride: int) -> tuple[int, ...]:
    if stride <= 0 or stride > TILE_SIZE or stride % 2:
        raise ValueError("stride must be an even integer in 1...256")
    maximum = SOURCE_SIZE - TILE_SIZE
    values = list(range(0, maximum + 1, stride))
    if values[-1] != maximum:
        values.append(maximum)
    return tuple(values)


def mask_stack(entry: V5NativeManifestEntry) -> np.ndarray:
    frames = decode_native_frames(
        entry.mask_video,
        start=entry.start_frame,
        count=NUM_FRAMES,
        pixel_format="gray",
    )
    if len(frames) != NUM_FRAMES:
        raise RuntimeError(
            f"decoded {len(frames)} masks for {entry.name}; expected {NUM_FRAMES}"
        )
    result: list[np.ndarray] = []
    for frame, origin in zip(frames, entry.origins, strict=True):
        cropped = crop_native_frame(
            frame,
            origin=origin,
            size=SOURCE_SIZE,
            mask=True,
        )
        if cropped.ndim == 3:
            cropped = np.max(cropped, axis=2)
        result.append(np.ascontiguousarray(cropped))
    return np.stack(result, axis=0)


def diverse_candidates(
    candidates: list[tuple[float, int, int]],
    *,
    maximum: int,
) -> list[tuple[float, int, int]]:
    if maximum <= 0 or len(candidates) <= maximum:
        return sorted(candidates, key=lambda item: (item[2], item[1]))
    remaining = list(candidates)
    first = max(remaining, key=lambda item: (item[0], -item[2], -item[1]))
    selected = [first]
    remaining.remove(first)
    while remaining and len(selected) < maximum:
        candidate = max(
            remaining,
            key=lambda item: (
                min(
                    (item[1] - chosen[1]) ** 2
                    + (item[2] - chosen[2]) ** 2
                    for chosen in selected
                ),
                item[0],
                -item[2],
                -item[1],
            ),
        )
        selected.append(candidate)
        remaining.remove(candidate)
    return sorted(selected, key=lambda item: (item[2], item[1]))


def build_manifest(
    *,
    input_path: Path,
    output_path: Path,
    stride: int,
    minimum_roi_extent: int,
    minimum_tile_mask_fraction: float,
    minimum_frame_tile_mask_fraction: float,
    max_tiles_per_entry: int,
    random_anchor_copies: int,
) -> dict[str, int | float]:
    if input_path.resolve() == output_path.resolve():
        raise ValueError("output manifest must differ from input manifest")
    if not 0 < minimum_tile_mask_fraction <= 1:
        raise ValueError("minimum tile mask fraction must be in (0, 1]")
    if not 0 < minimum_frame_tile_mask_fraction <= 1:
        raise ValueError("minimum per-frame tile mask fraction must be in (0, 1]")
    if not TILE_SIZE < minimum_roi_extent <= SOURCE_SIZE:
        raise ValueError("minimum ROI extent must be in 257...512")
    if max_tiles_per_entry < 0 or random_anchor_copies < 0:
        raise ValueError("tile and random-copy counts must be non-negative")

    raw_entries = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    entries = read_v5_native_manifest(input_path)
    if len(raw_entries) != len(entries):
        raise RuntimeError("typed and raw manifest lengths differ")

    offsets = axis_offsets(stride=stride)
    output_values: list[dict[str, object]] = []
    tiled_entries = 0
    for raw, entry in zip(raw_entries, entries, strict=True):
        masks = mask_stack(entry)
        mask = np.max(masks, axis=0)
        ys, xs = np.where(mask > 0)
        if not len(xs):
            continue
        bbox_left = int(xs.min())
        bbox_top = int(ys.min())
        bbox_right = int(xs.max()) + 1
        bbox_bottom = int(ys.max()) + 1
        bbox_width = bbox_right - bbox_left
        bbox_height = bbox_bottom - bbox_top
        if (
            bbox_width < minimum_roi_extent
            or bbox_height < minimum_roi_extent
        ):
            continue

        candidates: list[tuple[float, int, int]] = []
        for top in offsets:
            for left in offsets:
                fraction = float(
                    np.mean(mask[top : top + TILE_SIZE, left : left + TILE_SIZE] > 0)
                )
                minimum_frame_fraction = float(
                    np.min(
                        np.mean(
                            masks[
                                :,
                                top : top + TILE_SIZE,
                                left : left + TILE_SIZE,
                            ]
                            > 0,
                            axis=(1, 2),
                        )
                    )
                )
                if (
                    fraction >= minimum_tile_mask_fraction
                    and minimum_frame_fraction >= minimum_frame_tile_mask_fraction
                ):
                    candidates.append((fraction, left, top))
        candidates = diverse_candidates(
            candidates,
            maximum=max_tiles_per_entry,
        )
        for fraction, left, top in candidates:
            value = dict(raw)
            value["name"] = f"{raw['name']}:large-roi-{left:03d}-{top:03d}"
            value["forced_final_crop_offset"] = [left, top]
            value["large_roi_tile"] = {
                "bbox": [bbox_left, bbox_top, bbox_right, bbox_bottom],
                "mask_fraction": round(float(np.mean(mask > 0)), 8),
                "tile_mask_fraction": round(fraction, 8),
                "minimum_frame_tile_mask_fraction": round(
                    min(
                        float(
                            np.mean(
                                frame[
                                    top : top + TILE_SIZE,
                                    left : left + TILE_SIZE,
                                ]
                                > 0
                            )
                        )
                        for frame in masks
                    ),
                    8,
                ),
                "stride": stride,
                "version": 1,
            }
            output_values.append(value)
        if candidates:
            tiled_entries += 1

    for _ in range(random_anchor_copies):
        output_values.extend(dict(value) for value in raw_entries)
    if not output_values:
        raise RuntimeError("large-ROI selection produced an empty manifest")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in output_values
        ),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    tile_count = len(output_values) - random_anchor_copies * len(raw_entries)
    return {
        "input_entries": len(raw_entries),
        "large_roi_entries": tiled_entries,
        "native_tiles": tile_count,
        "random_anchor_entries": random_anchor_copies * len(raw_entries),
        "output_entries": len(output_values),
    }


def main() -> None:
    args = parse_args()
    summary = build_manifest(
        input_path=args.input,
        output_path=args.output,
        stride=args.stride,
        minimum_roi_extent=args.minimum_roi_extent,
        minimum_tile_mask_fraction=args.minimum_tile_mask_fraction,
        minimum_frame_tile_mask_fraction=args.minimum_frame_tile_mask_fraction,
        max_tiles_per_entry=args.max_tiles_per_entry,
        random_anchor_copies=args.random_anchor_copies,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
