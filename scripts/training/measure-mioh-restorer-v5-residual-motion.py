#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Measure crop-centred residual motion for the five V5 native ROI buckets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from lada.datasetcreation.restoration_dataset_metadata import (
    RestorationDatasetMetadataV2,
)
from lada.models.mioh_restorer.runner_v5 import round_to_even, select_v5_bucket
from lada.models.mioh_restorer.model_v5 import V5_BUCKETS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata-root", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-frames-per-clip", type=int, default=240)
    parser.add_argument("--smoothing-radius", type=int, default=4)
    parser.add_argument("--context-fraction", type=float, default=0.30)
    return parser.parse_args()


def metadata_paths(roots: list[Path], limit: int | None) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        paths.extend(
            path for path in sorted(root.glob("*.json")) if not path.name.startswith("._")
        )
    return paths[:limit] if limit is not None else paths


def read_video(path: Path, maximum: int, *, grayscale: bool = False) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open {path}")
    frames: list[np.ndarray] = []
    try:
        while len(frames) < maximum:
            ok, frame = capture.read()
            if not ok:
                break
            if grayscale:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
    finally:
        capture.release()
    return frames


def mask_box(mask: np.ndarray) -> tuple[float, float, int, int] | None:
    rows, columns = np.nonzero(mask > 127)
    if not len(rows):
        return None
    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    return (
        (left + right - 1) / 2.0,
        (top + bottom - 1) / 2.0,
        right - left,
        bottom - top,
    )


def fill_centres(
    boxes: list[tuple[float, float, int, int] | None],
    frame_width: int,
    frame_height: int,
) -> tuple[np.ndarray, np.ndarray]:
    centres = np.full((len(boxes), 2), np.nan, dtype=np.float32)
    sizes = np.zeros((len(boxes), 2), dtype=np.float32)
    for index, box in enumerate(boxes):
        if box is not None:
            centres[index] = box[:2]
            sizes[index] = box[2:]
    valid = np.flatnonzero(np.isfinite(centres[:, 0]))
    if not len(valid):
        centres[:] = (frame_width / 2.0, frame_height / 2.0)
        return centres, sizes
    for index in range(len(boxes)):
        if not np.isfinite(centres[index, 0]):
            nearest = valid[np.argmin(np.abs(valid - index))]
            centres[index] = centres[nearest]
            sizes[index] = sizes[nearest]
    return centres, sizes


def smooth_centres(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    kernel = np.ones(radius * 2 + 1, dtype=np.float32) / (radius * 2 + 1)
    return np.stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)],
        axis=1,
    )


def extract_native_crop(
    frame: np.ndarray,
    centre: np.ndarray,
    size: int,
    *,
    mask: bool = False,
) -> np.ndarray:
    x = round_to_even(float(centre[0]) - size / 2)
    y = round_to_even(float(centre[1]) - size / 2)
    height, width = frame.shape[:2]
    left, top = max(-x, 0), max(-y, 0)
    right, bottom = max(x + size - width, 0), max(y + size - height, 0)
    source = frame[max(y, 0) : min(y + size, height), max(x, 0) : min(x + size, width)]
    border = cv2.BORDER_CONSTANT if mask else cv2.BORDER_REPLICATE
    return cv2.copyMakeBorder(source, top, bottom, left, right, border, value=0)


def flow_magnitude(
    first: np.ndarray,
    second: np.ndarray,
    first_mask: np.ndarray,
    second_mask: np.ndarray,
) -> float | None:
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        first_gray,
        second_gray,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=21,
        iterations=4,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    roi = (first_mask > 127) | (second_mask > 127)
    if not np.any(roi):
        return None
    magnitude = np.linalg.norm(flow, axis=2)[roi]
    return float(np.percentile(magnitude, 75))


def summarize(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "maximum": float(array.max()),
    }


def main() -> int:
    args = parse_args()
    if args.max_frames_per_clip < 2:
        raise ValueError("at least two frames are required")
    samples: dict[int, list[float]] = {bucket: [] for bucket in V5_BUCKETS}
    failures: list[dict[str, str]] = []
    clip_count = 0
    for metadata_path in metadata_paths(args.metadata_root, args.limit):
        try:
            metadata = RestorationDatasetMetadataV2.from_json_file(metadata_path)
            video_path = metadata_path.parent / metadata.relative_nsfw_video_path
            mask_path = metadata_path.parent / metadata.relative_mask_video_path
            frames = read_video(video_path, args.max_frames_per_clip)
            masks = read_video(mask_path, args.max_frames_per_clip, grayscale=True)
            count = min(len(frames), len(masks))
            if count < 2:
                raise RuntimeError("clip has fewer than two decoded frames")
            frames, masks = frames[:count], masks[:count]
            height, width = frames[0].shape[:2]
            boxes = [mask_box(mask) for mask in masks]
            centres, sizes = fill_centres(boxes, width, height)
            centres = smooth_centres(centres, args.smoothing_radius)
            valid_sizes = sizes[np.any(sizes > 0, axis=1)]
            if not len(valid_sizes):
                continue
            roi_width, roi_height = np.percentile(valid_sizes, 95, axis=0)
            bucket = select_v5_bucket(
                float(roi_width),
                float(roi_height),
                context_fraction=args.context_fraction,
            )
            cropped_frames = [
                extract_native_crop(frame, centre, bucket)
                for frame, centre in zip(frames, centres, strict=True)
            ]
            cropped_masks = [
                extract_native_crop(mask, centre, bucket, mask=True)
                for mask, centre in zip(masks, centres, strict=True)
            ]
            for index in range(count - 1):
                value = flow_magnitude(
                    cropped_frames[index],
                    cropped_frames[index + 1],
                    cropped_masks[index],
                    cropped_masks[index + 1],
                )
                if value is not None and math.isfinite(value):
                    samples[bucket].append(value)
            clip_count += 1
        except Exception as error:  # report corrupt clips without losing the audit
            failures.append({"metadata": str(metadata_path), "error": str(error)})

    report = {
        "clips": clip_count,
        "failed_clips": failures,
        "metric": "p75 Farneback magnitude inside crop-centred ROI, pixels/frame",
        "buckets": {str(bucket): summarize(samples[bucket]) for bucket in V5_BUCKETS},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
