# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Compare the legacy and memory-efficient RF-DETR preprocessing paths.

The benchmark JSON produced for the Jasna v6 quality comparison contains the
clip name and frame index for each fixed validation sample. This script runs
both preprocessing paths through the same Core AI model and reports whether
query selection, boxes, scores, and binary masks remain equivalent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from lada.models.rfdetr.rfdetr_coreai_segmentation_model import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    RFDETRCoreAIRuntime,
    RFDETRCoreAISegmentationModel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--benchmark-json", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=576)
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--logit-classes", type=int, default=3)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--max-det", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def legacy_normalize(image: np.ndarray, resolution: int) -> np.ndarray:
    value = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    value = value.float().div_(255.0)
    value = F.interpolate(
        value,
        size=(resolution, resolution),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return value.sub_(mean).div_(std).contiguous().numpy()


def load_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"could not open {path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"could not read frame {frame_index} from {path}")
        return frame
    finally:
        capture.release()


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_union = np.any(first > 0, axis=0) if len(first) else None
    second_union = np.any(second > 0, axis=0) if len(second) else None
    if first_union is None and second_union is None:
        return 1.0
    if first_union is None or second_union is None:
        return 0.0
    union = np.logical_or(first_union, second_union).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(first_union, second_union).sum() / union)


def main() -> int:
    args = parse_args()
    benchmark = json.loads(args.benchmark_json.read_text())
    samples = benchmark["frames"]
    if args.limit is not None:
        samples = samples[: args.limit]

    runtime = RFDETRCoreAIRuntime(
        args.model,
        resolution=args.resolution,
        queries=args.queries,
        logit_classes=args.logit_classes,
    )
    model = RFDETRCoreAISegmentationModel(
        args.model,
        resolution=args.resolution,
        queries=args.queries,
        logit_classes=args.logit_classes,
        conf=args.confidence,
        max_det=args.max_det,
        runtime=runtime,
    )

    input_max_differences: list[float] = []
    input_mean_differences: list[float] = []
    mask_ious: list[float] = []
    box_max_differences: list[float] = []
    score_max_differences: list[float] = []
    count_mismatches = 0
    try:
        for sample in samples:
            clip_path = args.dataset_root / f"{sample['clip']}.mp4"
            frame = load_frame(clip_path, int(sample["frame"]))
            legacy_input = legacy_normalize(frame, args.resolution)
            optimized_input = model._normalize_one(
                torch.from_numpy(frame)
            ).numpy()
            input_difference = np.abs(legacy_input - optimized_input)
            input_max_differences.append(float(input_difference.max()))
            input_mean_differences.append(float(input_difference.mean()))

            legacy = runtime.infer_selected(
                legacy_input,
                conf=args.confidence,
                max_det=args.max_det,
            )
            optimized = runtime.infer_selected(
                optimized_input,
                conf=args.confidence,
                max_det=args.max_det,
            )
            if len(legacy[0]) != len(optimized[0]):
                count_mismatches += 1
            comparable = min(len(legacy[0]), len(optimized[0]))
            if comparable:
                box_max_differences.append(
                    float(
                        np.abs(
                            legacy[0][:comparable] - optimized[0][:comparable]
                        ).max()
                    )
                )
                score_max_differences.append(
                    float(
                        np.abs(
                            legacy[1][:comparable] - optimized[1][:comparable]
                        ).max()
                    )
                )
            mask_ious.append(mask_iou(legacy[2], optimized[2]))
    finally:
        runtime.close()

    report = {
        "samples": len(samples),
        "count_mismatches": count_mismatches,
        "input_max_abs": max(input_max_differences, default=0.0),
        "input_mean_abs": float(
            np.mean(input_mean_differences, dtype=np.float64)
        ),
        "mask_iou_min": min(mask_ious, default=1.0),
        "mask_iou_mean": float(np.mean(mask_ious, dtype=np.float64)),
        "box_max_abs": max(box_max_differences, default=0.0),
        "score_max_abs": max(score_max_differences, default=0.0),
    }
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
