#!/usr/bin/env python3
"""Compare restoration videos using stable frame-index-aligned ROI metrics.

The manifest fixes the source, reference, candidate videos, reference frame
offset, and sampled frame indices. ROI masks are derived once from the source
and reference pair, saved as PNG files, and reused by every candidate.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import structural_similarity


METRIC_NAMES = (
    "roi_psnr",
    "roi_ssim",
    "roi_edge_mae",
    "full_psnr",
    "full_ssim",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mask-directory", required=True, type=Path)
    parser.add_argument("--difference-threshold", type=float, default=18.0)
    parser.add_argument("--minimum-component-area", type=int, default=200)
    return parser.parse_args()


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    actual_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_index} from {path}")
    if actual_index != frame_index:
        raise RuntimeError(
            f"Frame seek mismatch for {path}: requested {frame_index}, "
            f"decoded {actual_index}"
        )
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def create_roi_mask(
    source: np.ndarray,
    reference: np.ndarray,
    difference_threshold: float,
    minimum_component_area: int,
) -> np.ndarray:
    difference = cv2.absdiff(source, reference).mean(axis=2)
    mask = (difference >= difference_threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
    )
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    filtered = np.zeros_like(mask)
    for label in range(1, component_count):
        if stats[label, cv2.CC_STAT_AREA] >= minimum_component_area:
            filtered[labels == label] = 255
    return filtered


def psnr_from_mse(mse: float) -> float:
    return 10.0 * math.log10((255.0 * 255.0) / max(mse, 1e-12))


def calculate_metrics(
    candidate: np.ndarray,
    reference: np.ndarray,
    mask_u8: np.ndarray,
) -> dict[str, float]:
    mask = mask_u8 > 0
    if not np.any(mask):
        raise RuntimeError("ROI mask is empty")

    candidate_float = candidate.astype(np.float32)
    reference_float = reference.astype(np.float32)
    error = candidate_float - reference_float
    roi_error = error[mask]

    _, ssim_map = structural_similarity(
        reference,
        candidate,
        channel_axis=2,
        data_range=255,
        full=True,
    )

    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_RGB2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    candidate_edges = cv2.Laplacian(candidate_gray, cv2.CV_32F, ksize=3)
    reference_edges = cv2.Laplacian(reference_gray, cv2.CV_32F, ksize=3)

    return {
        "roi_psnr": psnr_from_mse(float(np.mean(roi_error * roi_error))),
        "roi_ssim": float(np.mean(ssim_map[mask])),
        "roi_edge_mae": float(
            np.mean(np.abs(candidate_edges - reference_edges)[mask])
        ),
        "full_psnr": psnr_from_mse(float(np.mean(error * error))),
        "full_ssim": float(
            structural_similarity(
                reference,
                candidate,
                channel_axis=2,
                data_range=255,
            )
        ),
        "roi_pixels": int(np.count_nonzero(mask)),
    }


def average_metrics(frame_results: list[dict[str, Any]]) -> dict[str, float]:
    return {
        metric: float(np.mean([frame["metrics"][metric] for frame in frame_results]))
        for metric in (*METRIC_NAMES, "roi_pixels")
    }


def alignment_error(source: np.ndarray, reference: np.ndarray) -> float:
    """Return a robust full-frame error while ignoring the changed ROI tail."""
    difference = cv2.absdiff(source, reference).mean(axis=2).reshape(-1)
    cutoff = np.percentile(difference, 85.0)
    retained = difference[difference <= cutoff]
    return float(np.mean(retained))


def choose_reference_frame_offset(
    source_path: Path,
    reference_path: Path,
    frame_indices: list[int],
    requested_reference_frame_indices: list[int],
    search_radius: int,
) -> tuple[int, dict[str, float]]:
    if search_radius <= 0:
        return 0, {"0": 0.0}

    sampled_indices = frame_indices[: min(3, len(frame_indices))]
    sampled_reference_indices = requested_reference_frame_indices[
        : min(3, len(requested_reference_frame_indices))
    ]
    sources = {
        frame_index: read_frame(source_path, frame_index)
        for frame_index in sampled_indices
    }
    errors: dict[str, float] = {}
    for candidate_offset in range(-search_radius, search_radius + 1):
        if min(sampled_reference_indices) + candidate_offset < 0:
            continue
        candidate_errors = []
        for frame_index, reference_index in zip(
            sampled_indices,
            sampled_reference_indices,
            strict=True,
        ):
            reference = read_frame(reference_path, reference_index + candidate_offset)
            candidate_errors.append(alignment_error(sources[frame_index], reference))
        errors[str(candidate_offset)] = float(np.mean(candidate_errors))

    chosen = min(errors, key=errors.get)
    return int(chosen), errors


def evaluate_case(
    case: dict[str, Any],
    mask_directory: Path,
    difference_threshold: float,
    minimum_component_area: int,
) -> dict[str, Any]:
    case_name = case["name"]
    source_path = Path(case["source"])
    reference_path = Path(case["reference"])
    reference_search_radius = int(case.get("reference_start_frame_search_radius", 0))
    frame_indices = [int(index) for index in case["frame_indices"]]
    if "reference_frame_indices" in case:
        requested_reference_frame_indices = [
            int(index) for index in case["reference_frame_indices"]
        ]
    else:
        requested_reference_start_frame = int(case.get("reference_start_frame", 0))
        requested_reference_frame_indices = [
            requested_reference_start_frame + frame_index
            for frame_index in frame_indices
        ]
    if len(requested_reference_frame_indices) != len(frame_indices):
        raise ValueError(
            f"{case_name}: reference_frame_indices and frame_indices must have "
            "the same length"
        )

    for path in (source_path, reference_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    reference_frame_offset, reference_alignment_errors = choose_reference_frame_offset(
        source_path,
        reference_path,
        frame_indices,
        requested_reference_frame_indices,
        reference_search_radius,
    )
    reference_frame_indices = [
        index + reference_frame_offset
        for index in requested_reference_frame_indices
    ]

    shared_frames: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    case_mask_directory = mask_directory / case_name
    case_mask_directory.mkdir(parents=True, exist_ok=True)

    for frame_index, reference_index in zip(
        frame_indices,
        reference_frame_indices,
        strict=True,
    ):
        source = read_frame(source_path, frame_index)
        reference = read_frame(reference_path, reference_index)
        if source.shape != reference.shape:
            raise RuntimeError(
                f"Shape mismatch in {case_name} frame {frame_index}: "
                f"source={source.shape}, reference={reference.shape}"
            )

        mask_path = case_mask_directory / f"frame-{frame_index:06d}.png"
        if mask_path.is_file():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise RuntimeError(f"Could not read ROI mask: {mask_path}")
        else:
            mask = create_roi_mask(
                source,
                reference,
                difference_threshold,
                minimum_component_area,
            )
            if not cv2.imwrite(str(mask_path), mask):
                raise RuntimeError(f"Could not save ROI mask: {mask_path}")
        shared_frames[frame_index] = (source, reference, mask)

    candidate_results: dict[str, Any] = {}
    for candidate_name, candidate_value in case["candidates"].items():
        candidate_path = Path(candidate_value)
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)

        frame_results = []
        for frame_index, reference_index in zip(
            frame_indices,
            reference_frame_indices,
            strict=True,
        ):
            _, reference, mask = shared_frames[frame_index]
            candidate = read_frame(candidate_path, frame_index)
            if candidate.shape != reference.shape:
                raise RuntimeError(
                    f"Shape mismatch for {candidate_name} in {case_name} "
                    f"frame {frame_index}: candidate={candidate.shape}, "
                    f"reference={reference.shape}"
                )
            frame_results.append(
                {
                    "frame_index": frame_index,
                    "reference_frame_index": reference_index,
                    "metrics": calculate_metrics(candidate, reference, mask),
                }
            )

        candidate_results[candidate_name] = {
            "path": str(candidate_path),
            "average": average_metrics(frame_results),
            "frames": frame_results,
        }

    baseline_name = case.get("baseline")
    deltas: dict[str, dict[str, float]] = {}
    if baseline_name:
        baseline = candidate_results[baseline_name]["average"]
        for candidate_name, result in candidate_results.items():
            if candidate_name == baseline_name:
                continue
            deltas[candidate_name] = {
                metric: result["average"][metric] - baseline[metric]
                for metric in METRIC_NAMES
            }

    return {
        "name": case_name,
        "source": str(source_path),
        "reference": str(reference_path),
        "requested_reference_frame_indices": requested_reference_frame_indices,
        "reference_frame_indices": reference_frame_indices,
        "reference_frame_offset": reference_frame_offset,
        "reference_start_frame_search_radius": reference_search_radius,
        "reference_alignment_errors": reference_alignment_errors,
        "frame_indices": frame_indices,
        "baseline": baseline_name,
        "candidates": candidate_results,
        "deltas_from_baseline": deltas,
    }


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.mask_directory.mkdir(parents=True, exist_ok=True)

    results = {
        "protocol": {
            "alignment": "decoded frame index",
            "difference_threshold": args.difference_threshold,
            "minimum_component_area": args.minimum_component_area,
            "roi_mask": "source/reference RGB mean absolute difference plus morphology",
            "higher_is_better": ["roi_psnr", "roi_ssim", "full_psnr", "full_ssim"],
            "lower_is_better": ["roi_edge_mae"],
        },
        "cases": [
            evaluate_case(
                case,
                args.mask_directory,
                args.difference_threshold,
                args.minimum_component_area,
            )
            for case in manifest["cases"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    for case in results["cases"]:
        print(f"[{case['name']}] baseline={case['baseline']}")
        for candidate_name, candidate in case["candidates"].items():
            average = candidate["average"]
            print(
                f"  {candidate_name}: "
                f"ROI PSNR={average['roi_psnr']:.4f}, "
                f"ROI SSIM={average['roi_ssim']:.4f}, "
                f"Edge MAE={average['roi_edge_mae']:.4f}, "
                f"Full PSNR={average['full_psnr']:.4f}, "
                f"Full SSIM={average['full_ssim']:.4f}"
            )
        for candidate_name, delta in case["deltas_from_baseline"].items():
            print(
                f"  delta {candidate_name}: "
                f"ROI PSNR={delta['roi_psnr']:+.4f}, "
                f"ROI SSIM={delta['roi_ssim']:+.4f}, "
                f"Edge MAE={delta['roi_edge_mae']:+.4f}, "
                f"Full PSNR={delta['full_psnr']:+.4f}, "
                f"Full SSIM={delta['full_ssim']:+.4f}"
            )
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
