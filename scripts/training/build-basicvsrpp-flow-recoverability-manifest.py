#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Select flow-aligned, recoverable native windows for BasicVSR++ HF tuning.

This is an additive second-pass selector.  It consumes an existing V5 JSONL
manifest whose entries are native 512x512 windows and leaves that manifest
untouched.  Every candidate is decoded as nine exact source crops.  A single
shared 256x256 crop is then chosen around the temporal mask union; these are
integer slices, never resized training pixels.

Farneback flow is measured from the centre frame to each neighbour.  A window
is retained only when enough neighbours can be aligned with low photometric
residual, forward/backward flow consistency, and correlated high-frequency
content.  A separate row-alternation detector rejects likely interlaced or
combed material.  ``--analysis-size`` may downscale the *temporary metric
arrays* with INTER_AREA to reduce selection cost; it never changes paths,
origins, or pixels subsequently decoded by the training dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

import av
import cv2
import numpy as np

from lada.models.mioh_restorer.model_v5 import NUM_INPUT_FRAMES


NATIVE_SIZE = 512
ANALYSIS_CROP_SIZE = 256
CENTER = NUM_INPUT_FRAMES // 2


@dataclass(frozen=True)
class FlowThresholds:
    minimum_mask_fraction: float = 0.01
    minimum_center_hf_rms: float = 0.0025
    maximum_combing_score: float = 0.12
    maximum_aligned_residual: float = 0.085
    minimum_flow_valid_fraction: float = 0.60
    maximum_cycle_error: float = 2.0
    minimum_hf_correlation: float = 0.10
    minimum_recoverable_hf: float = 0.10
    minimum_good_neighbors: int = 2


@dataclass(frozen=True)
class FlowRecoverabilityMetrics:
    mask_fraction: float
    center_hf_rms: float
    combing_score: float
    good_neighbors: int
    aligned_residual: float
    unaligned_residual: float
    alignment_gain: float
    flow_valid_fraction: float
    flow_cycle_error: float
    hf_correlation: float
    recoverable_hf: float
    score: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quota", type=int, required=True)
    parser.add_argument(
        "--report",
        type=Path,
        help="atomic metrics sidecar (default: <output>.flow-metrics.json)",
    )
    parser.add_argument(
        "--analysis-size",
        type=int,
        default=ANALYSIS_CROP_SIZE,
        help=(
            "flow-analysis size in [64, 256]; values below 256 downscale only "
            "temporary metric arrays, never training pixels"
        ),
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--maximum-combing-score", type=float, default=0.12)
    parser.add_argument("--maximum-aligned-residual", type=float, default=0.085)
    parser.add_argument("--minimum-recoverable-hf", type=float, default=0.10)
    parser.add_argument("--minimum-good-neighbors", type=int, default=2)
    args = parser.parse_args(argv)
    if args.quota <= 0:
        parser.error("--quota must be positive")
    if not 64 <= args.analysis_size <= ANALYSIS_CROP_SIZE:
        parser.error("--analysis-size must be in [64, 256]")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    if not 0.0 <= args.maximum_combing_score <= 1.0:
        parser.error("--maximum-combing-score must be in [0, 1]")
    if args.maximum_aligned_residual <= 0:
        parser.error("--maximum-aligned-residual must be positive")
    if not 0.0 <= args.minimum_recoverable_hf <= 1.0:
        parser.error("--minimum-recoverable-hf must be in [0, 1]")
    if not 1 <= args.minimum_good_neighbors < NUM_INPUT_FRAMES:
        parser.error("--minimum-good-neighbors must be in [1, 8]")
    return args


def _resolve(value: object, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_native_512_entries(path: Path) -> list[dict[str, Any]]:
    """Load the V5 interchange format without discarding additive fields."""

    values: list[dict[str, Any]] = []
    root = path.parent
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("entry is not an object")
                origins = tuple(
                    (int(pair[0]), int(pair[1])) for pair in value["origins"]
                )
                reliability = tuple(float(item) for item in value["mask_reliability"])
                if int(value["bucket"]) != NATIVE_SIZE:
                    raise ValueError("flow selector requires native 512 entries")
                if int(value["start_frame"]) < 0:
                    raise ValueError("start_frame must be non-negative")
                if len(origins) != NUM_INPUT_FRAMES or len(reliability) != NUM_INPUT_FRAMES:
                    raise ValueError("expected nine origins and reliability values")
                if any(x % 2 or y % 2 for x, y in origins):
                    raise ValueError("origins must use even source pixels")
                if any(not math.isfinite(item) or not 0 <= item <= 1 for item in reliability):
                    raise ValueError("mask reliability must be finite and in [0, 1]")
                target = _resolve(value["target_video"], root)
                mask = _resolve(value["mask_video"], root)
                source_id = str(value["source_video_id"])
                name = str(value["name"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid native manifest line {line_number}: {error}"
                ) from error
            copied = dict(value)
            copied["_target_path"] = target
            copied["_mask_path"] = mask
            copied["_origins"] = origins
            copied["_source_id"] = source_id
            copied["_name"] = name
            values.append(copied)
    if not values:
        raise ValueError(f"native manifest is empty: {path}")
    return values


def crop_native(
    frame: np.ndarray,
    origin: tuple[int, int],
    *,
    mask: bool,
) -> np.ndarray:
    """Exact 512 crop: RGB edge replication or zero-padded masks."""

    if frame.ndim not in (2, 3) or min(frame.shape[:2]) <= 0:
        raise ValueError("invalid decoded frame")
    x, y = origin
    height, width = frame.shape[:2]
    if not mask:
        xs = np.clip(np.arange(x, x + NATIVE_SIZE), 0, width - 1)
        ys = np.clip(np.arange(y, y + NATIVE_SIZE), 0, height - 1)
        return np.ascontiguousarray(frame[ys[:, None], xs[None, :]])

    if frame.ndim == 3:
        frame = frame[..., 0]
    result = np.zeros((NATIVE_SIZE, NATIVE_SIZE), dtype=frame.dtype)
    sx0, sy0 = max(x, 0), max(y, 0)
    sx1, sy1 = min(x + NATIVE_SIZE, width), min(y + NATIVE_SIZE, height)
    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0 - x, sy0 - y
        result[dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] = frame[
            sy0:sy1, sx0:sx1
        ]
    return np.ascontiguousarray(result)


def roi_crop_offset(masks: Sequence[np.ndarray]) -> tuple[int, int]:
    """Return one deterministic 256 crop around the nine-frame mask union."""

    if len(masks) != NUM_INPUT_FRAMES:
        raise ValueError("ROI selection requires nine masks")
    union = np.zeros((NATIVE_SIZE, NATIVE_SIZE), dtype=np.uint8)
    for mask in masks:
        if mask.shape != union.shape:
            raise ValueError(f"expected a 512 mask, got {mask.shape}")
        union = np.maximum(union, mask > 0)
    ys, xs = np.nonzero(union)
    if len(xs):
        center_x = (int(xs.min()) + int(xs.max()) + 1) // 2
        center_y = (int(ys.min()) + int(ys.max()) + 1) // 2
    else:
        center_x = center_y = NATIVE_SIZE // 2
    maximum = NATIVE_SIZE - ANALYSIS_CROP_SIZE
    left = min(max(center_x - ANALYSIS_CROP_SIZE // 2, 0), maximum)
    top = min(max(center_y - ANALYSIS_CROP_SIZE // 2, 0), maximum)
    # Preserve phase consistency with V5/BasicVSR++ training crops.
    return top - top % 2, left - left % 2


def prepare_analysis_window(
    targets: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    analysis_size: int = ANALYSIS_CROP_SIZE,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    """Take shared exact 256 slices; optionally downscale metric copies only."""

    if len(targets) != NUM_INPUT_FRAMES or len(masks) != NUM_INPUT_FRAMES:
        raise ValueError("flow analysis requires nine targets and masks")
    top, left = roi_crop_offset(masks)
    target_crops = [
        np.ascontiguousarray(
            frame[top : top + ANALYSIS_CROP_SIZE, left : left + ANALYSIS_CROP_SIZE]
        )
        for frame in targets
    ]
    mask_crops = [
        np.ascontiguousarray(
            frame[top : top + ANALYSIS_CROP_SIZE, left : left + ANALYSIS_CROP_SIZE]
        )
        for frame in masks
    ]
    if any(frame.shape[:2] != (ANALYSIS_CROP_SIZE, ANALYSIS_CROP_SIZE) for frame in target_crops):
        raise ValueError("target native crop is incomplete")
    if analysis_size != ANALYSIS_CROP_SIZE:
        target_crops = [
            cv2.resize(frame, (analysis_size, analysis_size), interpolation=cv2.INTER_AREA)
            for frame in target_crops
        ]
        mask_crops = [
            cv2.resize(frame, (analysis_size, analysis_size), interpolation=cv2.INTER_AREA)
            for frame in mask_crops
        ]
    return target_crops, mask_crops, (top, left)


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise ValueError("target must be RGB")
    return cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def _hf(gray: np.ndarray) -> np.ndarray:
    return gray - cv2.GaussianBlur(
        gray, (0, 0), sigmaX=1.0, sigmaY=1.0, borderType=cv2.BORDER_REPLICATE
    )


def combing_score(gray: np.ndarray, region: np.ndarray) -> float:
    """Measure alternating scan lines while discounting isotropic texture.

    A pixel contributes only when the middle row is a strict extremum relative
    to both adjacent rows.  Progressive step edges therefore contribute zero,
    while A/B/A combing contributes at every affected row.  Subtracting the
    equivalent horizontal score avoids rejecting checkerboard-like texture.
    """

    if gray.ndim != 2 or region.shape != gray.shape or min(gray.shape) < 3:
        raise ValueError("invalid combing inputs")

    def axis_score(values: np.ndarray, valid: np.ndarray, axis: int) -> float:
        if axis == 0:
            before, middle, after = values[:-2], values[1:-1], values[2:]
            selected = valid[1:-1]
        else:
            before, middle, after = values[:, :-2], values[:, 1:-1], values[:, 2:]
            selected = valid[:, 1:-1]
        first = middle - before
        second = middle - after
        extrema = (first * second) > 0
        amplitude = np.abs(first + second) * 0.5
        weight = np.clip((amplitude - 2.0 / 255.0) / (24.0 / 255.0), 0.0, 1.0)
        active = selected & extrema
        denominator = max(int(selected.sum()), 1)
        return float((weight * active).sum() / denominator)

    vertical = axis_score(gray, region, 0)
    horizontal = axis_score(gray, region, 1)
    return max(0.0, vertical - horizontal)


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    return float(selected.mean()) if selected.size else float("inf")


def _masked_rms(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask]
    return float(np.sqrt(np.mean(np.square(selected)))) if selected.size else 0.0


def _masked_correlation(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    x = left[mask].astype(np.float64)
    y = right[mask].astype(np.float64)
    if x.size < 32:
        return 0.0
    x -= x.mean()
    y -= y.mean()
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(x, y) / denominator, -1.0, 1.0))


def _flow(
    source: np.ndarray,
    destination: np.ndarray,
) -> np.ndarray:
    # OpenCV accepts float32 here, but Farneback's polynomial-expansion
    # thresholds are calibrated for image-scale values.  Passing normalized
    # 0..1 luma makes ordinary motion collapse numerically toward zero.
    source_flow = np.ascontiguousarray(source * 255.0, dtype=np.float32)
    destination_flow = np.ascontiguousarray(destination * 255.0, dtype=np.float32)
    return cv2.calcOpticalFlowFarneback(
        source_flow,
        destination_flow,
        None,
        pyr_scale=0.5,
        levels=4,
        winsize=21,
        iterations=4,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )


def neighbor_metrics(
    center: np.ndarray,
    neighbor: np.ndarray,
    roi: np.ndarray,
    *,
    maximum_cycle_error: float,
) -> dict[str, float]:
    """Align ``neighbor`` into centre coordinates and measure transferable HF."""

    height, width = center.shape
    forward = _flow(center, neighbor)
    backward = _flow(neighbor, center)
    yy, xx = np.indices((height, width), dtype=np.float32)
    map_x = xx + forward[..., 0]
    map_y = yy + forward[..., 1]
    inside = (
        (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )
    aligned = cv2.remap(
        neighbor,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    backward_x = cv2.remap(
        backward[..., 0], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    backward_y = cv2.remap(
        backward[..., 1], map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    cycle = np.sqrt(
        np.square(forward[..., 0] + backward_x)
        + np.square(forward[..., 1] + backward_y)
    )
    valid = roi & inside & (cycle <= maximum_cycle_error)
    roi_pixels = max(int(roi.sum()), 1)
    valid_fraction = float(valid.sum() / roi_pixels)
    if not np.any(valid):
        return {
            "aligned_residual": float("inf"),
            "unaligned_residual": float("inf"),
            "flow_valid_fraction": 0.0,
            "flow_cycle_error": float("inf"),
            "hf_correlation": 0.0,
            "recoverable_hf": 0.0,
        }

    smooth_center = cv2.GaussianBlur(center, (0, 0), 1.0)
    smooth_neighbor = cv2.GaussianBlur(neighbor, (0, 0), 1.0)
    smooth_aligned = cv2.remap(
        smooth_neighbor,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    aligned_residual = _masked_mean(np.abs(smooth_center - smooth_aligned), valid)
    unaligned_residual = _masked_mean(np.abs(smooth_center - smooth_neighbor), valid)
    center_hf = _hf(center)
    neighbor_hf = _hf(neighbor)
    aligned_hf = cv2.remap(
        neighbor_hf,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    correlation = _masked_correlation(center_hf, aligned_hf, valid)
    center_energy = _masked_rms(center_hf, valid)
    neighbor_energy = _masked_rms(aligned_hf, valid)
    energy_ratio = min(center_energy, neighbor_energy) / max(
        center_energy, neighbor_energy, 1e-8
    )
    residual_quality = max(0.0, 1.0 - aligned_residual / 0.20)
    recoverable = max(0.0, correlation) * energy_ratio * residual_quality
    return {
        "aligned_residual": aligned_residual,
        "unaligned_residual": unaligned_residual,
        "flow_valid_fraction": valid_fraction,
        "flow_cycle_error": _masked_mean(cycle, valid),
        "hf_correlation": correlation,
        "recoverable_hf": float(np.clip(recoverable, 0.0, 1.0)),
    }


def evaluate_flow_window(
    targets: Sequence[np.ndarray],
    masks: Sequence[np.ndarray],
    *,
    analysis_size: int = ANALYSIS_CROP_SIZE,
    thresholds: FlowThresholds = FlowThresholds(),
) -> tuple[str | None, FlowRecoverabilityMetrics | None]:
    target_crops, mask_crops, _ = prepare_analysis_window(
        targets, masks, analysis_size=analysis_size
    )
    grays = [_gray(frame) for frame in target_crops]
    masks_float = []
    for frame in mask_crops:
        value = frame.astype(np.float32)
        if value.size and float(value.max()) > 1.0:
            value /= 255.0
        masks_float.append(value)
    center_roi = masks_float[CENTER] >= 0.5
    if not np.any(center_roi):
        return "empty_center_mask", None
    erosion_radius = max(1, int(round(analysis_size / 128.0)))
    erosion = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erosion_radius * 2 + 1,) * 2
    )
    center_roi = cv2.erode(center_roi.astype(np.uint8), erosion) > 0
    mask_fraction = float(center_roi.mean())
    if mask_fraction + 1e-12 < thresholds.minimum_mask_fraction:
        return "mask_fraction", None

    context_radius = max(2, int(round(analysis_size / 16.0)))
    context_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (context_radius * 2 + 1,) * 2
    )
    context = cv2.dilate(center_roi.astype(np.uint8), context_kernel) > 0
    combing = max(combing_score(gray, context) for gray in grays)
    if combing > thresholds.maximum_combing_score:
        return "combing", None

    center = grays[CENTER]
    center_hf_rms = _masked_rms(_hf(center), center_roi)
    if center_hf_rms + 1e-12 < thresholds.minimum_center_hf_rms:
        return "center_hf_rms", None
    scale = analysis_size / ANALYSIS_CROP_SIZE
    maximum_cycle_error = thresholds.maximum_cycle_error * scale
    values = [
        neighbor_metrics(
            center,
            gray,
            center_roi,
            maximum_cycle_error=maximum_cycle_error,
        )
        for index, gray in enumerate(grays)
        if index != CENTER
    ]
    good = [
        value
        for value in values
        if value["flow_valid_fraction"] >= thresholds.minimum_flow_valid_fraction
        and value["aligned_residual"] <= thresholds.maximum_aligned_residual
        and value["hf_correlation"] >= thresholds.minimum_hf_correlation
        and value["recoverable_hf"] >= thresholds.minimum_recoverable_hf
    ]
    if len(good) < thresholds.minimum_good_neighbors:
        residuals = sorted(
            value["aligned_residual"]
            for value in values
            if math.isfinite(value["aligned_residual"])
        )
        if (
            len(residuals) < thresholds.minimum_good_neighbors
            or residuals[thresholds.minimum_good_neighbors - 1]
            > thresholds.maximum_aligned_residual
        ):
            return "aligned_residual", None
        return "insufficient_recoverable_neighbors", None

    good.sort(
        key=lambda value: (
            -value["recoverable_hf"],
            value["aligned_residual"],
            -value["flow_valid_fraction"],
        )
    )
    chosen = good[: max(thresholds.minimum_good_neighbors, min(4, len(good)))]

    def average(name: str) -> float:
        return float(np.mean([value[name] for value in chosen]))

    aligned = average("aligned_residual")
    unaligned = average("unaligned_residual")
    gain = max(0.0, (unaligned - aligned) / max(unaligned, 1e-8))
    valid_fraction = average("flow_valid_fraction")
    cycle_error = average("flow_cycle_error")
    correlation = average("hf_correlation")
    recoverable = average("recoverable_hf")
    residual_quality = max(
        0.0, 1.0 - aligned / thresholds.maximum_aligned_residual
    )
    score = float(
        center_hf_rms
        * math.sqrt(mask_fraction)
        * recoverable
        * math.sqrt(valid_fraction)
        * (0.5 + 0.5 * gain)
        * (0.5 + 0.5 * residual_quality)
        * (1.0 - min(combing / max(thresholds.maximum_combing_score, 1e-8), 1.0))
    )
    return None, FlowRecoverabilityMetrics(
        mask_fraction=mask_fraction,
        center_hf_rms=center_hf_rms,
        combing_score=combing,
        good_neighbors=len(good),
        aligned_residual=aligned,
        unaligned_residual=unaligned,
        alignment_gain=gain,
        flow_valid_fraction=valid_fraction,
        flow_cycle_error=cycle_error,
        hf_correlation=correlation,
        recoverable_hf=recoverable,
        score=score,
    )


def _decode(path: Path, pixel_format: str) -> Iterator[np.ndarray]:
    with av.open(str(path), mode="r") as container:
        for frame in container.decode(video=0):
            yield frame.to_ndarray(format=pixel_format)


def evaluate_group(
    records: Sequence[dict[str, Any]],
    *,
    analysis_size: int,
    thresholds: FlowThresholds,
    decoder: Callable[..., Iterable[np.ndarray]] | None = None,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Decode one target/mask pair once, retaining only active native crops."""

    ordered = sorted(records, key=lambda value: (int(value["start_frame"]), value["_name"]))
    by_start: dict[int, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for position, record in enumerate(ordered):
        by_start[int(record["start_frame"])].append((position, record))
    final_index = max(int(record["start_frame"]) + NUM_INPUT_FRAMES - 1 for record in ordered)
    target_path = ordered[0]["_target_path"]
    mask_path = ordered[0]["_mask_path"]
    make = decoder or (lambda path, pixel_format: _decode(path, pixel_format))
    target_iterator = iter(make(target_path, pixel_format="rgb24"))
    mask_iterator = iter(make(mask_path, pixel_format="gray"))
    active: dict[int, tuple[dict[str, Any], list[np.ndarray], list[np.ndarray]]] = {}
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    decoded = 0
    try:
        for frame_index in range(final_index + 1):
            try:
                target_frame = next(target_iterator)
                mask_frame = next(mask_iterator)
            except StopIteration:
                break
            decoded = frame_index + 1
            for position, record in by_start.get(frame_index, ()):
                active[position] = (record, [], [])
            completed: list[int] = []
            for position, (record, targets, masks) in active.items():
                offset = frame_index - int(record["start_frame"])
                if not 0 <= offset < NUM_INPUT_FRAMES:
                    continue
                origin = record["_origins"][offset]
                targets.append(crop_native(target_frame, origin, mask=False))
                masks.append(crop_native(mask_frame, origin, mask=True))
                if len(targets) == NUM_INPUT_FRAMES:
                    try:
                        reason, metrics = evaluate_flow_window(
                            targets,
                            masks,
                            analysis_size=analysis_size,
                            thresholds=thresholds,
                        )
                    except (TypeError, ValueError, cv2.error, FloatingPointError):
                        reason, metrics = "invalid_candidate", None
                    if reason is None:
                        assert metrics is not None
                        record["_flow_recoverability"] = metrics
                        accepted.append(record)
                    else:
                        rejected[reason] += 1
                    completed.append(position)
            for position in completed:
                del active[position]
            if frame_index >= final_index:
                break
    finally:
        for iterator in (target_iterator, mask_iterator):
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
    for record in ordered:
        if int(record["start_frame"]) + NUM_INPUT_FRAMES > decoded:
            rejected["short_decode"] += 1
    return accepted, rejected


def _order(record: dict[str, Any]) -> tuple[float, str, int, str]:
    metrics: FlowRecoverabilityMetrics = record["_flow_recoverability"]
    return (
        -metrics.score,
        str(record["_target_path"]),
        int(record["start_frame"]),
        record["_name"],
    )


def source_balanced_select(
    candidates: Iterable[dict[str, Any]], quota: int
) -> list[dict[str, Any]]:
    """Round-robin ranked candidates so one long source cannot dominate."""

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int, str]] = set()
    for candidate in sorted(candidates, key=_order):
        key = (
            str(candidate["_target_path"]),
            int(candidate["start_frame"]),
            candidate["_name"],
        )
        if key in seen:
            continue
        seen.add(key)
        by_source[candidate["_source_id"]].append(candidate)
    selected: list[dict[str, Any]] = []
    depth = 0
    while len(selected) < quota:
        layer = [
            values[depth]
            for source, values in sorted(by_source.items())
            if depth < len(values)
        ]
        if not layer:
            break
        layer.sort(key=_order)
        selected.extend(layer[: quota - len(selected)])
        depth += 1
    if len(selected) != quota:
        raise RuntimeError(
            f"only {len(selected)} flow-recoverable windows remain; quota is {quota}"
        )
    return selected


def _public(record: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in record.items() if not key.startswith("_")}
    result["target_video"] = str(record["_target_path"])
    result["mask_video"] = str(record["_mask_path"])
    metrics: FlowRecoverabilityMetrics = record["_flow_recoverability"]
    result["flow_recoverability"] = {
        key: (int(value) if key == "good_neighbors" else round(float(value), 8))
        for key, value in asdict(metrics).items()
    }
    return result


def _jsonl(records: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )


def _json(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            destination.write(text)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "min": float(min(values)),
        "mean": float(np.mean(values)),
        "max": float(max(values)),
    }


def build_flow_recoverability_manifest(
    *,
    source_manifest: Path,
    output: Path,
    report: Path,
    quota: int,
    analysis_size: int = ANALYSIS_CROP_SIZE,
    thresholds: FlowThresholds = FlowThresholds(),
    decoder: Callable[..., Iterable[np.ndarray]] | None = None,
    progress_every: int = 25,
) -> dict[str, Any]:
    entries = load_native_512_entries(source_manifest)
    groups: dict[tuple[Path, Path], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[(entry["_target_path"], entry["_mask_path"])].append(entry)
    accepted: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for position, (_, records) in enumerate(
        sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))),
        start=1,
    ):
        try:
            group_accepted, group_rejected = evaluate_group(
                records,
                analysis_size=analysis_size,
                thresholds=thresholds,
                decoder=decoder,
            )
        except Exception:
            group_accepted = []
            group_rejected = Counter({"decode_error": len(records)})
        accepted.extend(group_accepted)
        rejected.update(group_rejected)
        if progress_every and (position == 1 or position % progress_every == 0):
            print(
                f"clips {position}/{len(groups)} | passed {len(accepted)} | "
                f"rejected {sum(rejected.values())}",
                flush=True,
            )

    selected = source_balanced_select(accepted, quota)
    selected.sort(
        key=lambda value: (
            value["_source_id"],
            str(value["_target_path"]),
            int(value["start_frame"]),
            value["_name"],
        )
    )
    public = [_public(record) for record in selected]
    manifest_text = _jsonl(public)
    metric_fields = tuple(asdict(selected[0]["_flow_recoverability"]))
    metric_summary = {
        name: _summary(
            [
                float(getattr(record["_flow_recoverability"], name))
                for record in selected
            ]
        )
        for name in metric_fields
    }
    per_source = Counter(record["_source_id"] for record in selected)
    report_value: dict[str, Any] = {
        "format": "basicvsrpp-flow-recoverability-v1",
        "source_manifest": str(source_manifest.resolve()),
        "output": str(output.resolve()),
        "quota": quota,
        "native_window_size": NATIVE_SIZE,
        "native_analysis_crop_size": ANALYSIS_CROP_SIZE,
        "analysis_size": analysis_size,
        "analysis_downscale_only": analysis_size != ANALYSIS_CROP_SIZE,
        "training_pixels_resized": 0,
        "flow": "OpenCV Farneback forward/backward",
        "thresholds": asdict(thresholds),
        "counts": {
            "input": len(entries),
            "decoded_clip_pairs": len(groups),
            "passed_filters": len(accepted),
            "selected": len(selected),
        },
        "rejections": dict(sorted(rejected.items())),
        "selected_per_source": dict(sorted(per_source.items())),
        "selected_metrics": metric_summary,
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
    }
    # The manifest is the commit marker.  A reader can verify that its digest
    # matches the sidecar if a process is interrupted between replacements.
    atomic_write_text(_json(report_value), report)
    atomic_write_text(manifest_text, output)
    return report_value


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    thresholds = FlowThresholds(
        maximum_combing_score=args.maximum_combing_score,
        maximum_aligned_residual=args.maximum_aligned_residual,
        minimum_recoverable_hf=args.minimum_recoverable_hf,
        minimum_good_neighbors=args.minimum_good_neighbors,
    )
    report = args.report or Path(str(args.output) + ".flow-metrics.json")
    result = build_flow_recoverability_manifest(
        source_manifest=args.source_manifest,
        output=args.output,
        report=report,
        quota=args.quota,
        analysis_size=args.analysis_size,
        thresholds=thresholds,
        progress_every=args.progress_every,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
