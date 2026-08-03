#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Select a deterministic, recoverable Native-HF 512 curriculum.

The source manifest is deliberately left untouched.  This command joins every
window to the Gemma decision for its *resolved target video*, applies strict
semantic, motion, phase, mask-context and native-detail gates, then writes an
exact, source-balanced quota.  Each video is decoded in one sequential pass:
only required grayscale mask frames are retained, while required RGB centre
frames are evaluated and released immediately.  No frame is resized or
rewritten.
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
from typing import Any, Callable, Iterable, Sequence

import av
import cv2
import numpy as np

from lada.models.mioh_restorer.model_v5 import NUM_INPUT_FRAMES, V5_BUCKETS


NATIVE_SIZE = 512
CENTER = NUM_INPUT_FRAMES // 2
DEFAULT_QUOTAS = {"train": 512, "validation": 112}


@dataclass(frozen=True)
class SparseDecodedFrames:
    frames: dict[int, np.ndarray]
    decoded_count: int


@dataclass(frozen=True)
class RecoverabilityThresholds:
    gemma_confidence: float = 0.90
    gemma_sharpness: float = 0.80
    gemma_clarity: float = 0.80
    gemma_occlusion: float = 0.10
    phase_diversity: float = 0.80
    motion_mean: float = 12.0
    motion_max: float = 32.0
    mask_reliability: float = 0.99
    context_valid_fraction: float = 0.98
    eroded_mask_fraction: float = 0.02
    luma_hf_rms: float = 0.0025


@dataclass(frozen=True)
class RecoverabilityMetrics:
    gemma_confidence: float
    gemma_sharpness: float
    gemma_clarity: float
    gemma_occlusion: float
    phase_diversity: float
    motion_mean: float
    motion_max: float
    mask_reliability: float
    context_valid_fraction: float
    eroded_mask_fraction: float
    luma_hf_rms: float
    score: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--classifications", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=tuple(DEFAULT_QUOTAS), required=True)
    parser.add_argument(
        "--quota",
        type=int,
        help="exact output size (default: train=512, validation=112)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="metrics sidecar (default: <output>.metrics.json)",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args(argv)
    if args.quota is not None and args.quota <= 0:
        parser.error("--quota must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every cannot be negative")
    return args


def _canonical_path(value: object, root: Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL line {line_number} in {path}: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL line {line_number} in {path} is not an object")
            records.append(record)
    if not records:
        raise ValueError(f"JSONL file is empty: {path}")
    return records


def visit_selected_decoded_frames(
    path: Path,
    *,
    selected_indices: Iterable[int],
    through_index: int,
    pixel_format: str,
    visitor: Callable[[int, np.ndarray], None],
    decoder: Callable[..., Iterable[np.ndarray]] | None = None,
) -> int:
    """Sequentially decode through ``through_index`` without retaining RGB.

    ``visitor`` is invoked only for requested indices.  The production path
    converts one PyAV frame at a time, so the caller controls the lifetime of
    every retained array.  The injectable decoder keeps deterministic tests
    independent from video codecs while preserving the same one-pass contract.

    The returned count is the number of sequential frames actually decoded.
    It is used to preserve the old full-clip decoder's ``short_decode`` gate.
    """

    wanted = frozenset(int(index) for index in selected_indices)
    if any(index < 0 for index in wanted):
        raise ValueError("selected frame indices must be non-negative")
    if through_index < -1:
        raise ValueError("through_index must be at least -1")
    if any(index > through_index for index in wanted):
        raise ValueError("selected frame index lies beyond through_index")
    if through_index < 0:
        return 0

    decoded_count = 0

    def consume(frames: Iterable[np.ndarray]) -> None:
        nonlocal decoded_count
        for index, frame in enumerate(frames):
            decoded_count = index + 1
            if index in wanted:
                visitor(index, frame)
            if index >= through_index:
                break

    if decoder is not None:
        consume(decoder(path, pixel_format=pixel_format))
    else:
        with av.open(str(path), mode="r") as container:
            for index, frame in enumerate(container.decode(video=0)):
                decoded_count = index + 1
                # Decoding remains sequential, but the expensive full-frame
                # ndarray conversion happens only for frames the caller keeps
                # or evaluates.
                if index in wanted:
                    visitor(index, frame.to_ndarray(format=pixel_format))
                if index >= through_index:
                    break
    return decoded_count


def decode_sparse_frames(
    path: Path,
    *,
    selected_indices: Iterable[int],
    through_index: int,
    pixel_format: str,
    decoder: Callable[..., Iterable[np.ndarray]] | None = None,
) -> SparseDecodedFrames:
    """Retain exactly the requested frames from one sequential decode pass."""

    retained: dict[int, np.ndarray] = {}

    def retain(index: int, frame: np.ndarray) -> None:
        retained[index] = frame

    decoded_count = visit_selected_decoded_frames(
        path,
        selected_indices=selected_indices,
        through_index=through_index,
        pixel_format=pixel_format,
        visitor=retain,
        decoder=decoder,
    )
    return SparseDecodedFrames(frames=retained, decoded_count=decoded_count)


def load_source_entries(path: Path) -> list[dict[str, Any]]:
    entries = load_jsonl(path)
    root = path.parent
    for line_number, entry in enumerate(entries, start=1):
        try:
            target = _canonical_path(entry["target_video"], root)
            mask = _canonical_path(entry["mask_video"], root)
            origins = tuple(
                (int(pair[0]), int(pair[1])) for pair in entry["origins"]
            )
            reliability = tuple(float(value) for value in entry["mask_reliability"])
            if len(origins) != NUM_INPUT_FRAMES or len(reliability) != NUM_INPUT_FRAMES:
                raise ValueError("expected nine origins and reliability values")
            if any(x % 2 or y % 2 for x, y in origins):
                raise ValueError("crop origins must use even source pixels")
            if any(
                not math.isfinite(value) or not 0.0 <= value <= 1.0
                for value in reliability
            ):
                raise ValueError("mask reliability must be finite and in [0, 1]")
            if int(entry["start_frame"]) < 0:
                raise ValueError("start_frame must be non-negative")
            bucket = int(entry["bucket"])
            if bucket not in V5_BUCKETS or bucket > NATIVE_SIZE:
                raise ValueError("unsupported native bucket")
            if (NATIVE_SIZE - bucket) % 2:
                raise ValueError("bucket cannot be centred losslessly in 512 pixels")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid source manifest line {line_number}: {error}") from error
        entry["_target_path"] = target
        entry["_mask_path"] = mask
    return entries


def load_gemma_decisions(path: Path) -> dict[Path, dict[str, Any]]:
    decisions: dict[Path, dict[str, Any]] = {}
    root = path.parent
    for line_number, record in enumerate(load_jsonl(path), start=1):
        if record.get("status") != "ok":
            continue
        decision = record.get("decision")
        video = record.get("video")
        if not isinstance(decision, dict) or not video:
            raise ValueError(f"invalid Gemma classification line {line_number}")
        key = _canonical_path(video, root)
        previous = decisions.get(key)
        if previous is not None and previous != decision:
            raise ValueError(f"conflicting Gemma decisions for {key}")
        decisions[key] = decision
    if not decisions:
        raise ValueError(f"no successful Gemma decisions in {path}")
    return decisions


def _score(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def gemma_rejection(
    decision: dict[str, Any] | None,
    thresholds: RecoverabilityThresholds,
) -> str | None:
    if decision is None:
        return "gemma_missing"
    if decision.get("usable") is not True:
        return "gemma_unusable"
    checks = (
        ("gemma_confidence", _score(decision.get("confidence")), thresholds.gemma_confidence, True),
        ("gemma_sharpness", _score(decision.get("sharpness")), thresholds.gemma_sharpness, True),
        ("gemma_clarity", _score(decision.get("clarity")), thresholds.gemma_clarity, True),
        ("gemma_occlusion", _score(decision.get("occlusion")), thresholds.gemma_occlusion, False),
    )
    for reason, value, limit, minimum in checks:
        if not math.isfinite(value) or (value < limit if minimum else value > limit):
            return reason
    return None


def recentered_origins(entry: dict[str, Any]) -> tuple[tuple[int, int], ...]:
    adjustment = (NATIVE_SIZE - int(entry["bucket"])) // 2
    return tuple(
        (int(pair[0]) - adjustment, int(pair[1]) - adjustment)
        for pair in entry["origins"]
    )


def phase5_diversity(origins: Sequence[tuple[int, int]]) -> float:
    if len(origins) != NUM_INPUT_FRAMES:
        raise ValueError("phase diversity requires nine origins")
    diversity = []
    for block_size in range(6, 13):
        phases = {
            (x % block_size, y % block_size)
            for x, y in origins[2:7]
        }
        diversity.append(len(phases) / 5.0)
    return float(np.mean(diversity))


def origin_motion(origins: Sequence[tuple[int, int]]) -> tuple[float, float]:
    if len(origins) != NUM_INPUT_FRAMES:
        raise ValueError("motion requires nine origins")
    distances = [
        math.hypot(right[0] - left[0], right[1] - left[1])
        for left, right in zip(origins, origins[1:])
    ]
    return float(np.mean(distances)), float(max(distances))


def static_rejection(
    entry: dict[str, Any],
    decision: dict[str, Any] | None,
    thresholds: RecoverabilityThresholds,
) -> tuple[str | None, dict[str, float]]:
    reason = gemma_rejection(decision, thresholds)
    if reason is not None:
        return reason, {}
    origins = recentered_origins(entry)
    phase = phase5_diversity(origins)
    if phase + 1e-12 < thresholds.phase_diversity:
        return "phase_diversity", {"phase_diversity": phase}
    motion_mean, motion_max = origin_motion(origins)
    if motion_mean > thresholds.motion_mean:
        return "motion_mean", {"motion_mean": motion_mean, "motion_max": motion_max}
    if motion_max > thresholds.motion_max:
        return "motion_max", {"motion_mean": motion_mean, "motion_max": motion_max}
    reliability_values = tuple(float(value) for value in entry["mask_reliability"])
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in reliability_values
    ):
        return "mask_reliability", {"mask_reliability": float("nan")}
    reliability = min(reliability_values)
    if reliability < thresholds.mask_reliability:
        return "mask_reliability", {"mask_reliability": reliability}
    assert decision is not None
    return None, {
        "gemma_confidence": _score(decision["confidence"]),
        "gemma_sharpness": _score(decision["sharpness"]),
        "gemma_clarity": _score(decision["clarity"]),
        "gemma_occlusion": _score(decision["occlusion"]),
        "phase_diversity": phase,
        "motion_mean": motion_mean,
        "motion_max": motion_max,
        "mask_reliability": reliability,
    }


def _crop_mask(frame: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
    if frame.ndim == 3:
        frame = frame[..., 0]
    height, width = frame.shape
    result = np.zeros((NATIVE_SIZE, NATIVE_SIZE), dtype=np.float32)
    x, y = origin
    sx0, sy0 = max(0, x), max(0, y)
    sx1, sy1 = min(width, x + NATIVE_SIZE), min(height, y + NATIVE_SIZE)
    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0 - x, sy0 - y
        result[dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] = frame[
            sy0:sy1, sx0:sx1
        ].astype(np.float32)
    if result.size and float(result.max()) > 1.0:
        result /= 255.0
    return np.clip(result, 0.0, 1.0)


def stabilized_masks(
    masks: Sequence[np.ndarray], origins: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, ...]:
    """NumPy equivalent of the production temporal mask stabilizer."""

    if len(masks) != NUM_INPUT_FRAMES or len(origins) != NUM_INPUT_FRAMES:
        raise ValueError("mask stabilization requires nine frames")
    spatial_radius = max(1, int(round(NATIVE_SIZE / 128.0)))
    kernel = np.ones((spatial_radius * 2 + 1,) * 2, dtype=np.uint8)
    guarded = [
        cv2.dilate(_crop_mask(frame, origin), kernel)
        for frame, origin in zip(masks, origins, strict=True)
    ]
    stable_masks: list[np.ndarray] = []
    for output_index in range(NUM_INPUT_FRAMES):
        stable = guarded[output_index].copy()
        first = max(0, output_index - 2)
        last = min(NUM_INPUT_FRAMES, output_index + 3)
        for neighbour in range(first, last):
            distance = abs(neighbour - output_index)
            if distance:
                stable = np.maximum(stable, guarded[neighbour] * (0.82**distance))
        feathered = cv2.blur(
            stable,
            (spatial_radius * 2 + 1, spatial_radius * 2 + 1),
            borderType=cv2.BORDER_CONSTANT,
        )
        stable_masks.append(np.maximum(stable, feathered).clip(0.0, 1.0))
    return tuple(stable_masks)


def stabilized_center_mask(
    masks: Sequence[np.ndarray], origins: Sequence[tuple[int, int]]
) -> np.ndarray:
    """Compatibility helper for the centre-frame eroded-detail gate."""

    return stabilized_masks(masks, origins)[CENTER]


def central_context_valid_fraction(
    stable_masks: Sequence[np.ndarray],
    target_frames: Sequence[np.ndarray],
    origins: Sequence[tuple[int, int]],
) -> float:
    """Return the worst valid 32px mask context over output frames 2..6."""

    if not (
        len(stable_masks) == len(target_frames) == len(origins) == NUM_INPUT_FRAMES
    ):
        raise ValueError("context validity requires nine aligned frames")
    context_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))
    fractions: list[float] = []
    for index in range(2, 7):
        binary = (stable_masks[index] >= 0.5).astype(np.uint8)
        context = cv2.dilate(binary, context_kernel)
        valid_scene = _valid_scene_mask(
            target_frames[index].shape[:2], origins[index]
        )
        context_pixels = int(context.sum())
        fractions.append(
            float((context * valid_scene).sum() / max(context_pixels, 1))
        )
    return min(fractions)


def _valid_scene_mask(shape: tuple[int, int], origin: tuple[int, int]) -> np.ndarray:
    height, width = shape
    x, y = origin
    valid = np.zeros((NATIVE_SIZE, NATIVE_SIZE), dtype=np.uint8)
    sx0, sy0 = max(0, x), max(0, y)
    sx1, sy1 = min(width, x + NATIVE_SIZE), min(height, y + NATIVE_SIZE)
    if sx1 > sx0 and sy1 > sy0:
        dx0, dy0 = sx0 - x, sy0 - y
        valid[dy0 : dy0 + sy1 - sy0, dx0 : dx0 + sx1 - sx0] = 1
    return valid


def _crop_replicate(frame: np.ndarray, origin: tuple[int, int]) -> np.ndarray:
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("empty target frame")
    x, y = origin
    horizontal = np.clip(np.arange(x, x + NATIVE_SIZE), 0, width - 1)
    vertical = np.clip(np.arange(y, y + NATIVE_SIZE), 0, height - 1)
    return np.ascontiguousarray(frame[vertical[:, None], horizontal[None, :]])


def luma_hf_residual(target: np.ndarray) -> np.ndarray:
    rgb = target.astype(np.float32) / 255.0
    luma = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    kernel = np.asarray((1.0, 4.0, 6.0, 4.0, 1.0), dtype=np.float32)
    kernel = np.outer(kernel, kernel)
    kernel /= kernel.sum()
    blurred = cv2.filter2D(luma, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return luma - blurred


def deterministic_score(values: dict[str, float]) -> float:
    """Rank only samples that are strong in every recoverability factor.

    The hard gates above already enforce semantic confidence, occlusion and
    motion limits.  A product is intentional here: unlike a weighted sum, a
    very sharp background cannot compensate for a tiny ROI, poor phase
    diversity, invalid crop context, or low clean-target detail.
    """

    return float(
        values["luma_hf_rms"]
        * math.sqrt(values["eroded_mask_fraction"])
        * values["phase_diversity"]
        * values["context_valid_fraction"]
        * math.sqrt(values["gemma_sharpness"] * values["gemma_clarity"])
    )


def evaluate_decoded_candidate(
    entry: dict[str, Any],
    decision: dict[str, Any],
    target_frames: Sequence[np.ndarray],
    mask_frames: Sequence[np.ndarray],
    thresholds: RecoverabilityThresholds,
) -> tuple[str | None, RecoverabilityMetrics | None]:
    reason, values = static_rejection(entry, decision, thresholds)
    if reason is not None:
        return reason, None
    start = int(entry["start_frame"])
    stop = start + NUM_INPUT_FRAMES
    if stop > len(target_frames) or stop > len(mask_frames):
        return "short_decode", None
    window_targets = target_frames[start:stop]
    window_masks = mask_frames[start:stop]
    origins = recentered_origins(entry)
    stable_masks = stabilized_masks(window_masks, origins)
    center_mask = stable_masks[CENTER]
    binary = (center_mask >= 0.5).astype(np.uint8)
    if not np.any(binary):
        return "empty_mask", None

    context_valid = central_context_valid_fraction(
        stable_masks, window_targets, origins
    )
    if context_valid + 1e-12 < thresholds.context_valid_fraction:
        return "context_valid_fraction", None

    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(binary, erosion_kernel)
    eroded_fraction = float(eroded.mean())
    if eroded_fraction + 1e-12 < thresholds.eroded_mask_fraction:
        return "eroded_mask_fraction", None

    target_crop = _crop_replicate(window_targets[CENTER], origins[CENTER])
    residual = luma_hf_residual(target_crop)
    hf_rms = float(np.sqrt(np.mean(np.square(residual[eroded > 0]))))
    if hf_rms + 1e-12 < thresholds.luma_hf_rms:
        return "luma_hf_rms", None

    values.update(
        context_valid_fraction=context_valid,
        eroded_mask_fraction=eroded_fraction,
        luma_hf_rms=hf_rms,
    )
    values["score"] = deterministic_score(values)
    return None, RecoverabilityMetrics(**values)


def evaluate_mask_candidate(
    entry: dict[str, Any],
    decision: dict[str, Any],
    window_masks: Sequence[np.ndarray],
    thresholds: RecoverabilityThresholds,
) -> tuple[str | None, dict[str, float] | None]:
    """Run every decoded gate that does not require RGB target pixels.

    Video streams have a fixed coded shape.  At this stage the mask frame
    shapes stand in for target shapes in the context-validity calculation.
    ``evaluate_target_candidate`` validates that contract before accepting the
    RGB-dependent result.
    """

    reason, values = static_rejection(entry, decision, thresholds)
    if reason is not None:
        return reason, None
    if len(window_masks) != NUM_INPUT_FRAMES:
        return "short_decode", None
    origins = recentered_origins(entry)
    stable_masks = stabilized_masks(window_masks, origins)
    center_mask = stable_masks[CENTER]
    binary = (center_mask >= 0.5).astype(np.uint8)
    if not np.any(binary):
        return "empty_mask", None

    # central_context_valid_fraction reads only frame.shape[:2].  Supplying
    # mask frames avoids retaining five full-resolution RGB frames.
    context_valid = central_context_valid_fraction(
        stable_masks, window_masks, origins
    )
    if context_valid + 1e-12 < thresholds.context_valid_fraction:
        return "context_valid_fraction", None

    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(binary, erosion_kernel)
    eroded_fraction = float(eroded.mean())
    if eroded_fraction + 1e-12 < thresholds.eroded_mask_fraction:
        return "eroded_mask_fraction", None

    values.update(
        context_valid_fraction=context_valid,
        eroded_mask_fraction=eroded_fraction,
    )
    return None, values


def evaluate_target_candidate(
    entry: dict[str, Any],
    target_center: np.ndarray,
    window_masks: Sequence[np.ndarray],
    mask_values: dict[str, float],
    thresholds: RecoverabilityThresholds,
) -> tuple[str | None, RecoverabilityMetrics | None]:
    """Finish one candidate and release its sole decoded RGB frame promptly."""

    if len(window_masks) != NUM_INPUT_FRAMES:
        return "short_decode", None
    target_shape = target_center.shape[:2]
    if target_center.ndim != 3 or target_center.shape[2] < 3:
        return "invalid_decoded_candidate", None
    if any(frame.shape[:2] != target_shape for frame in window_masks[2:7]):
        return "target_mask_shape_mismatch", None

    origins = recentered_origins(entry)
    stable_masks = stabilized_masks(window_masks, origins)
    binary = (stable_masks[CENTER] >= 0.5).astype(np.uint8)
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    eroded = cv2.erode(binary, erosion_kernel)
    target_crop = _crop_replicate(target_center, origins[CENTER])
    residual = luma_hf_residual(target_crop)
    hf_rms = float(np.sqrt(np.mean(np.square(residual[eroded > 0]))))
    if hf_rms + 1e-12 < thresholds.luma_hf_rms:
        return "luma_hf_rms", None

    values = dict(mask_values)
    values["luma_hf_rms"] = hf_rms
    values["score"] = deterministic_score(values)
    return None, RecoverabilityMetrics(**values)


def _candidate_order(item: dict[str, Any]) -> tuple[float, str, int, str]:
    metrics = item["_recoverability"]
    return (
        -float(metrics.score),
        str(item["_target_path"]),
        int(item["start_frame"]),
        str(item["name"]),
    )


def deduplicate_target_start(candidates: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    best: dict[tuple[str, int], dict[str, Any]] = {}
    for candidate in sorted(candidates, key=_candidate_order):
        key = (str(candidate["_target_path"]), int(candidate["start_frame"]))
        best.setdefault(key, candidate)
    return list(best.values())


def clip_nms(candidates: Iterable[dict[str, Any]], radius: int = 64) -> list[dict[str, Any]]:
    if radius <= 0:
        raise ValueError("NMS radius must be positive")
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_target[str(candidate["_target_path"])].append(candidate)
    selected: list[dict[str, Any]] = []
    for target in sorted(by_target):
        starts: list[int] = []
        for candidate in sorted(by_target[target], key=_candidate_order):
            start = int(candidate["start_frame"])
            if any(abs(start - previous) < radius for previous in starts):
                continue
            selected.append(candidate)
            starts.append(start)
    return selected


def source_balanced_select(
    candidates: Iterable[dict[str, Any]], quota: int
) -> list[dict[str, Any]]:
    if quota <= 0:
        raise ValueError("quota must be positive")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        by_source[str(candidate["source_video_id"])].append(candidate)
    for values in by_source.values():
        values.sort(key=_candidate_order)
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
        layer.sort(key=_candidate_order)
        selected.extend(layer[: quota - len(selected)])
        depth += 1
    if len(selected) != quota:
        raise RuntimeError(
            f"recoverability curriculum has {len(selected)} candidates; exact quota is {quota}"
        )
    return selected


def _public_entry(entry: dict[str, Any]) -> dict[str, Any]:
    public = {
        key: value
        for key, value in entry.items()
        if not key.startswith("_")
    }
    public["target_video"] = str(entry["_target_path"])
    public["mask_video"] = str(entry["_mask_path"])
    metrics: RecoverabilityMetrics = entry["_recoverability"]
    public["recoverability"] = {
        key: round(value, 8) for key, value in asdict(metrics).items()
    }
    return public


def _jsonl_text(records: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )


def _json_text(value: dict[str, Any]) -> str:
    return json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"


def atomic_write_text(value: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            destination.write(value)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_jsonl(records: Iterable[dict[str, Any]], output: Path) -> None:
    # Materialize before opening the temporary file so a failing generator
    # cannot disturb the previous public artifact.
    atomic_write_text(_jsonl_text(records), output)


def atomic_write_json(value: dict[str, Any], output: Path) -> None:
    atomic_write_text(_json_text(value), output)


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "min": float(min(values)),
        "mean": float(np.mean(values)),
        "max": float(max(values)),
    }


def build_recoverability_manifest(
    *,
    source_manifest: Path,
    classifications: Path,
    output: Path,
    split: str,
    quota: int,
    report: Path,
    thresholds: RecoverabilityThresholds = RecoverabilityThresholds(),
    decoder: Callable[..., Iterable[np.ndarray]] | None = None,
    progress_every: int = 25,
) -> dict[str, Any]:
    if split not in DEFAULT_QUOTAS:
        raise ValueError(f"invalid split: {split}")
    entries = load_source_entries(source_manifest)
    decisions = load_gemma_decisions(classifications)
    rejected: Counter[str] = Counter()
    groups: dict[tuple[Path, Path], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for entry in entries:
        decision = decisions.get(entry["_target_path"])
        reason, _ = static_rejection(entry, decision, thresholds)
        if reason is not None:
            rejected[reason] += 1
            continue
        assert decision is not None
        groups[(entry["_target_path"], entry["_mask_path"])].append((entry, decision))

    accepted: list[dict[str, Any]] = []
    for position, ((target_path, mask_path), group) in enumerate(
        sorted(groups.items(), key=lambda item: (str(item[0][0]), str(item[0][1]))),
        start=1,
    ):
        ordered_group = sorted(
            group,
            key=lambda item: (int(item[0]["start_frame"]), str(item[0]["name"])),
        )
        required_mask_indices = {
            frame_index
            for entry, _ in ordered_group
            for frame_index in range(
                int(entry["start_frame"]),
                int(entry["start_frame"]) + NUM_INPUT_FRAMES,
            )
        }
        through_index = max(required_mask_indices)
        try:
            sparse_masks = decode_sparse_frames(
                mask_path,
                selected_indices=required_mask_indices,
                through_index=through_index,
                pixel_format="gray",
                decoder=decoder,
            )
        except Exception:
            rejected["decode_error"] += len(group)
            continue

        # Each state contains only small scalar metadata.  The mask arrays are
        # shared through sparse_masks and RGB arrays never enter this list.
        states: list[
            tuple[dict[str, Any], str | None, dict[str, float] | None]
        ] = []
        target_centres: dict[int, list[int]] = defaultdict(list)
        for entry, decision in ordered_group:
            start = int(entry["start_frame"])
            stop = start + NUM_INPUT_FRAMES
            if stop > sparse_masks.decoded_count or any(
                index not in sparse_masks.frames for index in range(start, stop)
            ):
                reason, values = "short_decode", None
            else:
                window_masks = [
                    sparse_masks.frames[index] for index in range(start, stop)
                ]
                try:
                    reason, values = evaluate_mask_candidate(
                        entry, decision, window_masks, thresholds
                    )
                except (IndexError, TypeError, ValueError, cv2.error):
                    reason, values = "invalid_decoded_candidate", None
            state_index = len(states)
            states.append((entry, reason, values))
            if reason is None:
                target_centres[start + CENTER].append(state_index)

        target_results: dict[
            int, tuple[str | None, RecoverabilityMetrics | None]
        ] = {}

        def evaluate_target_frame(frame_index: int, frame: np.ndarray) -> None:
            for state_index in target_centres.get(frame_index, ()):
                entry, mask_reason, mask_values = states[state_index]
                assert mask_reason is None and mask_values is not None
                start = int(entry["start_frame"])
                window_masks = [
                    sparse_masks.frames[index]
                    for index in range(start, start + NUM_INPUT_FRAMES)
                ]
                try:
                    target_results[state_index] = evaluate_target_candidate(
                        entry, frame, window_masks, mask_values, thresholds
                    )
                except (IndexError, TypeError, ValueError, cv2.error):
                    target_results[state_index] = (
                        "invalid_decoded_candidate",
                        None,
                    )

        try:
            target_decoded_count = visit_selected_decoded_frames(
                target_path,
                selected_indices=target_centres,
                through_index=through_index,
                pixel_format="rgb24",
                visitor=evaluate_target_frame,
                decoder=decoder,
            )
        except Exception:
            rejected["decode_error"] += len(group)
            continue

        for state_index, (entry, mask_reason, _) in enumerate(states):
            start = int(entry["start_frame"])
            stop = start + NUM_INPUT_FRAMES
            # Match evaluate_decoded_candidate: short target or mask clips take
            # precedence over all decoded-content rejection reasons.
            if (
                stop > target_decoded_count
                or stop > sparse_masks.decoded_count
            ):
                reason, metrics = "short_decode", None
            elif mask_reason is not None:
                reason, metrics = mask_reason, None
            else:
                reason, metrics = target_results.get(
                    state_index, ("invalid_decoded_candidate", None)
                )
            try:
                if reason is None and metrics is None:
                    raise ValueError("accepted sparse candidate has no metrics")
            except (TypeError, ValueError):
                reason, metrics = "invalid_decoded_candidate", None
            if reason is not None:
                rejected[reason] += 1
                continue
            assert metrics is not None
            entry["_recoverability"] = metrics
            accepted.append(entry)
        if progress_every and (position == 1 or position % progress_every == 0):
            print(
                f"clips {position}/{len(groups)} | passed {len(accepted)} | "
                f"rejected {sum(rejected.values())}",
                flush=True,
            )

    deduplicated = deduplicate_target_start(accepted)
    nms = clip_nms(deduplicated, radius=64)
    selected = source_balanced_select(nms, quota)
    selected.sort(
        key=lambda item: (
            str(item["source_video_id"]),
            str(item["_target_path"]),
            int(item["start_frame"]),
            str(item["name"]),
        )
    )
    public = [_public_entry(entry) for entry in selected]
    metric_names = tuple(asdict(selected[0]["_recoverability"]))
    metrics_report = {
        name: _summary(
            [float(getattr(entry["_recoverability"], name)) for entry in selected]
        )
        for name in metric_names
    }
    per_source = Counter(str(entry["source_video_id"]) for entry in selected)
    manifest_text = _jsonl_text(public)
    report_value: dict[str, Any] = {
        "format": "mioh-native-hf-recoverability-v1",
        "split": split,
        "quota": quota,
        "source_manifest": str(source_manifest.resolve()),
        "classifications": str(classifications.resolve()),
        "output": str(output.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "thresholds": asdict(thresholds),
        "counts": {
            "input": len(entries),
            "decoded_clip_pairs": len(groups),
            "passed_filters": len(accepted),
            "after_target_start_dedup": len(deduplicated),
            "after_64_frame_nms": len(nms),
            "selected": len(selected),
        },
        "rejections": dict(sorted(rejected.items())),
        "selected_per_source": dict(sorted(per_source.items())),
        "selected_metrics": metrics_report,
    }
    # Materialize both complete payloads before replacing either public file.
    # The report is published first and the training manifest is the commit
    # marker.  Its digest makes an interrupted two-file update detectable.
    report_text = _json_text(report_value)
    atomic_write_text(report_text, report)
    atomic_write_text(manifest_text, output)
    return report_value


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    quota = args.quota or DEFAULT_QUOTAS[args.split]
    report = args.report or Path(str(args.output) + ".metrics.json")
    result = build_recoverability_manifest(
        source_manifest=args.source_manifest,
        classifications=args.classifications,
        output=args.output,
        split=args.split,
        quota=quota,
        report=report,
        progress_every=args.progress_every,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
