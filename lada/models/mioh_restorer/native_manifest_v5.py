# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Balanced native-resolution manifest construction for MiohRestorer V5.

The builder deliberately decodes masks only.  Clean RGB remains in its native
source video and is cropped without resampling by :mod:`native_dataset_v5` at
training time.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import heapq
from pathlib import Path
from typing import Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np

from lada.datasetcreation.restoration_dataset_metadata import RestorationDatasetMetadataV2

from .model_v5 import NUM_INPUT_FRAMES, QUALITY_OUTPUT_INDICES, V5_BUCKETS
from .native_dataset_v5 import decode_native_frames
from .runner_v5 import native_tile_offsets, round_to_even, select_v5_bucket, smooth_even_centers


MOTION_BUCKETS = ("static", "low", "medium", "high")
OCCUPANCY_BUCKETS = ("boundary", "preferred", "dense")
DEFAULT_BUCKET_RATIOS = {192: 0.10, 256: 0.30, 384: 0.60}
DEFAULT_OCCUPANCY_WEIGHTS = {"boundary": 0.12, "preferred": 0.68, "dense": 0.20}


@dataclass(frozen=True)
class BalancedNativeCandidate:
    entry: dict[str, object]
    base_window_id: str
    stable_score: int

    @property
    def bucket(self) -> int:
        return int(self.entry["bucket"])

    @property
    def motion_bucket(self) -> str:
        return str(self.entry["motion_bucket"])

    @property
    def occupancy_bucket(self) -> str:
        return str(self.entry["occupancy_bucket"])


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


def motion_bucket_for_centres(centres: Sequence[tuple[int, int]]) -> tuple[str, float]:
    if len(centres) < 2:
        return "static", 0.0
    distances = [
        float(np.hypot(current[0] - previous[0], current[1] - previous[1]))
        for previous, current in zip(centres, centres[1:])
    ]
    motion = float(np.mean(distances))
    if motion < 0.75:
        return "static", motion
    if motion < 2.5:
        return "low", motion
    if motion < 6.0:
        return "medium", motion
    return "high", motion


def occupancy_bucket_for_fraction(fraction: float) -> str | None:
    if 0.05 <= fraction < 0.20:
        return "boundary"
    if 0.20 <= fraction < 0.70:
        return "preferred"
    if 0.70 <= fraction <= 0.95:
        return "dense"
    return None


def mask_fraction_in_crop(frame: np.ndarray, origin: tuple[int, int], bucket: int) -> float:
    if frame.ndim == 3:
        frame = frame[..., 0]
    origin_x, origin_y = origin
    height, width = frame.shape[:2]
    x0, y0 = max(origin_x, 0), max(origin_y, 0)
    x1, y1 = min(origin_x + bucket, width), min(origin_y + bucket, height)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return float(np.count_nonzero(frame[y0:y1, x0:x1] > 0)) / float(bucket * bucket)


def stable_candidate_score(seed: int, identity: str) -> int:
    digest = hashlib.sha256(f"{seed}:{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _eligible_buckets(selected_bucket: int, *, minimum_bucket: int, maximum_bucket: int) -> tuple[int, ...]:
    minimum = max(selected_bucket, minimum_bucket)
    values = tuple(bucket for bucket in V5_BUCKETS if minimum <= bucket <= maximum_bucket)
    return values or (maximum_bucket,)


def candidates_for_metadata(
    metadata_path: Path,
    *,
    stride: int,
    context_fraction: float,
    tile_overlap: int,
    minimum_bucket: int = 192,
    maximum_bucket: int = 384,
    seed: int = 20260812,
    source_video_id: str | None = None,
    rejection_counts: MutableMapping[str, int] | None = None,
) -> Iterator[BalancedNativeCandidate]:
    """Yield lossless multi-bucket candidates for one short source clip."""

    if minimum_bucket not in V5_BUCKETS or maximum_bucket not in V5_BUCKETS:
        raise ValueError("minimum/maximum bucket must be a V5 bucket")
    if minimum_bucket > maximum_bucket:
        raise ValueError("minimum bucket must not exceed maximum bucket")
    if stride <= 0:
        raise ValueError("stride must be positive")
    rejection_counts = rejection_counts if rejection_counts is not None else Counter()
    metadata = RestorationDatasetMetadataV2.from_json_file(metadata_path)
    source_id = source_video_id or metadata.name
    target = (metadata_path.parent / metadata.relative_nsfw_video_path).resolve()
    mask_path = (metadata_path.parent / metadata.relative_mask_video_path).resolve()
    masks = decode_native_frames(mask_path, pixel_format="gray")
    frame_count = min(metadata.frames_count, len(masks))
    if frame_count < NUM_INPUT_FRAMES:
        rejection_counts["short_clip"] += 1
        return
    boxes = [mask_box(frame) for frame in masks[:frame_count]]
    centres = smooth_even_centers([value[0] if value else None for value in boxes])
    clip_identity = metadata_path.stem
    for start in range(0, frame_count - NUM_INPUT_FRAMES + 1, stride):
        window_boxes = boxes[start : start + NUM_INPUT_FRAMES]
        output_boxes = [window_boxes[index] for index in QUALITY_OUTPUT_INDICES]
        if any(value is None for value in output_boxes):
            rejection_counts["missing_output_mask"] += 1
            continue
        detected = [value for value in window_boxes if value is not None]
        maximum_width = max(value[1] for value in detected)
        maximum_height = max(value[2] for value in detected)
        selected_bucket = select_v5_bucket(
            maximum_width,
            maximum_height,
            context_fraction=context_fraction,
        )
        buckets = _eligible_buckets(
            selected_bucket,
            minimum_bucket=minimum_bucket,
            maximum_bucket=maximum_bucket,
        )
        window_centres = centres[start : start + NUM_INPUT_FRAMES]
        motion_bucket, motion_pixels = motion_bucket_for_centres(window_centres)
        for bucket in buckets:
            offsets = native_tile_offsets(
                maximum_width,
                maximum_height,
                bucket=bucket,
                context_fraction=context_fraction,
                overlap=tile_overlap,
            )
            for tile_index, (offset_x, offset_y) in enumerate(offsets):
                origins = tuple(
                    (round_to_even(horizontal + offset_x), round_to_even(vertical + offset_y))
                    for horizontal, vertical in window_centres
                )
                output_occupancies = tuple(
                    mask_fraction_in_crop(masks[start + index], origins[index], bucket)
                    for index in QUALITY_OUTPUT_INDICES
                )
                if any(value <= 0 for value in output_occupancies):
                    rejection_counts["empty_output_tile"] += 1
                    continue
                occupancy = float(np.mean(output_occupancies))
                occupancy_bucket = occupancy_bucket_for_fraction(occupancy)
                if occupancy_bucket is None:
                    rejection_counts["occupancy_outside_5_95"] += 1
                    continue
                identity = f"{source_id}:{clip_identity}:{start:06d}:b{bucket}:tile-{tile_index:02d}"
                base_window_id = f"{source_id}:{clip_identity}:{start:06d}:tile-{tile_index:02d}"
                yield BalancedNativeCandidate(
                    entry={
                        "name": identity,
                        "target_video": target,
                        "mask_video": mask_path,
                        "start_frame": start,
                        "bucket": bucket,
                        "origins": [[x, y] for x, y in origins],
                        "mask_reliability": [1.0 if value is not None else 0.5 for value in window_boxes],
                        "mosaic_block_size": metadata.base_mosaic_block_size.mosaic_size_v1_normal,
                        "source_video_id": source_id,
                        "motion_bucket": motion_bucket,
                        "motion_pixels_per_frame": round(motion_pixels, 6),
                        "mask_occupancy": round(occupancy, 8),
                        "occupancy_bucket": occupancy_bucket,
                        "roi_max_width": float(maximum_width),
                        "roi_max_height": float(maximum_height),
                    },
                    base_window_id=base_window_id,
                    stable_score=stable_candidate_score(seed, identity),
                )


class StratifiedCandidateReservoir:
    """Keep a deterministic bounded subset of each source stratum."""

    def __init__(self, per_stratum: int = 96) -> None:
        if per_stratum <= 0:
            raise ValueError("per-stratum capacity must be positive")
        self.per_stratum = per_stratum
        self._heaps: dict[tuple[int, str, str], list[tuple[int, str, BalancedNativeCandidate]]] = defaultdict(list)
        self.seen = 0

    def add(self, candidate: BalancedNativeCandidate) -> None:
        self.seen += 1
        key = (candidate.bucket, candidate.motion_bucket, candidate.occupancy_bucket)
        heap = self._heaps[key]
        item = (-candidate.stable_score, str(candidate.entry["name"]), candidate)
        if len(heap) < self.per_stratum:
            heapq.heappush(heap, item)
        elif candidate.stable_score < -heap[0][0]:
            heapq.heapreplace(heap, item)

    def candidates(self) -> list[BalancedNativeCandidate]:
        return [
            candidate
            for heap in self._heaps.values()
            for _, _, candidate in heap
        ]


def ratio_target_counts(total: int, ratios: Mapping[int, float]) -> dict[int, int]:
    if total <= 0:
        return {int(bucket): 0 for bucket in ratios}
    if not ratios or any(value < 0 for value in ratios.values()):
        raise ValueError("bucket ratios must be non-negative and non-empty")
    denominator = float(sum(ratios.values()))
    if denominator <= 0:
        raise ValueError("at least one bucket ratio must be positive")
    exact = {int(bucket): total * float(weight) / denominator for bucket, weight in ratios.items()}
    counts = {bucket: int(value) for bucket, value in exact.items()}
    remainder = total - sum(counts.values())
    order = sorted(exact, key=lambda bucket: (-(exact[bucket] - counts[bucket]), bucket))
    for bucket in order[:remainder]:
        counts[bucket] += 1
    return counts


def _select_bucket_candidates(
    candidates: Iterable[BalancedNativeCandidate],
    *,
    count: int,
    used_base_windows: set[str],
) -> list[BalancedNativeCandidate]:
    grouped: dict[tuple[str, str], list[BalancedNativeCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.motion_bucket, candidate.occupancy_bucket)].append(candidate)
    for values in grouped.values():
        values.sort(key=lambda value: (value.stable_score, str(value.entry["name"])))
    selected: list[BalancedNativeCandidate] = []
    selected_by_group: Counter[tuple[str, str]] = Counter()
    pointers: Counter[tuple[str, str]] = Counter()
    boundary_limit = max(1, int(count * 0.15))
    while len(selected) < count:
        available: list[tuple[str, str]] = []
        for key, values in grouped.items():
            if (
                key[1] == "boundary"
                and sum(
                    selected_count
                    for (_, occupancy), selected_count in selected_by_group.items()
                    if occupancy == "boundary"
                )
                >= boundary_limit
            ):
                continue
            pointer = pointers[key]
            while pointer < len(values) and values[pointer].base_window_id in used_base_windows:
                pointer += 1
            pointers[key] = pointer
            if pointer < len(values):
                available.append(key)
        if not available:
            break

        def deficit(key: tuple[str, str]) -> tuple[float, int, str, str]:
            motion, occupancy = key
            weight = 0.25 * DEFAULT_OCCUPANCY_WEIGHTS[occupancy]
            return (
                weight * count - selected_by_group[key],
                -pointers[key],
                motion,
                occupancy,
            )

        key = max(available, key=deficit)
        candidate = grouped[key][pointers[key]]
        pointers[key] += 1
        selected.append(candidate)
        selected_by_group[key] += 1
        used_base_windows.add(candidate.base_window_id)
    return selected


def select_balanced_source_candidates(
    candidates: Sequence[BalancedNativeCandidate],
    *,
    source_cap: int = 50,
    bucket_ratios: Mapping[int, float] = DEFAULT_BUCKET_RATIOS,
) -> list[BalancedNativeCandidate]:
    if source_cap <= 0:
        raise ValueError("source cap must be positive")
    target = ratio_target_counts(min(source_cap, len(candidates)), bucket_ratios)
    by_bucket: dict[int, list[BalancedNativeCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.bucket in target:
            by_bucket[candidate.bucket].append(candidate)
    selected: list[BalancedNativeCandidate] = []
    used_base_windows: set[str] = set()
    for bucket in sorted(target):
        selected.extend(
            _select_bucket_candidates(
                by_bucket[bucket],
                count=target[bucket],
                used_base_windows=used_base_windows,
            )
        )
    # Sparse sources can miss a requested size or exhaust distinct temporal
    # windows. Fill the remainder deterministically without violating the
    # one-window-per-source selection rule.
    if len(selected) < source_cap:
        selected_ids = {id(value) for value in selected}
        boundary_limit = max(1, int(source_cap * 0.15))
        selected_boundary = sum(value.occupancy_bucket == "boundary" for value in selected)
        remainder = sorted(candidates, key=lambda value: (value.stable_score, str(value.entry["name"])))
        for candidate in remainder:
            if id(candidate) in selected_ids or candidate.base_window_id in used_base_windows:
                continue
            if candidate.occupancy_bucket == "boundary" and selected_boundary >= boundary_limit:
                continue
            selected.append(candidate)
            used_base_windows.add(candidate.base_window_id)
            selected_boundary += candidate.occupancy_bucket == "boundary"
            if len(selected) >= source_cap:
                break
    return sorted(selected, key=lambda value: (candidate_sort_key(value)))


def candidate_sort_key(candidate: BalancedNativeCandidate) -> tuple[str, int, int, str]:
    return (
        str(candidate.entry["source_video_id"]),
        candidate.bucket,
        int(candidate.entry["start_frame"]),
        str(candidate.entry["name"]),
    )
