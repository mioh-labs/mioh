# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Choose representative LADA dataset clips from consecutive similar scenes."""

from __future__ import annotations

import json
import hashlib
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from lada.datasetcreation.gemma_video_selector import AnalysisWindow


CLIP_INDEX_PATTERN = re.compile(r"-(\d+)-$")
CLASSIFIER_VERSION = "representative-v1"


@dataclass(frozen=True)
class DatasetClip:
    video_path: Path
    metadata_path: Path
    mask_path: Path | None
    source_name: str
    clip_index: int
    frames_count: int
    fps: float
    scene_width: int
    scene_height: int

    @property
    def duration(self) -> float:
        return self.frames_count / self.fps if self.fps > 0 else 0.0

    @property
    def analysis_window(self) -> AnalysisWindow:
        return AnalysisWindow(0.0, self.duration)


def parse_clip_index(path: Path) -> int:
    # Dataset names end in "-000123-.mp4". Strip both suffixes first.
    stem = path.stem
    match = CLIP_INDEX_PATTERN.search(stem)
    if match is None:
        raise ValueError(f"cannot read dataset clip index from: {path.name}")
    return int(match.group(1))


def _resolve_relative(metadata_path: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = metadata_path.parent / path
    return path.resolve()


def load_dataset_clip(metadata_path: Path) -> DatasetClip:
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    video_path = _resolve_relative(metadata_path, payload.get("relative_nsfw_video_path"))
    if video_path is None:
        raise ValueError(f"relative_nsfw_video_path is missing: {metadata_path}")
    mask_path = _resolve_relative(metadata_path, payload.get("relative_mask_video_path"))
    scene_shape = payload.get("scene_shape") or [0, 0]
    return DatasetClip(
        video_path=video_path,
        metadata_path=metadata_path.resolve(),
        mask_path=mask_path,
        source_name=str(payload.get("name") or video_path.name),
        clip_index=parse_clip_index(video_path),
        frames_count=int(payload.get("frames_count") or 0),
        fps=float(payload.get("fps") or 0),
        scene_width=int(scene_shape[1] if len(scene_shape) > 1 else 0),
        scene_height=int(scene_shape[0] if scene_shape else 0),
    )


def collect_dataset_clips(
    dataset_root: Path,
    max_duration: float = 8.25,
    min_file_age: float = 15.0,
) -> tuple[list[DatasetClip], list[dict[str, str]]]:
    metadata_dir = dataset_root / "crop_unscaled_meta"
    if not metadata_dir.is_dir():
        raise FileNotFoundError(f"dataset metadata directory not found: {metadata_dir}")
    clips: list[DatasetClip] = []
    skipped: list[dict[str, str]] = []
    now = time.time()
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        if metadata_path.name.startswith("._"):
            continue
        try:
            clip = load_dataset_clip(metadata_path)
            required = [clip.video_path, clip.metadata_path]
            if clip.mask_path is not None:
                required.append(clip.mask_path)
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise ValueError(f"paired file is not ready: {', '.join(missing)}")
            if any(now - path.stat().st_mtime < min_file_age for path in required):
                raise ValueError("paired files are still being written")
            if clip.duration <= 0:
                raise ValueError("duration is zero")
            if max_duration > 0 and clip.duration > max_duration:
                raise ValueError(f"duration {clip.duration:.2f}s exceeds {max_duration:.2f}s")
            clips.append(clip)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            skipped.append({"metadata": str(metadata_path), "reason": str(error)})
    clips.sort(key=lambda clip: (clip.source_name, clip.clip_index, str(clip.video_path)))
    return clips, skipped


def build_representative_prompt(clip: DatasetClip, frame_count: int) -> str:
    return f"""This is an ordered contact sheet of {frame_count} frames from one short video clip.
The clip is {clip.duration:.2f} seconds long. It is a candidate for restoration-model training.

Evaluate it using this exact standard:
"The relevant subject is large and sharp, and body orientation, pose, and scene activity
are clearly distinguishable."

The result will be used to group consecutive clips with similar composition and keep the
best representative. Use stable, broad category keys: two visually and semantically
similar clips should receive the same keys. Do not identify people. Do not invent details.
Return exactly one JSON object and no markdown.

Required JSON shape:
{{
  "usable": true,
  "confidence": 0.0,
  "activity_key": "short_stable_snake_case_main_activity",
  "body_orientation": "front|back|profile|three_quarter|mixed|unclear",
  "pose": "standing|sitting|lying|kneeling|crouching|mixed|unclear",
  "framing": "extreme_close_up|close_up|medium|full_body|wide|mixed|unclear",
  "subject_scale": 0.0,
  "sharpness": 0.0,
  "clarity": 0.0,
  "occlusion": 0.0,
  "summary_ja": "brief non-graphic Japanese description",
  "reasons_ja": ["reason"]
}}

All scores are 0 to 1. subject_scale=1 means the relevant subject occupies a large useful
part of the crop. clarity measures how clearly orientation, pose and activity can be read.
occlusion=1 means heavily obstructed. Mark usable=false for blur, black frames, severe
occlusion, unclear content, or a subject too small to be useful.
For activity_key, describe the main activity only. Omit incidental details such as which
hand or limb is moving, so near-duplicate clips receive the same key.
"""


def normalize_category(value: Any) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "unclear").lower()).strip("_")
    return normalized or "unclear"


def classification_signature(decision: dict[str, Any]) -> dict[str, str]:
    return {
        "activity": normalize_category(decision.get("activity_key")),
        "orientation": normalize_category(decision.get("body_orientation")),
        "pose": normalize_category(decision.get("pose")),
        "framing": normalize_category(decision.get("framing")),
    }


def category_token_similarity(left: str, right: str) -> float:
    left_tokens = {token for token in left.split("_") if token}
    right_tokens = {token for token in right.split("_") if token}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def signature_similarity(left: dict[str, str], right: dict[str, str]) -> float:
    weights = {"activity": 0.4, "orientation": 0.2, "pose": 0.2, "framing": 0.2}
    # A genuine activity change must remain as a separate training example even
    # when camera position and body pose happen to be similar.
    left_activity = left.get("activity", "unclear")
    right_activity = right.get("activity", "unclear")
    activity_similarity = category_token_similarity(left_activity, right_activity)
    if (
        left_activity not in {"", "unclear", "mixed"}
        and right_activity not in {"", "unclear", "mixed"}
        and activity_similarity < 0.5
    ):
        return 0.0
    for key in ("orientation", "pose"):
        left_value = left.get(key, "unclear")
        right_value = right.get(key, "unclear")
        if (
            left_value not in {"", "unclear", "mixed"}
            and right_value not in {"", "unclear", "mixed"}
            and left_value != right_value
        ):
            return 0.0
    score = 0.0
    for key, weight in weights.items():
        left_value = left.get(key, "unclear")
        right_value = right.get(key, "unclear")
        if key == "activity" and activity_similarity >= 0.5:
            score += weight
        elif left_value == right_value and left_value not in {"", "unclear", "mixed"}:
            score += weight
    return score


def clamp_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def representative_score(clip: DatasetClip, decision: dict[str, Any]) -> float:
    # 768px is already a strong crop for the current source material. Above it,
    # semantic clarity and actual sharpness should decide rather than resolution.
    size_score = min(1.0, min(clip.scene_width, clip.scene_height) / 768.0)
    score = (
        0.20 * size_score
        + 0.20 * clamp_score(decision.get("subject_scale"))
        + 0.25 * clamp_score(decision.get("sharpness"))
        + 0.30 * clamp_score(decision.get("clarity"))
        + 0.05 * (1.0 - clamp_score(decision.get("occlusion")))
    )
    return score


def is_technically_usable(decision: dict[str, Any]) -> bool:
    """Judge training utility without inheriting a VLM's content-policy opinion."""
    return (
        clamp_score(decision.get("subject_scale")) >= 0.45
        and clamp_score(decision.get("sharpness")) >= 0.45
        and clamp_score(decision.get("clarity")) >= 0.50
        and clamp_score(decision.get("occlusion")) <= 0.65
    )


def group_consecutive_similar_clips(
    classified: list[tuple[DatasetClip, dict[str, Any]]],
    max_index_gap: int = 3,
    min_similarity: float = 0.6,
) -> list[list[tuple[DatasetClip, dict[str, Any]]]]:
    if not classified:
        return []
    ordered = sorted(classified, key=lambda item: item[0].clip_index)
    groups: list[list[tuple[DatasetClip, dict[str, Any]]]] = []
    for candidate in ordered:
        candidate_signature = classification_signature(candidate[1])
        best_group: list[tuple[DatasetClip, dict[str, Any]]] | None = None
        best_similarity = -1.0
        for group in reversed(groups):
            if candidate[0].clip_index - group[-1][0].clip_index > max_index_gap:
                continue
            similarity = max(
                signature_similarity(classification_signature(member[1]), candidate_signature)
                for member in group
                if 0 < candidate[0].clip_index - member[0].clip_index <= max_index_gap
            )
            if similarity >= min_similarity and similarity > best_similarity:
                best_group = group
                best_similarity = similarity
        if best_group is None:
            groups.append([candidate])
        else:
            best_group.append(candidate)
    groups.sort(key=lambda group: group[0][0].clip_index)
    return groups


def select_representatives(
    classified: Iterable[tuple[DatasetClip, dict[str, Any]]],
    max_index_gap: int = 3,
    min_similarity: float = 0.6,
    min_confidence: float = 0.6,
    max_per_source: int = 40,
) -> tuple[list[dict[str, Any]], list[DatasetClip]]:
    by_source: dict[str, list[tuple[DatasetClip, dict[str, Any]]]] = {}
    for clip, decision in classified:
        by_source.setdefault(clip.source_name, []).append((clip, decision))

    summaries: list[dict[str, Any]] = []
    representatives: list[DatasetClip] = []
    for source_name in sorted(by_source):
        source_summaries: list[dict[str, Any]] = []
        source_candidates: list[tuple[DatasetClip, float, int]] = []
        groups = group_consecutive_similar_clips(
            by_source[source_name], max_index_gap=max_index_gap, min_similarity=min_similarity
        )
        for group_index, group in enumerate(groups, start=1):
            ranked = sorted(
                group,
                key=lambda item: (representative_score(item[0], item[1]), -item[0].clip_index),
                reverse=True,
            )
            eligible = [
                item for item in ranked
                if is_technically_usable(item[1])
                and clamp_score(item[1].get("confidence")) >= min_confidence
            ]
            representative = eligible[0][0] if eligible else None
            if representative is not None:
                source_candidates.append((
                    representative,
                    representative_score(eligible[0][0], eligible[0][1]),
                    len(source_summaries),
                ))
            source_summaries.append({
                "source_name": source_name,
                "group_index": group_index,
                "signature": classification_signature(group[0][1]),
                "members": [str(item[0].video_path) for item in group],
                "representative": str(representative.video_path) if representative else None,
                "selected_after_source_cap": False,
                "ranking": [
                    {
                        "video": str(clip.video_path),
                        "score": round(representative_score(clip, decision), 6),
                        "model_usable": decision.get("usable") is True,
                        "technically_usable": is_technically_usable(decision),
                        "confidence": clamp_score(decision.get("confidence")),
                    }
                    for clip, decision in ranked
                ],
            })
        source_candidates.sort(key=lambda item: (-item[1], item[0].clip_index))
        if max_per_source > 0:
            source_candidates = source_candidates[:max_per_source]
        selected_group_indices = {item[2] for item in source_candidates}
        for group_index, summary in enumerate(source_summaries):
            summary["selected_after_source_cap"] = group_index in selected_group_indices
        representatives.extend(item[0] for item in source_candidates)
        summaries.extend(source_summaries)
    representatives.sort(key=lambda clip: (clip.source_name, clip.clip_index))
    return summaries, representatives


def split_representatives_by_source(
    clips: Iterable[DatasetClip],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[DatasetClip]]:
    test_ratio = 1.0 - train_ratio - validation_ratio
    if train_ratio <= 0 or validation_ratio < 0 or test_ratio < 0:
        raise ValueError("invalid train/validation/test ratios")
    by_source: dict[str, list[DatasetClip]] = {}
    for clip in clips:
        by_source.setdefault(clip.source_name, []).append(clip)
    for values in by_source.values():
        values.sort(key=lambda clip: clip.clip_index)

    total = sum(len(values) for values in by_source.values())
    ratios = {"train": train_ratio, "validation": validation_ratio, "test": test_ratio}
    targets = {name: total * ratio for name, ratio in ratios.items()}
    splits: dict[str, list[DatasetClip]] = {"train": [], "validation": [], "test": []}
    source_order = sorted(
        by_source,
        key=lambda source: (
            -len(by_source[source]),
            hashlib.sha256(f"{seed}:{source}".encode("utf-8")).hexdigest(),
        ),
    )
    active_splits = [name for name, ratio in ratios.items() if ratio > 0]
    for source in source_order:
        def shortage(split: str) -> tuple[float, str]:
            target = max(targets[split], 1.0)
            return ((targets[split] - len(splits[split])) / target, split)

        destination = max(active_splits, key=shortage)
        splits[destination].extend(by_source[source])
    for values in splits.values():
        values.sort(key=lambda clip: (clip.source_name, clip.clip_index))
    return splits


def limit_representatives_with_diversity(
    clips: Iterable[DatasetClip],
    decisions_by_video: dict[str, dict[str, Any]],
    limit: int,
) -> list[DatasetClip]:
    """Keep varied representatives, preferring quality within each variation.

    This is intended to cap a split *after* sources have been assigned to
    train/validation/test.  Consequently, clipping the training split cannot
    leak clips from one original source video into another split. Candidates
    are bucketed by activity/orientation/pose/framing and selected round-robin;
    the strongest clip is taken first within every semantic bucket.
    """
    clips = list(clips)
    if limit < 0:
        raise ValueError("limit cannot be negative")
    if limit == 0 or len(clips) <= limit:
        return sorted(clips, key=lambda clip: (clip.source_name, clip.clip_index))

    missing = [
        str(clip.video_path)
        for clip in clips
        if str(clip.video_path) not in decisions_by_video
    ]
    if missing:
        raise ValueError(f"missing Gemma decisions for {len(missing)} clips")

    buckets: dict[tuple[str, str, str, str], list[DatasetClip]] = {}
    for clip in clips:
        signature = classification_signature(decisions_by_video[str(clip.video_path)])
        key = (
            signature["activity"],
            signature["orientation"],
            signature["pose"],
            signature["framing"],
        )
        buckets.setdefault(key, []).append(clip)

    for bucket in buckets.values():
        bucket.sort(key=lambda clip: (
            -representative_score(clip, decisions_by_video[str(clip.video_path)]),
            clip.source_name,
            clip.clip_index,
        ))

    selected: list[DatasetClip] = []
    depth = 0
    while len(selected) < limit:
        candidates = [
            (key, bucket[depth])
            for key, bucket in buckets.items()
            if depth < len(bucket)
        ]
        if not candidates:
            break
        candidates.sort(key=lambda item: (
            -representative_score(
                item[1], decisions_by_video[str(item[1].video_path)]
            ),
            item[0],
            item[1].source_name,
            item[1].clip_index,
        ))
        selected.extend(clip for _, clip in candidates[:limit - len(selected)])
        depth += 1

    return sorted(
        selected, key=lambda clip: (clip.source_name, clip.clip_index)
    )


def split_targets_from_training_count(
    training_count: int,
    train_ratio: float,
    validation_ratio: float,
) -> dict[str, int]:
    """Derive integer 3-way targets while keeping training_count exact."""
    test_ratio = 1.0 - train_ratio - validation_ratio
    if training_count < 0:
        raise ValueError("training_count cannot be negative")
    if train_ratio <= 0 or validation_ratio < 0 or test_ratio < 0:
        raise ValueError("invalid train/validation/test ratios")
    if training_count == 0:
        return {"train": 0, "validation": 0, "test": 0}

    total = max(training_count, round(training_count / train_ratio))
    held_out = total - training_count
    held_out_ratio = validation_ratio + test_ratio
    if held_out_ratio == 0:
        validation_count = 0
    else:
        validation_count = round(held_out * validation_ratio / held_out_ratio)
    return {
        "train": training_count,
        "validation": validation_count,
        "test": held_out - validation_count,
    }


def materialize_representative_dataset(clips: Iterable[DatasetClip], output_root: Path, mode: str) -> None:
    clips = list(clips)
    directories = {
        "video": output_root / "crop_unscaled_img",
        "mask": output_root / "crop_unscaled_mask",
        "metadata": output_root / "crop_unscaled_meta",
    }
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    expected_names = {
        "video": {clip.video_path.name for clip in clips},
        "metadata": {clip.metadata_path.name for clip in clips},
        "mask": {clip.mask_path.name for clip in clips if clip.mask_path is not None},
    }
    for kind, directory in directories.items():
        for existing in directory.iterdir():
            if existing.is_symlink() and existing.name not in expected_names[kind]:
                # Another finishing/resumed selector may have removed the same
                # stale link after iterdir() returned it.
                existing.unlink(missing_ok=True)

    def create(source: Path, destination: Path):
        if destination.exists() or destination.is_symlink():
            return
        if mode == "symlink":
            destination.symlink_to(source)
        elif mode == "hardlink":
            os.link(source, destination)
        elif mode == "copy":
            shutil.copy2(source, destination)
        else:
            raise ValueError(f"unsupported materialization mode: {mode}")

    for clip in clips:
        create(clip.video_path, directories["video"] / clip.video_path.name)
        create(clip.metadata_path, directories["metadata"] / clip.metadata_path.name)
        if clip.mask_path is not None:
            create(clip.mask_path, directories["mask"] / clip.mask_path.name)


def remove_legacy_flat_representative_links(output_root: Path) -> None:
    """Remove only symlinks created by the pre-split flat output layout."""
    for name in ("crop_unscaled_img", "crop_unscaled_mask", "crop_unscaled_meta"):
        directory = output_root / name
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.is_symlink():
                # Cleanup must be restart-safe. A concurrently finishing older
                # run can remove a link between is_symlink() and unlink().
                path.unlink(missing_ok=True)
        try:
            directory.rmdir()
        except OSError:
            # Never remove real files or a user-managed non-empty directory.
            pass
