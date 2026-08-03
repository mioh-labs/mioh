# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import json
import hashlib
import runpy
from pathlib import Path

import numpy as np
import pytest


BUILDER = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "scripts"
        / "training"
        / "build-mioh-native-hf-recoverability-manifest.py"
    )
)

RecoverabilityThresholds = BUILDER["RecoverabilityThresholds"]


def decision(**updates):
    value = {
        "usable": True,
        "confidence": 0.95,
        "sharpness": 0.90,
        "clarity": 0.90,
        "occlusion": 0.05,
    }
    value.update(updates)
    return value


def entry(target: Path, mask: Path, *, source="source-a", start=0, tile=0):
    return {
        "name": f"{source}:{start:06d}:tile-{tile:02d}",
        "target_video": str(target),
        "mask_video": str(mask),
        "start_frame": start,
        "bucket": 512,
        "origins": [[index * 2, index * 4] for index in range(9)],
        "mask_reliability": [1.0] * 9,
        "mosaic_block_size": 8.0,
        "source_video_id": source,
    }


def synthetic_frames(frame_count=140):
    yy, xx = np.indices((512, 512))
    checker = (((xx // 2 + yy // 2) % 2) * 180 + 35).astype(np.uint8)
    target = np.stack((checker, np.roll(checker, 1, 0), np.roll(checker, 1, 1)), axis=2)
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[176:344, 176:344] = 255
    # Reusing immutable arrays keeps this focused test small while still
    # presenting a complete decoded clip to the builder.
    return [target] * frame_count, [mask] * frame_count


def test_strict_static_and_decoded_recoverability_filters(tmp_path: Path) -> None:
    evaluate = BUILDER["evaluate_decoded_candidate"]
    static = BUILDER["static_rejection"]
    thresholds = RecoverabilityThresholds()
    target_path = tmp_path / "clip.mp4"
    mask_path = tmp_path / "clip.mkv"
    candidate = entry(target_path, mask_path)
    candidate["_target_path"] = target_path.resolve()
    candidate["_mask_path"] = mask_path.resolve()
    targets, masks = synthetic_frames(9)

    reason, metrics = evaluate(candidate, decision(), targets, masks, thresholds)
    assert reason is None
    assert metrics is not None
    assert metrics.phase_diversity >= 0.80
    assert metrics.context_valid_fraction >= 0.98
    assert metrics.eroded_mask_fraction >= 0.02
    assert metrics.luma_hf_rms >= 0.0025

    reason, _ = static(candidate, decision(clarity=0.799), thresholds)
    assert reason == "gemma_clarity"
    unreliable = dict(candidate, mask_reliability=[1.0] * 8 + [0.98])
    reason, _ = static(unreliable, decision(), thresholds)
    assert reason == "mask_reliability"
    unreliable = dict(candidate, mask_reliability=[1.0] * 8 + [float("nan")])
    reason, _ = static(unreliable, decision(), thresholds)
    assert reason == "mask_reliability"
    no_phase = dict(candidate, origins=[[0, 0]] * 9)
    reason, _ = static(no_phase, decision(), thresholds)
    assert reason == "phase_diversity"

    flat_targets = [np.full((512, 512, 3), 128, np.uint8)] * 9
    reason, _ = evaluate(candidate, decision(), flat_targets, masks, thresholds)
    assert reason == "luma_hf_rms"


def test_sparse_evaluation_is_identical_to_full_decoded_evaluation(
    tmp_path: Path,
) -> None:
    candidate = entry(tmp_path / "clip.mp4", tmp_path / "clip.mkv")
    candidate["_target_path"] = (tmp_path / "clip.mp4").resolve()
    candidate["_mask_path"] = (tmp_path / "clip.mkv").resolve()
    targets, masks = synthetic_frames(9)
    thresholds = RecoverabilityThresholds()

    full_reason, full_metrics = BUILDER["evaluate_decoded_candidate"](
        candidate, decision(), targets, masks, thresholds
    )
    mask_reason, mask_values = BUILDER["evaluate_mask_candidate"](
        candidate, decision(), masks, thresholds
    )
    assert mask_reason is None
    assert mask_values is not None
    sparse_reason, sparse_metrics = BUILDER["evaluate_target_candidate"](
        candidate, targets[4], masks, mask_values, thresholds
    )

    assert sparse_reason == full_reason is None
    assert sparse_metrics == full_metrics


def test_sparse_decode_retains_only_requested_indices() -> None:
    yielded: list[int] = []

    def decoder(path, *, pixel_format):
        assert Path(path).name == "target.mp4"
        assert pixel_format == "rgb24"
        for index in range(20):
            yielded.append(index)
            yield np.full((4, 6, 3), index, dtype=np.uint8)

    decoded = BUILDER["decode_sparse_frames"](
        Path("target.mp4"),
        selected_indices={2, 5},
        through_index=7,
        pixel_format="rgb24",
        decoder=decoder,
    )
    assert decoded.decoded_count == 8
    assert sorted(decoded.frames) == [2, 5]
    assert int(decoded.frames[2][0, 0, 0]) == 2
    assert int(decoded.frames[5][0, 0, 0]) == 5
    # Sequential decode stops as soon as the final required window boundary is
    # known; frames 8..19 are never materialized.
    assert yielded == list(range(8))

    visited: list[int] = []
    BUILDER["visit_selected_decoded_frames"](
        Path("target.mp4"),
        selected_indices={1, 6},
        through_index=7,
        pixel_format="rgb24",
        visitor=lambda index, frame: visited.append(index),
        decoder=decoder,
    )
    assert visited == [1, 6]


def test_context_validity_uses_all_five_central_frames() -> None:
    context_valid = BUILDER["central_context_valid_fraction"]
    frames = [np.zeros((512, 512, 3), np.uint8)] * 9
    masks = [np.zeros((512, 512), np.float32) for _ in range(9)]
    for mask in masks:
        mask[176:344, 176:344] = 1.0
    origins = [(0, 0)] * 9
    assert context_valid(masks, frames, origins) == pytest.approx(1.0)
    # Only output frame 2 loses native context.  A centre-only check would
    # incorrectly retain this candidate.
    invalid_origins = list(origins)
    invalid_origins[2] = (-256, -256)
    assert context_valid(masks, frames, invalid_origins) < 0.98


def test_relative_gemma_paths_resolve_from_classification_file(tmp_path: Path) -> None:
    classification_dir = tmp_path / "classifications"
    classification_dir.mkdir()
    path = classification_dir / "values.jsonl"
    path.write_text(
        json.dumps(
            {"status": "ok", "video": "../clip.mp4", "decision": decision()}
        )
        + "\n",
        encoding="utf-8",
    )
    decisions = BUILDER["load_gemma_decisions"](path)
    assert (tmp_path / "clip.mp4").resolve() in decisions


def test_builder_is_deterministic_decodes_once_and_balances_sources(tmp_path: Path) -> None:
    source_manifest = tmp_path / "source.jsonl"
    classifications = tmp_path / "classifications.jsonl"
    output_a = tmp_path / "recoverable-a.jsonl"
    output_b = tmp_path / "recoverable-b.jsonl"
    report_a = tmp_path / "recoverable-a.metrics.json"
    report_b = tmp_path / "recoverable-b.metrics.json"
    target_a, mask_a = tmp_path / "a.mp4", tmp_path / "a.mkv"
    target_b, mask_b = tmp_path / "b.mp4", tmp_path / "b.mkv"
    rows = [
        entry(target_a, mask_a, source="source-a", start=0, tile=0),
        # Same target/start tile is deliberately better only by name tie-break;
        # exactly one survives target/start de-duplication.
        entry(target_a, mask_a, source="source-a", start=0, tile=1),
        entry(target_a, mask_a, source="source-a", start=64, tile=0),
        entry(target_b, mask_b, source="source-b", start=0, tile=0),
    ]
    source_manifest.write_text(
        "".join(json.dumps(value) + "\n" for value in rows), encoding="utf-8"
    )
    classifications.write_text(
        "".join(
            json.dumps(
                {"status": "ok", "video": str(path), "decision": decision()}
            )
            + "\n"
            for path in (target_a, target_b)
        ),
        encoding="utf-8",
    )
    targets, masks = synthetic_frames()

    # Compute the pre-optimization full-decoder reference using the original
    # candidate evaluator and selection pipeline.
    legacy_entries = BUILDER["load_source_entries"](source_manifest)
    legacy_decisions = BUILDER["load_gemma_decisions"](classifications)
    legacy_accepted = []
    for candidate in legacy_entries:
        reason, metrics = BUILDER["evaluate_decoded_candidate"](
            candidate,
            legacy_decisions[candidate["_target_path"]],
            targets,
            masks,
            RecoverabilityThresholds(),
        )
        assert reason is None and metrics is not None
        candidate["_recoverability"] = metrics
        legacy_accepted.append(candidate)
    legacy_selected = BUILDER["source_balanced_select"](
        BUILDER["clip_nms"](
            BUILDER["deduplicate_target_start"](legacy_accepted), radius=64
        ),
        2,
    )
    legacy_selected.sort(
        key=lambda value: (
            str(value["source_video_id"]),
            str(value["_target_path"]),
            int(value["start_frame"]),
            str(value["name"]),
        )
    )
    legacy_rows = [BUILDER["_public_entry"](value) for value in legacy_selected]

    def one_build(output: Path, report: Path):
        calls = []

        def decoder(path, *, pixel_format):
            calls.append((Path(path).name, pixel_format))
            return masks if pixel_format == "gray" else targets

        result = BUILDER["build_recoverability_manifest"](
            source_manifest=source_manifest,
            classifications=classifications,
            output=output,
            split="train",
            quota=2,
            report=report,
            decoder=decoder,
            progress_every=0,
        )
        assert sorted(calls) == [
            ("a.mkv", "gray"),
            ("a.mp4", "rgb24"),
            ("b.mkv", "gray"),
            ("b.mp4", "rgb24"),
        ]
        return result

    first = one_build(output_a, report_a)
    second = one_build(output_b, report_b)
    first_rows = [json.loads(line) for line in output_a.read_text().splitlines()]
    second_rows = [json.loads(line) for line in output_b.read_text().splitlines()]
    assert first_rows == legacy_rows
    assert first_rows == second_rows
    assert {row["source_video_id"] for row in first_rows} == {"source-a", "source-b"}
    assert all(Path(row["target_video"]).is_absolute() for row in first_rows)
    assert first["counts"]["input"] == 4
    assert first["counts"]["after_target_start_dedup"] == 3
    assert first["counts"]["selected"] == 2
    assert second["selected_metrics"] == first["selected_metrics"]
    assert first["manifest_sha256"] == hashlib.sha256(
        output_a.read_bytes()
    ).hexdigest()


def test_atomic_jsonl_preserves_previous_output_on_failure(tmp_path: Path) -> None:
    output = tmp_path / "curriculum.jsonl"
    output.write_text("previous\n", encoding="utf-8")

    def broken_records():
        yield {"name": "first"}
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic failure"):
        BUILDER["atomic_write_jsonl"](broken_records(), output)
    assert output.read_text(encoding="utf-8") == "previous\n"
    assert list(tmp_path.glob(".curriculum.jsonl.tmp-*")) == []


def test_score_requires_every_recoverability_factor() -> None:
    score = BUILDER["deterministic_score"]
    values = {
        "gemma_sharpness": 0.9,
        "gemma_clarity": 0.8,
        "phase_diversity": 0.9,
        "context_valid_fraction": 0.99,
        "eroded_mask_fraction": 0.04,
        "luma_hf_rms": 0.005,
    }
    baseline = score(values)
    assert baseline > 0
    for name in (
        "gemma_sharpness",
        "gemma_clarity",
        "phase_diversity",
        "context_valid_fraction",
        "eroded_mask_fraction",
        "luma_hf_rms",
    ):
        weakened = dict(values)
        weakened[name] *= 0.25
        assert score(weakened) < baseline
