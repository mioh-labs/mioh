# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import hashlib
import json
import runpy
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest


SELECTOR = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "scripts"
        / "training"
        / "build-basicvsrpp-flow-recoverability-manifest.py"
    )
)
FlowThresholds = SELECTOR["FlowThresholds"]
FlowRecoverabilityMetrics = SELECTOR["FlowRecoverabilityMetrics"]


def textured_window() -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = np.random.default_rng(73)
    noise = rng.normal(size=(512, 512)).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), 2.0)
    texture += 0.20 * cv2.GaussianBlur(noise, (0, 0), 0.5)
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    luma = np.rint(texture * 255.0).astype(np.uint8)
    base = np.stack(
        (luma, np.roll(luma, 2, axis=0), np.roll(luma, 1, axis=1)),
        axis=2,
    )
    frames = [
        cv2.warpAffine(
            base,
            np.float32(((1, 0, (index - 4) * 2), (0, 1, index - 4))),
            (512, 512),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        for index in range(9)
    ]
    masks = []
    for _ in range(9):
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[160:352, 160:352] = 255
        masks.append(mask)
    return frames, masks


def manifest_entry(target: Path, mask: Path, *, source: str, name: str) -> dict:
    return {
        "name": name,
        "target_video": str(target),
        "mask_video": str(mask),
        "start_frame": 0,
        "bucket": 512,
        "origins": [[0, 0]] * 9,
        "mask_reliability": [1.0] * 9,
        "mosaic_block_size": 8.0,
        "source_video_id": source,
        "recoverability": {"score": 0.5},
    }


def test_flow_alignment_recovers_translated_native_detail() -> None:
    frames, masks = textured_window()
    reason, metrics = SELECTOR["evaluate_flow_window"](frames, masks)
    assert reason is None
    assert metrics is not None
    assert metrics.good_neighbors >= 2
    assert metrics.aligned_residual < 0.002
    assert metrics.unaligned_residual > 0.02
    assert metrics.alignment_gain > 0.90
    assert metrics.hf_correlation > 0.95
    assert metrics.recoverable_hf > 0.90


def test_shared_roi_crop_is_an_exact_native_slice() -> None:
    yy, xx = np.indices((512, 512), dtype=np.uint16)
    frame = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=2).astype(np.uint8)
    masks = []
    for _ in range(9):
        mask = np.zeros((512, 512), dtype=np.uint8)
        mask[192:320, 192:320] = 255
        masks.append(mask)
    targets, selected_masks, offset = SELECTOR["prepare_analysis_window"](
        [frame] * 9, masks
    )
    assert offset == (128, 128)
    assert np.array_equal(targets[4], frame[128:384, 128:384])
    assert np.array_equal(selected_masks[4], masks[4][128:384, 128:384])
    assert not np.shares_memory(targets[4], frame)  # contiguous training-safe view copy

    downscaled, _, _ = SELECTOR["prepare_analysis_window"](
        [frame] * 9, masks, analysis_size=128
    )
    assert downscaled[4].shape == (128, 128, 3)


def test_combed_material_and_excessive_residual_are_rejected() -> None:
    frames, masks = textured_window()
    combed = np.zeros((512, 512, 3), dtype=np.uint8)
    combed[::2] = 255
    reason, metrics = SELECTOR["evaluate_flow_window"]([combed] * 9, masks)
    assert reason == "combing"
    assert metrics is None

    unrelated = [np.zeros_like(frames[0]) for _ in range(9)]
    unrelated[4] = frames[4]
    thresholds = replace(
        FlowThresholds(), maximum_combing_score=1.0
    )
    reason, metrics = SELECTOR["evaluate_flow_window"](
        unrelated, masks, thresholds=thresholds
    )
    assert reason == "aligned_residual"
    assert metrics is None


def test_source_balanced_builder_is_additive_deterministic_and_atomic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "native-512.jsonl"
    target = tmp_path / "target.mp4"
    mask = tmp_path / "mask.mkv"
    rows = [
        manifest_entry(target, mask, source="source-a", name="a-0"),
        manifest_entry(target, mask, source="source-a", name="a-1"),
        manifest_entry(target, mask, source="source-a", name="a-2"),
        manifest_entry(target, mask, source="source-b", name="b-0"),
    ]
    original = "".join(json.dumps(row) + "\n" for row in rows)
    source.write_text(original, encoding="utf-8")
    frames, masks = textured_window()
    calls: list[tuple[str, str]] = []

    def decoder(path: Path, *, pixel_format: str):
        calls.append((Path(path).name, pixel_format))
        return masks if pixel_format == "gray" else frames

    metric = FlowRecoverabilityMetrics(
        mask_fraction=0.5,
        center_hf_rms=0.04,
        combing_score=0.01,
        good_neighbors=4,
        aligned_residual=0.01,
        unaligned_residual=0.04,
        alignment_gain=0.75,
        flow_valid_fraction=0.95,
        flow_cycle_error=0.1,
        hf_correlation=0.8,
        recoverable_hf=0.7,
        score=0.02,
    )
    monkeypatch.setitem(
        SELECTOR,
        "evaluate_flow_window",
        lambda targets, masks, **kwargs: (None, metric),
    )

    output_a = tmp_path / "selected-a.jsonl"
    report_a = tmp_path / "selected-a.metrics.json"
    output_b = tmp_path / "selected-b.jsonl"
    report_b = tmp_path / "selected-b.metrics.json"
    build = SELECTOR["build_flow_recoverability_manifest"]
    first = build(
        source_manifest=source,
        output=output_a,
        report=report_a,
        quota=2,
        decoder=decoder,
        progress_every=0,
    )
    second = build(
        source_manifest=source,
        output=output_b,
        report=report_b,
        quota=2,
        decoder=decoder,
        progress_every=0,
    )

    first_rows = [json.loads(line) for line in output_a.read_text().splitlines()]
    second_rows = [json.loads(line) for line in output_b.read_text().splitlines()]
    assert first_rows == second_rows
    assert {row["source_video_id"] for row in first_rows} == {"source-a", "source-b"}
    assert all("recoverability" in row for row in first_rows)
    assert all("flow_recoverability" in row for row in first_rows)
    assert source.read_text(encoding="utf-8") == original
    assert first["manifest_sha256"] == hashlib.sha256(output_a.read_bytes()).hexdigest()
    assert second["selected_metrics"] == first["selected_metrics"]
    assert first["training_pixels_resized"] == 0
    assert calls == [
        ("target.mp4", "rgb24"),
        ("mask.mkv", "gray"),
        ("target.mp4", "rgb24"),
        ("mask.mkv", "gray"),
    ]
    assert not list(tmp_path.glob(".*.tmp-*"))
