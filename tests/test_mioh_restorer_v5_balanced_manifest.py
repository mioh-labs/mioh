# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from collections import Counter

import numpy as np
import importlib.util
from pathlib import Path
import pytest

from lada.models.mioh_restorer.native_manifest_v5 import (
    BalancedNativeCandidate,
    StratifiedCandidateReservoir,
    mask_fraction_in_crop,
    motion_bucket_for_centres,
    occupancy_bucket_for_fraction,
    ratio_target_counts,
    select_balanced_source_candidates,
)


def candidate(index: int, *, bucket: int, motion: str, occupancy: str) -> BalancedNativeCandidate:
    return BalancedNativeCandidate(
        entry={
            "name": f"sample-{index}-b{bucket}",
            "source_video_id": "source",
            "start_frame": index,
            "bucket": bucket,
            "motion_bucket": motion,
            "occupancy_bucket": occupancy,
        },
        base_window_id=f"window-{index}",
        stable_score=index,
    )


def test_motion_and_occupancy_buckets_cover_documented_ranges() -> None:
    assert motion_bucket_for_centres([(0, 0)] * 9)[0] == "static"
    assert motion_bucket_for_centres([(index * 2, 0) for index in range(9)])[0] == "low"
    assert motion_bucket_for_centres([(index * 4, 0) for index in range(9)])[0] == "medium"
    assert motion_bucket_for_centres([(index * 8, 0) for index in range(9)])[0] == "high"
    assert occupancy_bucket_for_fraction(0.049) is None
    assert occupancy_bucket_for_fraction(0.05) == "boundary"
    assert occupancy_bucket_for_fraction(0.20) == "preferred"
    assert occupancy_bucket_for_fraction(0.70) == "dense"
    assert occupancy_bucket_for_fraction(0.951) is None


def test_mask_fraction_uses_requested_native_crop_without_resizing() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:16, 8:16] = 255
    assert mask_fraction_in_crop(mask, (0, 0), 32) == 64 / (32 * 32)
    assert mask_fraction_in_crop(mask, (8, 8), 16) == 64 / (16 * 16)
    assert mask_fraction_in_crop(mask, (40, 40), 16) == 0.0


def test_ratio_targets_make_exact_10_30_60_source_cap() -> None:
    assert ratio_target_counts(50, {192: 0.10, 256: 0.30, 384: 0.60}) == {
        192: 5,
        256: 15,
        384: 30,
    }


def test_source_selection_caps_and_stratifies_without_duplicate_windows() -> None:
    values = []
    index = 0
    for bucket in (192, 256, 384):
        for motion in ("static", "low", "medium", "high"):
            for occupancy in ("boundary", "preferred", "dense"):
                for _ in range(20):
                    values.append(candidate(index, bucket=bucket, motion=motion, occupancy=occupancy))
                    index += 1
    selected = select_balanced_source_candidates(values, source_cap=50)
    assert len(selected) == 50
    assert Counter(item.bucket for item in selected) == {192: 5, 256: 15, 384: 30}
    assert len({item.base_window_id for item in selected}) == 50
    assert set(item.motion_bucket for item in selected) == {"static", "low", "medium", "high"}
    assert Counter(item.occupancy_bucket for item in selected)["boundary"] <= 7


def test_reservoir_keeps_smallest_stable_scores_per_stratum() -> None:
    reservoir = StratifiedCandidateReservoir(per_stratum=3)
    for index in (9, 2, 7, 1, 5):
        reservoir.add(candidate(index, bucket=384, motion="medium", occupancy="preferred"))
    assert sorted(item.stable_score for item in reservoir.candidates()) == [1, 2, 5]


def test_sampled_clip_source_ids_can_be_grouped(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts/training/build-mioh-restorer-v5-balanced-manifest.py"
    spec = importlib.util.spec_from_file_location("balanced_manifest_cli", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "sample.json"
    path.write_text('{"name":"FC2PPV-123-t456.mkv"}', encoding="utf-8")
    assert module.metadata_source_id(path, strip_sampled_timestamp=True) == "FC2PPV-123"
    path.write_text('{"name":"hhd800.com@FC2-PPV-2486345_3-t456.mkv"}', encoding="utf-8")
    assert module.metadata_source_id(
        path,
        strip_sampled_timestamp=True,
        canonicalize_fc2_source_id=True,
    ) == "FC2PPV-2486345"


def test_hq_training_rejects_sub_256_manifest_before_model_start(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts/training/train-mioh-restorer-v5.py"
    spec = importlib.util.spec_from_file_location("train_mioh_restorer_v5", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = tmp_path / "train.jsonl"
    manifest.write_text('{"bucket":192}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="requires native buckets >= 256"):
        module.validate_hq_manifest_minimum_bucket(manifest)
    manifest.write_text('{"bucket":256}\n{"bucket":384}\n', encoding="utf-8")
    module.validate_hq_manifest_minimum_bucket(manifest)
