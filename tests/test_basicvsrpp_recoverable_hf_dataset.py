# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from lada.models.basicvsrpp.mmagic.data_preprocessor import DataPreprocessor
from lada.models.basicvsrpp.mmagic.registry import DATASETS
from lada.models.basicvsrpp.recoverable_hf_dataset import (
    FINAL_CROP_SIZE,
    RecoverableHFMosaicVideoDataset,
    phase_block_average_mosaic,
)


def _write_manifest(tmp_path: Path, *, origins: list[list[int]] | None = None) -> Path:
    if origins is None:
        origins = [[32 + index * 2, 32 + index * 2] for index in range(9)]
    manifest = tmp_path / "recoverable.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "name": "source:000000:tile-00",
                "target_video": "target.mp4",
                "mask_video": "mask.mkv",
                "start_frame": 0,
                "bucket": 512,
                "source_video_id": "source",
                "origins": origins,
                "mask_reliability": [1.0] * 9,
                "mosaic_block_size": 8,
                "recoverability": {"score": 1.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _rgb_frames(*, size: int = 576) -> list[np.ndarray]:
    y, x = np.mgrid[:size, :size]
    return [
        np.stack(
            (
                (x + frame_index * 7) % 256,
                (y + frame_index * 11) % 256,
                (x + y + frame_index * 13) % 256,
            ),
            axis=2,
        ).astype(np.uint8)
        for frame_index in range(9)
    ]


def _patch_decode(
    monkeypatch,
    *,
    rgb: list[np.ndarray],
    masks: list[np.ndarray],
) -> None:
    def fake_decode(_path: Path, _start: int, pixel_format: str) -> list[np.ndarray]:
        values = rgb if pixel_format == "rgb24" else masks
        return [value.copy() for value in values]

    monkeypatch.setattr(
        RecoverableHFMosaicVideoDataset,
        "_decode",
        staticmethod(fake_decode),
    )


def test_phase_block_average_mosaic_uses_exact_cells() -> None:
    values = np.arange(8 * 8, dtype=np.uint8).reshape(8, 8)
    image = np.repeat(values[..., None], 3, axis=2)
    result = phase_block_average_mosaic(
        image, block_size=4, phase=(0, 0)
    )

    for top in (0, 4):
        for left in (0, 4):
            expected = np.uint8(
                np.rint(values[top : top + 4, left : left + 4].mean())
            )
            assert np.all(result[top : top + 4, left : left + 4] == expected)


def test_recoverable_hf_dataset_is_native_deterministic_and_preprocessor_compatible(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = _write_manifest(tmp_path)
    rgb = _rgb_frames()
    masks = [np.full((576, 576), 255, dtype=np.uint8) for _ in range(9)]
    _patch_decode(monkeypatch, rgb=rgb, masks=masks)
    dataset = RecoverableHFMosaicVideoDataset(
        manifest, training=False, seed=73
    )

    first = dataset[0]
    second = dataset[0]
    inputs = first["inputs"]
    sample = first["data_samples"]
    assert inputs.shape == (9, 3, FINAL_CROP_SIZE, FINAL_CROP_SIZE)
    assert inputs.dtype == torch.uint8
    assert sample.gt_img.shape == inputs.shape
    assert sample.gt_img.dtype == torch.uint8
    assert sample.mask.shape == (9, 1, FINAL_CROP_SIZE, FINAL_CROP_SIZE)
    torch.testing.assert_close(sample.mask, torch.ones_like(sample.mask))
    torch.testing.assert_close(first["inputs"], second["inputs"])
    torch.testing.assert_close(
        sample.mosaic_phase, second["data_samples"].mosaic_phase
    )
    assert sample.mosaic_block_size.item() in range(6, 13)

    block_size = int(sample.mosaic_block_size)
    final_phase = tuple(int(value) for value in sample.mosaic_phase[0])
    origin_x, origin_y = dataset.entries[0].origins[0]
    final_left, final_top = sample.metainfo["native_final_crop_offset"]
    final_x = origin_x + final_left
    final_y = origin_y + final_top
    source_phase = (
        (final_phase[0] + final_x) % block_size,
        (final_phase[1] + final_y) % block_size,
    )
    expected_target = rgb[0][final_y : final_y + 256, final_x : final_x + 256]
    expected_mosaic = phase_block_average_mosaic(
        rgb[0], block_size=block_size, phase=source_phase
    )[final_y : final_y + 256, final_x : final_x + 256]
    torch.testing.assert_close(
        sample.gt_img[0],
        torch.from_numpy(expected_target.transpose(2, 0, 1).copy()),
    )
    torch.testing.assert_close(
        inputs[0],
        torch.from_numpy(expected_mosaic.transpose(2, 0, 1).copy()),
    )

    raw_inputs = inputs.clone()
    raw_gt = sample.gt_img.clone()
    preprocessor = DataPreprocessor(
        mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0]
    )
    processed = preprocessor(
        {"inputs": [inputs], "data_samples": [sample]}, training=True
    )
    assert processed["inputs"].shape == (1, 9, 3, 256, 256)
    torch.testing.assert_close(
        processed["inputs"][0], raw_inputs.float() / 255.0
    )
    torch.testing.assert_close(
        processed["data_samples"].gt_img[0], raw_gt.float() / 255.0
    )
    assert DATASETS.get("RecoverableHFMosaicVideoDataset") is (
        RecoverableHFMosaicVideoDataset
    )


def test_recoverable_hf_dataset_composites_only_inside_mask(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = _write_manifest(tmp_path, origins=[[32, 32]] * 9)
    rgb = _rgb_frames()
    mask = np.zeros((576, 576), dtype=np.uint8)
    mask[240:336, 240:336] = 255
    _patch_decode(monkeypatch, rgb=rgb, masks=[mask] * 9)

    item = RecoverableHFMosaicVideoDataset(
        manifest, training=False, seed=9
    )[0]
    inputs = item["inputs"]
    sample = item["data_samples"]
    outside = sample.mask.expand_as(inputs) == 0
    inside = sample.mask.expand_as(inputs) > 0
    torch.testing.assert_close(inputs[outside], sample.gt_img[outside])
    assert torch.count_nonzero(inputs[inside] != sample.gt_img[inside]) > 0


def test_recoverable_hf_shared_crop_keeps_roi_at_512_edge(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = _write_manifest(tmp_path, origins=[[32, 32]] * 9)
    rgb = _rgb_frames()
    masks = []
    for frame_index in range(9):
        mask = np.zeros((576, 576), dtype=np.uint8)
        # In 512-crop coordinates this sits at x/y 448..503.  A fixed centre
        # 256 crop would miss it completely.
        shift = frame_index % 3
        mask[480 + shift : 536 + shift, 480 + shift : 536 + shift] = 255
        masks.append(mask)
    _patch_decode(monkeypatch, rgb=rgb, masks=masks)

    item = RecoverableHFMosaicVideoDataset(
        manifest, training=False, seed=19
    )[0]
    sample = item["data_samples"]
    assert sample.metainfo["native_final_crop_offset"] == (256, 256)
    assert torch.count_nonzero(sample.mask) > 0
    expected = rgb[0][288:544, 288:544]
    torch.testing.assert_close(
        sample.gt_img[0],
        torch.from_numpy(expected.transpose(2, 0, 1).copy()),
    )
    block_size = int(sample.mosaic_block_size)
    final_phase = tuple(int(value) for value in sample.mosaic_phase[0])
    source_phase = (
        (final_phase[0] + 288) % block_size,
        (final_phase[1] + 288) % block_size,
    )
    expected_mosaic = phase_block_average_mosaic(
        rgb[0], block_size=block_size, phase=source_phase
    )[288:544, 288:544]
    expected_mosaic_tensor = torch.from_numpy(
        expected_mosaic.transpose(2, 0, 1).copy()
    )
    inside = sample.mask[0].expand_as(sample.gt_img[0]) > 0
    torch.testing.assert_close(
        item["inputs"][0][inside], expected_mosaic_tensor[inside]
    )


def test_recoverable_hf_train_augmentations_flip_and_reverse_together(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = _write_manifest(tmp_path, origins=[[32, 32]] * 9)
    rgb = _rgb_frames()
    masks = [np.full((576, 576), 255, dtype=np.uint8) for _ in range(9)]
    _patch_decode(monkeypatch, rgb=rgb, masks=masks)

    class FixedRNG:
        def random(self) -> float:
            return 0.0

        def randint(self, _minimum: int, _maximum: int) -> int:
            return 8

        def randrange(self, _stop: int) -> int:
            return 0

    baseline = RecoverableHFMosaicVideoDataset(
        manifest, training=True, use_hflip=False, time_reverse=False
    )
    augmented = RecoverableHFMosaicVideoDataset(
        manifest, training=True, use_hflip=True, time_reverse=True
    )
    monkeypatch.setattr(baseline, "_rng_for_index", lambda _index: FixedRNG())
    monkeypatch.setattr(augmented, "_rng_for_index", lambda _index: FixedRNG())

    base_item = baseline[0]
    aug_item = augmented[0]
    torch.testing.assert_close(
        aug_item["inputs"], torch.flip(base_item["inputs"], dims=(0, 3))
    )
    torch.testing.assert_close(
        aug_item["data_samples"].gt_img,
        torch.flip(base_item["data_samples"].gt_img, dims=(0, 3)),
    )
    assert aug_item["data_samples"].metainfo["time_reversed"] is True
    assert aug_item["data_samples"].metainfo["hflip"] is True
