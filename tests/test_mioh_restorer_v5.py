# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import json

import numpy as np
import torch
from torch.nn import functional as F

from lada.models.mioh_restorer.model_v4 import shift2d
from lada.models.mioh_restorer.model_v5 import (
    FoldedPhaseShiftBank,
    MiohRestorerV5,
    MiohRestorerV5Config,
    MiohRestorerV5DecoderExportWrapper,
    MiohRestorerV5ExportWrapper,
    MiohRestorerV5StatefulExportWrapper,
    flatten_encoded_window,
)
from lada.models.mioh_restorer.curriculum_v5 import (
    V5_STAGES,
    previous_stage,
    stage_definition,
    stage_learning_rate,
)
from lada.models.mioh_restorer.losses_v5 import MiohRestorerV5Loss
from lada.models.mioh_restorer.runner_v5 import (
    MiohRestorerV5StreamingRunner,
    V5BucketHysteresis,
    cut_safe_window_indices,
    native_crop_for_center,
    native_tile_offsets,
    repair_isolated_mask_misses,
    required_v5_crop_size,
    select_v5_bucket,
    smooth_even_centers,
)
from lada.models.mioh_restorer.native_dataset_v5 import (
    MiohRestorerV5NativeDataset,
    V5BucketBatchSampler,
    crop_native_frame,
)


def tiny_config(*, quality: bool = True) -> MiohRestorerV5Config:
    return MiohRestorerV5Config(
        half_channels=8,
        quarter_channels=8,
        eighth_channels=8,
        sixteenth_channels=8,
        fusion_half_channels=8,
        fusion_quarter_channels=8,
        fusion_eighth_channels=8,
        fusion_sixteenth_channels=8,
        half_blocks=1,
        quarter_blocks=1,
        eighth_blocks=1,
        sixteenth_blocks=1,
        context_frames=5 if quality else 9,
        output_indices=(2, 3, 4, 5, 6) if quality else (4,),
    )


def sample_values(batch: int = 1, size: int = 32) -> torch.Tensor:
    values = torch.rand(batch, 9, 5, size, size)
    values[:, :, 3] = (values[:, :, 3] > 0.5).float()
    values[:, :, 4] = 1
    return values


def test_folded_phase_bank_exactly_matches_source_pixel_shifts() -> None:
    source = torch.rand(2, 3, 16, 16)
    packed = F.pixel_unshuffle(source, 2)
    bank = FoldedPhaseShiftBank(3)
    shifted = bank(packed)
    for index, (vertical, horizontal) in enumerate(bank.offsets):
        expected = F.pixel_unshuffle(
            shift2d(source, vertical, horizontal), 2
        )
        torch.testing.assert_close(shifted[:, index], expected, rtol=0, atol=0)


def test_v5_quality_is_identity_at_zero_initialization() -> None:
    model = MiohRestorerV5(tiny_config()).eval()
    values = sample_values(batch=2)
    with torch.no_grad():
        restored, confidence, base, texture = model.forward_components(values)
    torch.testing.assert_close(restored, values[:, 2:7, :3], rtol=0, atol=0)
    assert restored.shape == (2, 5, 3, 32, 32)
    assert confidence.shape == (2, 5, 1, 32, 32)
    assert torch.count_nonzero(base) == 0
    assert torch.count_nonzero(texture) == 0


def test_v5_shipping_is_single_center_output() -> None:
    model = MiohRestorerV5(tiny_config(quality=False)).eval()
    values = sample_values()
    with torch.no_grad():
        restored, confidence = model(values)
    torch.testing.assert_close(restored[:, 0], values[:, 4, :3], rtol=0, atol=0)
    assert confidence.shape == (1, 1, 1, 32, 32)


def test_v5_alignment_exposes_five_training_distributions() -> None:
    model = MiohRestorerV5(tiny_config(quality=False)).eval()
    with torch.no_grad():
        aligned, weights = model.alignment_diagnostics(sample_values())
    assert [value.shape[1] for value in weights] == [49, 9, 9, 9, 9]
    assert aligned.reliability.shape == (1, 1, 16, 16)
    assert aligned.entropy.shape == (1, 1, 16, 16)


def test_v5_split_decoder_matches_monolithic_model() -> None:
    model = MiohRestorerV5(tiny_config(quality=False)).eval()
    with torch.no_grad():
        model.decoder.base_head[-1].bias.fill_(0.1)
    values = sample_values()
    encoded = model.encode_window(values)
    decoder = MiohRestorerV5DecoderExportWrapper(model.decoder).eval()
    with torch.no_grad():
        expected = MiohRestorerV5ExportWrapper(model)(values.flatten(1, 2))
        actual = decoder(*flatten_encoded_window(encoded))
    for left, right in zip(expected, actual, strict=True):
        torch.testing.assert_close(left, right)


def test_v5_stateful_contract_matches_same_nine_frame_window() -> None:
    model = MiohRestorerV5(tiny_config(quality=False)).eval()
    values = sample_values()
    encoded = model.encode_window(values)
    old_states = tuple(
        torch.cat([frame[level] for frame in encoded[:-1]], dim=1)
        for level in range(5)
    )
    wrapper = MiohRestorerV5StatefulExportWrapper(model).eval()
    with torch.no_grad():
        expected = MiohRestorerV5ExportWrapper(model)(values.flatten(1, 2))
        actual = wrapper(values[:, -1], *old_states)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_native_bucket_selection_and_even_crop() -> None:
    assert select_v5_bucket(40, 60) == 128
    assert select_v5_bucket(100, 100) == 192
    assert select_v5_bucket(200, 180) == 384
    assert select_v5_bucket(400, 300) == 512
    crop = native_crop_for_center(
        11.2, 15.8, size=128, source_width=100, source_height=90
    )
    assert crop.x % 2 == 0 and crop.y % 2 == 0
    assert crop.pad_left > 0 and crop.pad_top > 0
    assert required_v5_crop_size(400, 300) > 512


def test_native_centres_interpolate_and_preserve_fold_phase() -> None:
    centres = smooth_even_centers([(10.0, 20.0), None, (18.0, 28.0)])
    assert len(centres) == 3
    assert all(horizontal % 2 == 0 and vertical % 2 == 0 for horizontal, vertical in centres)
    assert centres[0][0] <= centres[1][0] <= centres[2][0]


def test_large_native_roi_becomes_overlapping_tiles_without_resizing() -> None:
    offsets = native_tile_offsets(1000, 700, bucket=512, overlap=64)
    assert len(offsets) > 1
    assert all(x % 2 == 0 and y % 2 == 0 for x, y in offsets)
    assert min(x for x, _ in offsets) < max(x for x, _ in offsets)


def test_native_crop_pads_without_resizing() -> None:
    frame = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    cropped = crop_native_frame(frame, origin=(-2, -4), size=128)
    assert cropped.shape == (128, 128, 3)
    np.testing.assert_array_equal(cropped[0, 0], frame[0, 0])


def test_native_manifest_dataset_and_sampler_keep_bucket_shapes(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "train.jsonl"
    entries = []
    for index, bucket in enumerate((128, 128, 192)):
        entries.append(
            {
                "name": f"sample-{index}",
                "target_video": "target.mp4",
                "mask_video": "mask.mkv",
                "start_frame": 0,
                "bucket": bucket,
                "origins": [[0, 0]] * 9,
                "mask_reliability": [1.0] * 9,
                "mosaic_block_size": 12.0,
                "source_video_id": f"video-{index}",
            }
        )
    manifest.write_text("".join(json.dumps(value) + "\n" for value in entries))

    def fake_read(path, _start, pixel_format):
        size = 220
        if pixel_format == "gray":
            frame = np.zeros((size, size), dtype=np.uint8)
            frame[40:100, 60:120] = 255
        else:
            frame = np.full((size, size, 3), 127, dtype=np.uint8)
        return [frame.copy() for _ in range(9)]

    monkeypatch.setattr(MiohRestorerV5NativeDataset, "_read_frames", staticmethod(fake_read))
    monkeypatch.setattr(
        "lada.models.mioh_restorer.native_dataset_v5.addmosaic_base",
        lambda target, mask, *_args, **_kwargs: (target.copy(), mask.copy()),
    )
    dataset = MiohRestorerV5NativeDataset(
        manifest,
        degrade=False,
        horizontal_flip=False,
        time_reverse=False,
    )
    assert dataset[0]["inputs"].shape == (9, 5, 128, 128)
    assert dataset[2]["inputs"].shape == (9, 5, 192, 192)
    batches = list(V5BucketBatchSampler(dataset, batch_size=2, shuffle=False, drop_last=False))
    assert batches == [[0, 1], [2]]


def test_bucket_hysteresis_expands_at_boundary_and_delays_contraction() -> None:
    selector = V5BucketHysteresis(192, contraction_frames=18)
    assert selector.update(300, 300, at_window_boundary=False) == 192
    assert selector.update(300, 300, at_window_boundary=True) == 512
    for _ in range(17):
        assert selector.update(40, 40, at_window_boundary=True) == 512
    assert selector.update(40, 40, at_window_boundary=True) == 128


def test_cut_safe_windows_replicate_without_crossing_cut() -> None:
    assert cut_safe_window_indices(5, frame_count=12, cut_starts=(6,)) == (
        1, 2, 3, 4, 5, 5, 5, 5, 5
    )
    assert cut_safe_window_indices(6, frame_count=12, cut_starts=(6,)) == (
        6, 6, 6, 6, 6, 7, 8, 9, 10
    )


def test_mask_miss_repair_uses_nearest_direct_mask_and_lowers_reliability() -> None:
    masks = torch.zeros(1, 3, 1, 8, 8)
    masks[:, 0, :, 2:4, 2:4] = 1
    direct = torch.tensor([1.0, 0.0, 0.0]).reshape(1, 3, 1, 1, 1)
    origins = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]]])
    repaired, reliability = repair_isolated_mask_misses(masks, direct, origins)
    assert repaired[:, 1].sum() > 0
    assert reliability[0, 0, 0, 0, 0] == 1
    assert reliability[0, 1, 0, 0, 0] == 0.5


class _CenterIdentity(torch.nn.Module):
    def forward(self, values: torch.Tensor):
        return values[:, 4:5, :3], values[:, 4:5, 3:4]


def test_streaming_runner_covers_frames_and_respects_identity() -> None:
    frames = torch.rand(1, 7, 3, 32, 32)
    masks = torch.rand(1, 7, 1, 32, 32)
    reliability = torch.ones_like(masks)
    restored, confidence = MiohRestorerV5StreamingRunner(
        _CenterIdentity()
    ).restore(frames, masks, reliability, cut_starts=(3,))
    torch.testing.assert_close(restored, frames)
    torch.testing.assert_close(confidence, masks)


def test_v5_curriculum_is_six_independent_stages() -> None:
    assert [stage.stage_id for stage in V5_STAGES] == [1, 2, 3, 4, 5, 6]
    assert stage_definition("detail_recovery").stage_id == 4
    assert previous_stage(stage_definition(1)) is None
    assert previous_stage(stage_definition(6)).stage_id == 5
    assert stage_learning_rate(
        stage_definition(3), 1, total_steps=15_000
    ) < stage_learning_rate(stage_definition(3), 500, total_steps=15_000)


def test_v5_loss_backpropagates_and_requires_aligned_time_only_in_stage5() -> None:
    shape = (1, 5, 3, 16, 16)
    source = torch.rand(shape)
    target = torch.rand(shape)
    mask = torch.ones(1, 5, 1, 16, 16)
    base = torch.zeros(shape, requires_grad=True)
    texture = torch.zeros(shape, requires_grad=True)
    confidence = torch.full((1, 5, 1, 16, 16), 0.5, requires_grad=True)
    restored = source + mask * (base + confidence * texture)
    total, stats = MiohRestorerV5Loss(stage=4)(
        restored, confidence, base, texture, target, source, mask
    )
    total.backward()
    assert base.grad is not None and texture.grad is not None
    assert confidence.grad is not None
    assert set(stats) >= {"wavelet", "high_frequency", "confidence_mean"}
