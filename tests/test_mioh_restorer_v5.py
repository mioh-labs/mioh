# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from lada.models.mioh_restorer.adversarial import (
    SpectralUNetDiscriminator,
    TemporalPatchDiscriminator,
    discriminator_feature_matching_loss,
    discriminator_hinge_loss,
    discriminator_relativistic_pair_loss,
    discriminator_roi_patch_weights,
    generator_hinge_loss,
    generator_relativistic_pair_loss,
    roi_temporal_discriminator_input,
)
from lada.models.mioh_restorer.model_v5 import (
    FoldedPhaseShiftBank,
    MiohRestorerV5,
    MiohRestorerV5Config,
    MiohRestorerV5DecoderExportWrapper,
    MiohRestorerV5ExportWrapper,
    MiohRestorerV5StatefulExportWrapper,
    flatten_encoded_window,
    shift2d,
)
from lada.models.mioh_restorer.curriculum_v5 import (
    V5_STAGES,
    V5LossWeights,
    previous_stage,
    stage_definition,
    stage_learning_rate,
)
from lada.models.mioh_restorer.curriculum_v5_hq import (
    V5_HQ_STAGES,
    hq_learning_rate,
    hq_stage_definition,
)
from lada.models.mioh_restorer.losses_v5 import (
    MiohRestorerV5Loss,
    high_frequency,
    mask_boundary_band,
    masked_local_correlation,
    masked_projection_statistics,
)
from lada.models.mioh_restorer.model_v5_hq import (
    MiohRestorerV5HQ,
    MiohRestorerV5HQConfig,
    MiohRestorerV5HQExportWrapper,
    feather_texture_compositor_mask,
)
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
from lada.models.mioh_restorer.supervision_v5 import (
    V5PerceptualLoss,
    flow_aligned_temporal_tensors,
    known_motion_alignment_loss,
    natural_alignment_losses,
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


def test_v5_hq_roi_patchgan_input_preserves_generator_gradients() -> None:
    video = torch.rand(1, 5, 3, 128, 160, requires_grad=True)
    target = torch.rand_like(video)
    masks = torch.zeros(1, 5, 1, 128, 160)
    masks[:, :, :, 32:96, 48:112] = 1
    masks[:, 2, :, 20:108, 36:124] = 1

    spatial = roi_temporal_discriminator_input(
        video,
        target,
        masks,
        image_size=96,
        minimum_crop_size=80,
    )
    assert spatial.shape == (4, 7, 96, 96)
    assert torch.count_nonzero(spatial[:, 3:6]) > 0
    assert spatial[:, :3].amin() >= -1
    assert spatial[:, :3].amax() <= 1
    spatial[:, 3:6].mean().backward()
    assert video.grad is not None
    assert torch.count_nonzero(video.grad) > 0

    temporal = roi_temporal_discriminator_input(
        video.detach(),
        target,
        masks,
        image_size=96,
        minimum_crop_size=80,
        include_motion=True,
    )
    assert torch.count_nonzero(temporal[:, 3:6]) > 0


def test_v5_hq_roi_patchgan_losses_train_both_sides() -> None:
    target = torch.rand(1, 5, 3, 96, 96)
    fake = torch.rand_like(target, requires_grad=True)
    masks = torch.zeros(1, 5, 1, 96, 96)
    masks[:, :, :, 20:76, 24:72] = 1
    real_input = roi_temporal_discriminator_input(
        target, target, masks, image_size=64, minimum_crop_size=64
    )
    fake_input = roi_temporal_discriminator_input(
        fake, target, masks, image_size=64, minimum_crop_size=64
    )
    torch.testing.assert_close(fake_input[:, :3], real_input[:, :3])
    assert torch.count_nonzero(fake_input[:, 3:6] - real_input[:, 3:6]) > 0
    discriminator = TemporalPatchDiscriminator(base_channels=4)
    real_logits = discriminator(real_input)
    fake_logits = discriminator(fake_input.detach())
    weights = discriminator_roi_patch_weights(real_input, real_logits)
    discriminator_loss = discriminator_hinge_loss(real_logits, fake_logits, weights)
    discriminator_loss.backward()
    assert any(parameter.grad is not None for parameter in discriminator.parameters())

    discriminator.zero_grad(set_to_none=True)
    discriminator.requires_grad_(False)
    generator_logits = discriminator(fake_input)
    generator_loss = generator_hinge_loss(
        generator_logits,
        discriminator_roi_patch_weights(fake_input, generator_logits),
    )
    generator_loss.backward()
    assert fake.grad is not None
    assert torch.count_nonzero(fake.grad) > 0


def test_v5_hq_spectral_unet_discriminator_is_dense_and_trains_both_sides() -> None:
    target = torch.rand(1, 3, 3, 64, 64)
    fake = (target + 0.03 * torch.randn_like(target)).clamp(0, 1)
    fake.requires_grad_(True)
    masks = torch.zeros(1, 3, 1, 64, 64)
    masks[..., 16:48, 16:48] = 1
    real_input = roi_temporal_discriminator_input(
        target, target, masks, image_size=64, minimum_crop_size=64
    )
    fake_input = roi_temporal_discriminator_input(
        fake, target, masks, image_size=64, minimum_crop_size=64
    )
    discriminator = SpectralUNetDiscriminator(base_channels=4)
    real_logits = discriminator(real_input)
    fake_logits = discriminator(fake_input.detach())
    assert real_logits.shape == (2, 1, 64, 64)
    weights = discriminator_roi_patch_weights(real_input, real_logits)
    discriminator_hinge_loss(real_logits, fake_logits, weights).backward()
    assert any(parameter.grad is not None for parameter in discriminator.parameters())

    discriminator.zero_grad(set_to_none=True)
    discriminator.requires_grad_(False)
    generator_logits = discriminator(fake_input)
    generator_hinge_loss(
        generator_logits,
        discriminator_roi_patch_weights(fake_input, generator_logits),
    ).backward()
    assert fake.grad is not None
    assert torch.count_nonzero(fake.grad) > 0


def test_v5_hq_frozen_spectral_unet_still_trains_generator() -> None:
    target = torch.rand(1, 3, 3, 64, 64)
    fake = (target + 0.03 * torch.randn_like(target)).clamp(0, 1)
    fake.requires_grad_(True)
    masks = torch.zeros(1, 3, 1, 64, 64)
    masks[..., 16:48, 16:48] = 1
    fake_input = roi_temporal_discriminator_input(
        fake, target, masks, image_size=64, minimum_crop_size=64,
        normalize_secondary_rms=True,
    )
    discriminator = SpectralUNetDiscriminator(base_channels=4).eval()
    discriminator.requires_grad_(False)
    logits = discriminator(fake_input)
    generator_hinge_loss(
        logits, discriminator_roi_patch_weights(fake_input, logits)
    ).backward()
    assert fake.grad is not None and torch.count_nonzero(fake.grad) > 0
    assert all(parameter.grad is None for parameter in discriminator.parameters())


def test_v5_hq_roi_patchgan_feature_matching_trains_generator() -> None:
    target = torch.rand(1, 5, 3, 96, 96)
    fake = (target + torch.randn_like(target) * 0.03).clamp(0, 1)
    fake.requires_grad_(True)
    masks = torch.zeros(1, 5, 1, 96, 96)
    masks[:, :, :, 20:76, 24:72] = 1
    real_input = roi_temporal_discriminator_input(
        target,
        target,
        masks,
        image_size=64,
        minimum_crop_size=64,
        normalize_secondary_rms=True,
    )
    fake_input = roi_temporal_discriminator_input(
        fake,
        target,
        masks,
        image_size=64,
        minimum_crop_size=64,
        normalize_secondary_rms=True,
    )
    discriminator = TemporalPatchDiscriminator(base_channels=4)
    discriminator.requires_grad_(False)
    with torch.no_grad():
        _, real_features = discriminator.forward_features(real_input)
    _, fake_features = discriminator.forward_features(fake_input)
    loss = discriminator_feature_matching_loss(
        real_features,
        fake_features,
        fake_input,
    )
    assert float(loss.detach()) > 0
    loss.backward()
    assert fake.grad is not None
    assert torch.count_nonzero(fake.grad) > 0
    assert all(parameter.grad is None for parameter in discriminator.parameters())


def test_v5_hq_roi_patchgan_weights_exclude_unedited_context() -> None:
    discriminator_input = torch.zeros(1, 7, 64, 64)
    discriminator_input[:, 6, 24:40, 24:40] = 1
    logits = torch.zeros(1, 1, 8, 8)
    weights = discriminator_roi_patch_weights(discriminator_input, logits)
    assert weights.shape == logits.shape
    assert 0 < int(weights.sum()) < weights.numel()

    real_logits = torch.zeros_like(logits, requires_grad=True)
    fake_logits = torch.zeros_like(logits, requires_grad=True)
    loss = discriminator_hinge_loss(real_logits, fake_logits, weights)
    loss.backward()
    assert torch.count_nonzero(real_logits.grad[weights == 0]) == 0
    assert torch.count_nonzero(fake_logits.grad[weights == 0]) == 0


def test_v5_hq_relativistic_pair_losses_train_both_sides() -> None:
    weights = torch.zeros(1, 1, 4, 4)
    weights[..., 1:3, 1:3] = 1
    real = torch.randn(1, 1, 4, 4, requires_grad=True)
    fake = torch.randn(1, 1, 4, 4, requires_grad=True)
    discriminator_relativistic_pair_loss(real, fake, weights).backward()
    assert real.grad is not None and fake.grad is not None
    assert torch.count_nonzero(real.grad[weights == 0]) == 0
    assert torch.count_nonzero(fake.grad[weights == 0]) == 0

    real_generator = real.detach()
    fake_generator = fake.detach().requires_grad_(True)
    generator_relativistic_pair_loss(
        real_generator,
        fake_generator,
        weights,
    ).backward()
    assert fake_generator.grad is not None
    assert torch.count_nonzero(fake_generator.grad[weights > 0]) > 0
    assert torch.count_nonzero(fake_generator.grad[weights == 0]) == 0


def test_v5_hq_roi_patchgan_rms_normalization_removes_amplitude_shortcut() -> None:
    target = torch.full((1, 5, 3, 64, 64), 0.5)
    masks = torch.ones(1, 5, 1, 64, 64)
    checker = (torch.arange(64)[:, None] + torch.arange(64)[None, :]) % 2
    checker = (checker.float() * 2.0 - 1.0)[None, None, None]
    checker = checker.expand(1, 5, 3, -1, -1)
    weak = target + checker * 0.01
    strong = target + checker * 0.04
    weak_input = roi_temporal_discriminator_input(
        weak,
        target,
        masks,
        image_size=64,
        minimum_crop_size=64,
        normalize_secondary_rms=True,
    )
    strong_input = roi_temporal_discriminator_input(
        strong,
        target,
        masks,
        image_size=64,
        minimum_crop_size=64,
        normalize_secondary_rms=True,
    )
    torch.testing.assert_close(
        weak_input[:, 3:6],
        strong_input[:, 3:6],
        rtol=1e-4,
        atol=2e-5,
    )


def test_v5_hq_roi_patchgan_can_expose_candidate_rgb() -> None:
    target = torch.full((1, 5, 3, 64, 64), 0.5)
    fake = target.clone().requires_grad_(True)
    fake.data[:, :, :, 20:44, 20:44] += 0.1
    masks = torch.zeros(1, 5, 1, 64, 64)
    masks[:, :, :, 16:48, 16:48] = 1
    conditioned = roi_temporal_discriminator_input(
        fake,
        target,
        masks,
        image_size=64,
        minimum_crop_size=64,
    )
    candidate = roi_temporal_discriminator_input(
        fake,
        target,
        masks,
        image_size=64,
        minimum_crop_size=64,
        condition_on_target_rgb=False,
    )
    assert torch.count_nonzero(conditioned[:, :3] - candidate[:, :3]) > 0
    candidate[:, :3].mean().backward()
    assert fake.grad is not None
    assert torch.count_nonzero(fake.grad) > 0


def test_v5_hq_roi_patchgan_skips_empty_masks() -> None:
    values = torch.rand(1, 5, 3, 64, 64)
    result = roi_temporal_discriminator_input(
        values,
        values,
        torch.zeros(1, 5, 1, 64, 64),
        image_size=48,
    )
    assert result.shape == (0, 7, 48, 48)


class _ZeroSPyNet(torch.nn.Module):
    def forward(self, reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        del candidate
        return reference.new_zeros(reference.shape[0], 2, *reference.shape[-2:])


class _TinyRecurrentBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spynet = _ZeroSPyNet()
        self.gain = torch.nn.Parameter(torch.tensor(0.05))

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return rgb + self.gain * torch.tanh(rgb)


def tiny_hq_model(
    *,
    raw_temporal_candidates: bool = False,
    raw_temporal_encoder_channels: int = 0,
    raw_temporal_nearest_warp: bool = False,
    raw_temporal_zero_input: bool = False,
) -> MiohRestorerV5HQ:
    config = MiohRestorerV5HQConfig(
        backbone_channels=8,
        backbone_blocks=1,
        detail_channels=8,
        attention_channels=4,
        raw_temporal_candidates=raw_temporal_candidates,
        raw_temporal_encoder_channels=raw_temporal_encoder_channels,
        raw_temporal_nearest_warp=raw_temporal_nearest_warp,
        raw_temporal_zero_input=raw_temporal_zero_input,
    )
    return MiohRestorerV5HQ(config, backbone=_TinyRecurrentBackbone())


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


def test_native_crop_handles_rectangle_entirely_outside_frame() -> None:
    frame = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
    above = crop_native_frame(frame, origin=(-400, -400), size=128)
    below = crop_native_frame(frame, origin=(400, 400), size=128)
    assert above.shape == below.shape == (128, 128, 3)
    np.testing.assert_array_equal(above[64, 64], frame[0, 0])
    np.testing.assert_array_equal(below[64, 64], frame[-1, -1])

    mask = np.ones((8, 10), dtype=np.uint8) * 255
    outside_mask = crop_native_frame(
        mask, origin=(-400, -400), size=128, mask=True
    )
    assert outside_mask.shape == (128, 128)
    assert np.count_nonzero(outside_mask) == 0


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


def test_native_validation_mosaic_is_repeatable(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "name": "repeatable",
                "target_video": "target.mp4",
                "mask_video": "mask.mkv",
                "start_frame": 0,
                "bucket": 128,
                "origins": [[0, 0]] * 9,
                "mask_reliability": [1.0] * 9,
                "mosaic_block_size": 16.0,
                "source_video_id": "video",
            }
        )
        + "\n"
    )

    def fake_read(_path, _start, pixel_format):
        if pixel_format == "gray":
            frame = np.zeros((128, 128), dtype=np.uint8)
            frame[16:112, 16:112] = 255
        else:
            yy, xx = np.mgrid[:128, :128]
            frame = np.stack((xx, yy, (xx + yy) // 2), axis=-1).astype(np.uint8)
        return [frame.copy() for _ in range(9)]

    monkeypatch.setattr(MiohRestorerV5NativeDataset, "_read_frames", staticmethod(fake_read))
    dataset = MiohRestorerV5NativeDataset(
        manifest,
        output_indices=(2, 3, 4, 5, 6),
        degrade=False,
        horizontal_flip=False,
        time_reverse=False,
        deterministic=True,
    )
    first = dataset[0]
    _ = random.random()
    _ = np.random.rand()
    second = dataset[0]
    torch.testing.assert_close(first["inputs"], second["inputs"], rtol=0, atol=0)
    torch.testing.assert_close(first["masks"], second["masks"], rtol=0, atol=0)
    assert "loss_masks" in first
    assert float(first["loss_masks"].sum()) < float(first["masks"].sum())


def test_native_dataset_can_override_training_block_size(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "training.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "name": "small-block",
                "target_video": "target.mp4",
                "mask_video": "mask.mkv",
                "start_frame": 0,
                "bucket": 128,
                "origins": [[0, 0]] * 9,
                "mask_reliability": [1.0] * 9,
                "mosaic_block_size": 30.0,
                "source_video_id": "video",
            }
        )
        + "\n"
    )

    def fake_read(_path, _start, pixel_format):
        if pixel_format == "gray":
            frame = np.ones((128, 128), dtype=np.uint8) * 255
        else:
            frame = np.ones((128, 128, 3), dtype=np.uint8) * 127
        return [frame.copy() for _ in range(9)]

    seen = []

    def fake_parameters(block_size, **_kwargs):
        seen.append(block_size)
        return 8, "squa_avg", 1.0, 0

    monkeypatch.setattr(MiohRestorerV5NativeDataset, "_read_frames", staticmethod(fake_read))
    monkeypatch.setattr(
        "lada.models.mioh_restorer.native_dataset_v5.get_random_parameters_by_block_size",
        fake_parameters,
    )
    dataset = MiohRestorerV5NativeDataset(
        manifest,
        degrade=False,
        horizontal_flip=False,
        time_reverse=False,
        deterministic=True,
        mosaic_block_size_range=(6.0, 12.0),
    )
    _ = dataset[0]
    assert 6.0 <= seen[0] <= 12.0


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
        restored,
        confidence,
        base,
        texture,
        target,
        source,
        mask,
        perceptual=restored.new_tensor(0.25),
    )
    total.backward()
    assert base.grad is not None and texture.grad is not None
    assert confidence.grad is not None
    assert set(stats) >= {"wavelet", "high_frequency", "confidence_mean"}


def test_v5_hq_stage4_supervises_flow_aligned_roi_boundary() -> None:
    shape = (1, 5, 3, 16, 16)
    source = torch.rand(shape)
    target = torch.rand(shape)
    mask = torch.zeros(1, 5, 1, 16, 16)
    mask[..., 4:12, 4:12] = 1
    base = torch.zeros(shape, requires_grad=True)
    texture = torch.zeros(shape, requires_grad=True)
    confidence = torch.full((1, 5, 1, 16, 16), 0.5, requires_grad=True)
    restored = source + mask * (base + confidence * texture)
    aligned_restored, aligned_target, valid = flow_aligned_temporal_tensors(
        restored, target, mask, radius=1, scale=2
    )
    stage = hq_stage_definition(4)
    total, stats = MiohRestorerV5Loss(
        weights=stage.loss,
        boundary_temporal_weight=stage.boundary_temporal_weight,
    )(
        restored,
        confidence,
        base,
        texture,
        target,
        source,
        mask,
        aligned_previous_restored=aligned_restored,
        aligned_previous_target=aligned_target,
        temporal_valid=valid,
        perceptual=restored.new_tensor(0.25),
    )
    total.backward()
    assert stage.boundary_temporal_weight > 0
    assert stats["boundary_temporal"] > 0
    assert base.grad is not None


def test_v5_hf_ablation_supervises_final_amplitude_and_correlation() -> None:
    shape = (1, 5, 3, 16, 16)
    target = torch.rand(shape)
    source = torch.zeros_like(target)
    mask = torch.ones(1, 5, 1, 16, 16)
    base = torch.zeros(shape, requires_grad=True)
    texture = torch.zeros(shape, requires_grad=True)
    confidence = torch.full((1, 5, 1, 16, 16), 0.5, requires_grad=True)
    restored = source + mask * (base + confidence * texture)
    total, stats = MiohRestorerV5Loss(
        weights=hq_stage_definition(3).loss,
        base_loss_weight_override=0.0,
        candidate_loss_weight_override=0.0,
        confidence_loss_weight_override=0.03,
        hf_amplitude_weight=0.05,
        hf_correlation_weight=0.02,
        supervise_final_high_frequency=True,
    )(
        restored,
        confidence,
        base,
        texture,
        target,
        source,
        mask,
        perceptual=restored.new_tensor(0.25),
    )
    total.backward()
    assert stats["hf_amplitude_ratio"] < 0.01
    assert stats["hf_correlation"] == pytest.approx(0.0)
    assert texture.grad is not None and torch.count_nonzero(texture.grad) > 0


def test_v5_hf_ablation_disables_duplicate_candidate_loss() -> None:
    shape = (1, 5, 3, 8, 8)
    restored = torch.ones(shape)
    target = torch.zeros_like(restored)
    source = torch.zeros_like(restored)
    mask = torch.ones(1, 5, 1, 8, 8)
    confidence = torch.full_like(mask, 0.5)
    base = torch.zeros_like(restored)
    texture = torch.zeros_like(restored)
    candidate_only = V5LossWeights(
        reconstruction=0.0,
        candidate=1.0,
        base=0.0,
        high_frequency=0.0,
        wavelet=0.0,
        gradient=0.0,
        perceptual=0.0,
        temporal=0.0,
        confidence=0.0,
        confidence_regularization=0.0,
    )
    enabled, _ = MiohRestorerV5Loss(
        weights=candidate_only,
        supervise_final_high_frequency=True,
    )(restored, confidence, base, texture, target, source, mask)
    disabled, _ = MiohRestorerV5Loss(
        weights=candidate_only,
        candidate_loss_weight_override=0.0,
        supervise_final_high_frequency=True,
    )(restored, confidence, base, texture, target, source, mask)
    assert float(enabled) > 0.9
    assert float(disabled) == pytest.approx(0.0)


def test_v5_hf_projection_statistics_separate_signal_and_orthogonal_noise() -> None:
    target = torch.tensor((-1.0, 1.0, -1.0, 1.0)).reshape(1, 1, 1, 1, 4)
    orthogonal = torch.tensor((-1.0, -1.0, 1.0, 1.0)).reshape_as(target)
    prediction = 0.5 * target + 2.0 * orthogonal
    mask = torch.ones_like(target)
    projection, orthogonal_rms, orthogonal_energy = masked_projection_statistics(
        prediction, target, mask
    )
    assert float(projection) == pytest.approx(0.5)
    assert float(orthogonal_rms) == pytest.approx(2.0)
    assert float(orthogonal_energy) == pytest.approx(4.0)


def test_v5_local_hf_correlation_is_patchwise_and_backpropagates() -> None:
    target = torch.rand(1, 2, 3, 16, 16)
    mask = torch.zeros(1, 2, 1, 16, 16)
    mask[..., 2:14, 2:14] = 1
    identical = masked_local_correlation(target, target, mask, patch_size=8)
    assert float(identical) == pytest.approx(1.0, abs=1e-4)

    prediction = (target + 0.2 * torch.randn_like(target)).requires_grad_(True)
    correlation = masked_local_correlation(
        prediction, target, mask, patch_size=8
    )
    (1.0 - correlation).backward()
    assert float(correlation.detach()) < float(identical)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_v5_residual_local_hf_correlation_supervises_only_the_correction() -> None:
    shape = (1, 2, 3, 16, 16)
    reference = torch.rand(shape)
    target = reference + 0.1 * torch.randn(shape)
    mask = torch.ones(1, 2, 1, 16, 16)
    confidence = torch.full_like(mask, 0.5)
    zero = V5LossWeights(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    criterion = MiohRestorerV5Loss(
        weights=zero,
        hf_local_correlation_weight=1.0,
        hf_local_correlation_patch_size=8,
        hf_local_correlation_on_residual=True,
        supervise_final_high_frequency=True,
    )
    aligned = (reference + 0.5 * (target - reference)).requires_grad_(True)
    total, stats = criterion(
        aligned,
        confidence,
        torch.zeros_like(aligned),
        torch.zeros_like(aligned),
        target,
        reference,
        mask,
        hf_reference=reference,
    )
    total.backward()
    assert stats["hf_residual_local_correlation"] == pytest.approx(1.0, abs=1e-4)
    assert float(total.detach()) == pytest.approx(0.0, abs=1e-4)
    assert aligned.grad is not None and torch.isfinite(aligned.grad).all()

    with pytest.raises(ValueError, match="requires an HF reference"):
        criterion(
            target,
            confidence,
            torch.zeros_like(target),
            torch.zeros_like(target),
            target,
            reference,
            mask,
        )


def test_v5_normalized_residual_hf_reconstruction_has_unit_scale() -> None:
    shape = (1, 2, 3, 16, 16)
    reference = torch.rand(shape)
    target = reference + 0.01 * torch.randn(shape)
    restored = reference.clone().requires_grad_(True)
    mask = torch.ones(1, 2, 1, 16, 16)
    confidence = torch.full_like(mask, 0.5)
    zero = V5LossWeights(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    total, stats = MiohRestorerV5Loss(
        weights=zero,
        hf_residual_reconstruction_weight=1.0,
        supervise_final_high_frequency=True,
    )(
        restored,
        confidence,
        torch.zeros_like(restored),
        torch.zeros_like(restored),
        target,
        reference,
        mask,
        hf_reference=reference,
    )
    total.backward()
    assert 0.1 < stats["hf_residual_reconstruction"] < 2.0
    assert stats["hf_residual_target_rms"] > 0
    assert restored.grad is not None and torch.isfinite(restored.grad).all()


def test_v5_correction_target_objective_rewards_gt_projection_and_penalizes_noise() -> None:
    shape = (1, 2, 3, 16, 16)
    target = torch.rand(shape)
    reference = torch.zeros_like(target)
    restored = torch.zeros_like(target, requires_grad=True)
    mask = torch.ones(1, 2, 1, 16, 16)
    confidence = torch.full_like(mask, 0.5)
    zero = V5LossWeights(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    criterion = MiohRestorerV5Loss(
        weights=zero,
        hf_correction_target_projection_weight=0.25,
        hf_correction_target_orthogonal_energy_weight=1.0,
        supervise_final_high_frequency=True,
    )
    total, stats = criterion(
        restored,
        confidence,
        torch.zeros_like(restored),
        torch.zeros_like(restored),
        target,
        reference,
        mask,
        hf_reference=reference,
    )
    total.backward()
    assert stats["hf_correction_target_projection_gain"] == pytest.approx(0.0)
    assert stats["hf_correction_target_orthogonal_energy_ratio"] == pytest.approx(0.0)
    assert restored.grad is not None and torch.isfinite(restored.grad).all()

    updated = (-restored.grad).detach()
    updated_hf = high_frequency(updated)
    target_hf = high_frequency(target)
    projection, _, _ = masked_projection_statistics(updated_hf, target_hf, mask)
    assert float(projection) > 0

    with pytest.raises(ValueError, match="requires an HF reference"):
        criterion(
            restored,
            confidence,
            torch.zeros_like(restored),
            torch.zeros_like(restored),
            target,
            reference,
            mask,
        )


def test_v5_guard_ring_identity_loss_restores_clean_context() -> None:
    shape = (1, 2, 3, 8, 8)
    restored = torch.zeros(shape, requires_grad=True)
    restored.data[..., :2, :] = 0.5
    target = torch.zeros_like(restored)
    source = torch.zeros_like(restored)
    mask = torch.ones(1, 2, 1, 8, 8)
    guard_ring = torch.zeros_like(mask)
    guard_ring[..., :2, :] = 1
    confidence = torch.full_like(mask, 0.5)
    zero = V5LossWeights(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    total, stats = MiohRestorerV5Loss(
        weights=zero,
        guard_ring_identity_weight=1.0,
        supervise_final_high_frequency=True,
    )(
        restored,
        confidence,
        torch.zeros_like(restored),
        torch.zeros_like(restored),
        target,
        source,
        mask,
        guard_ring_mask=guard_ring,
    )
    total.backward()
    assert float(total.detach()) > 0
    assert stats["guard_ring_identity"] > 0
    assert restored.grad is not None and torch.count_nonzero(restored.grad) > 0


def test_v5_perceptual_native_mode_crops_without_resizing(monkeypatch) -> None:
    criterion = V5PerceptualLoss.__new__(V5PerceptualLoss)
    torch.nn.Module.__init__(criterion)
    criterion.image_size = 8
    criterion.preserve_native_scale = True
    restored = torch.rand(1, 5, 3, 16, 20)
    target = torch.rand_like(restored)
    mask = torch.zeros(1, 5, 1, 16, 20)
    mask[..., 6:10, 9:13] = 1
    cropped = criterion._native_roi_crop(restored, target, mask)
    assert cropped[0].shape == (1, 5, 3, 8, 8)
    assert cropped[2].shape == (1, 5, 1, 8, 8)


def test_v5_boundary_band_excludes_flat_interior_and_background() -> None:
    mask = torch.zeros(1, 2, 1, 24, 24)
    mask[..., 6:18, 6:18] = 1
    band = mask_boundary_band(mask, radius=2)
    assert band[0, 0, 0, 12, 12].item() == 0
    assert band[0, 0, 0, 0, 0].item() == 0
    assert band[0, 0, 0, 6, 12].item() > 0


def test_v5_known_motion_supervision_is_teacher_free_and_backpropagates() -> None:
    model = MiohRestorerV5(tiny_config()).train()
    values = sample_values()
    loss, stats = known_motion_alignment_loss(
        model, values, maximum_translation=8
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert stats["known_motion"] > 0
    assert model.decoder.alignment.phase_offset_bias.grad is not None


def test_v5_natural_alignment_self_supervision_backpropagates() -> None:
    model = MiohRestorerV5(tiny_config()).train()
    natural, feature, stats = natural_alignment_losses(model, sample_values())
    (natural + feature).backward()
    assert torch.isfinite(natural) and torch.isfinite(feature)
    assert stats["natural_motion"] > 0
    assert model.encoder.half_stage[0].weight.grad is not None


def test_v5_clean_gt_temporal_correspondence_preserves_gradient() -> None:
    restored = torch.rand(1, 5, 3, 32, 32, requires_grad=True)
    target = torch.rand_like(restored)
    mask = torch.ones(1, 5, 1, 32, 32)
    aligned_restored, aligned_target, valid = flow_aligned_temporal_tensors(
        restored, target, mask
    )
    assert aligned_restored.shape == (1, 4, 3, 32, 32)
    assert aligned_target.shape == aligned_restored.shape
    assert valid.shape == (1, 4, 1, 32, 32)
    aligned_restored.mean().backward()
    assert restored.grad is not None


def test_v5_hq_uses_recurrent_baseline_only_inside_roi() -> None:
    model = tiny_hq_model().eval()
    values = sample_values(size=16)
    with torch.no_grad():
        restored, confidence, base, texture = model.forward_components(values)
        recurrent = model.backbone(values[:, :, :3])[:, 2:7]
    source = values[:, 2:7, :3]
    mask = values[:, 2:7, 3:4]
    expected = source + mask * (recurrent - source)
    torch.testing.assert_close(restored, expected)
    torch.testing.assert_close((1 - mask) * restored, (1 - mask) * source)
    assert restored.shape == (1, 5, 3, 16, 16)
    assert confidence.shape == (1, 5, 1, 16, 16)
    assert torch.count_nonzero(texture) == 0
    torch.testing.assert_close(base, recurrent - source)


def test_v5_hq_refiner_and_flow_attention_backpropagate() -> None:
    model = tiny_hq_model().train()
    values = sample_values(size=16)
    restored, confidence, base, texture = model.forward_components(values)
    loss = restored.mean() + confidence.mean() + base.square().mean() + texture.mean()
    loss.backward()
    assert model.detail_encoder[0].weight.grad is not None
    assert model.deformable_attention.offset[-1].weight.grad is not None
    assert model.backbone.gain.grad is not None


def test_v5_hq_texture_mask_keeps_guard_ring_at_backbone_output() -> None:
    model = tiny_hq_model().eval()
    values = sample_values(size=16)
    compositor_mask = values[:, 2:7, 3:4].clone()
    effective_mask = compositor_mask.clone()
    effective_mask[..., :4, :] = 0
    with torch.no_grad():
        model.texture_head[-1].bias.fill_(0.25)
        restored, confidence, base, _texture = model.forward_components(
            values,
            texture_compositor_mask=effective_mask,
        )
    source = values[:, 2:7, :3]
    backbone_output = source + compositor_mask * base
    guard_ring = (compositor_mask - effective_mask).clamp(0, 1)
    torch.testing.assert_close(
        restored * guard_ring,
        backbone_output * guard_ring,
    )
    assert torch.count_nonzero(
        (restored - backbone_output) * effective_mask
    ) > 0
    assert torch.all(confidence > 0)


def test_v5_hq_texture_mask_feather_is_bounded_by_compositor() -> None:
    effective = torch.zeros(1, 1, 1, 9, 9)
    effective[..., 3:6, 3:6] = 1
    compositor = torch.ones_like(effective)
    compositor[..., :2, :] = 0
    feathered = feather_texture_compositor_mask(
        effective,
        compositor,
        radius=2,
    )
    torch.testing.assert_close(feathered[..., 3:6, 3:6], torch.ones(1, 1, 1, 3, 3))
    assert torch.all(feathered <= compositor)
    assert 0 < float(feathered[..., 2, 4]) < 1
    assert torch.count_nonzero(feathered[..., 0, :]) == 0


def test_v5_hq_raw_temporal_candidates_reach_fusion_without_breaking_identity() -> None:
    model = tiny_hq_model(raw_temporal_candidates=True).eval()
    values = sample_values(size=16)
    for frame in range(values.shape[1]):
        values[:, frame, :3].fill_(frame / 10)
    captured: list[torch.Tensor] = []
    handle = model.fusion[0].register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach())
    )
    with torch.no_grad():
        restored, _confidence, _base, texture = model.forward_components(values)
        recurrent = model.backbone(values[:, :, :3])[:, 2:7]
    handle.remove()
    mask = values[:, 2:7, 3:4]
    source = values[:, 2:7, :3]
    torch.testing.assert_close(restored, source + mask * (recurrent - source))
    assert torch.count_nonzero(texture) == 0
    assert captured and captured[0].shape[1] == 8 * 3 + 8 + 18
    # For output frame 2 the raw order is identity, previous, next.
    raw_candidates = captured[0][:, -18:-9]
    assert float(raw_candidates[:, :3].mean()) == pytest.approx(0.2)
    assert float(raw_candidates[:, 3:6].mean()) == pytest.approx(0.1)
    assert float(raw_candidates[:, 6:9].mean()) == pytest.approx(0.3)


def test_v5_hq_encodes_raw_candidates_before_flow_warp() -> None:
    model = tiny_hq_model(
        raw_temporal_candidates=True,
        raw_temporal_encoder_channels=4,
    ).eval()
    assert model.raw_temporal_encoder is not None
    values = sample_values(size=16)
    captured: list[torch.Tensor] = []
    handle = model.fusion[0].register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach())
    )
    with torch.no_grad():
        restored, _confidence, _base, texture = model.forward_components(values)
        recurrent = model.backbone(values[:, :, :3])[:, 2:7]
    handle.remove()
    mask = values[:, 2:7, 3:4]
    source = values[:, 2:7, :3]
    torch.testing.assert_close(restored, source + mask * (recurrent - source))
    assert torch.count_nonzero(texture) == 0
    assert captured and captured[0].shape[1] == 8 * 3 + 8 + 3 * 4 * 2


def test_v5_hq_nearest_raw_warp_preserves_observed_sample_values() -> None:
    raw = torch.tensor(
        [[[[[0.0, 1.0], [2.0, 3.0]]]]],
        dtype=torch.float32,
    )
    flow = torch.full((1, 1, 2, 2, 2), 0.5, dtype=torch.float32)
    bilinear = MiohRestorerV5HQ._aligned_raw_candidates(
        raw, (0,), flow, interpolation="bilinear"
    )
    nearest = MiohRestorerV5HQ._aligned_raw_candidates(
        raw, (0,), flow, interpolation="nearest"
    )
    assert not torch.equal(bilinear, nearest)
    assert set(nearest.flatten().tolist()).issubset(set(raw.flatten().tolist()))


def test_v5_hq_zero_raw_negative_control_keeps_raw_fusion_channels_zero() -> None:
    model = tiny_hq_model(
        raw_temporal_candidates=True,
        raw_temporal_zero_input=True,
    ).eval()
    captured: list[torch.Tensor] = []
    handle = model.fusion[0].register_forward_pre_hook(
        lambda _module, arguments: captured.append(arguments[0].detach())
    )
    with torch.no_grad():
        model.forward_components(sample_values(size=16))
    handle.remove()
    assert captured
    assert torch.count_nonzero(captured[0][:, -18:]) == 0


def test_v5_hq_export_wrapper_flattens_and_clamps() -> None:
    wrapper = MiohRestorerV5HQExportWrapper(tiny_hq_model()).eval()
    with torch.no_grad():
        rgb, confidence = wrapper(sample_values(size=16))
    assert rgb.shape == (1, 15, 16, 16)
    assert confidence.shape == (1, 5, 16, 16)
    assert torch.all((0 <= rgb) & (rgb <= 1))


def test_v5_hq_curriculum_unfreezes_recurrence_then_flow() -> None:
    assert [stage.stage_id for stage in V5_HQ_STAGES] == [1, 2, 3, 4, 5, 6]
    assert sum(stage.default_steps for stage in V5_HQ_STAGES) == 70_000
    assert not hq_stage_definition(2).train_backbone
    assert hq_stage_definition(3).train_backbone
    assert not hq_stage_definition(3).train_spynet
    assert hq_stage_definition(4).train_spynet
    stage = hq_stage_definition(4)
    assert hq_learning_rate(stage, 10_000, 10_000, 200) == pytest.approx(
        stage.end_learning_rate
    )
