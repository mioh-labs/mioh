# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

import copy

import pytest
import torch

from lada.models.mioh_restorer.losses_v4 import (
    MiohRestorerV4Loss,
    confidence_error_correlation,
    overlap_consistency_loss,
)
from lada.models.mioh_restorer.curriculum_v4 import (
    QUALITY_V4_STAGES,
    effective_loss_weights,
    previous_stage,
    stage_definition,
    stage_learning_rate,
)
from lada.models.mioh_restorer.model_v4 import (
    FixedShiftBank,
    HierarchicalAlignment27,
    MiohRestorerV4ExportWrapper,
    MiohRestorerV4Q,
    NormalizedShiftCorrelation,
    V41_NEW_STATE_PREFIXES,
    load_v4_state_for_v41_upgrade,
    make_offsets,
    shift2d,
)
from lada.models.mioh_restorer.runner_v4 import MiohRestorerV4WindowRunner


def tiny_model(*, mode: str = "batch", detail: bool = False) -> MiohRestorerV4Q:
    return MiohRestorerV4Q(
        execution_mode=mode,
        quarter_channels=8,
        eighth_channels=8,
        fusion_eighth_channels=8,
        fusion_quarter_channels=8,
        eighth_blocks=1,
        quarter_blocks=1,
        high_resolution_detail=detail,
    )


def test_shift2d_does_not_wrap_edges() -> None:
    values = torch.arange(9).reshape(1, 1, 3, 3).float()
    shifted = shift2d(values, 1, -1)
    assert shifted.tolist() == [[[[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [4.0, 5.0, 0.0]]]]


@pytest.mark.parametrize("dilation", (1, 2, 3, 4, 12))
def test_fixed_shift_bank_exactly_matches_explicit_shifts(dilation: int) -> None:
    values = torch.randn(2, 3, 31, 29)
    offsets = make_offsets(1, dilation)
    expected = torch.stack(
        [shift2d(values, vertical, horizontal) for vertical, horizontal in offsets],
        dim=1,
    )
    actual = FixedShiftBank(3, offsets)(values)
    # A one-hot convolution is mathematically identical.  Accelerated
    # convolution may round the copied float32 value by one ULP.
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_fixed_full121_shift_bank_exactly_matches_explicit_shifts() -> None:
    values = torch.randn(1, 2, 15, 17)
    offsets = make_offsets(5)
    expected = torch.stack(
        [shift2d(values, vertical, horizontal) for vertical, horizontal in offsets],
        dim=1,
    )
    actual = FixedShiftBank(2, offsets)(values)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_convolutional_correlation_matches_explicit_reference() -> None:
    offsets = make_offsets(1, 2)
    explicit = NormalizedShiftCorrelation(offsets).eval()
    convolutional = NormalizedShiftCorrelation(offsets, channels=4).eval()
    convolutional.load_state_dict(explicit.state_dict(), strict=False)
    reference = torch.randn(2, 4, 17, 19)
    target = torch.randn_like(reference)
    with torch.no_grad():
        expected_aligned, expected_weights = explicit(reference, target)
        actual_aligned, actual_weights = convolutional(reference, target)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(actual_aligned, expected_aligned, rtol=2e-6, atol=2e-6)


def test_hierarchical_alignment_has_required_reach() -> None:
    alignment = HierarchicalAlignment27()
    assert alignment.input_reach == 40
    assert len(alignment.offsets_eighth_coarse) == 9
    assert len(alignment.offsets_eighth_fine) == 9
    assert len(alignment.offsets_quarter_fine) == 9


def test_correlation_temperature_is_bounded() -> None:
    correlation = NormalizedShiftCorrelation(make_offsets(1))
    with torch.no_grad():
        correlation.raw_temperature.fill_(1_000)
    assert float(correlation.temperature.detach()) <= correlation.MAXIMUM_TEMPERATURE
    with torch.no_grad():
        correlation.raw_temperature.fill_(-1_000)
    assert float(correlation.temperature.detach()) >= correlation.MINIMUM_TEMPERATURE


def test_untrained_model_is_identity_and_preserves_roi_outside() -> None:
    model = tiny_model().eval()
    values = torch.rand(2, 9, 4, 16, 16)
    values[:, :, 3] = 0
    values[:, 2:7, 3, 4:12, 4:12] = 1
    with torch.no_grad():
        restored, confidence, base, texture = model.forward_components(values)
    expected = values[:, 2:7, :3]
    assert restored.shape == (2, 5, 3, 16, 16)
    assert confidence.shape == (2, 5, 1, 16, 16)
    assert torch.equal(restored, expected)
    assert torch.count_nonzero(base) == 0
    assert torch.count_nonzero(texture) == 0


def test_v41_whitelist_upgrade_is_exact_and_output_preserving() -> None:
    legacy = tiny_model().eval()
    with torch.no_grad():
        legacy.base_head_half[-1].bias.fill_(0.1)
        legacy.texture_head[-1].bias.fill_(0.05)
    upgraded = tiny_model(detail=True).eval()
    missing = load_v4_state_for_v41_upgrade(
        upgraded, copy.deepcopy(legacy.state_dict()), source_revision=1
    )
    assert missing
    assert all(key.startswith(V41_NEW_STATE_PREFIXES) for key in missing)
    values = torch.rand(1, 9, 4, 16, 16)
    with torch.no_grad():
        legacy_output = legacy(values)
        upgraded_output = upgraded(values)
    for expected, actual in zip(legacy_output, upgraded_output, strict=True):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_v41_whitelist_upgrade_rejects_silent_legacy_loss() -> None:
    legacy_state = copy.deepcopy(tiny_model().state_dict())
    legacy_state.pop("encoder.stem.0.weight")
    with pytest.raises(ValueError, match="whitelist"):
        load_v4_state_for_v41_upgrade(
            tiny_model(detail=True), legacy_state, source_revision=1
        )


def test_batch_and_serial_execution_match_for_multiple_samples() -> None:
    batch_model = tiny_model(mode="batch").eval()
    serial_model = tiny_model(mode="serial").eval()
    serial_model.load_state_dict(copy.deepcopy(batch_model.state_dict()))
    with torch.no_grad():
        batch_model.base_head_half[-1].bias.fill_(0.1)
        batch_model.texture_head[-1].bias.fill_(0.05)
        serial_model.load_state_dict(copy.deepcopy(batch_model.state_dict()))
    values = torch.rand(2, 9, 4, 16, 16)
    with torch.no_grad():
        batch_output = batch_model.forward_components(values)
        serial_output = serial_model.forward_components(values)
    for batched, serial in zip(batch_output, serial_output, strict=True):
        torch.testing.assert_close(batched, serial, rtol=1e-5, atol=1e-6)


def test_training_diagnostics_do_not_change_normal_output() -> None:
    model = tiny_model().eval()
    values = torch.rand(1, 9, 4, 16, 16)
    with torch.no_grad():
        normal = model.forward_components(values)
        diagnostic = model.forward_with_distillation(
            values, capture_alignment=True, capture_features=True
        )
    for expected, actual in zip(normal, diagnostic[:4], strict=True):
        torch.testing.assert_close(expected, actual)
    details = diagnostic[4]
    assert details["alignment_coarse"].shape == (1, 5, 4, 9, 2, 2)
    assert details["alignment_middle"].shape == (1, 5, 4, 9, 2, 2)
    assert details["alignment_fine"].shape == (1, 5, 4, 9, 4, 4)
    assert details["fused_eighth"].shape == (1, 5, 8, 2, 2)
    assert details["fused_quarter"].shape == (1, 5, 8, 4, 4)


def test_export_wrapper_clamps_only_at_export_boundary() -> None:
    model = tiny_model().eval()
    with torch.no_grad():
        model.base_head_half[-1].bias.fill_(2.0)
    values = torch.ones(1, 9, 4, 16, 16)
    raw, _confidence = model(values)
    assert raw.max() > 1
    flat = values.reshape(1, 36, 16, 16)
    exported, _confidence = MiohRestorerV4ExportWrapper(model)(flat)
    assert exported.max() <= 1


def test_v4_loss_is_roi_normalized_and_backpropagates() -> None:
    loss_fn = MiohRestorerV4Loss()
    shape = (1, 5, 3, 16, 16)
    input_rgb = torch.rand(shape)
    target = torch.rand(shape)
    mask = torch.zeros(1, 5, 1, 16, 16)
    mask[..., 4:12, 4:12] = 1
    base = torch.zeros(shape, requires_grad=True)
    texture = torch.zeros(shape, requires_grad=True)
    confidence = torch.full((1, 5, 1, 16, 16), 0.5, requires_grad=True)
    restored = input_rgb + mask * (base + confidence * texture)
    total, stats = loss_fn(
        restored, confidence, base, texture, target, input_rgb, mask
    )
    total.backward()
    assert torch.isfinite(total)
    assert base.grad is not None
    assert texture.grad is not None
    assert confidence.grad is not None
    assert set(stats) >= {
        "reconstruction",
        "temporal",
        "temporal_acceleration",
        "gradient",
        "structural",
        "confidence_mean",
        "confidence_std",
    }


def test_temporal_losses_use_only_shared_roi() -> None:
    loss_fn = MiohRestorerV4Loss()
    shape = (1, 5, 3, 8, 8)
    target = torch.zeros(shape)
    restored = target.clone()
    restored[:, 1, :, :4] = 1
    mask = torch.zeros(1, 5, 1, 8, 8)
    mask[:, 0, :, :4] = 1
    mask[:, 1, :, 4:] = 1
    confidence = torch.full((1, 5, 1, 8, 8), 0.5)
    _total, stats = loss_fn(
        restored,
        confidence,
        torch.zeros_like(target),
        torch.zeros_like(target),
        target,
        target,
        mask,
    )
    assert stats["temporal"] == 0


def test_quality_stages_are_independent_and_ramp_from_parent() -> None:
    assert stage_definition(1).name == "foundation"
    assert stage_definition("faithful-reconstruction").stage_id == 2
    assert stage_definition(3).default_steps == 20_000
    assert previous_stage(stage_definition(1)) is None
    assert previous_stage(stage_definition(5)).stage_id == 4
    with pytest.raises(ValueError):
        stage_definition(6)

    stage = stage_definition(2)
    previous = stage_definition(1).loss
    target = stage.loss
    first = effective_loss_weights(stage, 1, transition_steps=500)
    middle = effective_loss_weights(stage, 250, transition_steps=500)
    final = effective_loss_weights(stage, 500, transition_steps=500)
    assert previous.high_frequency < first.high_frequency < middle.high_frequency
    assert middle.high_frequency < final.high_frequency
    assert final == target
    assert stage_definition(1).spynet_alignment_weight > 0
    assert stage_definition(1).exact_motion_alignment_weight > 0
    assert stage_definition(2).feature_distillation_weight > 0
    assert all(
        item.spynet_alignment_weight == 0
        and item.exact_motion_alignment_weight == 0
        and item.feature_distillation_weight == 0
        for item in QUALITY_V4_STAGES[2:]
    )


def test_quality_curriculum_learning_rate_is_stage_local() -> None:
    stage = stage_definition(3)
    assert stage_learning_rate(
        stage, 1, total_steps=20_000, warmup_steps=500
    ) < stage_learning_rate(
        stage, 500, total_steps=20_000, warmup_steps=500
    )
    assert stage_learning_rate(
        stage, 20_000, total_steps=20_000, warmup_steps=500
    ) == pytest.approx(
        stage.end_learning_rate
    )
    assert [stage.stage_id for stage in QUALITY_V4_STAGES] == [1, 2, 3, 4, 5]


def test_gradient_checkpointed_model_backpropagates() -> None:
    model = tiny_model().train()
    model.enable_gradient_checkpointing()
    values = torch.rand(1, 9, 4, 16, 16)
    values[:, :, 3] = 1
    restored, _confidence = model(values)
    restored.mean().backward()
    assert model.encoder.stem[0].weight.grad is not None


def test_overlap_loss_uses_only_shared_frame_roi() -> None:
    first = torch.zeros(1, 5, 3, 8, 8)
    second = torch.zeros_like(first)
    first[:, -1, :, 2:6, 2:6] = 1
    first_mask = torch.zeros(1, 5, 1, 8, 8)
    second_mask = torch.zeros_like(first_mask)
    first_mask[:, -1, :, 2:6, 2:6] = 1
    assert overlap_consistency_loss(first, second, first_mask, second_mask) > 0


def test_confidence_error_correlation_handles_constant_input() -> None:
    confidence = torch.ones(1, 5, 1, 4, 4)
    error = torch.ones_like(confidence)
    mask = torch.ones_like(confidence)
    assert confidence_error_correlation(confidence, error, mask) == 0


class _IdentityWindowModel(torch.nn.Module):
    def forward(self, values: torch.Tensor):
        return values[:, 2:7, :3], values[:, 2:7, 3:4]


@pytest.mark.parametrize("frame_count", (1, 5, 9, 14))
def test_window_runner_covers_video_and_blends_overlap(frame_count: int) -> None:
    frames = torch.rand(1, frame_count, 3, 8, 8)
    masks = torch.rand(1, frame_count, 1, 8, 8)
    restored, confidence = MiohRestorerV4WindowRunner(
        _IdentityWindowModel()
    ).restore(frames, masks)
    torch.testing.assert_close(restored, frames)
    torch.testing.assert_close(confidence, masks)
