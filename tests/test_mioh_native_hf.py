# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lada.models.mioh_restorer import supervision_native_hf

from lada.models.mioh_restorer.losses_native_hf import (
    MiohNativeHF512Loss,
    NativeHFLossWeights,
    eroded_roi_mask,
    identity_normalized_missing_detail_loss,
    missing_detail_oracle,
    native_detail_innovation,
    phase_block_average,
)
from lada.models.mioh_restorer.losses_v5 import high_frequency
from lada.models.mioh_restorer.model_native_hf import (
    MiohNativeHF512,
    MiohNativeHF512ExportWrapper,
    NativeHF512Config,
)
from lada.models.mioh_restorer.native_dataset_v5 import V5NativeManifestEntry
from lada.models.mioh_restorer.native_hf_dataset import (
    phase_block_average_mosaic,
    recentered_origins,
)
from lada.models.mioh_restorer.supervision_native_hf import (
    NATIVE_HF_MAXIMUM_TRANSLATION,
    alignment_offsets_and_scales,
    hierarchical_motion_targets,
    known_motion_alignment_loss,
    make_known_motion_window,
)


def _tiny_config() -> NativeHF512Config:
    return NativeHF512Config(
        input_frames=5,
        output_indices=(2,),
        context_frames=3,
        half_channels=8,
        quarter_channels=8,
        eighth_channels=8,
        fusion_half_channels=8,
        fusion_quarter_channels=8,
        fusion_eighth_channels=8,
        coarse_radius=1,
        eighth_blocks=1,
        quarter_blocks=1,
        half_blocks=1,
    )


def _sample_values(*, size: int = 32) -> torch.Tensor:
    values = torch.rand(1, 5, 8, size, size)
    values[:, :, 3] = 0
    values[:, :, 3, size // 4 : size * 3 // 4, size // 4 : size * 3 // 4] = 1
    values[:, :, 4] = 1
    return values


def test_native_hf_known_motion_uses_target_zero_motion_and_padding_validity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = torch.zeros(1, 5, 8, 32, 32)
    values[:, 2, 0, 16, 16] = 1
    values[:, 2, 4] = 0  # Synthetic reliability must not inherit this value.
    draws = iter((torch.tensor([8]), torch.tensor([-4])))

    def fixed_randint(*_args: object, **kwargs: object) -> torch.Tensor:
        return next(draws).to(kwargs.get("device", "cpu"))

    monkeypatch.setattr(torch, "randint", fixed_randint)
    synthetic, motion = make_known_motion_window(
        values, maximum_translation=2.0
    )

    # Target zero is shifted (+2 vertical, -1 horizontal) from the unshifted
    # centre reference; its inverse alignment target is therefore (-2, +1).
    torch.testing.assert_close(motion, torch.tensor([[2.0, -1.0]]))
    assert synthetic[0, 0, 0, 18, 15].item() == 1
    assert synthetic[0, 2, 0, 16, 16].item() == 1
    assert synthetic[0, 4, 0, 14, 17].item() == 1

    torch.testing.assert_close(synthetic[:, 2, 4], torch.ones(1, 32, 32))
    assert torch.count_nonzero(synthetic[:, 0, 4, :2]) == 0
    assert torch.count_nonzero(synthetic[:, 0, 4, :, -1:]) == 0
    assert torch.all(synthetic[:, 0, 4, 2:, :-1] == 1)


def test_native_hf_known_motion_targets_decompose_inverse_motion_and_enforce_reach() -> None:
    model = MiohNativeHF512(_tiny_config())
    offsets, scales = alignment_offsets_and_scales(model)
    motion = torch.tensor(
        ((8.0, -8.0), (13.25, -7.5), (-14.75, 14.5)),
        dtype=torch.float32,
    )
    targets = hierarchical_motion_targets(
        motion, offsets, scales, dtype=torch.float32
    )

    assert len(targets) == 4
    expected = torch.zeros_like(motion)
    for target, stage_offsets, scale in zip(
        targets, offsets, scales, strict=True
    ):
        torch.testing.assert_close(
            target.sum(dim=1), torch.ones(motion.shape[0]), rtol=0, atol=1e-6
        )
        candidates = torch.tensor(stage_offsets, dtype=torch.float32) * scale
        expected += target @ candidates
    torch.testing.assert_close(expected, -motion, rtol=0, atol=0.08)

    with pytest.raises(ValueError, match="pyramid reach"):
        hierarchical_motion_targets(
            torch.tensor(((15.25, 0.0),)),
            offsets,
            scales,
            dtype=torch.float32,
        )
    with pytest.raises(ValueError, match=r"\+/-23px"):
        make_known_motion_window(
            _sample_values(),
            maximum_translation=NATIVE_HF_MAXIMUM_TRANSLATION + 0.25,
        )


def test_native_hf_alignment_diagnostics_are_normalized_distributions() -> None:
    model = MiohNativeHF512(_tiny_config()).eval()
    values = _sample_values(size=64)
    with torch.no_grad():
        aligned, distributions = model.alignment_diagnostics(values)

    assert len(distributions) == 4
    for distribution in distributions:
        assert distribution.ndim == 4
        torch.testing.assert_close(
            distribution.sum(dim=1),
            torch.ones_like(distribution[:, 0]),
            rtol=1e-5,
            atol=1e-6,
        )
    assert aligned.entropy.shape == (1, 1, 32, 32)
    assert torch.isfinite(aligned.entropy).all()


def test_native_hf_known_motion_loss_backpropagates_and_reports_exact_motion_epe() -> None:
    torch.manual_seed(20260802)
    model = MiohNativeHF512(_tiny_config()).train()
    values = _sample_values(size=64).repeat(3, 1, 1, 1, 1)
    loss, stats = known_motion_alignment_loss(
        model, values, maximum_translation=4.0
    )
    loss.backward()

    gradients = (
        model.encoder.half_stage[0].weight.grad,
        model.decoder.alignment.corr_8.offset_bias.grad,
        model.decoder.alignment.corr_4.offset_bias.grad,
        model.decoder.alignment.corr_2.offset_bias.grad,
        model.decoder.alignment.phase_bias.grad,
    )
    assert torch.isfinite(loss)
    assert all(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all()
        for gradient in gradients
        if gradient is not None
    )
    assert model.decoder.alignment.pair_gate[-1].weight.grad is None
    assert torch.count_nonzero(model.encoder.half_stage[0].weight.grad) > 0
    assert (
        torch.count_nonzero(model.decoder.alignment.corr_8.offset_bias.grad)
        + torch.count_nonzero(model.decoder.alignment.corr_4.offset_bias.grad)
        > 0
    )
    assert torch.count_nonzero(model.decoder.alignment.corr_2.offset_bias.grad) > 0
    assert torch.count_nonzero(model.decoder.alignment.phase_bias.grad) > 0
    for name in (
        "known_motion_epe",
        "known_motion_epe_p95",
        "known_motion_target_decomposition_epe",
        "known_reliability",
        "known_occlusion",
        "known_entropy",
    ):
        assert name in stats
        assert torch.isfinite(stats[name])
    assert stats["known_motion_teacher_forcing"] == 0
    assert stats["known_motion_epe_p95"] >= stats["known_motion_epe"]
    assert stats["known_motion_target_decomposition_epe"] < 0.1

    with pytest.raises(ValueError, match="this model's alignment reach"):
        known_motion_alignment_loss(
            model, values, maximum_translation=15.25
        )


def test_native_hf_known_motion_uses_pixelwise_ce_and_crops_gate_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MiohNativeHF512(_tiny_config())
    values = _sample_values(size=64)
    motion = torch.tensor(((8.0, 0.0),))
    offsets, scales = alignment_offsets_and_scales(model)
    targets = hierarchical_motion_targets(
        motion, offsets, scales, dtype=torch.float32
    )
    monkeypatch.setattr(
        supervision_native_hf,
        "make_known_motion_window",
        lambda window, maximum_translation: (window, motion.to(window.device)),
    )

    reliability = torch.zeros(1, 1, 32, 32)
    occlusion = torch.ones_like(reliability)
    reliability[..., 8:-8, 8:-8] = 1
    occlusion[..., 8:-8, 8:-8] = 0
    aligned = SimpleNamespace(
        reliability=reliability,
        occlusion=occlusion,
        entropy=torch.zeros_like(reliability),
    )

    def distributions(mode: str) -> tuple[torch.Tensor, ...]:
        result = []
        for target in targets:
            correct = target[..., None, None].expand(-1, -1, 32, 32)
            wrong = torch.roll(correct, shifts=1, dims=1)
            if mode == "perfect":
                student = correct
            elif mode == "uniform_spatial":
                student = 0.5 * (correct + wrong)
            else:
                student = correct.clone()
                student[..., :, 16:] = wrong[..., :, 16:]
            result.append(student)
        return tuple(result)

    def evaluate(mode: str) -> torch.Tensor:
        monkeypatch.setattr(
            model,
            "alignment_diagnostics",
            lambda *_args, **_kwargs: (aligned, distributions(mode)),
        )
        loss, _stats = known_motion_alignment_loss(
            model, values, maximum_translation=4.0
        )
        return loss

    perfect = evaluate("perfect")
    uniform_spatial = evaluate("uniform_spatial")
    spatially_wrong = evaluate("spatially_wrong")

    # Border reliability=0/occlusion=1 is synthetic padding and must not make
    # the otherwise-perfect gate target incur a large loss.
    # The non-zero remainder is the entropy of the soft subpixel phase target.
    assert perfect < 0.25
    # These students have the same spatially pooled distribution.  Pixelwise
    # CE must still penalize the one that is wrong over half the image.
    assert spatially_wrong > uniform_spatial + 1


def test_native_hf_zero_initialization_uses_base_inside_roi_and_source_outside() -> None:
    model = MiohNativeHF512(_tiny_config()).eval()
    values = _sample_values()
    source = values[:, 2:3, :3]
    mask = values[:, 2:3, 3:4]
    base = values[:, 2:3, 5:8]

    with torch.no_grad():
        restored, confidence, residual, returned_base = model.forward_components(values)

    expected = source + mask * (base - source)
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    torch.testing.assert_close(returned_base, base, rtol=0, atol=0)
    assert torch.count_nonzero(residual) == 0
    torch.testing.assert_close(confidence, torch.full_like(confidence, 0.5))
    assert torch.equal((1 - mask) * restored, (1 - mask) * source)


def test_native_hf_zero_initialized_detail_skip_receives_first_step_gradient() -> None:
    model = MiohNativeHF512(_tiny_config()).train()
    assert torch.count_nonzero(model.decoder.detail_skip.weight) == 0
    assert torch.count_nonzero(model.decoder.detail_skip.bias) == 0

    values = _sample_values()
    _restored, _confidence, residual, _base = model.forward_components(values)
    target_residual = torch.rand_like(residual) - 0.5
    (residual - target_residual).square().mean().backward()

    gradient = model.decoder.detail_skip.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_native_hf_detail_skip_cannot_read_guide_or_metadata_channels() -> None:
    model = MiohNativeHF512(_tiny_config())
    gate = model.decoder.detail_skip_gate.reshape(
        model.config.context_frames, -1
    )
    assert gate.shape[1] == 32
    assert torch.all(gate[:, :12] == 1)
    assert torch.count_nonzero(gate[:, 12:]) == 0


def test_native_hf_masked_composition_preserves_source_pixels_after_perturbation() -> None:
    model = MiohNativeHF512(_tiny_config()).eval()
    values = _sample_values()
    source = values[:, 2:3, :3]
    mask = values[:, 2:3, 3:4]
    with torch.no_grad():
        model.decoder.residual_head[-1].bias.copy_(
            torch.linspace(-0.25, 0.25, 12)
        )
        restored, _confidence, residual, _base = model.forward_components(values)

    outside = (mask == 0).expand_as(restored)
    assert torch.equal(restored[outside], source[outside])
    assert torch.count_nonzero(residual) > 0
    assert torch.count_nonzero((restored - source)[~outside]) > 0


def test_native_hf_flat_export_contract_matches_model_without_clamp() -> None:
    model = MiohNativeHF512(_tiny_config()).eval()
    wrapper = MiohNativeHF512ExportWrapper(model, clamp=False).eval()
    values = _sample_values()
    flat = values.flatten(1, 2)

    with torch.no_grad():
        expected_rgb, expected_confidence = model(values)
        actual_rgb, actual_confidence = wrapper(flat)

    torch.testing.assert_close(
        actual_rgb, expected_rgb.flatten(1, 2), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual_confidence, expected_confidence.flatten(1, 2), rtol=0, atol=0
    )
    assert actual_rgb.shape == (1, 3, 32, 32)
    assert actual_confidence.shape == (1, 1, 32, 32)


def test_native_hf_reliability_gates_inference_but_not_raw_training_confidence() -> None:
    model = MiohNativeHF512(_tiny_config()).eval()
    wrapper = MiohNativeHF512ExportWrapper(model, clamp=False).eval()
    values = _sample_values()
    values[:, :, 4] = 0.25
    with torch.no_grad():
        model.decoder.residual_head[-1].bias.copy_(
            torch.linspace(-0.25, 0.25, 12)
        )
        restored, raw_confidence, residual, base = model.forward_components(values)
        forward_restored, effective_confidence = model(values)
        export_restored, export_confidence = wrapper(values.flatten(1, 2))

    source = values[:, 2:3, :3]
    mask = values[:, 2:3, 3:4]
    expected_effective = raw_confidence * 0.25
    expected_restored = source + mask * (
        base - source + expected_effective * residual
    )
    torch.testing.assert_close(
        raw_confidence, torch.full_like(raw_confidence, 0.5), rtol=0, atol=0
    )
    torch.testing.assert_close(restored, expected_restored, rtol=0, atol=0)
    torch.testing.assert_close(forward_restored, restored, rtol=0, atol=0)
    torch.testing.assert_close(
        effective_confidence, expected_effective, rtol=0, atol=0
    )
    torch.testing.assert_close(
        export_restored, restored.flatten(1, 2), rtol=0, atol=0
    )
    torch.testing.assert_close(
        export_confidence, expected_effective.flatten(1, 2), rtol=0, atol=0
    )


def test_native_hf_training_graph_backpropagates_to_encoder_and_both_heads() -> None:
    model = MiohNativeHF512(_tiny_config()).train()
    with torch.no_grad():
        model.decoder.residual_head[-1].weight.normal_(std=1e-3)
        model.decoder.confidence_head[-1].weight.normal_(std=1e-3)
    values = _sample_values()
    restored, confidence, residual, _base = model.forward_components(values)
    target = torch.rand_like(restored)
    loss = (
        (restored - target).square().mean()
        + (residual - target).abs().mean()
        + confidence.mean()
    )
    loss.backward()

    gradients = (
        model.encoder.half_stage[0].weight.grad,
        model.decoder.residual_head[-1].weight.grad,
        model.decoder.confidence_head[-1].weight.grad,
    )
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)
    assert all(torch.count_nonzero(gradient) > 0 for gradient in gradients if gradient is not None)


def test_native_hf_observation_loss_respects_sample_weight() -> None:
    weights = NativeHFLossWeights(
        reconstruction=0,
        residual=0,
        missing_detail=0,
        non_detail_suppression=0,
        innovation=0,
        innovation_span=0,
        innovation_zero=0,
        fidelity_guard=0,
        high_frequency=0,
        gradient=0,
        wavelet=0,
        observation=1,
        low_frequency_drift=0,
        confidence=0,
        confidence_regularization=0,
    )
    criterion = MiohNativeHF512Loss(weights=weights)
    restored = torch.rand(1, 1, 3, 16, 16)
    target = torch.rand_like(restored)
    base = torch.rand_like(restored)
    source = torch.rand_like(restored)
    residual = torch.rand_like(restored)
    confidence = torch.full((1, 1, 1, 16, 16), 0.5)
    mask = torch.ones_like(confidence)
    observation = torch.zeros_like(restored)
    phases = torch.tensor([[[1, 2]]], dtype=torch.int64)
    block_size = torch.tensor([4], dtype=torch.int64)

    disabled_total, disabled_stats = criterion(
        restored,
        confidence,
        residual,
        base,
        target,
        source,
        mask,
        observation,
        phases,
        block_size,
        torch.tensor([0.0]),
    )
    enabled_total, enabled_stats = criterion(
        restored,
        confidence,
        residual,
        base,
        target,
        source,
        mask,
        observation,
        phases,
        block_size,
        torch.tensor([1.0]),
    )

    assert disabled_stats["observation"].item() == 0
    assert disabled_total.item() == 0
    assert enabled_stats["observation"].item() > 0
    torch.testing.assert_close(enabled_total, enabled_stats["observation"])


def test_native_hf_missing_detail_oracle_rejects_guide_smoothing() -> None:
    size = 24
    yy, xx = torch.meshgrid(
        torch.arange(size), torch.arange(size), indexing="ij"
    )
    pattern = ((xx + yy) % 2).float().mul(2).sub(1)
    base = (0.5 + 0.04 * pattern).reshape(1, 1, 1, size, size).repeat(1, 1, 3, 1, 1)
    target = (0.5 + 0.12 * pattern).reshape(1, 1, 1, size, size).repeat(1, 1, 3, 1, 1)
    mask = torch.ones(1, 1, 1, size, size)
    mask[..., :3, :] = 0
    mask[..., -3:, :] = 0
    mask[..., :, :3] = 0
    mask[..., :, -3:] = 0

    oracle, support = missing_detail_oracle(target, base, mask)
    assert torch.count_nonzero(support) > 0
    assert torch.count_nonzero(support[..., :2, :]) == 0
    assert torch.count_nonzero(support[..., -2:, :]) == 0
    smoothing = -0.8 * high_frequency(base)
    assert torch.sum(smoothing * oracle * support) < 0
    assert torch.sum(oracle.square() * support) > 0
    inner = eroded_roi_mask(mask)
    assert torch.count_nonzero(inner[..., :5, :]) == 0
    assert torch.count_nonzero(inner[..., -5:, :]) == 0


def test_native_hf_missing_detail_loss_is_identity_normalized() -> None:
    size = 24
    yy, xx = torch.meshgrid(
        torch.arange(size), torch.arange(size), indexing="ij"
    )
    pattern = ((xx + yy) % 2).float().mul(2).sub(1)
    base = (0.5 + 0.04 * pattern).reshape(
        1, 1, 1, size, size
    ).repeat(1, 1, 3, 1, 1)
    target = (0.5 + 0.12 * pattern).reshape(
        1, 1, 1, size, size
    ).repeat(1, 1, 3, 1, 1)
    mask = torch.ones(1, 1, 1, size, size)
    oracle, support = missing_detail_oracle(target, base, mask)
    identity = torch.zeros_like(oracle, requires_grad=True)

    identity_loss = identity_normalized_missing_detail_loss(
        identity, oracle, support
    )
    exact_loss = identity_normalized_missing_detail_loss(
        oracle, oracle, support
    )

    torch.testing.assert_close(identity_loss, torch.ones_like(identity_loss))
    assert exact_loss < 1e-3
    identity_loss.backward()
    assert identity.grad is not None
    assert torch.isfinite(identity.grad).all()
    assert torch.count_nonzero(identity.grad) > 0
    assert torch.sum(identity.grad * oracle * support) < 0
    assert torch.count_nonzero(identity.grad * (1.0 - support)) == 0


def test_native_hf_missing_detail_loss_handles_empty_support() -> None:
    correction = torch.randn(1, 1, 3, 8, 8, requires_grad=True)
    oracle = torch.zeros_like(correction)
    support = torch.zeros(1, 1, 1, 8, 8)

    loss = identity_normalized_missing_detail_loss(
        correction, oracle, support
    )

    assert torch.isfinite(loss)
    assert loss.item() == 0
    loss.backward()
    assert correction.grad is not None
    assert torch.count_nonzero(correction.grad) == 0


def test_native_hf_missing_detail_loss_prefers_oracle_direction() -> None:
    weights = NativeHFLossWeights(
        reconstruction=0,
        residual=0,
        missing_detail=1,
        non_detail_suppression=0,
        innovation=0,
        innovation_span=0,
        innovation_zero=0,
        fidelity_guard=0,
        high_frequency=0,
        gradient=0,
        wavelet=0,
        observation=0,
        low_frequency_drift=0,
        confidence=0,
        confidence_regularization=0,
    )
    criterion = MiohNativeHF512Loss(weights=weights)
    size = 24
    yy, xx = torch.meshgrid(
        torch.arange(size), torch.arange(size), indexing="ij"
    )
    pattern = ((xx + yy) % 2).float().mul(2).sub(1)
    base = (0.5 + 0.04 * pattern).reshape(1, 1, 1, size, size).repeat(1, 1, 3, 1, 1)
    target = (0.5 + 0.12 * pattern).reshape(1, 1, 1, size, size).repeat(1, 1, 3, 1, 1)
    mask = torch.ones(1, 1, 1, size, size)
    oracle, _support = missing_detail_oracle(target, base, mask)
    confidence = torch.ones_like(mask)
    phases = torch.zeros(1, 1, 2, dtype=torch.int64)
    block_size = torch.tensor([4], dtype=torch.int64)

    def evaluate(residual: torch.Tensor) -> torch.Tensor:
        restored = base + residual
        total, _stats = criterion(
            restored,
            confidence,
            residual,
            base,
            target,
            base,
            mask,
            target,
            phases,
            block_size,
            torch.tensor([0.0]),
        )
        return total

    exact = evaluate(oracle)
    zero = evaluate(torch.zeros_like(oracle))
    smoothing = evaluate(-0.8 * high_frequency(base))
    torch.testing.assert_close(zero, torch.ones_like(zero))
    assert exact < zero
    assert exact < smoothing
    assert smoothing > zero


def test_native_hf_innovation_rejects_input_filtering_and_recovers_new_texture() -> None:
    size = 64
    yy, xx = torch.meshgrid(
        torch.arange(size), torch.arange(size), indexing="ij"
    )
    base_pattern = torch.sin(xx.float() * torch.pi / 4)
    target_pattern = torch.sin(yy.float() * torch.pi / 3)
    base = (0.5 + 0.04 * base_pattern).reshape(
        1, 1, 1, size, size
    ).repeat(1, 1, 3, 1, 1)
    source = base.clone()
    target = base + 0.04 * target_pattern.reshape(
        1, 1, 1, size, size
    ).repeat(1, 1, 3, 1, 1)
    mask = torch.ones(1, 1, 1, size, size)
    exact_correction = target - base

    exact_loss, _exact_span, _exact_zero, exact_stats = native_detail_innovation(
        exact_correction, target, base, source, mask
    )
    zero_loss, _zero_span, _zero_guard, zero_stats = native_detail_innovation(
        torch.zeros_like(exact_correction), target, base, source, mask
    )
    filtered_loss, filtered_span, _filtered_zero, filtered_stats = (
        native_detail_innovation(
            -0.8 * high_frequency(base), target, base, source, mask
        )
    )

    assert exact_stats["innovation_valid_patches"] > 0
    assert exact_stats["innovation_ev_percent"] > 99
    assert exact_stats["innovation_correlation"] > 0.99
    assert exact_loss < zero_loss * 1e-3
    assert abs(float(zero_stats["innovation_ev_percent"])) < 1e-4
    assert abs(float(filtered_stats["innovation_ev_percent"])) < 1e-2
    assert filtered_span > 0
    assert filtered_loss == pytest.approx(float(zero_loss), rel=1e-3)


def test_native_hf_innovation_has_finite_detail_gradient_at_identity() -> None:
    size = 64
    yy, xx = torch.meshgrid(
        torch.arange(size), torch.arange(size), indexing="ij"
    )
    base = torch.full((1, 1, 3, size, size), 0.5)
    source = base.clone()
    target = base + 0.04 * torch.sin(yy.float() * torch.pi / 3).reshape(
        1, 1, 1, size, size
    ).repeat(1, 1, 3, 1, 1)
    mask = torch.ones(1, 1, 1, size, size)
    correction = torch.zeros_like(target, requires_grad=True)

    innovation, span, zero, _stats = native_detail_innovation(
        correction, target, base, source, mask
    )
    (innovation + 0.25 * span + 0.25 * zero).backward()
    assert correction.grad is not None
    assert torch.isfinite(correction.grad).all()
    assert torch.count_nonzero(correction.grad) > 0
    assert torch.sum(correction.grad * (target - base)) < 0


@pytest.mark.parametrize(
    ("height", "width", "block_size", "phase"),
    (
        (17, 23, 4, (0, 0)),
        (19, 21, 6, (2, 5)),
        (32, 27, 9, (8, 3)),
    ),
)
def test_known_phase_numpy_and_torch_mosaic_operators_match(
    height: int,
    width: int,
    block_size: int,
    phase: tuple[int, int],
) -> None:
    rng = np.random.default_rng(20260802 + height + width)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    expected = phase_block_average_mosaic(
        image, block_size=block_size, phase=phase
    )
    values = (
        torch.from_numpy(image.copy())
        .permute(2, 0, 1)
        .float()
        .reshape(1, 1, 3, height, width)
    )
    actual = phase_block_average(
        values,
        torch.tensor([block_size]),
        torch.tensor([[phase]], dtype=torch.int64),
    )
    actual_uint8 = actual.round().clamp(0, 255).to(torch.uint8)[0, 0].permute(1, 2, 0)
    np.testing.assert_array_equal(actual_uint8.numpy(), expected)


def test_recentered_origins_expand_native_crop_without_resampling() -> None:
    origins = tuple((200 + index * 2, 300 - index * 2) for index in range(9))
    entry = V5NativeManifestEntry(
        name="native-hf",
        target_video=Path("target.mp4"),
        mask_video=Path("mask.mkv"),
        start_frame=0,
        bucket=256,
        origins=origins,
        mask_reliability=(1.0,) * 9,
        mosaic_block_size=16.0,
        source_video_id="source",
    )

    expanded = recentered_origins(entry, native_size=512)
    assert expanded == tuple((x - 128, y - 128) for x, y in origins)
    with pytest.raises(ValueError, match="cannot shrink"):
        recentered_origins(entry, native_size=128)
