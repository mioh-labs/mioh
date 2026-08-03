# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""GT-primary losses for the Native-HF 512 prototype."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .losses_v5 import (
    gaussian_blur,
    gradient_loss,
    high_frequency,
    masked_charbonnier,
    masked_correlation,
    masked_mean,
    wavelet_loss,
)


@dataclass(frozen=True)
class NativeHFLossWeights:
    reconstruction: float = 1.0
    residual: float = 0.50
    missing_detail: float = 0.0
    non_detail_suppression: float = 0.0
    innovation: float = 0.25
    innovation_span: float = 0.05
    innovation_zero: float = 0.05
    fidelity_guard: float = 0.0
    high_frequency: float = 0.35
    gradient: float = 0.10
    wavelet: float = 0.10
    observation: float = 0.20
    low_frequency_drift: float = 0.05
    confidence: float = 0.10
    confidence_regularization: float = 1e-3


def eroded_roi_mask(mask: torch.Tensor, *, radius: int = 2) -> torch.Tensor:
    """Remove filter-contaminated pixels from the inside of an ROI mask."""

    if mask.ndim != 5 or mask.shape[2] != 1:
        raise ValueError("Native-HF ROI masks must be [B,O,1,H,W]")
    if radius < 0:
        raise ValueError("Native-HF ROI erosion radius must be non-negative")
    if radius == 0:
        return mask.clamp(0.0, 1.0)
    shape = mask.shape
    flat = mask.reshape(-1, 1, shape[-2], shape[-1]).clamp(0.0, 1.0)
    kernel = radius * 2 + 1
    eroded = 1.0 - F.max_pool2d(
        1.0 - flat, kernel_size=kernel, stride=1, padding=radius
    )
    # A binary interior is intentional: partially feathered pixels would mix
    # source and guide values and contaminate the five-pixel HF comparison.
    return (eroded >= 1.0 - 1e-6).to(mask.dtype).reshape_as(mask)


def missing_detail_oracle(
    target: torch.Tensor,
    base: torch.Tensor,
    mask: torch.Tensor,
    *,
    minimum_amplitude: float = 1.0 / 1024.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return GT detail that is present but weaker in the global guide.

    Only same-direction high-frequency coefficients whose GT magnitude exceeds
    the guide are retained.  Subtracting guide high frequency (a trivial blur)
    therefore points away from this oracle and cannot masquerade as native
    detail recovery.
    """

    if target.shape != base.shape or target.ndim != 5 or target.shape[2] != 3:
        raise ValueError("Native-HF oracle RGB tensors must be [B,O,3,H,W]")
    if mask.shape[:2] != target.shape[:2] or mask.shape[2:] != (
        1,
        target.shape[-2],
        target.shape[-1],
    ):
        raise ValueError("Native-HF oracle mask does not match RGB tensors")
    if minimum_amplitude < 0:
        raise ValueError("Native-HF oracle amplitude threshold must be non-negative")

    target_hf = high_frequency(target)
    base_hf = high_frequency(base)
    inner = eroded_roi_mask(mask, radius=2)
    support = (
        (target_hf * base_hf >= 0)
        & (target_hf.abs() > base_hf.abs() + minimum_amplitude)
    ).to(target.dtype) * inner
    oracle = target_hf.sign() * F.relu(target_hf.abs() - base_hf.abs())
    return oracle * support, support


def identity_normalized_missing_detail_loss(
    correction: torch.Tensor,
    detail_oracle: torch.Tensor,
    detail_support: torch.Tensor,
    *,
    charbonnier_epsilon: float = 1e-6,
    normalization_epsilon: float = 1e-8,
) -> torch.Tensor:
    """Measure same-sign missing detail relative to the identity baseline.

    ``detail_oracle`` and ``detail_support`` are the tensors returned by
    :func:`missing_detail_oracle`.  Dividing by the exact loss for a zero
    correction makes the identity prediction equal to one whenever supported
    missing detail exists, independent of that detail's absolute amplitude.
    An empty support has no training signal and returns zero.
    """

    if correction.shape != detail_oracle.shape or correction.ndim != 5:
        raise ValueError(
            "Native-HF correction and missing-detail oracle must match"
        )
    if detail_support.shape not in (
        correction.shape,
        (
            correction.shape[0],
            correction.shape[1],
            1,
            correction.shape[3],
            correction.shape[4],
        ),
    ):
        raise ValueError("Native-HF missing-detail support does not match")
    if charbonnier_epsilon <= 0 or normalization_epsilon <= 0:
        raise ValueError("Native-HF missing-detail epsilons must be positive")

    error = masked_charbonnier(
        correction,
        detail_oracle,
        detail_support,
        epsilon=charbonnier_epsilon,
    )
    identity_error = masked_charbonnier(
        torch.zeros_like(correction),
        detail_oracle,
        detail_support,
        epsilon=charbonnier_epsilon,
    ).detach()
    denominator_floor = max(
        normalization_epsilon, torch.finfo(identity_error.dtype).tiny
    )
    normalized = error / identity_error.clamp_min(denominator_floor)
    has_support = detail_support.expand_as(correction).sum() > 0
    return torch.where(has_support, normalized, error * 0.0)


def _luma(values: torch.Tensor) -> torch.Tensor:
    weights = values.new_tensor((0.2126, 0.7152, 0.0722)).reshape(
        1, 1, 3, 1, 1
    )
    return (values * weights).sum(dim=2, keepdim=True)


def _blur_single_channel(values: torch.Tensor) -> torch.Tensor:
    shape = values.shape
    flat = values.reshape(-1, 1, shape[-2], shape[-1])
    kernel = values.new_tensor((1.0, 4.0, 6.0, 4.0, 1.0))
    kernel = torch.outer(kernel, kernel)
    kernel = (kernel / kernel.sum()).reshape(1, 1, 5, 5)
    blurred = F.conv2d(
        F.pad(flat, (2, 2, 2, 2), mode="replicate"), kernel
    )
    return blurred.reshape(shape)


def _detail_bands(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    first = _blur_single_channel(values)
    second = _blur_single_channel(first)
    return values - first, first - second


def _patch_vectors(values: torch.Tensor, *, patch_size: int) -> torch.Tensor:
    if values.ndim != 5 or values.shape[2] != 1:
        raise ValueError("Native-HF patch values must be [B,O,1,H,W]")
    if patch_size <= 0:
        raise ValueError("Native-HF innovation patch size must be positive")
    batch, outputs, _channels, height, width = values.shape
    pad_bottom = (-height) % patch_size
    pad_right = (-width) % patch_size
    flat = values.reshape(batch * outputs, 1, height, width)
    if pad_bottom or pad_right:
        flat = F.pad(flat, (0, pad_right, 0, pad_bottom))
    padded_height, padded_width = flat.shape[-2:]
    patches = flat.reshape(
        batch * outputs,
        1,
        padded_height // patch_size,
        patch_size,
        padded_width // patch_size,
        patch_size,
    ).permute(0, 2, 4, 1, 3, 5)
    return patches.reshape(-1, patch_size * patch_size)


def _orthonormal_patch_basis(
    vectors: tuple[torch.Tensor, ...], *, epsilon: float = 1e-8
) -> tuple[torch.Tensor, ...]:
    basis: list[torch.Tensor] = []
    for vector in vectors:
        residual = vector
        # A second pass keeps nearly dependent base/source filters from
        # becoming a large numerical basis vector after normalization.
        for _pass in range(2):
            for existing in basis:
                residual = residual - (residual * existing).sum(
                    dim=1, keepdim=True
                ) * existing
        norm = torch.sqrt(residual.square().sum(dim=1, keepdim=True))
        source_norm = torch.sqrt(vector.square().sum(dim=1, keepdim=True))
        valid = norm > torch.maximum(
            norm.new_full((), epsilon), source_norm * 1e-4
        )
        normalized = residual / norm.clamp_min(epsilon)
        normalized = normalized * valid.to(normalized.dtype)
        basis.append(normalized)
    return tuple(basis)


def _project_patch_vectors(
    values: torch.Tensor, basis: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    projection = torch.zeros_like(values)
    for vector in basis:
        projection = projection + (values * vector).sum(
            dim=1, keepdim=True
        ) * vector
    return projection


def native_detail_innovation(
    correction: torch.Tensor,
    target: torch.Tensor,
    base: torch.Tensor,
    source: torch.Tensor,
    mask: torch.Tensor,
    *,
    patch_size: int = 64,
    minimum_rms: float = 1.5 / 255.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Separate true GT innovation from filtering of existing inputs.

    The nuisance space contains two frequency bands from both the global guide
    and the mosaic observation.  Per-patch Gram-Schmidt projection removes any
    correction obtainable by merely smoothing or sharpening those inputs.
    """

    if not (
        correction.shape == target.shape == base.shape == source.shape
        and target.ndim == 5
        and target.shape[2] == 3
    ):
        raise ValueError("Native-HF innovation RGB tensors must match")
    if minimum_rms <= 0:
        raise ValueError("Native-HF innovation RMS threshold must be positive")

    inner = eroded_roi_mask(mask, radius=2)
    mask_patches = _patch_vectors(inner, patch_size=patch_size)
    root_mask = torch.sqrt(mask_patches.clamp_min(0.0))
    valid_count = mask_patches.sum(dim=1)
    valid_patch = valid_count >= max(16.0, patch_size * patch_size / 64.0)

    correction_bands = _detail_bands(_luma(correction))
    target_bands = _detail_bands(_luma(target - base))
    base_bands = _detail_bands(_luma(base))
    source_bands = _detail_bands(_luma(source))
    nuisance_corrections = (*base_bands, *source_bands)
    nuisance_bands = tuple(
        _detail_bands(value) for value in nuisance_corrections
    )

    band_weights = (0.7, 0.3)
    innovation_loss = correction.new_zeros(())
    span_loss = correction.new_zeros(())
    zero_loss = correction.new_zeros(())
    target_energy_total = correction.new_zeros(())
    error_energy_total = correction.new_zeros(())
    prediction_energy_total = correction.new_zeros(())
    dot_total = correction.new_zeros(())
    span_energy_total = correction.new_zeros(())
    raw_energy_total = correction.new_zeros(())
    non_support_energy_total = correction.new_zeros(())
    support_count_total = correction.new_zeros(())

    for band_index, (weight, prediction_band, target_band) in enumerate(
        zip(band_weights, correction_bands, target_bands, strict=True)
    ):
        basis = _orthonormal_patch_basis(
            tuple(
                _patch_vectors(
                    value[band_index], patch_size=patch_size
                ).detach()
                * root_mask
                for value in nuisance_bands
            )
        )
        prediction = (
            _patch_vectors(prediction_band, patch_size=patch_size) * root_mask
        )
        truth = (
            _patch_vectors(target_band, patch_size=patch_size).detach()
            * root_mask
        )
        prediction_span = _project_patch_vectors(prediction, basis)
        truth_span = _project_patch_vectors(truth, basis)
        prediction_innovation = prediction - prediction_span
        truth_innovation = (truth - truth_span).detach()

        target_energy = truth_innovation.square().sum(dim=1)
        prediction_energy = prediction_innovation.square().sum(dim=1)
        error_energy = (prediction_innovation - truth_innovation).square().sum(
            dim=1
        )
        target_rms = torch.sqrt(
            target_energy / valid_count.clamp_min(1.0)
        )
        support = valid_patch & (target_rms >= minimum_rms)
        non_support = valid_patch & ~support
        support_float = support.to(correction.dtype)
        non_support_float = non_support.to(correction.dtype)
        support_count = support_float.sum()
        non_support_count = non_support_float.sum()
        normalization = target_energy + minimum_rms**2 * valid_count

        innovation_loss = innovation_loss + weight * (
            (error_energy / target_energy.clamp_min(1e-8)) * support_float
        ).sum() / support_count.clamp_min(1.0)
        span_energy = prediction_span.square().sum(dim=1)
        span_loss = span_loss + weight * (
            (span_energy / normalization.clamp_min(1e-8)) * support_float
        ).sum() / support_count.clamp_min(1.0)
        raw_energy = prediction.square().sum(dim=1)
        zero_loss = zero_loss + weight * (
            (
                raw_energy
                / (minimum_rms**2 * valid_count).clamp_min(1e-8)
            )
            * non_support_float
        ).sum() / non_support_count.clamp_min(1.0)

        target_energy_total = target_energy_total + weight * (
            target_energy * support_float
        ).sum()
        error_energy_total = error_energy_total + weight * (
            error_energy * support_float
        ).sum()
        prediction_energy_total = prediction_energy_total + weight * (
            prediction_energy * support_float
        ).sum()
        dot_total = dot_total + weight * (
            (prediction_innovation * truth_innovation).sum(dim=1)
            * support_float
        ).sum()
        span_energy_total = span_energy_total + weight * (
            span_energy * support_float
        ).sum()
        raw_energy_total = raw_energy_total + weight * (
            raw_energy * support_float
        ).sum()
        non_support_energy_total = non_support_energy_total + weight * (
            raw_energy * non_support_float
        ).sum()
        support_count_total = support_count_total + support_count

    has_innovation = (support_count_total > 0) & (target_energy_total > 1e-12)
    innovation_ev = torch.where(
        has_innovation,
        100.0
        * (1.0 - error_energy_total / target_energy_total.clamp_min(1e-12)),
        target_energy_total.new_zeros(()),
    )
    innovation_correlation = torch.where(
        has_innovation,
        dot_total
        / torch.sqrt(prediction_energy_total * target_energy_total + 1e-12),
        target_energy_total.new_zeros(()),
    )
    innovation_gain = torch.where(
        has_innovation,
        dot_total / target_energy_total.clamp_min(1e-12),
        target_energy_total.new_zeros(()),
    )
    non_support_energy_percent = torch.where(
        has_innovation,
        100.0
        * non_support_energy_total
        / target_energy_total.clamp_min(1e-12),
        target_energy_total.new_zeros(()),
    )
    stats = {
        "innovation_ev_percent": innovation_ev,
        "innovation_correlation": innovation_correlation,
        "innovation_gain": innovation_gain,
        "innovation_span_energy_percent": 100.0
        * span_energy_total
        / raw_energy_total.clamp_min(1e-12),
        "innovation_non_support_energy_percent": non_support_energy_percent,
        "innovation_valid_patches": support_count_total,
        "innovation_valid": has_innovation.to(correction.dtype),
        "innovation_positive": (
            has_innovation & (innovation_ev > 0)
        ).to(correction.dtype),
    }
    return innovation_loss, span_loss, zero_loss, stats


def _phase_block_average_one(
    values: torch.Tensor, *, block_size: int, phase_x: int, phase_y: int
) -> torch.Tensor:
    if values.ndim != 4 or values.shape[0] != 1:
        raise ValueError("phase block average operates on one BCHW sample")
    if block_size <= 1:
        raise ValueError("mosaic block size must exceed one")
    height, width = values.shape[-2:]
    phase_x %= block_size
    phase_y %= block_size
    left = (-phase_x) % block_size
    top = (-phase_y) % block_size
    right = (-(width + left)) % block_size
    bottom = (-(height + top)) % block_size
    padded = F.pad(values, (left, right, top, bottom), mode="replicate")
    reduced = F.avg_pool2d(padded, kernel_size=block_size, stride=block_size)
    expanded = reduced.repeat_interleave(block_size, dim=-2).repeat_interleave(
        block_size, dim=-1
    )
    return expanded[..., top : top + height, left : left + width]


def phase_block_average(
    values: torch.Tensor,
    block_sizes: torch.Tensor,
    phases: torch.Tensor,
) -> torch.Tensor:
    """Differentiable known-grid mosaic forward operator.

    Args:
        values: ``[B,O,3,H,W]``.
        block_sizes: ``[B]`` or scalar.
        phases: ``[B,O,2]`` in x/y order.
    """

    if values.ndim != 5 or values.shape[2] != 3:
        raise ValueError("phase mosaic values must be [B,O,3,H,W]")
    if phases.shape != (values.shape[0], values.shape[1], 2):
        raise ValueError("phase metadata does not match Native-HF outputs")
    block_sizes = block_sizes.reshape(-1)
    if block_sizes.numel() == 1 and values.shape[0] > 1:
        block_sizes = block_sizes.expand(values.shape[0])
    if block_sizes.numel() != values.shape[0]:
        raise ValueError("block-size metadata does not match batch")
    outputs: list[torch.Tensor] = []
    for batch_index in range(values.shape[0]):
        frame_outputs: list[torch.Tensor] = []
        block = int(block_sizes[batch_index].detach().cpu())
        for output_index in range(values.shape[1]):
            phase_x = int(phases[batch_index, output_index, 0].detach().cpu())
            phase_y = int(phases[batch_index, output_index, 1].detach().cpu())
            frame_outputs.append(
                _phase_block_average_one(
                    values[batch_index : batch_index + 1, output_index],
                    block_size=block,
                    phase_x=phase_x,
                    phase_y=phase_y,
                )
            )
        outputs.append(torch.stack(frame_outputs, dim=1))
    return torch.cat(outputs, dim=0)


class MiohNativeHF512Loss(nn.Module):
    def __init__(
        self,
        *,
        weights: NativeHFLossWeights | None = None,
    ) -> None:
        super().__init__()
        self.weights = weights or NativeHFLossWeights()

    def forward(
        self,
        restored: torch.Tensor,
        confidence: torch.Tensor,
        residual: torch.Tensor,
        base: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        mask: torch.Tensor,
        mosaic_observation: torch.Tensor,
        mosaic_phases: torch.Tensor,
        mosaic_block_size: torch.Tensor,
        observation_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        expected = target.shape
        if restored.shape != expected or base.shape != expected or source.shape != expected:
            raise ValueError("Native-HF RGB tensors must have identical shapes")
        if residual.shape != expected or mask.shape[:2] != expected[:2]:
            raise ValueError("Native-HF residual/mask shapes do not match target")

        reconstruction = masked_charbonnier(restored, target, mask)
        residual_target = high_frequency(target - base)
        residual_loss = masked_charbonnier(residual, residual_target, mask)
        baseline = source + mask * (base - source)
        effective_correction = restored - baseline
        detail_oracle, detail_support = missing_detail_oracle(
            target, base, mask
        )
        missing_detail = identity_normalized_missing_detail_loss(
            effective_correction, detail_oracle, detail_support
        )
        detail_interior = eroded_roi_mask(mask, radius=2).expand_as(restored)
        non_detail_support = (detail_interior - detail_support).clamp_min(0.0)
        non_detail_suppression = masked_charbonnier(
            effective_correction,
            torch.zeros_like(effective_correction),
            non_detail_support,
        )
        innovation, innovation_span, innovation_zero, innovation_stats = (
            native_detail_innovation(
                effective_correction,
                target,
                base,
                source,
                mask,
            )
        )
        hf_loss = masked_charbonnier(
            high_frequency(restored), high_frequency(target), mask
        )
        gradients = gradient_loss(restored, target, mask)
        wavelet = wavelet_loss(restored, target, mask)
        low_frequency_drift = masked_charbonnier(
            gaussian_blur(restored), gaussian_blur(base), mask
        )
        baseline_mse = masked_mean((baseline - target).square(), mask)
        restored_mse = masked_mean((restored - target).square(), mask)
        maximum_mse_ratio = restored_mse.new_tensor(10.0 ** (0.2 / 10.0))
        fidelity_guard = F.relu(
            restored_mse / baseline_mse.clamp_min(1e-12) - maximum_mse_ratio
        )

        remosaic_content = phase_block_average(
            restored, mosaic_block_size, mosaic_phases
        )
        remosaic = restored + mask * (remosaic_content - restored)
        observation_mask = mask * observation_weight.reshape(-1, 1, 1, 1, 1)
        observation = masked_charbonnier(
            remosaic, mosaic_observation, observation_mask
        )

        correction = mask * residual
        ungated_candidate = baseline + correction
        candidate_error = torch.mean(
            torch.abs(target - ungated_candidate), dim=2, keepdim=True
        )
        # Per-pixel least-squares gate: 0 means the learned correction points
        # away from GT, 1 means the complete correction is beneficial. Unlike
        # an absolute-error teacher, this cannot reward a useless residual
        # merely because the frozen global base is already good.
        desired = target - baseline
        confidence_target = (
            (desired * correction).sum(dim=2, keepdim=True)
            / correction.square().sum(dim=2, keepdim=True).clamp_min(1e-8)
        ).detach().clamp(0.0, 1.0)
        confidence_loss = masked_mean(
            torch.abs(confidence - confidence_target), mask
        )
        confidence_regularization = masked_mean(1.0 - confidence, mask)

        weights = self.weights
        total = (
            weights.reconstruction * reconstruction
            + weights.residual * residual_loss
            + weights.missing_detail * missing_detail
            + weights.non_detail_suppression * non_detail_suppression
            + weights.innovation * innovation
            + weights.innovation_span * innovation_span
            + weights.innovation_zero * innovation_zero
            + weights.fidelity_guard * fidelity_guard
            + weights.high_frequency * hf_loss
            + weights.gradient * gradients
            + weights.wavelet * wavelet
            + weights.observation * observation
            + weights.low_frequency_drift * low_frequency_drift
            + weights.confidence * confidence_loss
            + weights.confidence_regularization * confidence_regularization
        )
        stats = {
            "total": total.detach(),
            "reconstruction": reconstruction.detach(),
            "residual": residual_loss.detach(),
            "missing_detail": missing_detail.detach(),
            "non_detail_suppression": non_detail_suppression.detach(),
            "innovation": innovation.detach(),
            "innovation_span": innovation_span.detach(),
            "innovation_zero": innovation_zero.detach(),
            "fidelity_guard": fidelity_guard.detach(),
            **{
                name: value.detach()
                for name, value in innovation_stats.items()
            },
            "high_frequency": hf_loss.detach(),
            "gradient": gradients.detach(),
            "wavelet": wavelet.detach(),
            "observation": observation.detach(),
            "low_frequency_drift": low_frequency_drift.detach(),
            "confidence": confidence_loss.detach(),
            "confidence_mean": masked_mean(confidence, mask).detach(),
            "confidence_target_mean": masked_mean(
                confidence_target, mask
            ).detach(),
            "confidence_target_correlation": masked_correlation(
                confidence, confidence_target, mask
            ).detach(),
            "confidence_error_correlation": masked_correlation(
                confidence, -candidate_error, mask
            ).detach(),
        }
        return total, stats
