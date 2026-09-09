# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""ROI-normalized structure, detail, confidence and aligned-time V5 losses."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .curriculum_v5 import V5LossWeights, stage_definition


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def masked_correlation(
    left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    left_mean = masked_mean(left, mask)
    right_mean = masked_mean(right, mask)
    left_centered = left - left_mean
    right_centered = right - right_mean
    covariance = masked_mean(left_centered * right_centered, mask)
    denominator = torch.sqrt(
        masked_mean(left_centered.square(), mask)
        * masked_mean(right_centered.square(), mask)
        + 1e-12
    )
    return covariance / denominator


def masked_projection_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target projection gain and orthogonal residual ratios.

    Unlike ``correlation * amplitude_ratio``, these values use the same
    centered moments and are therefore an exact least-squares decomposition
    inside the supplied mask.  The energy ratio is variance-normalized; its
    square root is reported separately as an RMS amplitude ratio.
    """

    prediction_centered = prediction - masked_mean(prediction, mask)
    target_centered = target - masked_mean(target, mask)
    target_variance = masked_mean(target_centered.square(), mask).clamp_min(1e-12)
    prediction_variance = masked_mean(
        prediction_centered.square(), mask
    ).clamp_min(0.0)
    covariance = masked_mean(prediction_centered * target_centered, mask)
    projection_gain = covariance / target_variance
    orthogonal_energy_ratio = (
        prediction_variance - covariance.square() / target_variance
    ).clamp_min(0.0) / target_variance
    orthogonal_rms_ratio = torch.sqrt(orthogonal_energy_ratio)
    return projection_gain, orthogonal_rms_ratio, orthogonal_energy_ratio


def masked_local_correlation(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    patch_size: int = 32,
    stride: int | None = None,
    minimum_mask_fraction: float = 0.05,
    target_variance_floor: float = 1e-8,
) -> torch.Tensor:
    """Target-variance-weighted NCC over overlapping native-pixel patches."""

    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("local correlation tensors must be matching B,T,C,H,W")
    if mask.ndim != 5 or mask.shape[:2] != prediction.shape[:2] or mask.shape[2] != 1:
        raise ValueError("local correlation mask must be B,T,1,H,W")
    if patch_size <= 1:
        raise ValueError("local correlation patch size must exceed one")
    if not 0 < minimum_mask_fraction <= 1:
        raise ValueError("minimum mask fraction must be in (0, 1]")
    effective_patch_size = min(patch_size, prediction.shape[-2], prediction.shape[-1])
    if effective_patch_size <= 1:
        raise ValueError("local correlation tensors are too small")
    effective_stride = stride or max(1, effective_patch_size // 2)
    if effective_stride <= 0:
        raise ValueError("local correlation stride must be positive")

    shape = prediction.shape
    flat_prediction = prediction.reshape(-1, shape[2], shape[3], shape[4])
    flat_target = target.reshape_as(flat_prediction)
    flat_mask = mask.reshape(-1, 1, shape[3], shape[4]).clamp(0.0, 1.0)

    def pool(values: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(
            values,
            kernel_size=effective_patch_size,
            stride=effective_stride,
            ceil_mode=True,
        )

    coverage = pool(flat_mask)
    normalization = coverage.clamp_min(1e-6)
    prediction_mean = pool(flat_prediction * flat_mask) / normalization
    target_mean = pool(flat_target * flat_mask) / normalization
    prediction_second = pool(flat_prediction.square() * flat_mask) / normalization
    target_second = pool(flat_target.square() * flat_mask) / normalization
    cross = pool(flat_prediction * flat_target * flat_mask) / normalization
    prediction_variance = (prediction_second - prediction_mean.square()).clamp_min(0.0)
    target_variance = (target_second - target_mean.square()).clamp_min(0.0)
    covariance = cross - prediction_mean * target_mean
    correlation = covariance / torch.sqrt(
        prediction_variance * target_variance + 1e-12
    )
    valid = (
        (coverage >= minimum_mask_fraction)
        & (target_variance >= target_variance_floor)
    ).to(correlation.dtype)
    weights = valid * coverage * target_variance
    return (correlation * weights).sum() / weights.sum().clamp_min(1e-12)


def masked_charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    return masked_mean(
        torch.sqrt((prediction - target).square() + epsilon * epsilon), mask
    )


def gaussian_blur(values: torch.Tensor) -> torch.Tensor:
    shape = values.shape
    flat = values.reshape(-1, 3, shape[-2], shape[-1])
    kernel = values.new_tensor((1.0, 4.0, 6.0, 4.0, 1.0))
    kernel = torch.outer(kernel, kernel)
    kernel = (kernel / kernel.sum()).reshape(1, 1, 5, 5).repeat(3, 1, 1, 1)
    blurred = F.conv2d(F.pad(flat, (2, 2, 2, 2), mode="replicate"), kernel, groups=3)
    return blurred.reshape(shape)


def high_frequency(values: torch.Tensor) -> torch.Tensor:
    return values - gaussian_blur(values)


def image_gradients(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return values[..., 1:, :] - values[..., :-1, :], values[..., :, 1:] - values[..., :, :-1]


def gradient_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    pred_y, pred_x = image_gradients(prediction)
    target_y, target_x = image_gradients(target)
    mask_y = mask[..., 1:, :] * mask[..., :-1, :]
    mask_x = mask[..., :, 1:] * mask[..., :, :-1]
    return masked_charbonnier(pred_y, target_y, mask_y) + masked_charbonnier(
        pred_x, target_x, mask_x
    )


def haar_detail(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    top_left = values[..., 0::2, 0::2]
    top_right = values[..., 0::2, 1::2]
    bottom_left = values[..., 1::2, 0::2]
    bottom_right = values[..., 1::2, 1::2]
    horizontal = top_left - top_right + bottom_left - bottom_right
    vertical = top_left + top_right - bottom_left - bottom_right
    diagonal = top_left - top_right - bottom_left + bottom_right
    return horizontal * 0.5, vertical * 0.5, diagonal * 0.5


def wavelet_loss(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    reduced_mask = F.avg_pool2d(
        mask.reshape(-1, 1, mask.shape[-2], mask.shape[-1]), 2, 2
    ).reshape(*mask.shape[:-2], mask.shape[-2] // 2, mask.shape[-1] // 2)
    return sum(
        masked_charbonnier(pred, truth, reduced_mask)
        for pred, truth in zip(haar_detail(prediction), haar_detail(target), strict=True)
    ) / 3.0


def mask_boundary_band(mask: torch.Tensor, *, radius: int = 3) -> torch.Tensor:
    """Return a soft inner/outer band around an ROI mask boundary."""

    if mask.ndim != 5 or mask.shape[2] != 1:
        raise ValueError("boundary masks must be [B,T,1,H,W]")
    if radius <= 0:
        raise ValueError("boundary radius must be positive")
    shape = mask.shape
    flat = mask.reshape(-1, 1, shape[-2], shape[-1]).clamp(0.0, 1.0)
    kernel = radius * 2 + 1
    outer = F.max_pool2d(flat, kernel, stride=1, padding=radius)
    inner = 1.0 - F.max_pool2d(
        1.0 - flat, kernel, stride=1, padding=radius
    )
    return (outer - inner).clamp(0.0, 1.0).reshape_as(mask)


class MiohRestorerV5Loss(nn.Module):
    def __init__(
        self,
        *,
        stage: int | str = 3,
        weights: V5LossWeights | None = None,
        confidence_scale: float = 0.05,
        boundary_temporal_weight: float = 0.0,
        base_loss_weight_override: float | None = None,
        candidate_loss_weight_override: float | None = None,
        confidence_loss_weight_override: float | None = None,
        hf_amplitude_weight: float = 0.0,
        hf_correlation_weight: float = 0.0,
        hf_local_correlation_weight: float = 0.0,
        hf_local_correlation_patch_size: int = 32,
        hf_local_correlation_on_residual: bool = False,
        hf_residual_reconstruction_weight: float = 0.0,
        hf_correction_target_projection_weight: float = 0.0,
        hf_correction_target_orthogonal_energy_weight: float = 0.0,
        guard_ring_identity_weight: float = 0.0,
        supervise_final_high_frequency: bool = False,
    ) -> None:
        super().__init__()
        self.weights: V5LossWeights = weights or stage_definition(stage).loss
        self.confidence_scale = float(confidence_scale)
        self.boundary_temporal_weight = float(boundary_temporal_weight)
        self.base_loss_weight_override = base_loss_weight_override
        self.candidate_loss_weight_override = candidate_loss_weight_override
        self.confidence_loss_weight_override = confidence_loss_weight_override
        self.hf_amplitude_weight = float(hf_amplitude_weight)
        self.hf_correlation_weight = float(hf_correlation_weight)
        self.hf_local_correlation_weight = float(hf_local_correlation_weight)
        self.hf_local_correlation_patch_size = int(hf_local_correlation_patch_size)
        self.hf_local_correlation_on_residual = bool(
            hf_local_correlation_on_residual
        )
        self.hf_residual_reconstruction_weight = float(
            hf_residual_reconstruction_weight
        )
        self.hf_correction_target_projection_weight = float(
            hf_correction_target_projection_weight
        )
        self.hf_correction_target_orthogonal_energy_weight = float(
            hf_correction_target_orthogonal_energy_weight
        )
        self.guard_ring_identity_weight = float(guard_ring_identity_weight)
        self.supervise_final_high_frequency = bool(supervise_final_high_frequency)
        if self.boundary_temporal_weight < 0:
            raise ValueError("boundary temporal weight must be non-negative")
        if any(
            value is not None and value < 0
            for value in (
                base_loss_weight_override,
                candidate_loss_weight_override,
                confidence_loss_weight_override,
            )
        ) or min(
            self.hf_amplitude_weight,
            self.hf_correlation_weight,
            self.hf_local_correlation_weight,
            self.hf_residual_reconstruction_weight,
            self.hf_correction_target_projection_weight,
            self.hf_correction_target_orthogonal_energy_weight,
            self.guard_ring_identity_weight,
        ) < 0:
            raise ValueError("V5 diagnostic loss weights must be non-negative")
        if self.hf_local_correlation_patch_size <= 1:
            raise ValueError("HF local correlation patch size must exceed one")

    def forward(
        self,
        restored: torch.Tensor,
        confidence: torch.Tensor,
        base: torch.Tensor,
        texture: torch.Tensor,
        target: torch.Tensor,
        source: torch.Tensor,
        mask: torch.Tensor,
        *,
        aligned_previous_restored: torch.Tensor | None = None,
        aligned_previous_target: torch.Tensor | None = None,
        temporal_valid: torch.Tensor | None = None,
        perceptual: torch.Tensor | None = None,
        guard_ring_mask: torch.Tensor | None = None,
        hf_reference: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weights = self.weights
        reconstruction = masked_charbonnier(restored, target, mask)
        candidate = (
            restored
            if self.supervise_final_high_frequency
            else source + mask * (base + texture)
        )
        candidate_loss = masked_charbonnier(candidate, target, mask)
        base_loss = masked_charbonnier(
            source + mask * base, gaussian_blur(target), mask
        )
        hf_loss = masked_charbonnier(
            high_frequency(restored) if self.supervise_final_high_frequency else mask * texture,
            high_frequency(target) if self.supervise_final_high_frequency else mask * high_frequency(target),
            mask,
        )
        gradients = gradient_loss(restored, target, mask)
        wavelet = wavelet_loss(restored, target, mask)
        restored_hf = high_frequency(restored)
        target_hf = high_frequency(target)
        restored_hf_amplitude = torch.sqrt(
            masked_mean(restored_hf.square(), mask).clamp_min(1e-12)
        )
        target_hf_amplitude = torch.sqrt(
            masked_mean(target_hf.square(), mask).clamp_min(1e-12)
        )
        hf_amplitude_ratio = restored_hf_amplitude / target_hf_amplitude.clamp_min(1e-6)
        hf_amplitude = torch.abs(torch.log(hf_amplitude_ratio.clamp_min(1e-6)))
        hf_correlation_value = masked_correlation(restored_hf, target_hf, mask).clamp(-1, 1)
        hf_correlation = 1.0 - hf_correlation_value
        hf_local_correlation_value = masked_local_correlation(
            restored_hf,
            target_hf,
            mask,
            patch_size=self.hf_local_correlation_patch_size,
        ).clamp(-1, 1)
        hf_residual_local_correlation_value = restored.new_zeros(())
        hf_residual_reconstruction = restored.new_zeros(())
        hf_residual_target_rms = restored.new_zeros(())
        hf_residual_projection_gain = restored.new_zeros(())
        hf_residual_orthogonal_rms_ratio = restored.new_zeros(())
        hf_residual_orthogonal_energy_ratio = restored.new_zeros(())
        hf_correction_target_projection_gain = restored.new_zeros(())
        hf_correction_target_orthogonal_rms_ratio = restored.new_zeros(())
        hf_correction_target_orthogonal_energy_ratio = restored.new_zeros(())
        if hf_reference is not None:
            detached_reference = hf_reference.detach()
            correction_hf = high_frequency(restored - detached_reference)
            needed_correction_hf = high_frequency(target - detached_reference)
            hf_residual_target_rms = torch.sqrt(
                masked_mean(needed_correction_hf.square(), mask).clamp_min(1e-12)
            )
            residual_scale = hf_residual_target_rms.detach().clamp_min(1e-4)
            hf_residual_reconstruction = masked_charbonnier(
                correction_hf / residual_scale,
                needed_correction_hf / residual_scale,
                mask,
            )
            hf_residual_local_correlation_value = masked_local_correlation(
                correction_hf,
                needed_correction_hf,
                mask,
                patch_size=self.hf_local_correlation_patch_size,
            ).clamp(-1, 1)
            (
                hf_residual_projection_gain,
                hf_residual_orthogonal_rms_ratio,
                hf_residual_orthogonal_energy_ratio,
            ) = masked_projection_statistics(
                correction_hf, needed_correction_hf, mask
            )
            (
                hf_correction_target_projection_gain,
                hf_correction_target_orthogonal_rms_ratio,
                hf_correction_target_orthogonal_energy_ratio,
            ) = masked_projection_statistics(correction_hf, target_hf, mask)
        if (
            self.hf_local_correlation_on_residual
            or self.hf_residual_reconstruction_weight > 0
            or self.hf_correction_target_projection_weight > 0
            or self.hf_correction_target_orthogonal_energy_weight > 0
        ):
            if hf_reference is None:
                raise ValueError(
                    "HF residual supervision requires an HF reference"
                )
        if self.hf_local_correlation_on_residual:
            hf_local_correlation_objective_value = (
                hf_residual_local_correlation_value
            )
        else:
            hf_local_correlation_objective_value = hf_local_correlation_value
        hf_local_correlation = 1.0 - hf_local_correlation_objective_value
        (
            hf_projection_gain,
            hf_orthogonal_rms_ratio,
            hf_orthogonal_energy_ratio,
        ) = masked_projection_statistics(restored_hf, target_hf, mask)
        guard_ring_identity = restored.new_zeros(())
        if guard_ring_mask is not None:
            guard_ring_identity = masked_charbonnier(
                restored, target, guard_ring_mask
            )

        error = torch.mean(torch.abs(target - candidate), dim=2, keepdim=True)
        confidence_target = torch.exp(-error.detach() / self.confidence_scale)
        confidence_loss = masked_mean(
            torch.abs(confidence - confidence_target), mask
        )
        confidence_regularization = masked_mean(1.0 - confidence, mask)

        temporal = restored.new_zeros(())
        temporal_acceleration = restored.new_zeros(())
        boundary_temporal = restored.new_zeros(())
        if weights.temporal or self.boundary_temporal_weight:
            if (
                aligned_previous_restored is None
                or aligned_previous_target is None
                or temporal_valid is None
            ):
                raise ValueError("flow-aligned temporal tensors are required for this stage")
            current_delta = restored[:, 1:] - aligned_previous_restored
            target_delta = target[:, 1:] - aligned_previous_target
            temporal_error = current_delta - target_delta
            if weights.temporal:
                temporal = masked_charbonnier(
                    current_delta, target_delta, mask[:, 1:] * temporal_valid
                )
                if temporal_error.shape[1] > 1:
                    acceleration_valid = (
                        mask[:, 2:]
                        * temporal_valid[:, 1:]
                        * temporal_valid[:, :-1]
                    )
                    temporal_acceleration = masked_charbonnier(
                        temporal_error[:, 1:] - temporal_error[:, :-1],
                        torch.zeros_like(temporal_error[:, 1:]),
                        acceleration_valid,
                    )
            if self.boundary_temporal_weight:
                boundary = mask_boundary_band(mask[:, 1:])
                boundary_temporal = masked_charbonnier(
                    current_delta,
                    target_delta,
                    boundary * temporal_valid,
                )

        if weights.perceptual and perceptual is None:
            raise ValueError("perceptual loss value is required for this stage")
        perceptual_loss = restored.new_zeros(()) if perceptual is None else perceptual
        base_weight = (
            weights.base
            if self.base_loss_weight_override is None
            else self.base_loss_weight_override
        )
        # When the final output carries the high-frequency supervision the
        # candidate term degenerates into a second copy of ``reconstruction``,
        # so it must be disabled instead of silently reweighting it.
        candidate_weight = (
            weights.candidate
            if self.candidate_loss_weight_override is None
            else self.candidate_loss_weight_override
        )
        confidence_weight = (
            weights.confidence
            if self.confidence_loss_weight_override is None
            else self.confidence_loss_weight_override
        )
        total = (
            weights.reconstruction * reconstruction
            + candidate_weight * candidate_loss
            + base_weight * base_loss
            + weights.high_frequency * hf_loss
            + weights.wavelet * wavelet
            + weights.gradient * gradients
            + weights.perceptual * perceptual_loss
            + weights.temporal * (temporal + 0.25 * temporal_acceleration)
            + self.boundary_temporal_weight * boundary_temporal
            + confidence_weight * confidence_loss
            + weights.confidence_regularization * confidence_regularization
            + self.hf_amplitude_weight * hf_amplitude
            + self.hf_correlation_weight * hf_correlation
            + self.hf_local_correlation_weight * hf_local_correlation
            + self.hf_residual_reconstruction_weight
            * hf_residual_reconstruction
            - self.hf_correction_target_projection_weight
            * hf_correction_target_projection_gain
            + self.hf_correction_target_orthogonal_energy_weight
            * hf_correction_target_orthogonal_energy_ratio
            + self.guard_ring_identity_weight * guard_ring_identity
        )
        stats = {
            "total": float(total.detach()),
            "reconstruction": float(reconstruction.detach()),
            "candidate": float(candidate_loss.detach()),
            "base": float(base_loss.detach()),
            "high_frequency": float(hf_loss.detach()),
            "wavelet": float(wavelet.detach()),
            "gradient": float(gradients.detach()),
            "perceptual": float(perceptual_loss.detach()),
            "temporal": float(temporal.detach()),
            "temporal_acceleration": float(temporal_acceleration.detach()),
            "boundary_temporal": float(boundary_temporal.detach()),
            "confidence": float(confidence_loss.detach()),
            "confidence_mean": float(masked_mean(confidence, mask).detach()),
            "confidence_std": float(confidence.detach().std()),
            "confidence_error_correlation": float(
                masked_correlation(confidence, -error, mask).detach()
            ),
            "hf_amplitude_ratio": float(hf_amplitude_ratio.detach()),
            "hf_correlation": float(hf_correlation_value.detach()),
            "hf_amplitude_loss": float(hf_amplitude.detach()),
            "hf_correlation_loss": float(hf_correlation.detach()),
            "hf_local_correlation": float(hf_local_correlation_value.detach()),
            "hf_local_correlation_objective": float(
                hf_local_correlation_objective_value.detach()
            ),
            "hf_local_correlation_loss": float(hf_local_correlation.detach()),
            "hf_projection_gain": float(hf_projection_gain.detach()),
            "hf_orthogonal_rms_ratio": float(hf_orthogonal_rms_ratio.detach()),
            "hf_orthogonal_energy_ratio": float(
                hf_orthogonal_energy_ratio.detach()
            ),
            "hf_residual_local_correlation": float(
                hf_residual_local_correlation_value.detach()
            ),
            "hf_residual_reconstruction": float(
                hf_residual_reconstruction.detach()
            ),
            "hf_residual_target_rms": float(hf_residual_target_rms.detach()),
            "hf_residual_projection_gain": float(
                hf_residual_projection_gain.detach()
            ),
            "hf_residual_orthogonal_rms_ratio": float(
                hf_residual_orthogonal_rms_ratio.detach()
            ),
            "hf_residual_orthogonal_energy_ratio": float(
                hf_residual_orthogonal_energy_ratio.detach()
            ),
            "hf_correction_target_projection_gain": float(
                hf_correction_target_projection_gain.detach()
            ),
            "hf_correction_target_orthogonal_rms_ratio": float(
                hf_correction_target_orthogonal_rms_ratio.detach()
            ),
            "hf_correction_target_orthogonal_energy_ratio": float(
                hf_correction_target_orthogonal_energy_ratio.detach()
            ),
            "guard_ring_identity": float(guard_ring_identity.detach()),
        }
        return total, stats
