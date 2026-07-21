# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""ROI-normalized losses for MiohRestorerV4Q."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def masked_charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    values = torch.sqrt(
        (prediction.float() - target.float()).square() + epsilon * epsilon
    )
    weights = mask.float().expand_as(values)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def masked_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    weights = mask.float().expand_as(prediction)
    values = (prediction.float() - target.float()).abs()
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


class MiohRestorerV4Loss(nn.Module):
    def __init__(
        self,
        *,
        candidate_weight: float = 0.5,
        high_frequency_weight: float = 0.1,
        base_weight: float = 0.05,
        confidence_weight: float = 0.2,
        confidence_regularization_weight: float = 1e-3,
        temporal_weight: float = 0.2,
        temporal_acceleration_weight: float = 0.0,
        gradient_weight: float = 0.0,
        structural_weight: float = 0.0,
        confidence_scale: float = 0.1,
    ) -> None:
        super().__init__()
        weights = (
            candidate_weight,
            high_frequency_weight,
            base_weight,
            confidence_weight,
            confidence_regularization_weight,
            temporal_weight,
            temporal_acceleration_weight,
            gradient_weight,
            structural_weight,
        )
        if any(value < 0 for value in weights) or confidence_scale <= 0:
            raise ValueError("V4 loss weights must be non-negative")
        self.candidate_weight = candidate_weight
        self.high_frequency_weight = high_frequency_weight
        self.base_weight = base_weight
        self.confidence_weight = confidence_weight
        self.confidence_regularization_weight = (
            confidence_regularization_weight
        )
        self.temporal_weight = temporal_weight
        self.temporal_acceleration_weight = temporal_acceleration_weight
        self.gradient_weight = gradient_weight
        self.structural_weight = structural_weight
        self.confidence_scale = confidence_scale
        kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        kernel = torch.outer(kernel, kernel)
        kernel = (kernel / kernel.sum()).reshape(1, 1, 5, 5)
        self.register_buffer("gaussian_kernel", kernel.repeat(3, 1, 1, 1))

    def gaussian_blur(self, values: torch.Tensor) -> torch.Tensor:
        shape = values.shape
        flattened = values.reshape(-1, 3, shape[-2], shape[-1])
        kernel = self.gaussian_kernel.to(
            device=flattened.device, dtype=flattened.dtype
        )
        blurred = F.conv2d(
            F.pad(flattened, (2, 2, 2, 2), mode="replicate"),
            kernel,
            groups=3,
        )
        return blurred.reshape(shape)

    def forward(
        self,
        restored: torch.Tensor,
        confidence: torch.Tensor,
        base: torch.Tensor,
        texture: torch.Tensor,
        target: torch.Tensor,
        input_rgb: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        residual_target = target - input_rgb
        base_target = self.gaussian_blur(residual_target)
        texture_target = residual_target - base_target
        candidate = input_rgb + mask * (base + texture)
        reconstruction = masked_charbonnier(restored, target, mask)
        candidate_loss = masked_charbonnier(candidate, target, mask)
        high_frequency = masked_charbonnier(
            texture, texture_target, mask
        )
        base_loss = masked_charbonnier(base, base_target, mask)
        error = (target - candidate).abs().mean(dim=2, keepdim=True)
        confidence_target = torch.exp(
            -error.detach() / self.confidence_scale
        )
        confidence_loss = masked_l1(
            confidence, confidence_target, mask
        )
        confidence_regularization = (
            ((1.0 - confidence.float()) * mask.float()).sum()
            / mask.float().sum().clamp_min(1.0)
        )
        if restored.shape[1] > 1:
            # Only pixels editable in both frames provide a valid motion target.
            pair_mask = torch.minimum(mask[:, 1:], mask[:, :-1])
            temporal = masked_charbonnier(
                restored[:, 1:] - restored[:, :-1],
                target[:, 1:] - target[:, :-1],
                pair_mask,
            )
        else:
            temporal = restored.new_zeros(())
        if restored.shape[1] > 2 and self.temporal_acceleration_weight > 0:
            acceleration_mask = torch.minimum(
                torch.minimum(mask[:, 2:], mask[:, 1:-1]), mask[:, :-2]
            )
            temporal_acceleration = masked_charbonnier(
                restored[:, 2:] - 2.0 * restored[:, 1:-1] + restored[:, :-2],
                target[:, 2:] - 2.0 * target[:, 1:-1] + target[:, :-2],
                acceleration_mask,
            )
        else:
            temporal_acceleration = restored.new_zeros(())
        if self.gradient_weight > 0:
            restored_dx = restored[..., :, 1:] - restored[..., :, :-1]
            target_dx = target[..., :, 1:] - target[..., :, :-1]
            mask_dx = torch.minimum(mask[..., :, 1:], mask[..., :, :-1])
            restored_dy = restored[..., 1:, :] - restored[..., :-1, :]
            target_dy = target[..., 1:, :] - target[..., :-1, :]
            mask_dy = torch.minimum(mask[..., 1:, :], mask[..., :-1, :])
            gradient = 0.5 * (
                masked_l1(restored_dx, target_dx, mask_dx)
                + masked_l1(restored_dy, target_dy, mask_dy)
            )
        else:
            gradient = restored.new_zeros(())
        structural = (
            multiscale_structural_loss(restored, target, mask)
            if self.structural_weight > 0
            else restored.new_zeros(())
        )
        total = (
            reconstruction
            + self.candidate_weight * candidate_loss
            + self.high_frequency_weight * high_frequency
            + self.base_weight * base_loss
            + self.confidence_weight * confidence_loss
            + self.confidence_regularization_weight
            * confidence_regularization
            + self.temporal_weight * temporal
            + self.temporal_acceleration_weight * temporal_acceleration
            + self.gradient_weight * gradient
            + self.structural_weight * structural
        )
        return total, {
            "reconstruction": reconstruction,
            "candidate": candidate_loss,
            "high_frequency": high_frequency,
            "base": base_loss,
            "confidence": confidence_loss,
            "confidence_regularization": confidence_regularization,
            "temporal": temporal,
            "temporal_acceleration": temporal_acceleration,
            "gradient": gradient,
            "structural": structural,
            "confidence_mean": (
                (confidence.float() * mask.float()).sum()
                / mask.float().sum().clamp_min(1.0)
            ),
            "confidence_std": masked_standard_deviation(confidence, mask),
        }


def masked_standard_deviation(
    values: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    weights = mask.float().expand_as(values)
    count = weights.sum().clamp_min(1.0)
    mean = (values.float() * weights).sum() / count
    variance = ((values.float() - mean).square() * weights).sum() / count
    return torch.sqrt(variance.clamp_min(0.0))


def multiscale_structural_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    levels: int = 3,
) -> torch.Tensor:
    """ROI-normalized SSIM-style loss at full, half and quarter scale."""

    prediction = prediction.reshape(-1, 3, *prediction.shape[-2:])
    target = target.reshape(-1, 3, *target.shape[-2:])
    mask = mask.reshape(-1, 1, *mask.shape[-2:])
    losses: list[torch.Tensor] = []
    c1 = 0.01**2
    c2 = 0.03**2
    for level in range(levels):
        mean_prediction = F.avg_pool2d(prediction, 7, stride=1, padding=3)
        mean_target = F.avg_pool2d(target, 7, stride=1, padding=3)
        variance_prediction = F.avg_pool2d(
            prediction.square(), 7, stride=1, padding=3
        ) - mean_prediction.square()
        variance_target = F.avg_pool2d(
            target.square(), 7, stride=1, padding=3
        ) - mean_target.square()
        covariance = F.avg_pool2d(
            prediction * target, 7, stride=1, padding=3
        ) - mean_prediction * mean_target
        similarity = (
            (2.0 * mean_prediction * mean_target + c1)
            * (2.0 * covariance + c2)
            / (
                (mean_prediction.square() + mean_target.square() + c1)
                * (variance_prediction + variance_target + c2)
            ).clamp_min(1e-8)
        )
        losses.append(masked_l1(similarity, torch.ones_like(similarity), mask) * 0.5)
        if level + 1 < levels:
            prediction = F.avg_pool2d(prediction, 2, stride=2)
            target = F.avg_pool2d(target, 2, stride=2)
            mask = F.max_pool2d(mask, 2, stride=2)
    return sum(losses) / len(losses)


def overlap_consistency_loss(
    first_restored: torch.Tensor,
    second_restored: torch.Tensor,
    first_mask: torch.Tensor,
    second_mask: torch.Tensor,
) -> torch.Tensor:
    """Deprecated: adjacent V4 windows produce the shared frame identically.

    Kept for checkpoint-era callers and tests.  It must not be used for V4
    training because both sides have the same local five-frame context and the
    loss has zero gradient.
    """

    overlap_mask = torch.maximum(first_mask[:, -1:], second_mask[:, :1])
    return masked_charbonnier(
        first_restored[:, -1:],
        second_restored[:, :1],
        overlap_mask,
    )


@torch.no_grad()
def confidence_error_correlation(
    confidence: torch.Tensor,
    error: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    selected = mask > 0
    confidence_values = confidence.expand_as(error)[selected].float()
    error_values = (-error)[selected].float()
    if confidence_values.numel() < 2:
        return confidence.new_zeros(())
    confidence_values = confidence_values - confidence_values.mean()
    error_values = error_values - error_values.mean()
    denominator = torch.sqrt(
        confidence_values.square().sum()
        * error_values.square().sum()
    )
    if float(denominator) <= 1e-12:
        return confidence.new_zeros(())
    return (confidence_values * error_values).sum() / denominator
