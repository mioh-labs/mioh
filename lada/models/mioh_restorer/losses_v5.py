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
    ) -> None:
        super().__init__()
        self.weights: V5LossWeights = weights or stage_definition(stage).loss
        self.confidence_scale = float(confidence_scale)
        self.boundary_temporal_weight = float(boundary_temporal_weight)
        if self.boundary_temporal_weight < 0:
            raise ValueError("boundary temporal weight must be non-negative")

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
    ) -> tuple[torch.Tensor, dict[str, float]]:
        weights = self.weights
        reconstruction = masked_charbonnier(restored, target, mask)
        candidate = source + mask * (base + texture)
        candidate_loss = masked_charbonnier(candidate, target, mask)
        base_loss = masked_charbonnier(
            source + mask * base, gaussian_blur(target), mask
        )
        hf_loss = masked_charbonnier(
            mask * texture, mask * high_frequency(target), mask
        )
        gradients = gradient_loss(restored, target, mask)
        wavelet = wavelet_loss(restored, target, mask)

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
        total = (
            weights.reconstruction * reconstruction
            + weights.candidate * candidate_loss
            + weights.base * base_loss
            + weights.high_frequency * hf_loss
            + weights.wavelet * wavelet
            + weights.gradient * gradients
            + weights.perceptual * perceptual_loss
            + weights.temporal * (temporal + 0.25 * temporal_acceleration)
            + self.boundary_temporal_weight * boundary_temporal
            + weights.confidence * confidence_loss
            + weights.confidence_regularization * confidence_regularization
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
        }
        return total, stats
