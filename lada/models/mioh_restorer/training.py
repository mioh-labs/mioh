# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Training utilities for the flow-free Mioh restoration model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .model import MiohRestorerV1


@dataclass(frozen=True)
class RestorationLoss:
    total: torch.Tensor
    pixel: torch.Tensor
    gradient: torch.Tensor
    temporal: torch.Tensor
    high_frequency: torch.Tensor
    perceptual: torch.Tensor
    structural: torch.Tensor


class MaskedVGG16PerceptualLoss(nn.Module):
    """Frozen VGG16 feature loss evaluated only around restoration masks.

    A subset of frames is resized before feature extraction so long recurrent
    sequences remain practical on Apple Silicon.  The target branch is
    detached, while the restored branch keeps its gradient into the restorer.
    """

    FEATURE_LAYERS = (3, 8, 15)  # relu1_2, relu2_2, relu3_3
    FEATURE_WEIGHTS = (0.1, 0.2, 1.0)

    def __init__(
        self,
        *,
        frame_stride: int = 4,
        image_size: int = 224,
        features: nn.Sequential | None = None,
    ) -> None:
        super().__init__()
        if frame_stride <= 0:
            raise ValueError("frame_stride must be positive")
        if image_size < 32:
            raise ValueError("image_size must be at least 32")
        if features is None:
            from torchvision.models import VGG16_Weights, vgg16

            features = vgg16(
                weights=VGG16_Weights.IMAGENET1K_V1,
                progress=True,
            ).features[: max(self.FEATURE_LAYERS) + 1]
        if len(features) <= max(self.FEATURE_LAYERS):
            raise ValueError("VGG feature extractor does not contain required layers")
        self.features = features.eval()
        self.features.requires_grad_(False)
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        )

    @staticmethod
    def _flatten_sampled_frames(values: torch.Tensor, stride: int) -> torch.Tensor:
        sampled = values[:, ::stride]
        return sampled.reshape(-1, *sampled.shape[2:])

    def _extract(self, values: torch.Tensor) -> list[torch.Tensor]:
        result: list[torch.Tensor] = []
        for index, layer in enumerate(self.features):
            values = layer(values)
            if index in self.FEATURE_LAYERS:
                result.append(values)
        return result

    def forward(
        self,
        restored: torch.Tensor,
        target: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        if restored.shape != target.shape or restored.ndim != 5:
            raise ValueError("restored and target must have matching B,T,C,H,W shapes")
        if masks.shape != (restored.shape[0], restored.shape[1], 1, *restored.shape[-2:]):
            raise ValueError("masks do not match restored tensor dimensions")

        restored_frames = self._flatten_sampled_frames(restored, self.frame_stride)
        target_frames = self._flatten_sampled_frames(target, self.frame_stride)
        feature_masks = self._flatten_sampled_frames(masks, self.frame_stride)
        size = (self.image_size, self.image_size)
        restored_frames = F.interpolate(
            restored_frames, size=size, mode="bilinear", align_corners=False
        )
        target_frames = F.interpolate(
            target_frames, size=size, mode="bilinear", align_corners=False
        )
        feature_masks = F.interpolate(
            feature_masks, size=size, mode="bilinear", align_corners=False
        )
        # Include context touched by VGG's receptive field at the ROI boundary.
        feature_masks = F.max_pool2d(feature_masks, kernel_size=5, stride=1, padding=2)
        restored_frames = (restored_frames - self.mean) / self.std
        target_frames = (target_frames - self.mean) / self.std

        restored_features = self._extract(restored_frames)
        with torch.no_grad():
            target_features = self._extract(target_frames)
        losses = []
        for weight, restored_feature, target_feature in zip(
            self.FEATURE_WEIGHTS,
            restored_features,
            target_features,
            strict=True,
        ):
            layer_mask = F.interpolate(
                feature_masks,
                size=restored_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            losses.append(
                weight * _masked_mean((restored_feature - target_feature).abs(), layer_mask)
            )
        return sum(losses) / sum(self.FEATURE_WEIGHTS)


def run_training_sequence(
    model: MiohRestorerV1,
    frames: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Unroll a sequence using the exact state contract used at inference."""
    if frames.ndim != 5 or frames.shape[2] != 3:
        raise ValueError("frames must have shape [B,T,3,H,W]")
    if masks.ndim != 5 or masks.shape[2] != 1:
        raise ValueError("masks must have shape [B,T,1,H,W]")
    if frames.shape[:2] != masks.shape[:2] or frames.shape[-2:] != masks.shape[-2:]:
        raise ValueError("frames and masks must have matching batch, time and image dimensions")
    if frames.shape[1] % model.chunk_frames:
        raise ValueError("training sequence length must be divisible by chunk_frames")

    state = model.initial_state(frames)
    restored_chunks: list[torch.Tensor] = []
    for start in range(0, frames.shape[1], model.chunk_frames):
        restored, state = model(
            frames[:, start : start + model.chunk_frames],
            masks[:, start : start + model.chunk_frames],
            state,
        )
        restored_chunks.append(restored)
    return torch.cat(restored_chunks, dim=1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded_mask = mask.expand_as(values)
    return (values * expanded_mask).sum() / expanded_mask.sum().clamp_min(1.0)


def masked_charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    masks: torch.Tensor,
    *,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target tensors must have identical shapes")
    if masks.shape != (prediction.shape[0], prediction.shape[1], 1, *prediction.shape[-2:]):
        raise ValueError("masks do not match prediction tensor dimensions")
    return _masked_mean(
        torch.sqrt((prediction - target).square() + epsilon**2),
        masks,
    )


def masked_multiscale_structural_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    masks: torch.Tensor,
    *,
    frame_stride: int = 2,
    levels: int = 3,
    kernel_size: int = 7,
) -> torch.Tensor:
    """Masked multi-scale SSIM-style structural loss for video frames."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must have matching B,T,C,H,W shapes")
    if masks.shape != (prediction.shape[0], prediction.shape[1], 1, *prediction.shape[-2:]):
        raise ValueError("masks do not match prediction tensor dimensions")
    if frame_stride <= 0 or levels <= 0 or kernel_size <= 1 or kernel_size % 2 == 0:
        raise ValueError("invalid structural loss settings")

    prediction = prediction[:, ::frame_stride].reshape(
        -1, 3, *prediction.shape[-2:]
    )
    target = target[:, ::frame_stride].reshape(-1, 3, *target.shape[-2:])
    masks = masks[:, ::frame_stride].reshape(-1, 1, *masks.shape[-2:])
    losses: list[torch.Tensor] = []
    c1 = 0.01**2
    c2 = 0.03**2
    padding = kernel_size // 2
    for level in range(levels):
        mu_prediction = F.avg_pool2d(
            prediction, kernel_size, stride=1, padding=padding
        )
        mu_target = F.avg_pool2d(target, kernel_size, stride=1, padding=padding)
        variance_prediction = F.avg_pool2d(
            prediction.square(), kernel_size, stride=1, padding=padding
        ) - mu_prediction.square()
        variance_target = F.avg_pool2d(
            target.square(), kernel_size, stride=1, padding=padding
        ) - mu_target.square()
        covariance = F.avg_pool2d(
            prediction * target, kernel_size, stride=1, padding=padding
        ) - mu_prediction * mu_target
        numerator = (2.0 * mu_prediction * mu_target + c1) * (
            2.0 * covariance + c2
        )
        denominator = (
            mu_prediction.square() + mu_target.square() + c1
        ) * (variance_prediction + variance_target + c2)
        ssim = numerator / denominator.clamp_min(torch.finfo(prediction.dtype).eps)
        structural_error = ((1.0 - ssim) * 0.5).clamp(0.0, 1.0)
        expanded_mask = F.max_pool2d(masks, 3, stride=1, padding=1)
        losses.append(_masked_mean(structural_error, expanded_mask))
        if level + 1 < levels:
            prediction = F.avg_pool2d(prediction, 2, stride=2)
            target = F.avg_pool2d(target, 2, stride=2)
            masks = F.max_pool2d(masks, 2, stride=2)
    return sum(losses) / len(losses)


def masked_high_frequency_loss(
    restored: torch.Tensor,
    target: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Compare Laplacian detail inside the restoration region."""
    if restored.shape != target.shape:
        raise ValueError("restored and target tensors must have identical shapes")
    if masks.shape != (restored.shape[0], restored.shape[1], 1, *restored.shape[-2:]):
        raise ValueError("masks do not match restored tensor dimensions")
    flat_restored = restored.reshape(-1, 3, *restored.shape[-2:])
    flat_target = target.reshape(-1, 3, *target.shape[-2:])
    flat_masks = masks.reshape(-1, 1, *masks.shape[-2:])
    kernel = restored.new_tensor(
        ((0.0, -1.0, 0.0), (-1.0, 4.0, -1.0), (0.0, -1.0, 0.0))
    ).view(1, 1, 3, 3).expand(3, 1, 3, 3)
    restored_detail = F.conv2d(flat_restored, kernel, padding=1, groups=3)
    target_detail = F.conv2d(flat_target, kernel, padding=1, groups=3)
    # Ignore padding edges and include a one-pixel neighborhood of the ROI.
    detail_mask = F.max_pool2d(flat_masks, kernel_size=3, stride=1, padding=1)
    detail_mask[..., (0, -1), :] = 0
    detail_mask[..., :, (0, -1)] = 0
    return _masked_mean((restored_detail - target_detail).abs(), detail_mask)


def restoration_loss(
    restored: torch.Tensor,
    target: torch.Tensor,
    masks: torch.Tensor,
    *,
    gradient_weight: float = 0.2,
    temporal_weight: float = 0.1,
    high_frequency_weight: float = 0.0,
    perceptual_weight: float = 0.0,
    perceptual: torch.Tensor | None = None,
    structural_weight: float = 0.0,
    structural: torch.Tensor | None = None,
    charbonnier_epsilon: float = 1e-3,
) -> RestorationLoss:
    """Mask-weighted pixel, edge and temporal reconstruction objective."""
    if restored.shape != target.shape:
        raise ValueError("restored and target tensors must have identical shapes")
    if masks.shape != (restored.shape[0], restored.shape[1], 1, *restored.shape[-2:]):
        raise ValueError("masks do not match restored tensor dimensions")

    error = torch.sqrt((restored - target).square() + charbonnier_epsilon**2)
    pixel = _masked_mean(error, masks)

    restored_dx = restored[..., :, 1:] - restored[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    mask_dx = torch.minimum(masks[..., :, 1:], masks[..., :, :-1])
    restored_dy = restored[..., 1:, :] - restored[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    mask_dy = torch.minimum(masks[..., 1:, :], masks[..., :-1, :])
    gradient = 0.5 * (
        _masked_mean((restored_dx - target_dx).abs(), mask_dx)
        + _masked_mean((restored_dy - target_dy).abs(), mask_dy)
    )

    if restored.shape[1] > 1:
        restored_dt = restored[:, 1:] - restored[:, :-1]
        target_dt = target[:, 1:] - target[:, :-1]
        mask_dt = torch.minimum(masks[:, 1:], masks[:, :-1])
        temporal = _masked_mean((restored_dt - target_dt).abs(), mask_dt)
    else:
        temporal = restored.new_zeros(())

    high_frequency = (
        masked_high_frequency_loss(restored, target, masks)
        if high_frequency_weight > 0
        else restored.new_zeros(())
    )
    if perceptual is None:
        if perceptual_weight > 0:
            raise ValueError("perceptual loss value is required when its weight is positive")
        perceptual = restored.new_zeros(())
    if structural is None:
        if structural_weight > 0:
            raise ValueError("structural loss value is required when its weight is positive")
        structural = restored.new_zeros(())
    total = (
        pixel
        + gradient_weight * gradient
        + temporal_weight * temporal
        + high_frequency_weight * high_frequency
        + perceptual_weight * perceptual
        + structural_weight * structural
    )
    return RestorationLoss(
        total,
        pixel,
        gradient,
        temporal,
        high_frequency,
        perceptual,
        structural,
    )


def masked_psnr(
    prediction: torch.Tensor,
    target: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """Compute PSNR inside the restoration mask for tensors in [0, 1]."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target tensors must have identical shapes")
    if masks.shape != (
        prediction.shape[0],
        prediction.shape[1],
        1,
        *prediction.shape[-2:],
    ):
        raise ValueError("masks do not match prediction tensor dimensions")
    mse = _masked_mean((prediction - target).square(), masks)
    return -10.0 * torch.log10(mse.clamp_min(torch.finfo(mse.dtype).tiny))
