# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Self-contained training supervision for MiohRestorer V5.

No pretrained restoration or optical-flow network is used here. Stage 1 gets
analytic motion labels from synthetic translations, Stage 2 learns natural
alignment from clean non-mosaic context, and Stages 5/6 build detached local
correspondences directly from paired clean ground truth.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from .losses_v5 import masked_charbonnier, masked_mean
from .model_v5 import (
    CENTER_INDEX,
    NUM_INPUT_FRAMES,
    MiohRestorerV5,
    V5EncodedFrame,
    shift2d,
)


def _translate(
    values: torch.Tensor,
    vertical: torch.Tensor,
    horizontal: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Translate every BCHW sample by a potentially fractional pixel offset."""

    if values.ndim != 4 or vertical.shape != (values.shape[0],) or horizontal.shape != vertical.shape:
        raise ValueError("translation inputs have incompatible shapes")
    batch, _channels, height, width = values.shape
    theta = values.new_zeros(batch, 2, 3)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = -2.0 * horizontal / max(width - 1, 1)
    theta[:, 1, 2] = -2.0 * vertical / max(height - 1, 1)
    grid = F.affine_grid(theta, values.shape, align_corners=True)
    return F.grid_sample(
        values,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )


def make_known_motion_window(
    values: torch.Tensor,
    *,
    maximum_translation: float = 40.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a nine-frame synthetic trajectory and return frame-0 motion.

    All frames originate from the same real source frame, so the exact motion
    is known without an external teacher. Quarter-pixel phases teach the folded
    path to interpolate rather than only select integer candidates.
    """

    if values.ndim != 5 or values.shape[1] != NUM_INPUT_FRAMES:
        raise ValueError("known-motion source must be [B,9,C,H,W]")
    if maximum_translation <= 0:
        raise ValueError("maximum translation must be positive")
    batch = values.shape[0]
    device = values.device
    steps = max(1, int(round(maximum_translation * 4)))
    vertical = torch.randint(-steps, steps + 1, (batch,), device=device).to(values.dtype) / 4
    horizontal = torch.randint(-steps, steps + 1, (batch,), device=device).to(values.dtype) / 4
    too_small = vertical.abs() + horizontal.abs() < 1
    horizontal = torch.where(too_small, horizontal.new_full((), 1.0), horizontal)
    source = values[:, CENTER_INDEX]
    frames = []
    for index in range(NUM_INPUT_FRAMES):
        factor = (CENTER_INDEX - index) / CENTER_INDEX
        shifted_rgb = _translate(
            source[:, :3], vertical * factor, horizontal * factor, mode="bilinear"
        )
        shifted_aux = _translate(
            source[:, 3:], vertical * factor, horizontal * factor, mode="nearest"
        )
        frames.append(torch.cat((shifted_rgb, shifted_aux), dim=1))
    motion_frame_zero = torch.stack((vertical, horizontal), dim=1)
    return torch.stack(frames, dim=1), motion_frame_zero


def _alignment_offsets_and_scales(model: MiohRestorerV5) -> tuple[tuple[tuple[int, int], ...], tuple[int, ...]]:
    alignment = model.decoder.alignment
    offsets = (
        *alignment.offset_sets_16,
        alignment.offsets_8,
        alignment.offsets_4,
        alignment.offsets_2,
        alignment.phase_bank.offsets,
    )
    scales = (16,) * len(alignment.offset_sets_16) + (8, 4, 2, 1)
    return offsets, scales


def _hierarchical_targets(
    motion: torch.Tensor,
    offsets: Sequence[Sequence[tuple[int, int]]],
    scales: Sequence[int],
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Greedily decompose the inverse motion over V5's search pyramid."""

    if len(offsets) != len(scales):
        raise ValueError("alignment offsets and scales do not match")
    remaining = -motion.float()
    targets: list[torch.Tensor] = []
    for stage_offsets, scale in zip(offsets, scales, strict=True):
        candidates = torch.tensor(stage_offsets, device=motion.device, dtype=torch.float32)
        candidates = candidates * float(scale)
        distance = (remaining[:, None] - candidates[None]).square().sum(dim=2)
        if scale == 1:
            distribution = torch.softmax(-distance / 0.35, dim=1)
            selected = (distribution[..., None] * candidates[None]).sum(dim=1)
        else:
            index = distance.argmin(dim=1)
            distribution = F.one_hot(index, len(stage_offsets)).float()
            selected = candidates[index]
        remaining = remaining - selected
        targets.append(distribution.to(dtype=dtype))
    return tuple(targets)


def known_motion_alignment_loss(
    model: MiohRestorerV5,
    values: torch.Tensor,
    *,
    maximum_translation: float = 40.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    synthetic, motion = make_known_motion_window(
        values, maximum_translation=maximum_translation
    )
    aligned, weights = model.alignment_diagnostics(
        synthetic, reference_index=CENTER_INDEX, target_index=0
    )
    offsets, scales = _alignment_offsets_and_scales(model)
    targets = _hierarchical_targets(motion, offsets, scales, dtype=values.dtype)
    losses = []
    accuracies = []
    for student, target in zip(weights, targets, strict=True):
        # Synthetic motion is global. Averaging away boundaries makes this a
        # stable distribution target for every native bucket.
        height, width = student.shape[-2:]
        margin_y = max(1, height // 8)
        margin_x = max(1, width // 8)
        pooled = student[..., margin_y : height - margin_y, margin_x : width - margin_x]
        if pooled.numel() == 0:
            pooled = student
        pooled = pooled.mean(dim=(-2, -1)).clamp_min(1e-7)
        losses.append(-(target * pooled.log()).sum(dim=1).mean())
        accuracies.append((pooled.argmax(dim=1) == target.argmax(dim=1)).float().mean())
    reliability = -torch.log(aligned.reliability.clamp_min(1e-6)).mean()
    occlusion = -torch.log((1.0 - aligned.occlusion).clamp_min(1e-6)).mean()
    loss = sum(losses) / len(losses) + 0.1 * (reliability + occlusion)
    return loss, {
        "known_motion": float(loss.detach()),
        "known_motion_accuracy": float(torch.stack(accuracies).mean().detach()),
        "known_reliability": float(aligned.reliability.mean().detach()),
    }


def _normalized_feature_error(
    aligned: torch.Tensor, reference: torch.Tensor, validity: torch.Tensor
) -> torch.Tensor:
    aligned = F.normalize(aligned.float(), dim=1)
    reference = F.normalize(reference.detach().float(), dim=1)
    mask = F.interpolate(validity, size=aligned.shape[-2:], mode="nearest")
    return masked_mean((aligned - reference).abs(), mask)


def natural_alignment_losses(
    model: MiohRestorerV5,
    values: torch.Tensor,
    *,
    target_indices: tuple[int, ...] = (0, 8),
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Photometric and feature consistency on real nine-frame motion."""

    encoded = model.encode_window(values)
    reference = encoded[CENTER_INDEX]
    reference_values = values[:, CENTER_INDEX]
    natural_losses = []
    feature_losses = []
    gate_losses = []
    reliability_means = []
    for index in target_indices:
        aligned, _weights = model.decoder.alignment.forward_with_diagnostics(
            reference, encoded[index]
        )
        aligned_values = F.pixel_shuffle(aligned.packed, 2)
        clean_context = (1.0 - reference_values[:, 3:4]) * (
            1.0 - aligned_values[:, 3:4]
        )
        clean_context = clean_context * aligned_values[:, 4:5].clamp(0, 1)
        difference = (reference_values[:, :3] - aligned_values[:, :3]).abs()
        natural_losses.append(masked_mean(difference, clean_context))
        feature_losses.append(
            sum(
                _normalized_feature_error(aligned_feature, reference_feature, clean_context)
                for aligned_feature, reference_feature in zip(
                    (aligned.half, aligned.quarter, aligned.eighth, aligned.sixteenth),
                    (reference.half, reference.quarter, reference.eighth, reference.sixteenth),
                    strict=True,
                )
            )
            / 4.0
        )
        gate_target = torch.exp(
            -difference.detach().mean(dim=1, keepdim=True) / 0.08
        )
        gate_target = F.interpolate(
            gate_target, size=aligned.reliability.shape[-2:], mode="area"
        )
        valid_gate = F.interpolate(
            clean_context, size=aligned.reliability.shape[-2:], mode="nearest"
        )
        gate_losses.append(
            masked_mean(
                F.binary_cross_entropy(
                    aligned.reliability, gate_target, reduction="none"
                ),
                valid_gate,
            )
            + masked_mean(
                F.binary_cross_entropy(
                    aligned.occlusion, 1.0 - gate_target, reduction="none"
                ),
                valid_gate,
            )
        )
        reliability_means.append(aligned.reliability.mean())
    natural = sum(natural_losses) / len(natural_losses) + 0.1 * sum(gate_losses) / len(gate_losses)
    feature = sum(feature_losses) / len(feature_losses)
    return natural, feature, {
        "natural_motion": float(natural.detach()),
        "self_feature_consistency": float(feature.detach()),
        "natural_reliability": float(torch.stack(reliability_means).mean().detach()),
    }


def _local_clean_alignment_weights(
    current: torch.Tensor,
    previous: torch.Tensor,
    *,
    radius: int = 2,
    scale: int = 4,
    temperature: float = 0.08,
) -> tuple[torch.Tensor, tuple[tuple[int, int], ...]]:
    current_small = F.avg_pool2d(current, scale, scale)
    previous_small = F.avg_pool2d(previous, scale, scale)
    current_small = F.normalize(current_small.float(), dim=1)
    previous_small = F.normalize(previous_small.float(), dim=1)
    offsets = tuple(
        (vertical, horizontal)
        for vertical in range(-radius, radius + 1)
        for horizontal in range(-radius, radius + 1)
    )
    candidates = torch.stack(
        [shift2d(previous_small, vertical, horizontal) for vertical, horizontal in offsets],
        dim=1,
    )
    logits = (current_small.unsqueeze(1) * candidates).sum(dim=2) / temperature
    weights = torch.softmax(logits, dim=1).detach()
    return weights, offsets


def _warp_from_low_resolution_weights(
    values: torch.Tensor,
    weights: torch.Tensor,
    offsets: Sequence[tuple[int, int]],
    *,
    scale: int,
) -> torch.Tensor:
    full_weights = F.interpolate(
        weights, size=values.shape[-2:], mode="bilinear", align_corners=False
    )
    candidates = torch.stack(
        [shift2d(values, vertical * scale, horizontal * scale) for vertical, horizontal in offsets],
        dim=1,
    )
    return (full_weights.unsqueeze(2) * candidates).sum(dim=1)


def flow_aligned_temporal_tensors(
    restored: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    radius: int = 2,
    scale: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build detached clean-GT correspondences for temporal supervision."""

    if restored.shape != target.shape or restored.shape[1] < 2:
        raise ValueError("temporal supervision requires at least two outputs")
    aligned_restored = []
    aligned_target = []
    valid = []
    for index in range(1, target.shape[1]):
        weights, offsets = _local_clean_alignment_weights(
            target[:, index], target[:, index - 1], radius=radius, scale=scale
        )
        previous_restored = _warp_from_low_resolution_weights(
            restored[:, index - 1], weights, offsets, scale=scale
        )
        previous_target = _warp_from_low_resolution_weights(
            target[:, index - 1], weights, offsets, scale=scale
        )
        previous_mask = _warp_from_low_resolution_weights(
            mask[:, index - 1], weights, offsets, scale=scale
        )
        photometric = torch.exp(
            -(target[:, index] - previous_target).detach().abs().mean(dim=1, keepdim=True)
            / 0.08
        )
        aligned_restored.append(previous_restored)
        aligned_target.append(previous_target)
        valid.append(previous_mask.clamp(0, 1) * photometric)
    return (
        torch.stack(aligned_restored, dim=1),
        torch.stack(aligned_target, dim=1),
        torch.stack(valid, dim=1),
    )


class V5PerceptualLoss(nn.Module):
    """Frozen VGG16 ROI feature loss owned by the V5 training stack."""

    FEATURE_LAYERS = (3, 8, 15)
    FEATURE_WEIGHTS = (0.1, 0.2, 1.0)

    def __init__(self, *, image_size: int = 224) -> None:
        super().__init__()
        if image_size < 32:
            raise ValueError("perceptual image size must be at least 32")
        from torchvision.models import VGG16_Weights, vgg16

        self.features = vgg16(
            weights=VGG16_Weights.IMAGENET1K_V1, progress=True
        ).features[: max(self.FEATURE_LAYERS) + 1].eval()
        self.features.requires_grad_(False)
        self.image_size = image_size
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
        )

    def _extract(self, values: torch.Tensor) -> list[torch.Tensor]:
        result = []
        for index, layer in enumerate(self.features):
            values = layer(values)
            if index in self.FEATURE_LAYERS:
                result.append(values)
        return result

    def forward(
        self, restored: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        shape = restored.shape
        if restored.shape != target.shape or restored.ndim != 5:
            raise ValueError("perceptual tensors must be matching B,T,C,H,W")
        restored = restored.reshape(-1, 3, *shape[-2:])
        target = target.reshape(-1, 3, *shape[-2:])
        mask = mask.reshape(-1, 1, *shape[-2:])
        size = (self.image_size, self.image_size)
        restored = F.interpolate(restored, size=size, mode="bilinear", align_corners=False)
        target = F.interpolate(target, size=size, mode="bilinear", align_corners=False)
        mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
        mask = F.max_pool2d(mask, 5, stride=1, padding=2)
        restored_features = self._extract((restored - self.mean) / self.std)
        with torch.no_grad():
            target_features = self._extract((target - self.mean) / self.std)
        losses = []
        for weight, prediction, truth in zip(
            self.FEATURE_WEIGHTS, restored_features, target_features, strict=True
        ):
            layer_mask = F.interpolate(mask, size=prediction.shape[-2:], mode="bilinear", align_corners=False)
            losses.append(weight * masked_mean((prediction - truth).abs(), layer_mask))
        return sum(losses) / sum(self.FEATURE_WEIGHTS)
