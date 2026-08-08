# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import MODELS


def _masked_charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    mask = mask.to(dtype=prediction.dtype)
    if mask.shape[1] == 1 and prediction.shape[1] != 1:
        mask = mask.expand(-1, prediction.shape[1], -1, -1)
    error = torch.sqrt((prediction - target).square() + eps * eps)
    denominator = mask.sum().clamp_min(1.0)
    return (error * mask).sum() / denominator


def _dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)


def _known_phase_block_average(
    image: torch.Tensor,
    *,
    block_size: int,
    phase: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average complete known-grid cells in one CHW crop.

    Recoverable-HF mosaics are generated before the 256-pixel crop is taken.
    Pixels outside that crop are therefore unavailable when a cell crosses a
    crop edge.  This helper returns a second tensor marking only complete cells
    so those partial measurements never receive an incorrect replicate-pad
    constraint.
    """

    if image.ndim != 3:
        raise ValueError("known-phase block average expects a CHW image")
    if block_size <= 1:
        raise ValueError("mosaic block size must exceed one pixel")

    _, height, width = image.shape
    phase_x = int(phase[0]) % block_size
    phase_y = int(phase[1]) % block_size
    end_x = width - ((width - phase_x) % block_size)
    end_y = height - ((height - phase_y) % block_size)
    if end_x <= phase_x or end_y <= phase_y:
        return torch.zeros_like(image), image.new_zeros((1, height, width))

    region = image[:, phase_y:end_y, phase_x:end_x].unsqueeze(0)
    pooled = F.avg_pool2d(region, kernel_size=block_size, stride=block_size)
    expanded = pooled.repeat_interleave(block_size, dim=2).repeat_interleave(
        block_size, dim=3
    )
    averaged = F.pad(
        expanded,
        (phase_x, width - end_x, phase_y, height - end_y),
    ).squeeze(0)
    valid = F.pad(
        image.new_ones((1, end_y - phase_y, end_x - phase_x)),
        (phase_x, width - end_x, phase_y, height - end_y),
    )
    return averaged, valid


@MODELS.register_module()
class KnownGridMosaicConsistencyLoss(nn.Module):
    """Enforce the exact synthetic mosaic measurement inside its ROI.

    The loss is intentionally defined only for data carrying an explicit
    square block-average size, per-frame grid phase, and an observation-valid
    sample weight.  It must not be silently applied to midpoint, rectangular,
    resized, compressed, or otherwise approximate mosaic operators.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        eps: float = 1e-6,
        full_mask_threshold: float = 0.999,
        dead_zone: float = 0.5 / 255.0,
    ) -> None:
        super().__init__()
        if loss_weight < 0:
            raise ValueError("mosaic consistency loss weight cannot be negative")
        self.loss_weight = float(loss_weight)
        self.eps = float(eps)
        if not 0.0 < full_mask_threshold <= 1.0:
            raise ValueError("full-mask threshold must be in (0, 1]")
        if dead_zone < 0:
            raise ValueError("mosaic consistency dead zone cannot be negative")
        self.full_mask_threshold = float(full_mask_threshold)
        self.dead_zone = float(dead_zone)

    def forward(
        self,
        prediction: torch.Tensor,
        observation: torch.Tensor,
        mask: torch.Tensor,
        phases: torch.Tensor,
        block_sizes: torch.Tensor,
        observation_weight: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.ndim != 5 or observation.shape != prediction.shape:
            raise ValueError(
                "mosaic consistency expects matching [B,T,C,H,W] tensors"
            )
        batch, frames, _, height, width = prediction.shape
        if mask.shape != (batch, frames, 1, height, width):
            raise ValueError("mosaic consistency mask shape does not match video")
        if phases.shape != (batch, frames, 2):
            raise ValueError("mosaic phases must have shape [B,T,2]")

        flat_block_sizes = block_sizes.reshape(-1)
        flat_weights = observation_weight.reshape(-1)
        if flat_block_sizes.numel() != batch or flat_weights.numel() != batch:
            raise ValueError(
                "mosaic block sizes and observation weights must have one value per sample"
            )

        # Copy the small metadata tensors once.  Reading every scalar directly
        # from MPS would otherwise introduce one synchronization per frame.
        phase_values = phases.detach().cpu().tolist()
        block_values = flat_block_sizes.detach().cpu().tolist()

        measurements: list[torch.Tensor] = []
        measurement_masks: list[torch.Tensor] = []
        alpha = mask.to(dtype=prediction.dtype).clamp(0.0, 1.0)
        for batch_index in range(batch):
            sample_measurements: list[torch.Tensor] = []
            sample_masks: list[torch.Tensor] = []
            block_size = int(block_values[batch_index])
            for frame_index in range(frames):
                frame_alpha = alpha[batch_index, frame_index]
                averaged, valid = _known_phase_block_average(
                    prediction[batch_index, frame_index],
                    block_size=block_size,
                    phase=tuple(phase_values[batch_index][frame_index]),
                )
                hard_roi = (
                    frame_alpha >= self.full_mask_threshold
                ).to(dtype=prediction.dtype)
                roi_coverage, _ = _known_phase_block_average(
                    hard_roi,
                    block_size=block_size,
                    phase=tuple(phase_values[batch_index][frame_index]),
                )
                # Only cells wholly covered by the unfeathered ROI have a
                # pure block-average observation.  Boundary cells contain an
                # alpha composite and would otherwise bias even the GT image.
                complete_roi_cells = (
                    roi_coverage >= 1.0 - 1e-6
                ).to(dtype=prediction.dtype)
                sample_measurements.append(averaged)
                sample_masks.append(valid * complete_roi_cells)
            measurements.append(torch.stack(sample_measurements, dim=0))
            measurement_masks.append(torch.stack(sample_masks, dim=0))

        measurement_video = torch.stack(measurements, dim=0)
        valid_video = torch.stack(measurement_masks, dim=0)
        effective_mask = (
            valid_video
            * flat_weights.to(device=prediction.device, dtype=prediction.dtype)
            .view(batch, 1, 1, 1, 1)
        )
        flat_prediction = measurement_video.reshape(
            batch * frames, prediction.shape[2], height, width
        )
        flat_observation = observation.reshape_as(flat_prediction)
        flat_mask = effective_mask.reshape(batch * frames, 1, height, width)
        expanded_mask = flat_mask.expand_as(flat_prediction)
        residual = (
            (flat_prediction - flat_observation).abs() - self.dead_zone
        ).clamp_min(0.0)
        robust_error = torch.sqrt(residual.square() + self.eps * self.eps)
        robust_error = robust_error - self.eps
        denominator = expanded_mask.sum().clamp_min(1.0)
        return self.loss_weight * (
            robust_error * expanded_mask
        ).sum() / denominator


@MODELS.register_module()
class ROIPixelLoss(nn.Module):
    """Charbonnier reconstruction loss normalized by the restoration ROI.

    A full-frame reconstruction loss is dominated by the already-clean area
    outside the mosaic.  This loss gives every sample comparable weight based
    on the affected pixels while retaining a configurable guard band around
    the mask edge.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        mask_dilation: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.mask_dilation = mask_dilation
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = _dilate(mask, self.mask_dilation)
        return self.loss_weight * _masked_charbonnier(
            prediction, target, mask, self.eps
        )


@MODELS.register_module()
class ROIHighFrequencyLoss(nn.Module):
    """Match gradients and Laplacian detail inside the restoration ROI."""

    def __init__(
        self,
        loss_weight: float = 1.0,
        gradient_weight: float = 1.0,
        laplacian_weight: float = 0.5,
        mask_dilation: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.gradient_weight = gradient_weight
        self.laplacian_weight = laplacian_weight
        self.mask_dilation = mask_dilation
        self.eps = eps

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ) / 8.0
        sobel_y = sobel_x.t()
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
        )
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3), persistent=False)
        self.register_buffer(
            'laplacian', laplacian.view(1, 1, 3, 3), persistent=False
        )

    @staticmethod
    def _filter(image: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        channels = image.shape[1]
        return F.conv2d(
            image,
            kernel.expand(channels, 1, -1, -1),
            padding=1,
            groups=channels,
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = _dilate(mask, self.mask_dilation)
        pred_x = self._filter(prediction, self.sobel_x)
        target_x = self._filter(target, self.sobel_x)
        pred_y = self._filter(prediction, self.sobel_y)
        target_y = self._filter(target, self.sobel_y)
        pred_lap = self._filter(prediction, self.laplacian)
        target_lap = self._filter(target, self.laplacian)

        gradient_loss = (
            _masked_charbonnier(pred_x, target_x, mask, self.eps)
            + _masked_charbonnier(pred_y, target_y, mask, self.eps)
        )
        laplacian_loss = _masked_charbonnier(
            pred_lap, target_lap, mask, self.eps
        )
        return self.loss_weight * (
            self.gradient_weight * gradient_loss
            + self.laplacian_weight * laplacian_loss
        )


@MODELS.register_module()
class ROITemporalDifferenceLoss(nn.Module):
    """Match real inter-frame changes rather than smoothing moving detail."""

    def __init__(
        self,
        loss_weight: float = 1.0,
        mask_dilation: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.mask_dilation = mask_dilation
        self.eps = eps

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if prediction.shape[1] < 2:
            return prediction.sum() * 0.0
        prediction_delta = prediction[:, 1:] - prediction[:, :-1]
        target_delta = target[:, 1:] - target[:, :-1]
        temporal_mask = torch.maximum(mask[:, 1:], mask[:, :-1])

        b, t, c, h, w = prediction_delta.shape
        prediction_delta = prediction_delta.reshape(b * t, c, h, w)
        target_delta = target_delta.reshape(b * t, c, h, w)
        temporal_mask = temporal_mask.reshape(b * t, 1, h, w)
        temporal_mask = _dilate(temporal_mask, self.mask_dilation)
        return self.loss_weight * _masked_charbonnier(
            prediction_delta,
            target_delta,
            temporal_mask,
            self.eps,
        )
