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
