# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Training-only adversarial components for MiohRestorerV2."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TemporalPatchDiscriminator(nn.Module):
    """Judge spatial appearance and adjacent-frame motion inside the ROI."""

    def __init__(self, base_channels: int = 32) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        # RGB current frame + RGB temporal difference + one mask channel.
        channels = (7, base_channels, base_channels * 2, base_channels * 4, base_channels * 8)
        layers: list[nn.Module] = []
        for index in range(len(channels) - 1):
            layers.extend(
                (
                    nn.Conv2d(
                        channels[index],
                        channels[index + 1],
                        4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.2),
                )
            )
        layers.extend(
            (
                nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(base_channels * 8, 1, 3, padding=1),
            )
        )
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 4 or values.shape[1] != 7:
            raise ValueError("discriminator input must have shape [N,7,H,W]")
        return self.network(values)


def temporal_discriminator_input(
    video: torch.Tensor,
    target: torch.Tensor,
    masks: torch.Tensor,
    *,
    frame_stride: int = 4,
    image_size: int = 192,
) -> torch.Tensor:
    """Build masked current-frame/motion pairs with clean target context."""
    if video.shape != target.shape or video.ndim != 5:
        raise ValueError("video and target must have matching B,T,C,H,W shapes")
    if masks.shape != (video.shape[0], video.shape[1], 1, *video.shape[-2:]):
        raise ValueError("masks do not match video")
    if video.shape[1] < 2:
        raise ValueError("temporal discriminator requires at least two frames")
    if frame_stride <= 0 or image_size < 32:
        raise ValueError("invalid temporal discriminator sampling settings")

    composed = video * masks + target * (1.0 - masks)
    current = composed[:, 1::frame_stride]
    previous = composed[:, 0:-1:frame_stride]
    pair_count = min(current.shape[1], previous.shape[1])
    current = current[:, :pair_count]
    previous = previous[:, :pair_count]
    pair_mask = torch.minimum(
        masks[:, 1::frame_stride][:, :pair_count],
        masks[:, 0:-1:frame_stride][:, :pair_count],
    )
    motion = current - previous
    values = torch.cat((current, motion, pair_mask), dim=2)
    values = values.reshape(-1, 7, *values.shape[-2:])
    return F.interpolate(
        values,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )


def discriminator_hinge_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
) -> torch.Tensor:
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()
