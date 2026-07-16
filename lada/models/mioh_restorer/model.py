# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Flow-free recurrent restoration prototype for replacing BasicVSR++."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.conv1(features))
        return F.relu(features + self.conv2(residual))


class MiohRestorerV1(nn.Module):
    """Small mask-aware recurrent model with an explicit persistent state.

    The fixed four-frame micro-batch amortizes Core AI transport overhead while
    the recurrent state makes the same weights usable for arbitrarily long
    clips. The model intentionally uses only convolution, pointwise activation,
    concatenation and pixel shuffle so it can target PyTorch, Core ML and Core
    AI without BasicVSR++'s optical-flow or deformable-alignment kernels.
    """

    DEFAULT_CHUNK_FRAMES = 4
    DEFAULT_CHANNELS = 64
    DEFAULT_BLOCKS = 8
    DOWNSCALE = 4

    def __init__(
        self,
        *,
        chunk_frames: int = DEFAULT_CHUNK_FRAMES,
        channels: int = DEFAULT_CHANNELS,
        num_blocks: int = DEFAULT_BLOCKS,
    ) -> None:
        super().__init__()
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if num_blocks < 0:
            raise ValueError("num_blocks must not be negative")
        self.chunk_frames = chunk_frames
        self.channels = channels
        self.num_blocks = num_blocks

        self.encoder = nn.Sequential(
            nn.Conv2d(4, channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.update_gate = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.candidate = nn.Conv2d(channels * 2, channels, 3, padding=1)
        self.refinement = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(num_blocks))
        )
        self.up1 = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.up2 = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.output_head = nn.Conv2d(channels, 3, 3, padding=1)

        # The untrained deployment prototype is an exact identity restorer.
        # Training can then learn only the masked residual it needs to change.
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def state_shape(
        self,
        *,
        batch_size: int = 1,
        image_height: int = 256,
        image_width: int = 256,
    ) -> tuple[int, int, int, int]:
        if image_height % self.DOWNSCALE or image_width % self.DOWNSCALE:
            raise ValueError("image dimensions must be divisible by 4")
        return (
            batch_size,
            self.channels,
            image_height // self.DOWNSCALE,
            image_width // self.DOWNSCALE,
        )

    def initial_state(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        return frames.new_zeros(
            self.state_shape(
                batch_size=frames.shape[0],
                image_height=frames.shape[-2],
                image_width=frames.shape[-1],
            )
        )

    def forward(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
        history: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = history
        restored_frames: list[torch.Tensor] = []
        for frame_index in range(self.chunk_frames):
            frame = frames[:, frame_index]
            mask = masks[:, frame_index]
            encoded = self.encoder(torch.cat((frame, mask), dim=1))
            recurrent = torch.cat((encoded, hidden), dim=1)
            update = torch.sigmoid(self.update_gate(recurrent))
            candidate = torch.tanh(self.candidate(recurrent))
            hidden = update * hidden + (1.0 - update) * candidate

            features = self.refinement(hidden)
            features = F.relu(F.pixel_shuffle(self.up1(features), 2))
            features = F.relu(F.pixel_shuffle(self.up2(features), 2))
            residual = torch.tanh(self.output_head(features))
            restored_frames.append(torch.clamp(frame + residual * mask, 0.0, 1.0))

        return torch.stack(restored_frames, dim=1), hidden
