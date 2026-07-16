# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Bidirectional multi-scale Mioh restoration model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .model import MiohRestorerV1, ResidualBlock


class MultiScaleFusionRefiner(nn.Module):
    """Fuse forward/backward predictions and restore full-resolution detail."""

    INPUT_CHANNELS = 13  # frame, mask, forward, backward, absolute difference

    def __init__(
        self,
        *,
        full_channels: int = 32,
        half_channels: int = 64,
        quarter_channels: int = 96,
        detail_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if min(full_channels, half_channels, quarter_channels) <= 0:
            raise ValueError("fusion channels must be positive")
        if detail_scale <= 0:
            raise ValueError("detail_scale must be positive")
        self.full_channels = full_channels
        self.half_channels = half_channels
        self.quarter_channels = quarter_channels
        self.detail_scale = detail_scale

        self.full_encoder = nn.Sequential(
            nn.Conv2d(self.INPUT_CHANNELS, full_channels, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(full_channels),
            ResidualBlock(full_channels),
        )
        self.half_encoder = nn.Sequential(
            nn.Conv2d(full_channels, half_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(half_channels),
            ResidualBlock(half_channels),
        )
        self.quarter_encoder = nn.Sequential(
            nn.Conv2d(half_channels, quarter_channels, 3, stride=2, padding=1),
            nn.ReLU(),
            ResidualBlock(quarter_channels),
            ResidualBlock(quarter_channels),
            ResidualBlock(quarter_channels),
            ResidualBlock(quarter_channels),
        )
        self.up_half = nn.Conv2d(quarter_channels, half_channels * 4, 3, padding=1)
        self.half_fusion = nn.Sequential(
            nn.Conv2d(half_channels * 2, half_channels, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(half_channels),
        )
        self.up_full = nn.Conv2d(half_channels, full_channels * 4, 3, padding=1)
        self.full_fusion = nn.Sequential(
            nn.Conv2d(full_channels * 2, full_channels, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(full_channels),
        )
        self.direction_gate = nn.Conv2d(full_channels, 1, 3, padding=1)
        self.detail_head = nn.Conv2d(full_channels, 3, 3, padding=1)

        # Start from an even directional blend without inventing extra detail.
        nn.init.zeros_(self.direction_gate.weight)
        nn.init.zeros_(self.direction_gate.bias)
        nn.init.zeros_(self.detail_head.weight)
        nn.init.zeros_(self.detail_head.bias)

    def forward(
        self,
        frame: torch.Tensor,
        mask: torch.Tensor,
        forward_prediction: torch.Tensor,
        backward_prediction: torch.Tensor,
    ) -> torch.Tensor:
        difference = (forward_prediction - backward_prediction).abs()
        inputs = torch.cat(
            (
                frame,
                mask,
                forward_prediction,
                backward_prediction,
                difference,
            ),
            dim=1,
        )
        full = self.full_encoder(inputs)
        half = self.half_encoder(full)
        quarter = self.quarter_encoder(half)
        half_up = F.pixel_shuffle(self.up_half(quarter), 2)
        half = self.half_fusion(torch.cat((half, half_up), dim=1))
        full_up = F.pixel_shuffle(self.up_full(half), 2)
        full = self.full_fusion(torch.cat((full, full_up), dim=1))

        gate = torch.sigmoid(self.direction_gate(full))
        directional = (
            gate * forward_prediction + (1.0 - gate) * backward_prediction
        )
        detail = torch.tanh(self.detail_head(full)) * self.detail_scale
        residual = directional - frame + detail
        return torch.clamp(frame + residual * mask, 0.0, 1.0)


class MiohRestorerV2(nn.Module):
    """Fixed-window bidirectional restorer designed for Core ML/Core AI.

    Two independent V1-compatible recurrent branches traverse a complete
    temporal window in opposite directions.  A learned multi-scale fusion
    network then selects directionally reliable content and adds a bounded
    full-resolution detail residual.
    """

    DEFAULT_WINDOW_FRAMES = 24
    DEFAULT_CHUNK_FRAMES = 4
    DEFAULT_CHANNELS = 96
    DEFAULT_BLOCKS = 12
    DEFAULT_IMAGE_SIZE = 384
    DEFAULT_FUSION_FULL_CHANNELS = 32
    DEFAULT_FUSION_HALF_CHANNELS = 64
    DEFAULT_FUSION_QUARTER_CHANNELS = 96

    def __init__(
        self,
        *,
        window_frames: int = DEFAULT_WINDOW_FRAMES,
        chunk_frames: int = DEFAULT_CHUNK_FRAMES,
        channels: int = DEFAULT_CHANNELS,
        num_blocks: int = DEFAULT_BLOCKS,
        fusion_full_channels: int = DEFAULT_FUSION_FULL_CHANNELS,
        fusion_half_channels: int = DEFAULT_FUSION_HALF_CHANNELS,
        fusion_quarter_channels: int = DEFAULT_FUSION_QUARTER_CHANNELS,
        detail_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if window_frames <= 0 or window_frames % chunk_frames:
            raise ValueError("window_frames must be positive and divisible by chunk_frames")
        self.window_frames = window_frames
        self.chunk_frames = chunk_frames
        self.channels = channels
        self.num_blocks = num_blocks
        self.forward_branch = MiohRestorerV1(
            chunk_frames=chunk_frames,
            channels=channels,
            num_blocks=num_blocks,
        )
        self.backward_branch = MiohRestorerV1(
            chunk_frames=chunk_frames,
            channels=channels,
            num_blocks=num_blocks,
        )
        self.fusion = MultiScaleFusionRefiner(
            full_channels=fusion_full_channels,
            half_channels=fusion_half_channels,
            quarter_channels=fusion_quarter_channels,
            detail_scale=detail_scale,
        )

    @property
    def fusion_full_channels(self) -> int:
        return self.fusion.full_channels

    @property
    def fusion_half_channels(self) -> int:
        return self.fusion.half_channels

    @property
    def fusion_quarter_channels(self) -> int:
        return self.fusion.quarter_channels

    @property
    def detail_scale(self) -> float:
        return self.fusion.detail_scale

    def _validate_inputs(self, frames: torch.Tensor, masks: torch.Tensor) -> None:
        if frames.ndim != 5 or frames.shape[1:3] != (self.window_frames, 3):
            raise ValueError(
                f"frames must have shape [B,{self.window_frames},3,H,W]"
            )
        if masks.shape != (
            frames.shape[0],
            self.window_frames,
            1,
            *frames.shape[-2:],
        ):
            raise ValueError("masks do not match frames")
        if frames.shape[-2] % 4 or frames.shape[-1] % 4:
            raise ValueError("frame dimensions must be divisible by 4")

    def _run_branch(
        self,
        branch: MiohRestorerV1,
        frames: torch.Tensor,
        masks: torch.Tensor,
        *,
        reverse: bool,
    ) -> torch.Tensor:
        if reverse:
            frames = torch.flip(frames, dims=(1,))
            masks = torch.flip(masks, dims=(1,))
        state = branch.initial_state(frames)
        outputs: list[torch.Tensor] = []
        for start in range(0, self.window_frames, self.chunk_frames):
            restored, state = branch(
                frames[:, start : start + self.chunk_frames],
                masks[:, start : start + self.chunk_frames],
                state,
            )
            outputs.append(restored)
        result = torch.cat(outputs, dim=1)
        return torch.flip(result, dims=(1,)) if reverse else result

    def directional_predictions(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_inputs(frames, masks)
        forward = self._run_branch(
            self.forward_branch,
            frames,
            masks,
            reverse=False,
        )
        backward = self._run_branch(
            self.backward_branch,
            frames,
            masks,
            reverse=True,
        )
        return forward, backward

    def forward_with_directions(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        forward, backward = self.directional_predictions(frames, masks)
        restored_frames: list[torch.Tensor] = []
        for frame_index in range(self.window_frames):
            restored_frames.append(
                self.fusion(
                    frames[:, frame_index],
                    masks[:, frame_index],
                    forward[:, frame_index],
                    backward[:, frame_index],
                )
            )
        return torch.stack(restored_frames, dim=1), forward, backward

    def forward(self, frames: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        restored, _forward, _backward = self.forward_with_directions(frames, masks)
        return restored

    def initialize_branches_from_v1(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Copy a compatible V1 checkpoint into both temporal directions."""
        self.forward_branch.load_state_dict(state_dict, strict=True)
        self.backward_branch.load_state_dict(state_dict, strict=True)
