# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Fixed-window inference and one-frame overlap blending for V4."""

from __future__ import annotations

import torch

from .model_v4 import NUM_INPUT_FRAMES, NUM_OUTPUT_FRAMES


class MiohRestorerV4WindowRunner:
    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def restore(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.ndim != 5 or masks.ndim != 5:
            raise ValueError("frames and masks must be [B,T,C,H,W]")
        if masks.shape != (
            frames.shape[0],
            frames.shape[1],
            1,
            frames.shape[-2],
            frames.shape[-1],
        ):
            raise ValueError("V4 runner masks do not match frames")
        if frames.shape[1] == 0:
            raise ValueError("V4 runner needs at least one frame")
        batch, frame_count, _, height, width = frames.shape
        restored_sum = frames.new_zeros(frames.shape)
        confidence_sum = frames.new_zeros(
            batch, frame_count, 1, height, width
        )
        weight_sum = frames.new_zeros(
            batch, frame_count, 1, height, width
        )
        output_start = 0
        while output_start < frame_count:
            input_start = output_start - 2
            indices = [
                min(max(input_start + offset, 0), frame_count - 1)
                for offset in range(NUM_INPUT_FRAMES)
            ]
            window_frames = torch.stack(
                [frames[:, index] for index in indices], dim=1
            )
            window_masks = torch.stack(
                [masks[:, index] for index in indices], dim=1
            )
            values = torch.cat((window_frames, window_masks), dim=2)
            restored, confidence = self.model(values)
            if restored.shape[1] != NUM_OUTPUT_FRAMES:
                raise ValueError("V4 runner requires a five-output model")
            for local_index in range(NUM_OUTPUT_FRAMES):
                frame_index = output_start + local_index
                if frame_index >= frame_count:
                    break
                weight = 0.5 if local_index in (0, 4) else 1.0
                restored_sum[:, frame_index] += restored[:, local_index] * weight
                confidence_sum[:, frame_index] += (
                    confidence[:, local_index] * weight
                )
                weight_sum[:, frame_index] += weight
            output_start += 4
        return (
            restored_sum / weight_sum.clamp_min(1e-6),
            confidence_sum / weight_sum.clamp_min(1e-6),
        )
