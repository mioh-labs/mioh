# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""ANE-first fixed-window mosaic restoration model.

MiohRestorerV4Q consumes nine RGB+mask frames and restores the middle five.
Each output owns its reference coordinate system and aligns a local five-frame
context to that reference.  The five contexts are folded into the batch axis
for the shared fusion and decoder.  The graph deliberately avoids optical
flow, grid sampling, deformable convolution and recurrent state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


NUM_INPUT_FRAMES = 9
NUM_OUTPUT_FRAMES = 5
OUTPUT_START = 2
OUTPUT_INDICES = tuple(range(OUTPUT_START, OUTPUT_START + NUM_OUTPUT_FRAMES))
LOCAL_CONTEXT_RADIUS = 2


def make_offsets(radius: int, dilation: int = 1) -> tuple[tuple[int, int], ...]:
    if radius < 0 or dilation <= 0:
        raise ValueError("shift radius must be non-negative and dilation positive")
    return tuple(
        (vertical * dilation, horizontal * dilation)
        for vertical in range(-radius, radius + 1)
        for horizontal in range(-radius, radius + 1)
    )


def shift2d(
    values: torch.Tensor,
    vertical: int,
    horizontal: int,
) -> torch.Tensor:
    """Translate content with static padding and slicing only."""

    if vertical == 0 and horizontal == 0:
        return values
    height, width = values.shape[-2:]
    left = max(horizontal, 0)
    right = max(-horizontal, 0)
    top = max(vertical, 0)
    bottom = max(-vertical, 0)
    padded = F.pad(values, (left, right, top, bottom))
    return padded[..., bottom : bottom + height, right : right + width]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.conv2(self.activation(self.conv1(values)))


class NormalizedShiftCorrelation(nn.Module):
    """Core ML friendly cosine correlation over a fixed shift bank."""

    MINIMUM_TEMPERATURE = 0.1
    MAXIMUM_TEMPERATURE = 1.5

    def __init__(
        self,
        offsets: Sequence[tuple[int, int]],
        *,
        initial_temperature: float = 0.5,
        center_bias: float = 1.0,
    ) -> None:
        super().__init__()
        if not offsets:
            raise ValueError("correlation needs at least one shift")
        if not self.MINIMUM_TEMPERATURE < initial_temperature < self.MAXIMUM_TEMPERATURE:
            raise ValueError("initial temperature is outside the supported range")
        self.offsets = tuple((int(y), int(x)) for y, x in offsets)
        fraction = (
            (initial_temperature - self.MINIMUM_TEMPERATURE)
            / (self.MAXIMUM_TEMPERATURE - self.MINIMUM_TEMPERATURE)
        )
        self.raw_temperature = nn.Parameter(
            torch.tensor(math.log(fraction / (1.0 - fraction)))
        )
        self.offset_bias = nn.Parameter(
            torch.zeros(1, len(self.offsets), 1, 1)
        )
        if (0, 0) in self.offsets:
            with torch.no_grad():
                self.offset_bias[:, self.offsets.index((0, 0))].fill_(
                    center_bias
                )

    @property
    def temperature(self) -> torch.Tensor:
        span = self.MAXIMUM_TEMPERATURE - self.MINIMUM_TEMPERATURE
        return self.MINIMUM_TEMPERATURE + span * torch.sigmoid(
            self.raw_temperature
        )

    @staticmethod
    def _normalize(values: torch.Tensor) -> torch.Tensor:
        inverse_norm = torch.rsqrt(
            values.float().square().sum(dim=1, keepdim=True) + 1e-6
        ).to(dtype=values.dtype)
        return values * inverse_norm

    def forward(
        self,
        reference: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reference.shape != target.shape:
            raise ValueError("reference and target feature shapes must match")
        normalized_reference = self._normalize(reference)
        normalized_target = self._normalize(target)
        # ``Tensor.new_ones`` is not accepted by the current Core ML Torch
        # frontend.  Derive the fixed validity plane from an existing tensor.
        validity_source = torch.ones_like(target[:, :1])
        shifted_targets: list[torch.Tensor] = []
        scores: list[torch.Tensor] = []
        for vertical, horizontal in self.offsets:
            shifted_target = shift2d(target, vertical, horizontal)
            shifted_normalized = shift2d(
                normalized_target, vertical, horizontal
            )
            valid = shift2d(validity_source, vertical, horizontal)
            score = (normalized_reference * shifted_normalized).sum(
                dim=1, keepdim=True
            )
            score = score + (valid - 1.0) * 10_000.0
            shifted_targets.append(shifted_target)
            scores.append(score)
        logits = torch.cat(scores, dim=1) + self.offset_bias
        weights = torch.softmax(logits / self.temperature, dim=1)
        aligned = torch.zeros_like(target)
        for index, shifted_target in enumerate(shifted_targets):
            aligned = aligned + weights[:, index : index + 1] * shifted_target
        return aligned, weights


def apply_shift_weights(
    target: torch.Tensor,
    weights: torch.Tensor,
    offsets: Sequence[tuple[int, int]],
) -> torch.Tensor:
    if weights.shape[1] != len(offsets):
        raise ValueError("shift weights and offsets do not match")
    aligned = torch.zeros_like(target)
    for index, (vertical, horizontal) in enumerate(offsets):
        aligned = aligned + weights[:, index : index + 1] * shift2d(
            target, vertical, horizontal
        )
    return aligned


class HierarchicalAlignment27(nn.Module):
    """Three nine-way banks covering +/-40 input pixels."""

    input_reach = 40

    def __init__(self) -> None:
        super().__init__()
        self.offsets_eighth_coarse = make_offsets(1, 3)
        self.offsets_eighth_fine = make_offsets(1, 1)
        self.offsets_quarter_fine = make_offsets(1, 2)
        self.offsets_quarter_coarse = make_offsets(1, 6)
        self.offsets_quarter_mid = make_offsets(1, 2)
        self.eighth_coarse = NormalizedShiftCorrelation(
            self.offsets_eighth_coarse
        )
        self.eighth_fine = NormalizedShiftCorrelation(
            self.offsets_eighth_fine
        )
        self.quarter_fine = NormalizedShiftCorrelation(
            self.offsets_quarter_fine
        )

    def forward(
        self,
        reference_eighth: torch.Tensor,
        target_eighth: torch.Tensor,
        reference_quarter: torch.Tensor,
        target_quarter: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        aligned_eighth, coarse = self.eighth_coarse(
            reference_eighth, target_eighth
        )
        aligned_eighth, middle = self.eighth_fine(
            reference_eighth, aligned_eighth
        )
        coarse_quarter = F.interpolate(coarse, scale_factor=2, mode="nearest")
        middle_quarter = F.interpolate(middle, scale_factor=2, mode="nearest")
        aligned_quarter = apply_shift_weights(
            target_quarter,
            coarse_quarter,
            self.offsets_quarter_coarse,
        )
        aligned_quarter = apply_shift_weights(
            aligned_quarter,
            middle_quarter,
            self.offsets_quarter_mid,
        )
        aligned_quarter, fine = self.quarter_fine(
            reference_quarter, aligned_quarter
        )
        return aligned_eighth, aligned_quarter, (coarse, middle, fine)


class FullAlignment121(nn.Module):
    """Exhaustive correctness baseline; not intended for V4 training."""

    input_reach = 48

    def __init__(self) -> None:
        super().__init__()
        self.offsets_eighth = make_offsets(5)
        self.offsets_quarter = tuple(
            (vertical * 2, horizontal * 2)
            for vertical, horizontal in self.offsets_eighth
        )
        self.offsets_quarter_fine = make_offsets(1, 2)
        self.eighth = NormalizedShiftCorrelation(self.offsets_eighth)
        self.quarter_fine = NormalizedShiftCorrelation(
            self.offsets_quarter_fine
        )

    def forward(
        self,
        reference_eighth: torch.Tensor,
        target_eighth: torch.Tensor,
        reference_quarter: torch.Tensor,
        target_quarter: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
    ]:
        aligned_eighth, coarse = self.eighth(
            reference_eighth, target_eighth
        )
        coarse_quarter = F.interpolate(coarse, scale_factor=2, mode="nearest")
        aligned_quarter = apply_shift_weights(
            target_quarter,
            coarse_quarter,
            self.offsets_quarter,
        )
        aligned_quarter, fine = self.quarter_fine(
            reference_quarter, aligned_quarter
        )
        return aligned_eighth, aligned_quarter, (coarse, fine)


class SharedFrameEncoder(nn.Module):
    def __init__(
        self,
        *,
        stem_channels: int = 32,
        half_channels: int = 48,
        quarter_channels: int = 64,
        eighth_channels: int = 96,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(4, stem_channels, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(stem_channels),
        )
        # ``nn.Module.half`` is an existing dtype-conversion method, so this
        # stage must not use ``half`` as its attribute name.
        self.half_stage = nn.Sequential(
            nn.Conv2d(stem_channels, half_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(half_channels),
        )
        self.quarter = nn.Sequential(
            nn.Conv2d(half_channels, quarter_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(quarter_channels),
        )
        self.eighth = nn.Sequential(
            nn.Conv2d(quarter_channels, eighth_channels, 3, stride=2, padding=1),
            nn.SiLU(),
            ResidualBlock(eighth_channels),
        )

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.stem(values)
        values = self.half_stage(values)
        quarter = self.quarter(values)
        eighth = self.eighth(quarter)
        return quarter, eighth


class MiohRestorerV4Q(nn.Module):
    """Quality-reference fixed-window model with output-specific alignment."""

    ARCHITECTURE_REVISION = 1

    def __init__(
        self,
        *,
        alignment_variant: str = "hier27",
        execution_mode: str = "batch",
        quarter_channels: int = 64,
        eighth_channels: int = 96,
        fusion_eighth_channels: int = 192,
        fusion_quarter_channels: int = 96,
        eighth_blocks: int = 10,
        quarter_blocks: int = 4,
    ) -> None:
        super().__init__()
        if alignment_variant not in ("hier27", "full121"):
            raise ValueError("unknown V4 alignment variant")
        if execution_mode not in ("batch", "serial", "center1"):
            raise ValueError("unknown V4 execution mode")
        self.alignment_variant = alignment_variant
        self.execution_mode = execution_mode
        self.quarter_channels = quarter_channels
        self.eighth_channels = eighth_channels
        self.fusion_eighth_channels = fusion_eighth_channels
        self.fusion_quarter_channels = fusion_quarter_channels
        self.eighth_blocks = eighth_blocks
        self.quarter_blocks = quarter_blocks
        self.gradient_checkpointing = False
        self.encoder = SharedFrameEncoder(
            quarter_channels=quarter_channels,
            eighth_channels=eighth_channels,
        )
        self.alignment: HierarchicalAlignment27 | FullAlignment121 = (
            HierarchicalAlignment27()
            if alignment_variant == "hier27"
            else FullAlignment121()
        )
        context_frames = LOCAL_CONTEXT_RADIUS * 2 + 1
        self.reduce_eighth = nn.Sequential(
            nn.Conv2d(
                eighth_channels * context_frames,
                fusion_eighth_channels,
                1,
            ),
            nn.SiLU(),
        )
        self.body_eighth = nn.Sequential(
            *(ResidualBlock(fusion_eighth_channels) for _ in range(eighth_blocks))
        )
        self.up_eighth_to_quarter = nn.Sequential(
            nn.Conv2d(
                fusion_eighth_channels,
                fusion_quarter_channels * 4,
                3,
                padding=1,
            ),
            nn.PixelShuffle(2),
            nn.SiLU(),
        )
        self.reduce_quarter = nn.Sequential(
            nn.Conv2d(
                fusion_quarter_channels + quarter_channels * context_frames,
                fusion_quarter_channels,
                1,
            ),
            nn.SiLU(),
        )
        self.body_quarter = nn.Sequential(
            *(ResidualBlock(fusion_quarter_channels) for _ in range(quarter_blocks))
        )
        self.up_quarter_to_half = nn.Sequential(
            nn.Conv2d(
                fusion_quarter_channels,
                64 * 4,
                3,
                padding=1,
            ),
            nn.PixelShuffle(2),
            nn.SiLU(),
        )
        self.base_head_half = nn.Sequential(
            nn.Conv2d(64, 48, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, 3, 3, padding=1),
        )
        self.up_half_to_full = nn.Sequential(
            nn.Conv2d(64, 48 * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.SiLU(),
        )
        self.texture_head = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(48),
            nn.Conv2d(48, 3, 3, padding=1),
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(48, 24, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(24, 1, 3, padding=1),
        )
        self._zero_initialize_heads()

    def _zero_initialize_heads(self) -> None:
        for head in (
            self.base_head_half,
            self.texture_head,
            self.confidence_head,
        ):
            output = head[-1]
            if not isinstance(output, nn.Conv2d):
                raise TypeError("V4 output head must end in a convolution")
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    @property
    def output_indices(self) -> tuple[int, ...]:
        if self.execution_mode == "center1":
            return (NUM_INPUT_FRAMES // 2,)
        return OUTPUT_INDICES

    def enable_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Trade extra training compute for substantially lower activation memory."""

        self.gradient_checkpointing = enabled

    def _validate_input(self, values: torch.Tensor) -> None:
        if values.ndim != 5 or values.shape[1:3] != (NUM_INPUT_FRAMES, 4):
            raise ValueError("V4 input must have shape [B,9,4,H,W]")
        if values.shape[-2] % 8 or values.shape[-1] % 8:
            raise ValueError("V4 frame dimensions must be divisible by eight")

    def _encode_frames(
        self, values: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.training and self.gradient_checkpointing:
            return [
                checkpoint(
                    self.encoder,
                    values[:, index],
                    use_reentrant=False,
                )
                for index in range(NUM_INPUT_FRAMES)
            ]
        return [self.encoder(values[:, index]) for index in range(NUM_INPUT_FRAMES)]

    def _align_pair(
        self,
        reference_eighth: torch.Tensor,
        target_eighth: torch.Tensor,
        reference_quarter: torch.Tensor,
        target_quarter: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def align(
            ref_eighth: torch.Tensor,
            tgt_eighth: torch.Tensor,
            ref_quarter: torch.Tensor,
            tgt_quarter: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            aligned_eighth, aligned_quarter, _weights = self.alignment(
                ref_eighth,
                tgt_eighth,
                ref_quarter,
                tgt_quarter,
            )
            return aligned_eighth, aligned_quarter

        if self.training and self.gradient_checkpointing:
            return checkpoint(
                align,
                reference_eighth,
                target_eighth,
                reference_quarter,
                target_quarter,
                use_reentrant=False,
            )
        return align(
            reference_eighth,
            target_eighth,
            reference_quarter,
            target_quarter,
        )

    def _align_pair_with_weights(
        self,
        reference_eighth: torch.Tensor,
        target_eighth: torch.Tensor,
        reference_quarter: torch.Tensor,
        target_quarter: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """Training-only alignment path retaining the three shift banks."""

        if not isinstance(self.alignment, HierarchicalAlignment27):
            raise TypeError("V4 distillation requires hierarchical alignment")

        def align(
            ref_eighth: torch.Tensor,
            tgt_eighth: torch.Tensor,
            ref_quarter: torch.Tensor,
            tgt_quarter: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            aligned_eighth, aligned_quarter, weights = self.alignment(
                ref_eighth,
                tgt_eighth,
                ref_quarter,
                tgt_quarter,
            )
            return aligned_eighth, aligned_quarter, *weights

        if self.training and self.gradient_checkpointing:
            values = checkpoint(
                align,
                reference_eighth,
                target_eighth,
                reference_quarter,
                target_quarter,
                use_reentrant=False,
            )
        else:
            values = align(
                reference_eighth,
                target_eighth,
                reference_quarter,
                target_quarter,
            )
        return values[0], values[1], (values[2], values[3], values[4])

    def _aligned_context(
        self,
        features: list[tuple[torch.Tensor, torch.Tensor]],
        reference_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference_quarter, reference_eighth = features[reference_index]
        quarters: list[torch.Tensor] = []
        eighths: list[torch.Tensor] = []
        for frame_index in range(
            reference_index - LOCAL_CONTEXT_RADIUS,
            reference_index + LOCAL_CONTEXT_RADIUS + 1,
        ):
            target_quarter, target_eighth = features[frame_index]
            if frame_index == reference_index:
                aligned_eighth = target_eighth
                aligned_quarter = target_quarter
            else:
                aligned_eighth, aligned_quarter = self._align_pair(
                    reference_eighth,
                    target_eighth,
                    reference_quarter,
                    target_quarter,
                )
            eighths.append(aligned_eighth)
            quarters.append(aligned_quarter)
        return torch.cat(eighths, dim=1), torch.cat(quarters, dim=1)

    def _aligned_context_with_diagnostics(
        self,
        features: list[tuple[torch.Tensor, torch.Tensor]],
        reference_index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        tuple[
            tuple[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]], ...
        ],
    ]:
        reference_quarter, reference_eighth = features[reference_index]
        quarters: list[torch.Tensor] = []
        eighths: list[torch.Tensor] = []
        records: list[
            tuple[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = []
        for frame_index in range(
            reference_index - LOCAL_CONTEXT_RADIUS,
            reference_index + LOCAL_CONTEXT_RADIUS + 1,
        ):
            target_quarter, target_eighth = features[frame_index]
            if frame_index == reference_index:
                aligned_eighth = target_eighth
                aligned_quarter = target_quarter
            else:
                aligned_eighth, aligned_quarter, weights = (
                    self._align_pair_with_weights(
                        reference_eighth,
                        target_eighth,
                        reference_quarter,
                        target_quarter,
                    )
                )
                records.append((frame_index, weights))
            eighths.append(aligned_eighth)
            quarters.append(aligned_quarter)
        return (
            torch.cat(eighths, dim=1),
            torch.cat(quarters, dim=1),
            tuple(records),
        )

    def _decode_context(
        self,
        context_eighth: torch.Tensor,
        context_quarter: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        eighth = self.reduce_eighth(context_eighth)
        if self.training and self.gradient_checkpointing and self.eighth_blocks:
            eighth = checkpoint_sequential(
                self.body_eighth,
                self.eighth_blocks,
                eighth,
                use_reentrant=False,
            )
        else:
            eighth = self.body_eighth(eighth)
        quarter = self.up_eighth_to_quarter(eighth)
        quarter = self.reduce_quarter(
            torch.cat((quarter, context_quarter), dim=1)
        )
        if self.training and self.gradient_checkpointing and self.quarter_blocks:
            quarter = checkpoint_sequential(
                self.body_quarter,
                self.quarter_blocks,
                quarter,
                use_reentrant=False,
            )
        else:
            quarter = self.body_quarter(quarter)
        half = self.up_quarter_to_half(quarter)
        base_half = self.base_head_half(half)
        base = F.interpolate(
            base_half,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        full = self.up_half_to_full(half)
        texture = self.texture_head(full)
        confidence = torch.sigmoid(self.confidence_head(full))
        if return_features:
            return base, texture, confidence, eighth, quarter
        return base, texture, confidence

    @staticmethod
    def _restore_output_major_layout(
        values: torch.Tensor,
        *,
        batch: int,
        output_frames: int,
    ) -> torch.Tensor:
        return values.reshape(
            output_frames, batch, values.shape[1], *values.shape[-2:]
        ).transpose(0, 1)

    def _forward_components_impl(
        self,
        values: torch.Tensor,
        *,
        capture_alignment: bool,
        capture_features: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        self._validate_input(values)
        features = self._encode_frames(values)
        contexts: list[tuple[torch.Tensor, torch.Tensor]] = []
        alignment_records: list[
            tuple[tuple[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]], ...]
        ] = []
        for index in self.output_indices:
            if capture_alignment:
                eighth, quarter, records = self._aligned_context_with_diagnostics(
                    features, index
                )
                contexts.append((eighth, quarter))
                alignment_records.append(records)
            else:
                contexts.append(self._aligned_context(features, index))

        batch = values.shape[0]
        fused_eighth: torch.Tensor | None = None
        fused_quarter: torch.Tensor | None = None
        if self.execution_mode in ("batch", "center1"):
            context_eighth = torch.cat([context[0] for context in contexts], dim=0)
            context_quarter = torch.cat([context[1] for context in contexts], dim=0)
            decoded = self._decode_context(
                context_eighth,
                context_quarter,
                return_features=capture_features,
            )
            base_flat, texture_flat, confidence_flat = decoded[:3]
            output_frames = len(self.output_indices)
            base = self._restore_output_major_layout(
                base_flat, batch=batch, output_frames=output_frames
            )
            texture = self._restore_output_major_layout(
                texture_flat, batch=batch, output_frames=output_frames
            )
            confidence = self._restore_output_major_layout(
                confidence_flat, batch=batch, output_frames=output_frames
            )
            if capture_features:
                fused_eighth = self._restore_output_major_layout(
                    decoded[3], batch=batch, output_frames=output_frames
                )
                fused_quarter = self._restore_output_major_layout(
                    decoded[4], batch=batch, output_frames=output_frames
                )
        else:
            decoded = [
                self._decode_context(
                    *context, return_features=capture_features
                )
                for context in contexts
            ]
            base = torch.stack([item[0] for item in decoded], dim=1)
            texture = torch.stack([item[1] for item in decoded], dim=1)
            confidence = torch.stack([item[2] for item in decoded], dim=1)
            if capture_features:
                fused_eighth = torch.stack([item[3] for item in decoded], dim=1)
                fused_quarter = torch.stack([item[4] for item in decoded], dim=1)

        indices = self.output_indices
        rgb = torch.stack([values[:, index, :3] for index in indices], dim=1)
        mask = torch.stack([values[:, index, 3:4] for index in indices], dim=1)
        restored = rgb + mask * (base + confidence * texture)
        diagnostics: dict[str, torch.Tensor] = {}
        if capture_alignment:
            banks: list[list[torch.Tensor]] = [[], [], []]
            for records in alignment_records:
                for bank_index in range(3):
                    banks[bank_index].append(
                        torch.stack(
                            [weights[bank_index] for _frame, weights in records],
                            dim=1,
                        )
                    )
            diagnostics["alignment_coarse"] = torch.stack(banks[0], dim=1)
            diagnostics["alignment_middle"] = torch.stack(banks[1], dim=1)
            diagnostics["alignment_fine"] = torch.stack(banks[2], dim=1)
        if capture_features:
            if fused_eighth is None or fused_quarter is None:
                raise AssertionError("requested V4 features were not retained")
            diagnostics["fused_eighth"] = fused_eighth
            diagnostics["fused_quarter"] = fused_quarter
        return restored, confidence, base, texture, diagnostics

    def forward_components(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        restored, confidence, base, texture, _diagnostics = (
            self._forward_components_impl(
                values,
                capture_alignment=False,
                capture_features=False,
            )
        )
        return restored, confidence, base, texture

    def forward_with_distillation(
        self,
        values: torch.Tensor,
        *,
        capture_alignment: bool = False,
        capture_features: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Training-only diagnostics; never called by the export wrapper."""

        if not capture_alignment and not capture_features:
            raise ValueError("at least one distillation signal must be requested")
        return self._forward_components_impl(
            values,
            capture_alignment=capture_alignment,
            capture_features=capture_features,
        )

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        restored, confidence, _base, _texture = self.forward_components(values)
        return restored, confidence


class MiohRestorerV4ExportWrapper(nn.Module):
    """Fixed flat I/O and inference-only clamping for Apple conversion."""

    def __init__(self, model: MiohRestorerV4Q) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, flat_frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = flat_frames.shape
        values = flat_frames.reshape(
            batch, NUM_INPUT_FRAMES, 4, height, width
        )
        restored, confidence = self.model(values)
        return (
            torch.clamp(restored, 0.0, 1.0).flatten(1, 2),
            confidence.flatten(1, 2),
        )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
