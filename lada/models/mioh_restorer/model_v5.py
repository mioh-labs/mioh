# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""MiohRestorer V5: native-bucket, folded-space video restoration.

The deployment graph never applies a learned convolution at source resolution.
Each frame is folded with PixelUnshuffle, encoded once, aligned to the output
reference in five fixed stages, fused, and returned through one PixelShuffle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import torch
from torch import nn
from torch.nn import functional as F

NUM_INPUT_FRAMES = 9
NUM_QUALITY_OUTPUT_FRAMES = 5
QUALITY_OUTPUT_INDICES = (2, 3, 4, 5, 6)
CENTER_INDEX = 4
FRAME_CHANNELS = 5  # RGB, mosaic mask, mask reliability
FOLD_FACTOR = 2
FOLDED_FRAME_CHANNELS = FRAME_CHANNELS * FOLD_FACTOR * FOLD_FACTOR
V5_BUCKETS = (128, 192, 256, 384, 512)


def make_offsets(
    radius: int, dilation: int = 1
) -> tuple[tuple[int, int], ...]:
    if radius < 0 or dilation <= 0:
        raise ValueError("shift radius must be non-negative and dilation positive")
    return tuple(
        (vertical * dilation, horizontal * dilation)
        for vertical in range(-radius, radius + 1)
        for horizontal in range(-radius, radius + 1)
    )


def shift2d(values: torch.Tensor, vertical: int, horizontal: int) -> torch.Tensor:
    """Translate content using static zero padding and slicing."""

    if vertical == 0 and horizontal == 0:
        return values
    height, width = values.shape[-2:]
    left = max(horizontal, 0)
    right = max(-horizontal, 0)
    top = max(vertical, 0)
    bottom = max(-vertical, 0)
    padded = F.pad(values, (left, right, top, bottom))
    return padded[..., bottom : bottom + height, right : right + width]


class FixedShiftBank(nn.Module):
    """Generate a complete square shift bank with one frozen convolution."""

    def __init__(
        self, channels: int, offsets: Sequence[tuple[int, int]]
    ) -> None:
        super().__init__()
        if channels <= 0 or not offsets:
            raise ValueError("shift-bank channels and offsets must be non-empty")
        self.channels = int(channels)
        self.offsets = tuple((int(y), int(x)) for y, x in offsets)
        nonzero = [abs(value) for offset in self.offsets for value in offset if value]
        dilation = math.gcd(*nonzero) if nonzero else 1
        radius = max(
            max(abs(vertical), abs(horizontal))
            for vertical, horizontal in self.offsets
        ) // dilation
        expected = {
            (vertical * dilation, horizontal * dilation)
            for vertical in range(-radius, radius + 1)
            for horizontal in range(-radius, radius + 1)
        }
        if set(self.offsets) != expected or len(self.offsets) != len(expected):
            raise ValueError("fixed shift bank requires a complete square grid")
        self.dilation = dilation
        self.radius = radius
        kernel_size = radius * 2 + 1
        kernels = torch.zeros(
            self.channels * len(self.offsets), 1, kernel_size, kernel_size
        )
        for channel in range(self.channels):
            for index, (vertical, horizontal) in enumerate(self.offsets):
                kernel_y = radius - vertical // dilation
                kernel_x = radius - horizontal // dilation
                kernels[channel * len(self.offsets) + index, 0, kernel_y, kernel_x] = 1
        self.register_buffer("kernels", kernels, persistent=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] != self.channels:
            raise ValueError(
                f"shift bank expected {self.channels} channels, got {values.shape[1]}"
            )
        shifted = F.conv2d(
            values,
            self.kernels.to(dtype=values.dtype),
            padding=self.radius * self.dilation,
            dilation=self.dilation,
            groups=self.channels,
        )
        batch, _channels, height, width = shifted.shape
        return shifted.reshape(
            batch, self.channels, len(self.offsets), height, width
        ).permute(0, 2, 1, 3, 4)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.activation = nn.SiLU()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.conv2(self.activation(self.conv1(values)))


class NormalizedShiftCorrelation(nn.Module):
    """Cosine correlation over a frozen V5 shift bank."""

    MINIMUM_TEMPERATURE = 0.1
    MAXIMUM_TEMPERATURE = 1.5

    def __init__(
        self,
        offsets: Sequence[tuple[int, int]],
        *,
        channels: int,
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
        self.offset_bias = nn.Parameter(torch.zeros(1, len(self.offsets), 1, 1))
        self.shift_bank = FixedShiftBank(channels, self.offsets)
        self.validity_bank = FixedShiftBank(1, self.offsets)
        if (0, 0) in self.offsets:
            with torch.no_grad():
                self.offset_bias[:, self.offsets.index((0, 0))].fill_(center_bias)

    @property
    def temperature(self) -> torch.Tensor:
        span = self.MAXIMUM_TEMPERATURE - self.MINIMUM_TEMPERATURE
        return self.MINIMUM_TEMPERATURE + span * torch.sigmoid(self.raw_temperature)

    @staticmethod
    def _normalize(values: torch.Tensor) -> torch.Tensor:
        inverse = torch.rsqrt(
            values.float().square().sum(dim=1, keepdim=True) + 1e-6
        ).to(values.dtype)
        return values * inverse

    def forward(
        self, reference: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reference.shape != target.shape:
            raise ValueError("reference and target feature shapes must match")
        normalized_reference = self._normalize(reference)
        normalized_target = self._normalize(target)
        shifted_targets = self.shift_bank(target)
        shifted_normalized = self.shift_bank(normalized_target)
        valid = self.validity_bank(torch.ones_like(target[:, :1])).squeeze(2)
        logits = (
            normalized_reference.unsqueeze(1) * shifted_normalized
        ).sum(dim=2)
        logits = logits + (valid - 1.0) * 10_000.0 + self.offset_bias
        weights = torch.softmax(logits / self.temperature, dim=1)
        return (weights.unsqueeze(2) * shifted_targets).sum(dim=1), weights


def apply_shift_weights(
    target: torch.Tensor,
    weights: torch.Tensor,
    offsets: Sequence[tuple[int, int]],
    shift_bank: FixedShiftBank,
) -> torch.Tensor:
    if weights.shape[1] != len(offsets):
        raise ValueError("shift weights and offsets do not match")
    return (weights.unsqueeze(2) * shift_bank(target)).sum(dim=1)


@dataclass(frozen=True)
class MiohRestorerV5Config:
    half_channels: int
    quarter_channels: int
    eighth_channels: int
    sixteenth_channels: int
    fusion_half_channels: int
    fusion_quarter_channels: int
    fusion_eighth_channels: int
    fusion_sixteenth_channels: int
    half_blocks: int
    quarter_blocks: int
    eighth_blocks: int
    sixteenth_blocks: int
    context_frames: int
    output_indices: tuple[int, ...]
    coarse_mode: str = "full49"

    @classmethod
    def quality(cls, *, context_frames: int = 7) -> "MiohRestorerV5Config":
        return cls(
            half_channels=64,
            quarter_channels=96,
            eighth_channels=160,
            sixteenth_channels=224,
            fusion_half_channels=96,
            fusion_quarter_channels=128,
            fusion_eighth_channels=192,
            fusion_sixteenth_channels=256,
            half_blocks=2,
            quarter_blocks=2,
            eighth_blocks=3,
            sixteenth_blocks=4,
            context_frames=context_frames,
            output_indices=QUALITY_OUTPUT_INDICES,
            coarse_mode="full49",
        )

    @classmethod
    def shipping(
        cls,
        *,
        context_frames: int = 9,
        factorized_coarse: bool = False,
    ) -> "MiohRestorerV5Config":
        return cls(
            half_channels=32,
            quarter_channels=48,
            eighth_channels=80,
            sixteenth_channels=128,
            fusion_half_channels=48,
            fusion_quarter_channels=64,
            fusion_eighth_channels=96,
            fusion_sixteenth_channels=128,
            half_blocks=1,
            quarter_blocks=2,
            eighth_blocks=2,
            sixteenth_blocks=2,
            context_frames=context_frames,
            output_indices=(CENTER_INDEX,),
            coarse_mode="factorized18" if factorized_coarse else "full49",
        )

    def validate(self) -> None:
        if self.context_frames not in (5, 7, 9):
            raise ValueError("V5 context must contain 5, 7 or 9 frames")
        if not self.output_indices:
            raise ValueError("V5 requires at least one output")
        if any(index < 0 or index >= NUM_INPUT_FRAMES for index in self.output_indices):
            raise ValueError("V5 output index is outside the input window")
        if self.coarse_mode not in ("full49", "factorized18"):
            raise ValueError("unknown V5 coarse-alignment mode")


class V5EncodedFrame(NamedTuple):
    packed: torch.Tensor
    half: torch.Tensor
    quarter: torch.Tensor
    eighth: torch.Tensor
    sixteenth: torch.Tensor


class FoldedPhaseShiftBank(nn.Module):
    """Exact +/-1 source-pixel shifts in a 2x PixelUnshuffle layout.

    A source-pixel shift becomes a fixed phase-channel permutation plus at most
    one folded-grid displacement. Expressing the operation as one frozen sparse
    convolution avoids returning to a full-resolution feature plane.
    """

    def __init__(self, source_channels: int) -> None:
        super().__init__()
        if source_channels <= 0:
            raise ValueError("phase shift bank needs source channels")
        self.source_channels = int(source_channels)
        self.packed_channels = self.source_channels * 4
        self.offsets = make_offsets(1)
        kernels = torch.zeros(
            len(self.offsets) * self.packed_channels,
            self.packed_channels,
            3,
            3,
        )
        for candidate, (vertical, horizontal) in enumerate(self.offsets):
            for channel in range(self.source_channels):
                for output_y in range(2):
                    for output_x in range(2):
                        source_y_unwrapped = output_y - vertical
                        source_x_unwrapped = output_x - horizontal
                        folded_y = source_y_unwrapped // 2
                        folded_x = source_x_unwrapped // 2
                        source_y = source_y_unwrapped % 2
                        source_x = source_x_unwrapped % 2
                        output_phase = output_y * 2 + output_x
                        source_phase = source_y * 2 + source_x
                        output_channel = (
                            candidate * self.packed_channels
                            + channel * 4
                            + output_phase
                        )
                        input_channel = channel * 4 + source_phase
                        kernels[
                            output_channel,
                            input_channel,
                            1 + folded_y,
                            1 + folded_x,
                        ] = 1
        self.register_buffer("kernels", kernels, persistent=False)

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.shape[1] != self.packed_channels:
            raise ValueError(
                f"phase bank expected {self.packed_channels} channels, "
                f"got {packed.shape[1]}"
            )
        shifted = F.conv2d(packed, self.kernels.to(dtype=packed.dtype), padding=1)
        batch, _channels, height, width = shifted.shape
        return shifted.reshape(
            batch, len(self.offsets), self.packed_channels, height, width
        )


class MiohRestorerV5Encoder(nn.Module):
    """Shared per-frame encoder; all learned work starts at half resolution."""

    def __init__(self, config: MiohRestorerV5Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.half_stage = nn.Sequential(
            nn.Conv2d(FOLDED_FRAME_CHANNELS, config.half_channels, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(config.half_channels),
        )
        self.quarter_stage = nn.Sequential(
            nn.Conv2d(
                config.half_channels,
                config.quarter_channels,
                3,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            ResidualBlock(config.quarter_channels),
        )
        self.eighth_stage = nn.Sequential(
            nn.Conv2d(
                config.quarter_channels,
                config.eighth_channels,
                3,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            ResidualBlock(config.eighth_channels),
        )
        self.sixteenth_stage = nn.Sequential(
            nn.Conv2d(
                config.eighth_channels,
                config.sixteenth_channels,
                3,
                stride=2,
                padding=1,
            ),
            nn.SiLU(),
            ResidualBlock(config.sixteenth_channels),
        )

    def forward(self, frame: torch.Tensor) -> V5EncodedFrame:
        if frame.ndim != 4 or frame.shape[1] != FRAME_CHANNELS:
            raise ValueError("V5 encoder input must be [B,5,H,W]")
        if frame.shape[-2] % 16 or frame.shape[-1] % 16:
            raise ValueError("V5 frame dimensions must be divisible by 16")
        packed = F.pixel_unshuffle(frame, FOLD_FACTOR)
        half = self.half_stage(packed)
        quarter = self.quarter_stage(half)
        eighth = self.eighth_stage(quarter)
        sixteenth = self.sixteenth_stage(eighth)
        return V5EncodedFrame(packed, half, quarter, eighth, sixteenth)


class V5AlignedFrame(NamedTuple):
    packed: torch.Tensor
    half: torch.Tensor
    quarter: torch.Tensor
    eighth: torch.Tensor
    sixteenth: torch.Tensor
    reliability: torch.Tensor
    occlusion: torch.Tensor
    entropy: torch.Tensor


class V5PyramidAlignment(nn.Module):
    """Five-stage folded-space alignment with reliability diagnostics."""

    MINIMUM_TEMPERATURE = 0.1
    MAXIMUM_TEMPERATURE = 1.5

    def __init__(self, config: MiohRestorerV5Config) -> None:
        super().__init__()
        self.config = config
        self.offset_sets_16 = (
            (make_offsets(3, 1),)
            if config.coarse_mode == "full49"
            else (make_offsets(1, 3), make_offsets(1, 1))
        )
        self.offsets_8 = make_offsets(1, 1)
        self.offsets_4 = make_offsets(1, 1)
        self.offsets_2 = make_offsets(1, 1)
        self.corr_16 = nn.ModuleList(
            NormalizedShiftCorrelation(offsets, channels=config.sixteenth_channels)
            for offsets in self.offset_sets_16
        )
        self.corr_8 = NormalizedShiftCorrelation(
            self.offsets_8, channels=config.eighth_channels
        )
        self.corr_4 = NormalizedShiftCorrelation(
            self.offsets_4, channels=config.quarter_channels
        )
        self.corr_2 = NormalizedShiftCorrelation(
            self.offsets_2, channels=config.half_channels
        )

        self.prewarp_8 = nn.ModuleList(
            FixedShiftBank(
                config.eighth_channels,
                tuple((y * 2, x * 2) for y, x in offsets),
            )
            for offsets in self.offset_sets_16
        )
        self.prewarp_4 = nn.ModuleList(
            [
                *(
                    FixedShiftBank(
                        config.quarter_channels,
                        tuple((y * 4, x * 4) for y, x in offsets),
                    )
                    for offsets in self.offset_sets_16
                ),
                FixedShiftBank(config.quarter_channels, make_offsets(1, 2)),
            ]
        )
        self.prewarp_2 = nn.ModuleList(
            [
                *(
                    FixedShiftBank(
                        config.half_channels,
                        tuple((y * 8, x * 8) for y, x in offsets),
                    )
                    for offsets in self.offset_sets_16
                ),
                FixedShiftBank(config.half_channels, make_offsets(1, 4)),
                FixedShiftBank(config.half_channels, make_offsets(1, 2)),
            ]
        )
        self.prewarp_packed = nn.ModuleList(
            [
                *(
                    FixedShiftBank(
                        FOLDED_FRAME_CHANNELS,
                        tuple((y * 8, x * 8) for y, x in offsets),
                    )
                    for offsets in self.offset_sets_16
                ),
                FixedShiftBank(FOLDED_FRAME_CHANNELS, make_offsets(1, 4)),
                FixedShiftBank(FOLDED_FRAME_CHANNELS, make_offsets(1, 2)),
                FixedShiftBank(FOLDED_FRAME_CHANNELS, make_offsets(1, 1)),
            ]
        )

        self.phase_bank = FoldedPhaseShiftBank(FRAME_CHANNELS)
        self.phase_validity_bank = FoldedPhaseShiftBank(1)
        phase_channels = max(16, config.half_channels // 2)
        self.phase_descriptor = nn.Sequential(
            nn.Conv2d(FOLDED_FRAME_CHANNELS, phase_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(phase_channels, phase_channels, 1),
        )
        self.phase_offset_bias = nn.Parameter(torch.zeros(1, 9, 1, 1))
        self.phase_offset_bias.data[:, 4].fill_(1.0)
        initial_temperature = 0.5
        fraction = (
            (initial_temperature - self.MINIMUM_TEMPERATURE)
            / (self.MAXIMUM_TEMPERATURE - self.MINIMUM_TEMPERATURE)
        )
        self.phase_raw_temperature = nn.Parameter(
            torch.tensor(math.log(fraction / (1 - fraction)))
        )
        gate_channels = config.half_channels * 3 + 1
        self.pair_gate = nn.Sequential(
            nn.Conv2d(gate_channels, max(16, config.half_channels // 2), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(max(16, config.half_channels // 2), 2, 1),
        )
        with torch.no_grad():
            self.pair_gate[-1].weight.zero_()
            self.pair_gate[-1].bias.copy_(torch.tensor((2.0, -2.0)))

    @property
    def phase_temperature(self) -> torch.Tensor:
        span = self.MAXIMUM_TEMPERATURE - self.MINIMUM_TEMPERATURE
        return self.MINIMUM_TEMPERATURE + span * torch.sigmoid(
            self.phase_raw_temperature
        )

    @staticmethod
    def _resize_weights(weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(weights, size=target.shape[-2:], mode="nearest")

    @staticmethod
    def _cosine(values: torch.Tensor) -> torch.Tensor:
        return values * torch.rsqrt(
            values.float().square().sum(dim=1, keepdim=True) + 1e-6
        ).to(values.dtype)

    @staticmethod
    def _apply_shift(
        target: torch.Tensor,
        weights: torch.Tensor,
        bank: FixedShiftBank,
    ) -> torch.Tensor:
        return apply_shift_weights(target, weights, bank.offsets, bank)

    def _phase_align(
        self, reference: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates = self.phase_bank(target)
        batch, count, channels, height, width = candidates.shape
        reference_descriptor = self._cosine(self.phase_descriptor(reference))
        candidate_descriptor = self.phase_descriptor(
            candidates.reshape(batch * count, channels, height, width)
        ).reshape(batch, count, -1, height, width)
        candidate_descriptor = candidate_descriptor * torch.rsqrt(
            candidate_descriptor.float().square().sum(dim=2, keepdim=True) + 1e-6
        ).to(candidate_descriptor.dtype)
        logits = (
            reference_descriptor.unsqueeze(1) * candidate_descriptor
        ).sum(dim=2)
        validity_source = torch.ones_like(reference[:, :4])
        valid = self.phase_validity_bank(validity_source).amin(dim=2)
        logits = logits + (valid - 1.0) * 10_000.0 + self.phase_offset_bias
        weights = torch.softmax(logits / self.phase_temperature, dim=1)
        aligned = (weights.unsqueeze(2) * candidates).sum(dim=1)
        entropy = -(
            weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True) / math.log(count)
        return aligned, weights, entropy

    def _forward_impl(
        self,
        reference: V5EncodedFrame,
        target: V5EncodedFrame,
    ) -> tuple[V5AlignedFrame, tuple[torch.Tensor, ...]]:
        aligned_16 = target.sixteenth
        weights_16: list[torch.Tensor] = []
        for correlation in self.corr_16:
            aligned_16, weights = correlation(reference.sixteenth, aligned_16)
            weights_16.append(weights)

        aligned_8 = target.eighth
        for weights, bank in zip(weights_16, self.prewarp_8, strict=True):
            aligned_8 = self._apply_shift(
                aligned_8,
                self._resize_weights(weights, aligned_8),
                bank,
            )
        aligned_8, weights_8 = self.corr_8(reference.eighth, aligned_8)

        aligned_4 = target.quarter
        propagated_4 = (*weights_16, weights_8)
        for weights, bank in zip(propagated_4, self.prewarp_4, strict=True):
            aligned_4 = self._apply_shift(
                aligned_4, self._resize_weights(weights, aligned_4), bank
            )
        aligned_4, weights_4 = self.corr_4(reference.quarter, aligned_4)

        aligned_2 = target.half
        for weights, bank in zip(
            (*weights_16, weights_8, weights_4), self.prewarp_2, strict=True
        ):
            aligned_2 = self._apply_shift(
                aligned_2, self._resize_weights(weights, aligned_2), bank
            )
        aligned_2, weights_2 = self.corr_2(reference.half, aligned_2)

        aligned_packed = target.packed
        for weights, bank in zip(
            (*weights_16, weights_8, weights_4, weights_2),
            self.prewarp_packed,
            strict=True,
        ):
            aligned_packed = self._apply_shift(
                aligned_packed,
                self._resize_weights(weights, aligned_packed),
                bank,
            )
        aligned_packed, phase_weights, entropy = self._phase_align(
            reference.packed, aligned_packed
        )

        reliability_phases = aligned_packed[:, 16:20].mean(dim=1, keepdim=True)
        pair_values = torch.cat(
            (
                reference.half,
                aligned_2,
                torch.abs(reference.half - aligned_2),
                reliability_phases,
            ),
            dim=1,
        )
        gates = torch.sigmoid(self.pair_gate(pair_values))
        reliability = gates[:, :1]
        occlusion = gates[:, 1:2]
        return (
            V5AlignedFrame(
                aligned_packed,
                aligned_2,
                aligned_4,
                aligned_8,
                aligned_16,
                reliability,
                occlusion,
                entropy,
            ),
            (*weights_16, weights_8, weights_4, weights_2, phase_weights),
        )

    def forward(
        self,
        reference: V5EncodedFrame,
        target: V5EncodedFrame,
    ) -> V5AlignedFrame:
        aligned, _weights = self._forward_impl(reference, target)
        return aligned

    def forward_with_diagnostics(
        self,
        reference: V5EncodedFrame,
        target: V5EncodedFrame,
    ) -> tuple[V5AlignedFrame, tuple[torch.Tensor, ...]]:
        """Training-only access to the five alignment distributions."""

        return self._forward_impl(reference, target)


class MiohRestorerV5FusionDecoder(nn.Module):
    """Output-specific alignment, reliability fusion and folded decoder."""

    def __init__(self, config: MiohRestorerV5Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.alignment = V5PyramidAlignment(config)
        contexts = config.context_frames
        self.reduce_16 = nn.Sequential(
            nn.Conv2d(
                config.sixteenth_channels * contexts,
                config.fusion_sixteenth_channels,
                1,
            ),
            nn.SiLU(),
        )
        self.body_16 = nn.Sequential(
            *(ResidualBlock(config.fusion_sixteenth_channels) for _ in range(config.sixteenth_blocks))
        )
        self.up_16_to_8 = nn.Sequential(
            nn.Conv2d(
                config.fusion_sixteenth_channels,
                config.fusion_eighth_channels * 4,
                3,
                padding=1,
            ),
            nn.PixelShuffle(2),
            nn.SiLU(),
        )
        self.reduce_8 = nn.Sequential(
            nn.Conv2d(
                config.fusion_eighth_channels + config.eighth_channels * contexts,
                config.fusion_eighth_channels,
                1,
            ),
            nn.SiLU(),
        )
        self.body_8 = nn.Sequential(
            *(ResidualBlock(config.fusion_eighth_channels) for _ in range(config.eighth_blocks))
        )
        self.up_8_to_4 = nn.Sequential(
            nn.Conv2d(
                config.fusion_eighth_channels,
                config.fusion_quarter_channels * 4,
                3,
                padding=1,
            ),
            nn.PixelShuffle(2),
            nn.SiLU(),
        )
        self.reduce_4 = nn.Sequential(
            nn.Conv2d(
                config.fusion_quarter_channels + config.quarter_channels * contexts,
                config.fusion_quarter_channels,
                1,
            ),
            nn.SiLU(),
        )
        self.body_4 = nn.Sequential(
            *(ResidualBlock(config.fusion_quarter_channels) for _ in range(config.quarter_blocks))
        )
        self.up_4_to_2 = nn.Sequential(
            nn.Conv2d(
                config.fusion_quarter_channels,
                config.fusion_half_channels * 4,
                3,
                padding=1,
            ),
            nn.PixelShuffle(2),
            nn.SiLU(),
        )
        packed_fusion_channels = max(16, config.fusion_half_channels // 2)
        self.reduce_packed = nn.Sequential(
            nn.Conv2d(
                FOLDED_FRAME_CHANNELS * contexts,
                packed_fusion_channels,
                1,
            ),
            nn.SiLU(),
        )
        self.reduce_2 = nn.Sequential(
            nn.Conv2d(
                config.fusion_half_channels
                + config.half_channels * contexts
                + packed_fusion_channels,
                config.fusion_half_channels,
                1,
            ),
            nn.SiLU(),
        )
        self.body_2 = nn.Sequential(
            *(ResidualBlock(config.fusion_half_channels) for _ in range(config.half_blocks))
        )
        self.base_head = nn.Sequential(
            nn.Conv2d(config.fusion_half_channels, config.fusion_half_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(config.fusion_half_channels, 12, 3, padding=1),
        )
        self.texture_head = nn.Sequential(
            nn.Conv2d(config.fusion_half_channels, config.fusion_half_channels, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(config.fusion_half_channels),
            nn.Conv2d(config.fusion_half_channels, 12, 3, padding=1),
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(config.fusion_half_channels, max(16, config.fusion_half_channels // 2), 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(max(16, config.fusion_half_channels // 2), 4, 3, padding=1),
        )
        self._zero_initialize_outputs()

    def _zero_initialize_outputs(self) -> None:
        for head in (self.base_head, self.texture_head, self.confidence_head):
            output = head[-1]
            if not isinstance(output, nn.Conv2d):
                raise TypeError("V5 head must end with convolution")
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def _context_indices(self, reference_index: int) -> tuple[int, ...]:
        count = self.config.context_frames
        start = min(
            max(reference_index - count // 2, 0), NUM_INPUT_FRAMES - count
        )
        return tuple(range(start, start + count))

    @staticmethod
    def _gate(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return value * F.interpolate(gate, size=value.shape[-2:], mode="nearest")

    def _aligned_context(
        self,
        encoded: tuple[V5EncodedFrame, ...],
        reference_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = encoded[reference_index]
        packed: list[torch.Tensor] = []
        half: list[torch.Tensor] = []
        quarter: list[torch.Tensor] = []
        eighth: list[torch.Tensor] = []
        sixteenth: list[torch.Tensor] = []
        for index in self._context_indices(reference_index):
            target = encoded[index]
            if index == reference_index:
                gate = torch.ones_like(reference.half[:, :1])
                aligned = V5AlignedFrame(
                    target.packed,
                    target.half,
                    target.quarter,
                    target.eighth,
                    target.sixteenth,
                    gate,
                    torch.zeros_like(gate),
                    torch.zeros_like(gate),
                )
            else:
                aligned = self.alignment(reference, target)
                gate = aligned.reliability * (1.0 - aligned.occlusion)
            packed.append(self._gate(aligned.packed, gate))
            half.append(self._gate(aligned.half, gate))
            quarter.append(self._gate(aligned.quarter, gate))
            eighth.append(self._gate(aligned.eighth, gate))
            sixteenth.append(self._gate(aligned.sixteenth, gate))
        return (
            torch.cat(packed, dim=1),
            torch.cat(half, dim=1),
            torch.cat(quarter, dim=1),
            torch.cat(eighth, dim=1),
            torch.cat(sixteenth, dim=1),
        )

    def _decode(
        self,
        packed: torch.Tensor,
        half_context: torch.Tensor,
        quarter_context: torch.Tensor,
        eighth_context: torch.Tensor,
        sixteenth_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sixteenth = self.body_16(self.reduce_16(sixteenth_context))
        eighth = self.up_16_to_8(sixteenth)
        eighth = self.body_8(
            self.reduce_8(torch.cat((eighth, eighth_context), dim=1))
        )
        quarter = self.up_8_to_4(eighth)
        quarter = self.body_4(
            self.reduce_4(torch.cat((quarter, quarter_context), dim=1))
        )
        half = self.up_4_to_2(quarter)
        packed_features = self.reduce_packed(packed)
        half = self.body_2(
            self.reduce_2(torch.cat((half, half_context, packed_features), dim=1))
        )
        base = F.pixel_shuffle(self.base_head(half), 2)
        texture = F.pixel_shuffle(self.texture_head(half), 2)
        confidence = torch.sigmoid(
            F.pixel_shuffle(self.confidence_head(half), 2)
        )
        return base, texture, confidence

    @staticmethod
    def _restore_output_layout(
        values: torch.Tensor, batch: int, outputs: int
    ) -> torch.Tensor:
        return values.reshape(outputs, batch, values.shape[1], *values.shape[-2:]).transpose(0, 1)

    def forward_components_from_encoded(
        self,
        encoded: tuple[V5EncodedFrame, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(encoded) != NUM_INPUT_FRAMES:
            raise ValueError("V5 decoder needs nine encoded frames")
        contexts = [
            self._aligned_context(encoded, reference)
            for reference in self.config.output_indices
        ]
        decoded = self._decode(
            *(torch.cat([context[index] for context in contexts], dim=0) for index in range(5))
        )
        batch = encoded[0].packed.shape[0]
        outputs = len(self.config.output_indices)
        base = self._restore_output_layout(decoded[0], batch, outputs)
        texture = self._restore_output_layout(decoded[1], batch, outputs)
        confidence = self._restore_output_layout(decoded[2], batch, outputs)

        reference_packed = torch.cat(
            [encoded[index].packed for index in self.config.output_indices], dim=0
        )
        reference_values = F.pixel_shuffle(reference_packed, 2)
        reference_values = self._restore_output_layout(
            reference_values, batch, outputs
        )
        rgb = reference_values[:, :, :3]
        mask = reference_values[:, :, 3:4]
        restored = rgb + mask * (base + confidence * texture)
        return restored, confidence, base, texture

    def forward_from_encoded(
        self, encoded: tuple[V5EncodedFrame, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        restored, confidence, _base, _texture = self.forward_components_from_encoded(
            encoded
        )
        return restored, confidence


class MiohRestorerV5(nn.Module):
    def __init__(self, config: MiohRestorerV5Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.encoder = MiohRestorerV5Encoder(config)
        self.decoder = MiohRestorerV5FusionDecoder(config)

    @classmethod
    def quality(cls, *, context_frames: int = 7) -> "MiohRestorerV5":
        return cls(MiohRestorerV5Config.quality(context_frames=context_frames))

    @classmethod
    def shipping(
        cls,
        *,
        context_frames: int = 9,
        factorized_coarse: bool = False,
    ) -> "MiohRestorerV5":
        return cls(
            MiohRestorerV5Config.shipping(
                context_frames=context_frames,
                factorized_coarse=factorized_coarse,
            )
        )

    def encode_window(self, values: torch.Tensor) -> tuple[V5EncodedFrame, ...]:
        self._validate_input(values)
        return tuple(self.encoder(values[:, index]) for index in range(NUM_INPUT_FRAMES))

    @staticmethod
    def _validate_input(values: torch.Tensor) -> None:
        if values.ndim != 5 or values.shape[1:3] != (
            NUM_INPUT_FRAMES,
            FRAME_CHANNELS,
        ):
            raise ValueError("V5 input must have shape [B,9,5,H,W]")
        if values.shape[-2] != values.shape[-1]:
            raise ValueError("V5 Stage 0 supports square ROI buckets")
        if values.shape[-1] not in V5_BUCKETS and values.shape[-1] < 32:
            raise ValueError("V5 test inputs must be at least 32 pixels")
        if values.shape[-2] % 16 or values.shape[-1] % 16:
            raise ValueError("V5 frame dimensions must be divisible by 16")

    def forward_components(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.decoder.forward_components_from_encoded(self.encode_window(values))

    def alignment_diagnostics(
        self,
        values: torch.Tensor,
        *,
        reference_index: int = CENTER_INDEX,
        target_index: int = 0,
    ) -> tuple[V5AlignedFrame, tuple[torch.Tensor, ...]]:
        """Training-only pair diagnostics for known/natural motion losses."""

        if reference_index == target_index:
            raise ValueError("alignment diagnostics need two different frames")
        encoded = self.encode_window(values)
        return self.decoder.alignment.forward_with_diagnostics(
            encoded[reference_index], encoded[target_index]
        )

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        restored, confidence, _base, _texture = self.forward_components(values)
        return restored, confidence


class MiohRestorerV5ExportWrapper(nn.Module):
    """Flat monolithic Core ML contract."""

    def __init__(self, model: MiohRestorerV5) -> None:
        super().__init__()
        self.model = model

    def forward(self, flat_frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _channels, height, width = flat_frames.shape
        values = flat_frames.reshape(
            batch, NUM_INPUT_FRAMES, FRAME_CHANNELS, height, width
        )
        restored, confidence = self.model(values)
        return torch.clamp(restored, 0.0, 1.0).flatten(1, 2), confidence.flatten(1, 2)


class MiohRestorerV5EncoderExportWrapper(nn.Module):
    def __init__(self, encoder: MiohRestorerV5Encoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, frame: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.encoder(frame))


class MiohRestorerV5DecoderExportWrapper(nn.Module):
    """Explicit feature-ring contract for the split V5-S deployment."""

    def __init__(self, decoder: MiohRestorerV5FusionDecoder) -> None:
        super().__init__()
        if decoder.config.output_indices != (CENTER_INDEX,):
            raise ValueError("split V5 decoder is defined for the shipping center output")
        self.decoder = decoder

    @staticmethod
    def _frames(flat: torch.Tensor, channels: int) -> tuple[torch.Tensor, ...]:
        batch, _channels, height, width = flat.shape
        values = flat.reshape(batch, NUM_INPUT_FRAMES, channels, height, width)
        return tuple(values[:, index] for index in range(NUM_INPUT_FRAMES))

    def forward(
        self,
        packed: torch.Tensor,
        half: torch.Tensor,
        quarter: torch.Tensor,
        eighth: torch.Tensor,
        sixteenth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.decoder.config
        packed_frames = self._frames(packed, FOLDED_FRAME_CHANNELS)
        half_frames = self._frames(half, config.half_channels)
        quarter_frames = self._frames(quarter, config.quarter_channels)
        eighth_frames = self._frames(eighth, config.eighth_channels)
        sixteenth_frames = self._frames(sixteenth, config.sixteenth_channels)
        encoded = tuple(
            V5EncodedFrame(*items)
            for items in zip(
                packed_frames,
                half_frames,
                quarter_frames,
                eighth_frames,
                sixteenth_frames,
                strict=True,
            )
        )
        restored, confidence = self.decoder.forward_from_encoded(encoded)
        return torch.clamp(restored, 0.0, 1.0).flatten(1, 2), confidence.flatten(1, 2)


class MiohRestorerV5StatefulExportWrapper(nn.Module):
    """Single-model V5-S contract with explicit eight-frame feature state."""

    def __init__(self, model: MiohRestorerV5) -> None:
        super().__init__()
        if model.config.output_indices != (CENTER_INDEX,):
            raise ValueError("stateful V5 export is defined for the shipping model")
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.config = model.config

    @staticmethod
    def _old_frames(
        flat: torch.Tensor, channels: int
    ) -> tuple[torch.Tensor, ...]:
        batch, _channels, height, width = flat.shape
        values = flat.reshape(batch, NUM_INPUT_FRAMES - 1, channels, height, width)
        return tuple(values[:, index] for index in range(NUM_INPUT_FRAMES - 1))

    @staticmethod
    def _next_state(frames: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return torch.cat(frames[1:], dim=1)

    def forward(
        self,
        current_frame: torch.Tensor,
        packed_state: torch.Tensor,
        half_state: torch.Tensor,
        quarter_state: torch.Tensor,
        eighth_state: torch.Tensor,
        sixteenth_state: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        current = self.encoder(current_frame)
        packed = self._old_frames(packed_state, FOLDED_FRAME_CHANNELS) + (current.packed,)
        half = self._old_frames(half_state, self.config.half_channels) + (current.half,)
        quarter = self._old_frames(quarter_state, self.config.quarter_channels) + (current.quarter,)
        eighth = self._old_frames(eighth_state, self.config.eighth_channels) + (current.eighth,)
        sixteenth = self._old_frames(
            sixteenth_state, self.config.sixteenth_channels
        ) + (current.sixteenth,)
        encoded = tuple(
            V5EncodedFrame(*items)
            for items in zip(packed, half, quarter, eighth, sixteenth, strict=True)
        )
        restored, confidence = self.decoder.forward_from_encoded(encoded)
        return (
            torch.clamp(restored, 0.0, 1.0).flatten(1, 2),
            confidence.flatten(1, 2),
            self._next_state(packed),
            self._next_state(half),
            self._next_state(quarter),
            self._next_state(eighth),
            self._next_state(sixteenth),
        )


def flatten_encoded_window(
    encoded: tuple[V5EncodedFrame, ...]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(encoded) != NUM_INPUT_FRAMES:
        raise ValueError("V5 feature ring must contain nine frames")
    return tuple(
        torch.cat([frame[index] for frame in encoded], dim=1)
        for index in range(5)
    )  # type: ignore[return-value]


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
