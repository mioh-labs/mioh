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
V41_NEW_STATE_PREFIXES = (
    "detail_full_projection.",
    "detail_half_projection.",
    "detail_alignment.",
    "detail_half_fusion.",
    "detail_full_fusion.",
    "detail_output.",
)


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


class FixedShiftBank(nn.Module):
    """Produce a static shift bank with one depthwise convolution.

    The frozen one-hot kernels are exactly the same translations as
    :func:`shift2d`, including zero padding at frame boundaries.  Keeping the
    candidates identical while expressing the bank as a convolution avoids
    expanding every shift into a separate pad/slice subgraph during Core ML
    conversion.  No candidate is removed or approximated.
    """

    def __init__(
        self,
        channels: int,
        offsets: Sequence[tuple[int, int]],
    ) -> None:
        super().__init__()
        if channels <= 0 or not offsets:
            raise ValueError("shift-bank channels and offsets must be non-empty")
        self.channels = int(channels)
        self.offsets = tuple((int(y), int(x)) for y, x in offsets)

        nonzero = [
            abs(value)
            for offset in self.offsets
            for value in offset
            if value
        ]
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
    """Core ML friendly cosine correlation over a fixed shift bank."""

    MINIMUM_TEMPERATURE = 0.1
    MAXIMUM_TEMPERATURE = 1.5

    def __init__(
        self,
        offsets: Sequence[tuple[int, int]],
        *,
        channels: int | None = None,
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
        self.shift_bank = (
            FixedShiftBank(channels, self.offsets) if channels is not None else None
        )
        self.validity_bank = FixedShiftBank(1, self.offsets)
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
        if self.shift_bank is None:
            shifted_targets = torch.stack(
                [shift2d(target, y, x) for y, x in self.offsets], dim=1
            )
            shifted_normalized = torch.stack(
                [shift2d(normalized_target, y, x) for y, x in self.offsets],
                dim=1,
            )
        else:
            shifted_targets = self.shift_bank(target)
            shifted_normalized = self.shift_bank(normalized_target)
        valid = self.validity_bank(validity_source).squeeze(2)
        logits = (
            normalized_reference.unsqueeze(1) * shifted_normalized
        ).sum(dim=2)
        logits = logits + (valid - 1.0) * 10_000.0 + self.offset_bias
        weights = torch.softmax(logits / self.temperature, dim=1)
        aligned = (weights.unsqueeze(2) * shifted_targets).sum(dim=1)
        return aligned, weights


def apply_shift_weights(
    target: torch.Tensor,
    weights: torch.Tensor,
    offsets: Sequence[tuple[int, int]],
    shift_bank: FixedShiftBank | None = None,
) -> torch.Tensor:
    if weights.shape[1] != len(offsets):
        raise ValueError("shift weights and offsets do not match")
    shifted = (
        shift_bank(target)
        if shift_bank is not None
        else torch.stack(
            [shift2d(target, y, x) for y, x in offsets], dim=1
        )
    )
    return (weights.unsqueeze(2) * shifted).sum(dim=1)


class HierarchicalAlignment27(nn.Module):
    """Three nine-way banks covering +/-40 input pixels."""

    input_reach = 40

    def __init__(self, *, eighth_channels: int = 96, quarter_channels: int = 64) -> None:
        super().__init__()
        self.offsets_eighth_coarse = make_offsets(1, 3)
        self.offsets_eighth_fine = make_offsets(1, 1)
        self.offsets_quarter_fine = make_offsets(1, 2)
        self.offsets_quarter_coarse = make_offsets(1, 6)
        self.offsets_quarter_mid = make_offsets(1, 2)
        self.eighth_coarse = NormalizedShiftCorrelation(
            self.offsets_eighth_coarse, channels=eighth_channels
        )
        self.eighth_fine = NormalizedShiftCorrelation(
            self.offsets_eighth_fine, channels=eighth_channels
        )
        self.quarter_fine = NormalizedShiftCorrelation(
            self.offsets_quarter_fine, channels=quarter_channels
        )
        self.quarter_coarse_bank = FixedShiftBank(
            quarter_channels, self.offsets_quarter_coarse
        )
        self.quarter_mid_bank = FixedShiftBank(
            quarter_channels, self.offsets_quarter_mid
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
            self.quarter_coarse_bank,
        )
        aligned_quarter = apply_shift_weights(
            aligned_quarter,
            middle_quarter,
            self.offsets_quarter_mid,
            self.quarter_mid_bank,
        )
        aligned_quarter, fine = self.quarter_fine(
            reference_quarter, aligned_quarter
        )
        return aligned_eighth, aligned_quarter, (coarse, middle, fine)


class HighResolutionDetailAlignment(nn.Module):
    """Refine hierarchical motion down to a one-input-pixel grid.

    The existing V4 alignment ends on a quarter-resolution dilation-two bank,
    whose candidates are eight input pixels apart.  Reusing those weights at
    high resolution without refinement would blend displaced texture and blur
    it.  This module first carries the three coarse banks to half/full
    resolution, then adds two half-resolution banks and one full-resolution
    bank.  All operations remain static shifts, correlation, softmax and
    elementwise blending.
    """

    def __init__(
        self,
        alignment: HierarchicalAlignment27,
        *,
        full_channels: int,
        half_channels: int,
    ) -> None:
        super().__init__()
        self.offsets_eighth_coarse = alignment.offsets_eighth_coarse
        self.offsets_eighth_fine = alignment.offsets_eighth_fine
        self.offsets_quarter_fine = alignment.offsets_quarter_fine
        self.offsets_half_coarse = make_offsets(1, 2)
        self.offsets_half_fine = make_offsets(1, 1)
        self.offsets_full_fine = make_offsets(1, 1)
        self.half_coarse = NormalizedShiftCorrelation(
            self.offsets_half_coarse, channels=half_channels
        )
        self.half_fine = NormalizedShiftCorrelation(
            self.offsets_half_fine, channels=half_channels
        )
        self.full_fine = NormalizedShiftCorrelation(
            self.offsets_full_fine, channels=full_channels
        )
        original_half_offsets = (
            self._scaled_offsets(self.offsets_eighth_coarse, 4),
            self._scaled_offsets(self.offsets_eighth_fine, 4),
            self._scaled_offsets(self.offsets_quarter_fine, 2),
        )
        packed_channels = full_channels * 4
        self.original_half_banks = nn.ModuleList(
            FixedShiftBank(half_channels, offsets)
            for offsets in original_half_offsets
        )
        self.original_packed_banks = nn.ModuleList(
            FixedShiftBank(packed_channels, offsets)
            for offsets in original_half_offsets
        )
        self.packed_half_coarse_bank = FixedShiftBank(
            packed_channels, self.offsets_half_coarse
        )
        self.packed_half_fine_bank = FixedShiftBank(
            packed_channels, self.offsets_half_fine
        )

    @staticmethod
    def _scaled_offsets(
        offsets: Sequence[tuple[int, int]], factor: int
    ) -> tuple[tuple[int, int], ...]:
        return tuple(
            (vertical * factor, horizontal * factor)
            for vertical, horizontal in offsets
        )

    @staticmethod
    def _resize_weights(
        weights: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return F.interpolate(weights, size=target.shape[-2:], mode="nearest")

    def _apply_original_banks(
        self,
        target: torch.Tensor,
        weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        *,
        eighth_scale: int,
        quarter_scale: int,
        shift_banks: Sequence[FixedShiftBank],
    ) -> torch.Tensor:
        coarse, middle, fine = weights
        target = apply_shift_weights(
            target,
            self._resize_weights(coarse, target),
            self._scaled_offsets(
                self.offsets_eighth_coarse, eighth_scale
            ),
            shift_banks[0],
        )
        target = apply_shift_weights(
            target,
            self._resize_weights(middle, target),
            self._scaled_offsets(
                self.offsets_eighth_fine, eighth_scale
            ),
            shift_banks[1],
        )
        return apply_shift_weights(
            target,
            self._resize_weights(fine, target),
            self._scaled_offsets(
                self.offsets_quarter_fine, quarter_scale
            ),
            shift_banks[2],
        )

    def forward(
        self,
        reference_full: torch.Tensor,
        target_full: torch.Tensor,
        reference_half: torch.Tensor,
        target_half: torch.Tensor,
        coarse_weights: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aligned_half = self._apply_original_banks(
            target_half,
            coarse_weights,
            eighth_scale=4,
            quarter_scale=2,
            shift_banks=self.original_half_banks,
        )
        aligned_half, half_coarse = self.half_coarse(
            reference_half, aligned_half
        )
        aligned_half, half_fine = self.half_fine(reference_half, aligned_half)

        # Carry full-resolution texture through the coarse banks without
        # replaying 45 memory-bound shifts on the 384x384 plane.  Space-to-
        # depth is lossless: all four pixel phases remain separate channels at
        # half resolution.  Only the final +/-1px bank runs at full size.
        packed_full = F.pixel_unshuffle(target_full, 2)
        packed_full = self._apply_original_banks(
            packed_full,
            coarse_weights,
            eighth_scale=4,
            quarter_scale=2,
            shift_banks=self.original_packed_banks,
        )
        packed_full = apply_shift_weights(
            packed_full,
            self._resize_weights(half_coarse, packed_full),
            self.offsets_half_coarse,
            self.packed_half_coarse_bank,
        )
        packed_full = apply_shift_weights(
            packed_full,
            self._resize_weights(half_fine, packed_full),
            self.offsets_half_fine,
            self.packed_half_fine_bank,
        )
        aligned_full = F.pixel_shuffle(packed_full, 2)
        aligned_full, _full_fine = self.full_fine(
            reference_full, aligned_full
        )
        return aligned_full, aligned_half


class FullAlignment121(nn.Module):
    """Exhaustive correctness baseline; not intended for V4 training."""

    input_reach = 48

    def __init__(self, *, eighth_channels: int = 96, quarter_channels: int = 64) -> None:
        super().__init__()
        self.offsets_eighth = make_offsets(5)
        self.offsets_quarter = tuple(
            (vertical * 2, horizontal * 2)
            for vertical, horizontal in self.offsets_eighth
        )
        self.offsets_quarter_fine = make_offsets(1, 2)
        self.eighth = NormalizedShiftCorrelation(
            self.offsets_eighth, channels=eighth_channels
        )
        self.quarter_fine = NormalizedShiftCorrelation(
            self.offsets_quarter_fine, channels=quarter_channels
        )
        self.quarter_bank = FixedShiftBank(quarter_channels, self.offsets_quarter)

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
            self.quarter_bank,
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        full = self.stem(values)
        half = self.half_stage(full)
        quarter = self.quarter(half)
        eighth = self.eighth(quarter)
        return full, half, quarter, eighth


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
        high_resolution_detail: bool = False,
        detail_full_channels: int = 32,
        detail_half_channels: int = 48,
        detail_fusion_channels: int = 64,
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
        self.high_resolution_detail = bool(high_resolution_detail)
        self.detail_full_channels = int(detail_full_channels)
        self.detail_half_channels = int(detail_half_channels)
        self.detail_fusion_channels = int(detail_fusion_channels)
        self.architecture_revision = 2 if self.high_resolution_detail else 1
        self.gradient_checkpointing = False
        self.encoder = SharedFrameEncoder(
            quarter_channels=quarter_channels,
            eighth_channels=eighth_channels,
        )
        self.alignment: HierarchicalAlignment27 | FullAlignment121 = (
            HierarchicalAlignment27(
                eighth_channels=eighth_channels,
                quarter_channels=quarter_channels,
            )
            if alignment_variant == "hier27"
            else FullAlignment121(
                eighth_channels=eighth_channels,
                quarter_channels=quarter_channels,
            )
        )
        if self.high_resolution_detail:
            if not isinstance(self.alignment, HierarchicalAlignment27):
                raise ValueError(
                    "high-resolution detail requires hierarchical alignment"
                )
            self.detail_full_projection = nn.Sequential(
                nn.Conv2d(32, self.detail_full_channels, 1),
                nn.SiLU(),
            )
            self.detail_half_projection = nn.Sequential(
                nn.Conv2d(48, self.detail_half_channels, 1),
                nn.SiLU(),
            )
            self.detail_alignment = HighResolutionDetailAlignment(
                self.alignment,
                full_channels=self.detail_full_channels,
                half_channels=self.detail_half_channels,
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
        if self.high_resolution_detail:
            self.detail_half_fusion = nn.Sequential(
                nn.Conv2d(
                    self.detail_half_channels * context_frames,
                    self.detail_half_channels,
                    1,
                ),
                nn.SiLU(),
                ResidualBlock(self.detail_half_channels),
                nn.Conv2d(
                    self.detail_half_channels,
                    self.detail_full_channels * 4,
                    3,
                    padding=1,
                ),
                nn.PixelShuffle(2),
                nn.SiLU(),
            )
            self.detail_full_fusion = nn.Sequential(
                nn.Conv2d(
                    48
                    + self.detail_full_channels
                    + self.detail_full_channels * context_frames,
                    self.detail_fusion_channels,
                    1,
                ),
                nn.SiLU(),
                ResidualBlock(self.detail_fusion_channels),
                ResidualBlock(self.detail_fusion_channels),
            )
            self.detail_output = nn.Conv2d(
                self.detail_fusion_channels, 3, 3, padding=1
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
        if self.high_resolution_detail:
            nn.init.zeros_(self.detail_output.weight)
            nn.init.zeros_(self.detail_output.bias)

    def high_resolution_detail_modules(self) -> tuple[nn.Module, ...]:
        if not self.high_resolution_detail:
            return ()
        return (
            self.detail_full_projection,
            self.detail_half_projection,
            self.detail_alignment,
            self.detail_half_fusion,
            self.detail_full_fusion,
            self.detail_output,
        )

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
    ) -> list[
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor | None,
            torch.Tensor | None,
        ]
    ]:
        if self.training and self.gradient_checkpointing:
            encoded = [
                checkpoint(
                    self.encoder,
                    values[:, index],
                    use_reentrant=False,
                )
                for index in range(NUM_INPUT_FRAMES)
            ]
        else:
            encoded = [
                self.encoder(values[:, index]) for index in range(NUM_INPUT_FRAMES)
            ]
        features = []
        for full, half, quarter, eighth in encoded:
            if self.high_resolution_detail:
                detail_full = self.detail_full_projection(full)
                detail_half = self.detail_half_projection(half)
            else:
                detail_full = None
                detail_half = None
            features.append((quarter, eighth, detail_full, detail_half))
        return features

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
        features: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
            ]
        ],
        reference_index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        reference_quarter, reference_eighth, reference_full, reference_half = (
            features[reference_index]
        )
        quarters: list[torch.Tensor] = []
        eighths: list[torch.Tensor] = []
        fulls: list[torch.Tensor] = []
        halves: list[torch.Tensor] = []
        for frame_index in range(
            reference_index - LOCAL_CONTEXT_RADIUS,
            reference_index + LOCAL_CONTEXT_RADIUS + 1,
        ):
            target_quarter, target_eighth, target_full, target_half = features[
                frame_index
            ]
            if frame_index == reference_index:
                aligned_eighth = target_eighth
                aligned_quarter = target_quarter
                aligned_full = target_full
                aligned_half = target_half
            else:
                if self.high_resolution_detail:
                    aligned_eighth, aligned_quarter, weights = (
                        self._align_pair_with_weights(
                            reference_eighth,
                            target_eighth,
                            reference_quarter,
                            target_quarter,
                        )
                    )
                    if any(
                        item is None
                        for item in (
                            reference_full,
                            target_full,
                            reference_half,
                            target_half,
                        )
                    ):
                        raise AssertionError("V4.1 detail features are missing")
                    aligned_full, aligned_half = self.detail_alignment(
                        reference_full,
                        target_full,
                        reference_half,
                        target_half,
                        weights,
                    )
                else:
                    aligned_eighth, aligned_quarter = self._align_pair(
                        reference_eighth,
                        target_eighth,
                        reference_quarter,
                        target_quarter,
                    )
                    aligned_full = None
                    aligned_half = None
            eighths.append(aligned_eighth)
            quarters.append(aligned_quarter)
            if self.high_resolution_detail:
                if aligned_full is None or aligned_half is None:
                    raise AssertionError("V4.1 aligned detail is missing")
                fulls.append(aligned_full)
                halves.append(aligned_half)
        return (
            torch.cat(eighths, dim=1),
            torch.cat(quarters, dim=1),
            torch.cat(fulls, dim=1) if fulls else None,
            torch.cat(halves, dim=1) if halves else None,
        )

    def _aligned_context_with_diagnostics(
        self,
        features: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
            ]
        ],
        reference_index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        tuple[
            tuple[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]], ...
        ],
    ]:
        reference_quarter, reference_eighth, reference_full, reference_half = (
            features[reference_index]
        )
        quarters: list[torch.Tensor] = []
        eighths: list[torch.Tensor] = []
        fulls: list[torch.Tensor] = []
        halves: list[torch.Tensor] = []
        records: list[
            tuple[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        ] = []
        for frame_index in range(
            reference_index - LOCAL_CONTEXT_RADIUS,
            reference_index + LOCAL_CONTEXT_RADIUS + 1,
        ):
            target_quarter, target_eighth, target_full, target_half = features[
                frame_index
            ]
            if frame_index == reference_index:
                aligned_eighth = target_eighth
                aligned_quarter = target_quarter
                aligned_full = target_full
                aligned_half = target_half
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
                if self.high_resolution_detail:
                    if any(
                        item is None
                        for item in (
                            reference_full,
                            target_full,
                            reference_half,
                            target_half,
                        )
                    ):
                        raise AssertionError("V4.1 detail features are missing")
                    aligned_full, aligned_half = self.detail_alignment(
                        reference_full,
                        target_full,
                        reference_half,
                        target_half,
                        weights,
                    )
                else:
                    aligned_full = None
                    aligned_half = None
            eighths.append(aligned_eighth)
            quarters.append(aligned_quarter)
            if self.high_resolution_detail:
                if aligned_full is None or aligned_half is None:
                    raise AssertionError("V4.1 aligned detail is missing")
                fulls.append(aligned_full)
                halves.append(aligned_half)
        return (
            torch.cat(eighths, dim=1),
            torch.cat(quarters, dim=1),
            torch.cat(fulls, dim=1) if fulls else None,
            torch.cat(halves, dim=1) if halves else None,
            tuple(records),
        )

    def _decode_context(
        self,
        context_eighth: torch.Tensor,
        context_quarter: torch.Tensor,
        context_full: torch.Tensor | None = None,
        context_half: torch.Tensor | None = None,
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
        if self.high_resolution_detail:
            if context_full is None or context_half is None:
                raise ValueError("V4.1 decoder requires full and half detail")
            detail_half = self.detail_half_fusion(context_half)
            detail = self.detail_full_fusion(
                torch.cat((full, detail_half, context_full), dim=1)
            )
            texture = texture + self.detail_output(detail)
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
        contexts: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
            ]
        ] = []
        alignment_records: list[
            tuple[tuple[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]], ...]
        ] = []
        for index in self.output_indices:
            if capture_alignment:
                eighth, quarter, full, half, records = (
                    self._aligned_context_with_diagnostics(features, index)
                )
                contexts.append((eighth, quarter, full, half))
                alignment_records.append(records)
            else:
                contexts.append(self._aligned_context(features, index))

        batch = values.shape[0]
        fused_eighth: torch.Tensor | None = None
        fused_quarter: torch.Tensor | None = None
        if self.execution_mode in ("batch", "center1"):
            context_eighth = torch.cat([context[0] for context in contexts], dim=0)
            context_quarter = torch.cat([context[1] for context in contexts], dim=0)
            context_full = (
                torch.cat([context[2] for context in contexts], dim=0)
                if self.high_resolution_detail
                else None
            )
            context_half = (
                torch.cat([context[3] for context in contexts], dim=0)
                if self.high_resolution_detail
                else None
            )
            decoded = self._decode_context(
                context_eighth,
                context_quarter,
                context_full,
                context_half,
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


def load_v4_state_for_v41_upgrade(
    model: MiohRestorerV4Q,
    state_dict: dict[str, torch.Tensor],
    *,
    source_revision: int,
) -> tuple[str, ...]:
    """Load V4/V4.1 weights without hiding accidental incompatibilities.

    A revision-1 source must provide every legacy key and may omit exactly the
    explicitly listed V4.1 modules.  A revision-2 source must be an exact
    match.  Shape changes and unexpected keys are always fatal.
    """

    if not model.high_resolution_detail:
        raise ValueError("V4.1 upgrade loading requires high-resolution detail")
    current = model.state_dict()
    current_keys = set(current)
    source_keys = set(state_dict)
    unexpected = sorted(source_keys - current_keys)
    if unexpected:
        raise ValueError(f"unexpected checkpoint keys: {unexpected}")
    mismatched = sorted(
        key
        for key in source_keys
        if tuple(state_dict[key].shape) != tuple(current[key].shape)
    )
    if mismatched:
        raise ValueError(f"checkpoint tensor shapes changed: {mismatched}")
    missing = sorted(current_keys - source_keys)
    allowed_new = sorted(
        key
        for key in current_keys
        if key.startswith(V41_NEW_STATE_PREFIXES)
    )
    expected_missing = allowed_new if source_revision < 2 else []
    if missing != expected_missing:
        raise ValueError(
            "checkpoint missing keys do not exactly match the V4.1 whitelist: "
            f"missing={missing}, expected={expected_missing}"
        )
    merged = dict(current)
    merged.update(state_dict)
    model.load_state_dict(merged, strict=True)
    return tuple(missing)
