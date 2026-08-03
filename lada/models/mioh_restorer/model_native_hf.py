# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Native-resolution temporal high-frequency refiner for mosaic ROIs.

The model is deliberately *not* another complete restoration network.  It
consumes the native mosaic window together with a frozen 256px BasicVSR++
global reconstruction and predicts only a band-limited residual.  Keeping the
global solution outside this graph prevents independent high-resolution tiles
from inventing different colours or low-frequency shapes.

Input channels per frame are::

    RGB mosaic (3), mosaic mask (1), mask reliability (1), global base RGB (3)

All learned work starts after PixelUnshuffle(2).  The final PixelShuffle is the
only return to source resolution, which preserves the four source-pixel phases
without placing a convolution on the 512x512 plane.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from lada.models.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet

from .model_v5 import (
    FixedShiftBank,
    FoldedPhaseShiftBank,
    NormalizedShiftCorrelation,
    ResidualBlock,
    apply_shift_weights,
    make_offsets,
)


# The frozen global guide still sees the full nine-frame BasicVSR++ window.
# The native refiner only needs the centre five because its job is local HF
# recovery; this keeps 512px activation memory bounded.
NATIVE_HF_GUIDE_FRAMES = 9
NATIVE_HF_INPUT_FRAMES = 5
NATIVE_HF_CENTER = 2
NATIVE_HF_FRAME_CHANNELS = 8
NATIVE_HF_SOURCE_CHANNELS = 3
NATIVE_HF_MASK_CHANNEL = 3
NATIVE_HF_RELIABILITY_CHANNEL = 4
NATIVE_HF_BASE_START = 5
NATIVE_HF_FOLD = 2
NATIVE_HF_PACKED_CHANNELS = NATIVE_HF_FRAME_CHANNELS * NATIVE_HF_FOLD**2
NATIVE_HF_MODEL_INITIALIZATION_SEED = 20260802
NATIVE_HF_INITIALIZATION_RECIPE = "frozen-analytic-alignment-v2"


def _gaussian_blur_rgb(values: torch.Tensor) -> torch.Tensor:
    """Fixed 5x5 Gaussian used to remove low-frequency residual drift."""

    kernel = values.new_tensor((1.0, 4.0, 6.0, 4.0, 1.0))
    kernel = torch.outer(kernel, kernel)
    kernel = (kernel / kernel.sum()).reshape(1, 1, 5, 5).repeat(3, 1, 1, 1)
    return F.conv2d(
        F.pad(values, (2, 2, 2, 2), mode="replicate"), kernel, groups=3
    )


def band_limit_residual(values: torch.Tensor) -> torch.Tensor:
    """Force a residual to remain in the high-frequency branch."""

    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("native HF residual must be [B,3,H,W]")
    return values - _gaussian_blur_rgb(values)


@dataclass(frozen=True)
class NativeHF512Config:
    """Fixed prototype contract; tiny values may be supplied by unit tests."""

    input_frames: int = NATIVE_HF_INPUT_FRAMES
    output_indices: tuple[int, ...] = (NATIVE_HF_CENTER,)
    context_frames: int = 5
    half_channels: int = 32
    quarter_channels: int = 48
    eighth_channels: int = 64
    fusion_half_channels: int = 48
    fusion_quarter_channels: int = 64
    fusion_eighth_channels: int = 96
    coarse_radius: int = 2
    eighth_blocks: int = 2
    quarter_blocks: int = 2
    half_blocks: int = 2

    def validate(self) -> None:
        if self.input_frames not in (5, 7, 9):
            raise ValueError("the Native-HF refiner supports 5, 7 or 9 frames")
        if self.context_frames not in (3, 5, 7, 9):
            raise ValueError("context_frames must be 3, 5, 7 or 9")
        if not self.output_indices:
            raise ValueError("at least one Native-HF output is required")
        if any(index < 0 or index >= self.input_frames for index in self.output_indices):
            raise ValueError("Native-HF output index is outside the input window")
        dimensions = (
            self.half_channels,
            self.quarter_channels,
            self.eighth_channels,
            self.fusion_half_channels,
            self.fusion_quarter_channels,
            self.fusion_eighth_channels,
            self.coarse_radius,
            self.eighth_blocks,
            self.quarter_blocks,
            self.half_blocks,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("Native-HF dimensions must be positive")


class NativeHFEncodedFrame(NamedTuple):
    packed: torch.Tensor
    half: torch.Tensor
    quarter: torch.Tensor
    eighth: torch.Tensor


class NativeHFAlignedFrame(NamedTuple):
    packed: torch.Tensor
    half: torch.Tensor
    quarter: torch.Tensor
    eighth: torch.Tensor
    reliability: torch.Tensor
    occlusion: torch.Tensor
    entropy: torch.Tensor


class NativeHFEncoder(nn.Module):
    """Shared frame encoder; no learned op runs at native resolution."""

    def __init__(self, config: NativeHF512Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.half_stage = nn.Sequential(
            nn.Conv2d(NATIVE_HF_PACKED_CHANNELS, config.half_channels, 3, padding=1),
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
        feature_gate = torch.ones(1, NATIVE_HF_PACKED_CHANNELS, 1, 1)
        feature_gate[
            :,
            NATIVE_HF_MASK_CHANNEL * 4 : (NATIVE_HF_RELIABILITY_CHANNEL + 1) * 4,
        ] = 0
        self.register_buffer("feature_gate", feature_gate, persistent=False)

    def forward(self, frame: torch.Tensor) -> NativeHFEncodedFrame:
        if frame.ndim != 4 or frame.shape[1] != NATIVE_HF_FRAME_CHANNELS:
            raise ValueError("Native-HF frame must be [B,8,H,W]")
        if frame.shape[-2] % 8 or frame.shape[-1] % 8:
            raise ValueError("Native-HF dimensions must be divisible by eight")
        packed = F.pixel_unshuffle(frame, NATIVE_HF_FOLD)
        # Mask geometry is intentionally excluded from correlation features.
        # Otherwise exact-motion pretraining can solve the synthetic task from
        # the translated mask edge and fail to transfer to real texture.
        half = self.half_stage(packed * self.feature_gate.to(packed.dtype))
        quarter = self.quarter_stage(half)
        eighth = self.eighth_stage(quarter)
        return NativeHFEncodedFrame(packed, half, quarter, eighth)


class NativeHFPyramidAlignment(nn.Module):
    """Coarse-to-fine fixed-shift alignment ending at +/-1 source pixel."""

    def __init__(self, config: NativeHF512Config) -> None:
        super().__init__()
        self.config = config
        self.offsets_8 = make_offsets(config.coarse_radius)
        self.offsets_4 = make_offsets(1)
        self.offsets_2 = make_offsets(1)
        self.corr_8 = NormalizedShiftCorrelation(
            self.offsets_8,
            channels=config.eighth_channels,
            initial_temperature=0.02,
            minimum_temperature=0.01,
            maximum_temperature=0.25,
        )
        self.corr_4 = NormalizedShiftCorrelation(
            self.offsets_4,
            channels=config.quarter_channels,
            initial_temperature=0.02,
            minimum_temperature=0.01,
            maximum_temperature=0.25,
        )
        self.detail_corr_8 = NormalizedShiftCorrelation(
            tuple((vertical * 2, horizontal * 2) for vertical, horizontal in self.offsets_8),
            channels=24,
            initial_temperature=0.02,
            center_bias=0.0,
            minimum_temperature=0.01,
            maximum_temperature=0.25,
        )
        self.detail_corr_4 = NormalizedShiftCorrelation(
            tuple((vertical * 2, horizontal * 2) for vertical, horizontal in self.offsets_4),
            channels=24,
            initial_temperature=0.02,
            center_bias=0.0,
            minimum_temperature=0.01,
            maximum_temperature=0.25,
        )
        self.detail_descriptor_channels = 24
        self.corr_2 = NormalizedShiftCorrelation(
            self.offsets_2,
            channels=self.detail_descriptor_channels,
            initial_temperature=0.02,
            center_bias=0.0,
            minimum_temperature=0.01,
            maximum_temperature=0.25,
        )
        self.prewarp_4_from_8 = FixedShiftBank(
            config.quarter_channels,
            tuple((vertical * 2, horizontal * 2) for vertical, horizontal in self.offsets_8),
        )
        self.prewarp_2_from_8 = FixedShiftBank(
            config.half_channels,
            tuple((vertical * 4, horizontal * 4) for vertical, horizontal in self.offsets_8),
        )
        self.prewarp_2_from_4 = FixedShiftBank(
            config.half_channels,
            tuple((vertical * 2, horizontal * 2) for vertical, horizontal in self.offsets_4),
        )
        self.prewarp_2_from_2 = FixedShiftBank(
            config.half_channels, self.offsets_2
        )
        self.prewarp_packed_from_8 = FixedShiftBank(
            NATIVE_HF_PACKED_CHANNELS,
            tuple((vertical * 4, horizontal * 4) for vertical, horizontal in self.offsets_8),
        )
        self.prewarp_packed_from_4 = FixedShiftBank(
            NATIVE_HF_PACKED_CHANNELS,
            tuple((vertical * 2, horizontal * 2) for vertical, horizontal in self.offsets_4),
        )
        self.prewarp_packed_from_2 = FixedShiftBank(
            NATIVE_HF_PACKED_CHANNELS, self.offsets_2
        )
        self.phase_bank = FoldedPhaseShiftBank(NATIVE_HF_FRAME_CHANNELS)
        self.phase_validity_bank = FoldedPhaseShiftBank(1)
        self.phase_bias = nn.Parameter(torch.zeros(1, 9, 1, 1))
        with torch.no_grad():
            self.phase_bias[:, 4].fill_(1.0)
        phase_fraction = (0.02 - 0.01) / (0.25 - 0.01)
        self.phase_temperature = nn.Parameter(
            torch.tensor(float(torch.logit(torch.tensor(phase_fraction))))
        )
        gate_input_channels = config.half_channels * 3 + 2
        gate_channels = max(16, config.half_channels // 2)
        self.pair_gate = nn.Sequential(
            nn.Conv2d(gate_input_channels, gate_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(gate_channels, 2, 1),
        )
        with torch.no_grad():
            self.pair_gate[-1].weight.zero_()
            self.pair_gate[-1].bias.copy_(torch.tensor((2.0, -2.0)))

    @staticmethod
    def _resize_weights(weights: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(weights, size=target.shape[-2:], mode="nearest")

    @staticmethod
    def _apply_shift(
        target: torch.Tensor, weights: torch.Tensor, bank: FixedShiftBank
    ) -> torch.Tensor:
        return apply_shift_weights(target, weights, bank.offsets, bank)

    @staticmethod
    def _hard_straight_through(weights: torch.Tensor) -> torch.Tensor:
        """Select one integer shift while retaining softmax gradients."""

        index = weights.argmax(dim=1, keepdim=True)
        hard = torch.zeros_like(weights).scatter_(1, index, 1.0)
        return hard + weights - weights.detach()

    @staticmethod
    def _standardize_candidates(values: torch.Tensor) -> torch.Tensor:
        centered = values - values.mean(dim=1, keepdim=True)
        return centered * torch.rsqrt(
            centered.float().square().mean(dim=1, keepdim=True) + 1e-6
        ).to(centered.dtype)

    def _fuse_correlations(
        self, learned: torch.Tensor, detail: torch.Tensor
    ) -> torch.Tensor:
        learned = self._resize_weights(learned, detail)
        learned_score = self._standardize_candidates(
            learned.clamp_min(1e-7).log()
        )
        detail_score = self._standardize_candidates(
            detail.clamp_min(1e-7).log()
        )
        # The learned coarse representation supplies the global structure;
        # native local contrast breaks aliases between neighbouring offsets.
        # A small local support window rejects isolated candidate flips without
        # imposing one rigid motion on the complete ROI.  This is especially
        # important for the 4px correction stage, where a one-candidate error
        # otherwise survives every finer stage.
        score = detail_score + 4.0 * learned_score
        score = F.avg_pool2d(
            score, 5, stride=1, padding=2, count_include_pad=False
        )
        return torch.softmax(score, dim=1)

    @staticmethod
    def _normalize(values: torch.Tensor) -> torch.Tensor:
        return values * torch.rsqrt(
            values.float().square().sum(dim=1, keepdim=True) + 1e-6
        ).to(values.dtype)

    @staticmethod
    def _detail_descriptor(packed: torch.Tensor) -> torch.Tensor:
        """Fixed local-contrast RGB/base descriptor for 1-2px alignment.

        Coarse learned features are robust to large displacement.  At the two
        finest stages, however, raw native phase is the signal we must retain;
        a random nonlinear descriptor made all nine candidates nearly equal.
        Source and global-base RGB are used, while mask/reliability are
        excluded so they cannot become a synthetic-motion shortcut.
        """

        source = packed[:, : NATIVE_HF_SOURCE_CHANNELS * 4]
        base = packed[
            :,
            NATIVE_HF_BASE_START * 4 :
            (NATIVE_HF_BASE_START + NATIVE_HF_SOURCE_CHANNELS) * 4,
        ]
        values = torch.cat((source, base), dim=1)
        return values - F.avg_pool2d(values, 3, stride=1, padding=1)

    def _phase_align(
        self,
        reference: torch.Tensor,
        target: torch.Tensor,
        teacher_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates = self.phase_bank(target)
        batch, count, channels, height, width = candidates.shape
        reference_descriptor = self._normalize(self._detail_descriptor(reference))
        candidate_descriptor = self._detail_descriptor(
            candidates.reshape(batch * count, channels, height, width)
        ).reshape(batch, count, -1, height, width)
        candidate_descriptor = candidate_descriptor * torch.rsqrt(
            candidate_descriptor.float().square().sum(dim=2, keepdim=True) + 1e-6
        ).to(candidate_descriptor.dtype)
        logits = (reference_descriptor[:, None] * candidate_descriptor).sum(dim=2)
        valid = self.phase_validity_bank(torch.ones_like(reference[:, :4])).amin(dim=2)
        temperature = 0.01 + 0.24 * torch.sigmoid(self.phase_temperature)
        weights = torch.softmax(
            (logits + (valid - 1.0) * 10_000.0 + self.phase_bias) / temperature,
            dim=1,
        )
        alignment_weights = weights
        if teacher_weights is not None:
            if teacher_weights.shape != weights.shape[:2]:
                raise ValueError("phase teacher weights do not match candidates")
            alignment_weights = teacher_weights[..., None, None].expand_as(weights)
        aligned = (alignment_weights[:, :, None] * candidates).sum(dim=1)
        entropy = -(
            weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True) / torch.log(weights.new_tensor(float(count)))
        return aligned, entropy, weights

    def _forward_impl(
        self,
        reference: NativeHFEncodedFrame,
        target: NativeHFEncodedFrame,
        teacher_weights: Sequence[torch.Tensor] | None = None,
    ) -> tuple[NativeHFAlignedFrame, tuple[torch.Tensor, ...]]:
        if teacher_weights is not None and len(teacher_weights) != 4:
            raise ValueError("alignment teacher requires four shift distributions")

        _soft_aligned_8, learned_weights_8 = self.corr_8(
            reference.eighth, target.eighth
        )
        reference_detail = self._detail_descriptor(reference.packed)
        target_detail = self._detail_descriptor(target.packed)
        _detail_aligned_8, detail_weights_8 = self.detail_corr_8(
            F.avg_pool2d(reference_detail, 2, stride=2),
            F.avg_pool2d(target_detail, 2, stride=2),
        )
        weights_8 = self._fuse_correlations(
            learned_weights_8, detail_weights_8
        )
        warp_8 = self._hard_straight_through(weights_8)
        if teacher_weights is not None:
            warp_8 = teacher_weights[0][..., None, None].expand_as(weights_8)
        aligned_8 = self._apply_shift(
            target.eighth,
            self._resize_weights(warp_8, target.eighth),
            self.corr_8.shift_bank,
        )

        quarter = self._apply_shift(
            target.quarter,
            self._resize_weights(warp_8, target.quarter),
            self.prewarp_4_from_8,
        )
        _soft_aligned_4, learned_weights_4 = self.corr_4(
            reference.quarter, quarter
        )
        packed = self._apply_shift(
            target.packed,
            self._resize_weights(warp_8, target.packed),
            self.prewarp_packed_from_8,
        )
        target_detail = self._detail_descriptor(packed)
        _detail_aligned_4, detail_weights_4 = self.detail_corr_4(
            reference_detail, target_detail
        )
        weights_4 = self._fuse_correlations(
            learned_weights_4, detail_weights_4
        )
        warp_4 = self._hard_straight_through(weights_4)
        if teacher_weights is not None:
            warp_4 = teacher_weights[1][..., None, None].expand_as(weights_4)
        aligned_4 = self._apply_shift(
            quarter,
            self._resize_weights(warp_4, quarter),
            self.corr_4.shift_bank,
        )

        packed = self._apply_shift(
            packed,
            self._resize_weights(warp_4, packed),
            self.prewarp_packed_from_4,
        )
        reference_detail = self._detail_descriptor(reference.packed)
        target_detail = self._detail_descriptor(packed)
        _aligned_detail, weights_2 = self.corr_2(
            reference_detail, target_detail
        )
        warp_2 = self._hard_straight_through(weights_2)
        if teacher_weights is not None:
            warp_2 = teacher_weights[2][..., None, None].expand_as(weights_2)

        half = self._apply_shift(
            target.half,
            self._resize_weights(warp_8, target.half),
            self.prewarp_2_from_8,
        )
        half = self._apply_shift(
            half,
            self._resize_weights(warp_4, half),
            self.prewarp_2_from_4,
        )
        aligned_2 = self._apply_shift(
            half,
            self._resize_weights(warp_2, half),
            self.prewarp_2_from_2,
        )
        packed = self._apply_shift(
            packed,
            self._resize_weights(warp_2, packed),
            self.prewarp_packed_from_2,
        )
        aligned_packed, entropy, weights_phase = self._phase_align(
            reference.packed,
            packed,
            None if teacher_weights is None else teacher_weights[3],
        )

        reliability = aligned_packed[
            :,
            NATIVE_HF_RELIABILITY_CHANNEL * 4 : (NATIVE_HF_RELIABILITY_CHANNEL + 1) * 4,
        ].mean(dim=1, keepdim=True)
        mask = aligned_packed[
            :,
            NATIVE_HF_MASK_CHANNEL * 4 : (NATIVE_HF_MASK_CHANNEL + 1) * 4,
        ].mean(dim=1, keepdim=True)
        pair = torch.cat(
            (
                reference.half,
                aligned_2,
                torch.abs(reference.half - aligned_2),
                reliability,
                mask,
            ),
            dim=1,
        )
        gates = torch.sigmoid(self.pair_gate(pair))
        return (
            NativeHFAlignedFrame(
                aligned_packed,
                aligned_2,
                aligned_4,
                aligned_8,
                gates[:, :1] * reliability,
                gates[:, 1:2],
                entropy,
            ),
            (weights_8, weights_4, weights_2, weights_phase),
        )

    def forward(
        self, reference: NativeHFEncodedFrame, target: NativeHFEncodedFrame
    ) -> NativeHFAlignedFrame:
        aligned, _weights = self._forward_impl(reference, target)
        return aligned

    def forward_with_diagnostics(
        self,
        reference: NativeHFEncodedFrame,
        target: NativeHFEncodedFrame,
        *,
        teacher_weights: Sequence[torch.Tensor] | None = None,
    ) -> tuple[NativeHFAlignedFrame, tuple[torch.Tensor, ...]]:
        """Training-only access to the four alignment distributions."""

        return self._forward_impl(
            reference, target, teacher_weights=teacher_weights
        )


class NativeHFFusionDecoder(nn.Module):
    def __init__(self, config: NativeHF512Config) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.alignment = NativeHFPyramidAlignment(config)
        count = config.context_frames
        self.reduce_8 = nn.Sequential(
            nn.Conv2d(config.eighth_channels * count, config.fusion_eighth_channels, 1),
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
                config.fusion_quarter_channels + config.quarter_channels * count,
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
        packed_channels = max(24, config.fusion_half_channels // 2)
        self.reduce_packed = nn.Sequential(
            nn.Conv2d(NATIVE_HF_PACKED_CHANNELS * count, packed_channels, 1),
            nn.SiLU(),
        )
        self.reduce_2 = nn.Sequential(
            nn.Conv2d(
                config.fusion_half_channels
                + config.half_channels * count
                + packed_channels,
                config.fusion_half_channels,
                1,
            ),
            nn.SiLU(),
        )
        self.body_2 = nn.Sequential(
            *(ResidualBlock(config.fusion_half_channels) for _ in range(config.half_blocks))
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(config.fusion_half_channels, config.fusion_half_channels, 3, padding=1),
            nn.SiLU(),
            ResidualBlock(config.fusion_half_channels),
            nn.Conv2d(config.fusion_half_channels, 12, 3, padding=1),
        )
        # A zero-initialized linear path keeps native packed samples visible to
        # the residual output from the first optimizer step.  Without this
        # path the zero-initialized deep head initially blocks gradients to the
        # encoder, and random fusion features converge to the safe zero-HF
        # solution before temporal alignment becomes useful.
        self.detail_skip = nn.Conv2d(
            NATIVE_HF_PACKED_CHANNELS * count, 12, 3, padding=1
        )
        # Stage-one's short native-detail path must not see the BasicVSR++
        # base RGB.  Otherwise this single convolution can learn the trivial
        # shortcut -HF(base), which improves the old metric by blurring guide
        # halos without recovering any native detail.  The deeper fusion path
        # remains base-aware and can learn cleanup during joint calibration.
        detail_skip_gate = torch.zeros(
            1, NATIVE_HF_PACKED_CHANNELS * count, 1, 1
        )
        for context_index in range(count):
            start = context_index * NATIVE_HF_PACKED_CHANNELS
            detail_skip_gate[
                :, start : start + NATIVE_HF_SOURCE_CHANNELS * 4
            ] = 1
        self.register_buffer(
            "detail_skip_gate", detail_skip_gate, persistent=False
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(
                config.fusion_half_channels,
                max(16, config.fusion_half_channels // 2),
                3,
                padding=1,
            ),
            nn.SiLU(),
            nn.Conv2d(max(16, config.fusion_half_channels // 2), 4, 3, padding=1),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.detail_skip.weight)
        nn.init.zeros_(self.detail_skip.bias)
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.zeros_(self.confidence_head[-1].bias)

    def _context_indices(self, reference: int) -> tuple[int, ...]:
        start = min(
            max(reference - self.config.context_frames // 2, 0),
            self.config.input_frames - self.config.context_frames,
        )
        return tuple(range(start, start + self.config.context_frames))

    @staticmethod
    def _gate(value: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        return value * F.interpolate(gate, size=value.shape[-2:], mode="nearest")

    def _aligned_context(
        self, encoded: Sequence[NativeHFEncodedFrame], reference_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        reference = encoded[reference_index]
        packed_values: list[torch.Tensor] = []
        half_values: list[torch.Tensor] = []
        quarter_values: list[torch.Tensor] = []
        eighth_values: list[torch.Tensor] = []
        for index in self._context_indices(reference_index):
            if index == reference_index:
                gate = torch.ones_like(reference.half[:, :1])
                aligned = NativeHFAlignedFrame(
                    reference.packed,
                    reference.half,
                    reference.quarter,
                    reference.eighth,
                    gate,
                    torch.zeros_like(gate),
                    torch.zeros_like(gate),
                )
            else:
                aligned = self.alignment(reference, encoded[index])
                gate = aligned.reliability * (1.0 - aligned.occlusion)
            packed_values.append(self._gate(aligned.packed, gate))
            half_values.append(self._gate(aligned.half, gate))
            quarter_values.append(self._gate(aligned.quarter, gate))
            eighth_values.append(self._gate(aligned.eighth, gate))
        return (
            torch.cat(packed_values, dim=1),
            torch.cat(half_values, dim=1),
            torch.cat(quarter_values, dim=1),
            torch.cat(eighth_values, dim=1),
        )

    def _decode(
        self,
        packed: torch.Tensor,
        half_context: torch.Tensor,
        quarter_context: torch.Tensor,
        eighth_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        eighth = self.body_8(self.reduce_8(eighth_context))
        quarter = self.up_8_to_4(eighth)
        quarter = self.body_4(
            self.reduce_4(torch.cat((quarter, quarter_context), dim=1))
        )
        half = self.up_4_to_2(quarter)
        packed_features = self.reduce_packed(packed)
        half = self.body_2(
            self.reduce_2(torch.cat((half, half_context, packed_features), dim=1))
        )
        raw_residual = F.pixel_shuffle(
            self.residual_head(half)
            + self.detail_skip(packed * self.detail_skip_gate.to(packed.dtype)),
            2,
        )
        confidence = torch.sigmoid(F.pixel_shuffle(self.confidence_head(half), 2))
        return band_limit_residual(raw_residual), confidence

    def forward(
        self, encoded: Sequence[NativeHFEncodedFrame]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contexts = [
            self._aligned_context(encoded, index)
            for index in self.config.output_indices
        ]
        residual, confidence = self._decode(
            *(torch.cat([context[level] for context in contexts], dim=0) for level in range(4))
        )
        batch = encoded[0].packed.shape[0]
        outputs = len(self.config.output_indices)
        shape = (outputs, batch, residual.shape[1], *residual.shape[-2:])
        residual = residual.reshape(shape).transpose(0, 1)
        confidence_shape = (outputs, batch, 1, *confidence.shape[-2:])
        confidence = confidence.reshape(confidence_shape).transpose(0, 1)
        return residual, confidence


class MiohNativeHF512(nn.Module):
    """High-frequency residual model with exact source-pixel ROI preservation."""

    def __init__(self, config: NativeHF512Config | None = None) -> None:
        super().__init__()
        self.config = config or NativeHF512Config()
        self.config.validate()
        self.encoder = NativeHFEncoder(self.config)
        self.decoder = NativeHFFusionDecoder(self.config)

    def encode_window(self, values: torch.Tensor) -> tuple[NativeHFEncodedFrame, ...]:
        if values.ndim != 5 or values.shape[1:3] != (
            self.config.input_frames,
            NATIVE_HF_FRAME_CHANNELS,
        ):
            raise ValueError(
                f"Native-HF input must be [B,{self.config.input_frames},8,H,W]"
            )
        batch, frames, channels, height, width = values.shape
        flat = values.reshape(batch * frames, channels, height, width)
        encoded = self.encoder(flat)
        return tuple(
            NativeHFEncodedFrame(
                *(level.reshape(batch, frames, level.shape[1], *level.shape[-2:])[:, index]
                  for level in encoded)
            )
            for index in range(frames)
        )

    def alignment_diagnostics(
        self,
        values: torch.Tensor,
        *,
        reference_index: int = NATIVE_HF_CENTER,
        target_index: int = 0,
        teacher_weights: Sequence[torch.Tensor] | None = None,
    ) -> tuple[NativeHFAlignedFrame, tuple[torch.Tensor, ...]]:
        if not 0 <= reference_index < self.config.input_frames:
            raise ValueError("Native-HF alignment reference is outside the window")
        if not 0 <= target_index < self.config.input_frames:
            raise ValueError("Native-HF alignment target is outside the window")
        encoded = self.encode_window(values)
        return self.decoder.alignment.forward_with_diagnostics(
            encoded[reference_index],
            encoded[target_index],
            teacher_weights=teacher_weights,
        )

    def forward_components(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encode_window(values)
        residual, raw_confidence = self.decoder(encoded)
        indices = list(self.config.output_indices)
        source = values[:, indices, :3]
        mask = values[:, indices, NATIVE_HF_MASK_CHANNEL : NATIVE_HF_MASK_CHANNEL + 1].clamp(0, 1)
        reliability = values[
            :, indices, NATIVE_HF_RELIABILITY_CHANNEL : NATIVE_HF_RELIABILITY_CHANNEL + 1
        ].clamp(0, 1)
        base = values[:, indices, NATIVE_HF_BASE_START : NATIVE_HF_BASE_START + 3]
        effective_confidence = raw_confidence * reliability
        restored = source + mask * (
            base - source + effective_confidence * residual
        )
        # Training supervises the model's own confidence. Reliability is an
        # external observation-quality gate and must not make that target
        # unreachable when reliability is below one.
        return restored, raw_confidence, residual, base

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        restored, raw_confidence, _residual, _base = self.forward_components(values)
        indices = list(self.config.output_indices)
        reliability = values[
            :,
            indices,
            NATIVE_HF_RELIABILITY_CHANNEL : NATIVE_HF_RELIABILITY_CHANNEL + 1,
        ].clamp(0, 1)
        return restored, raw_confidence * reliability


class MiohNativeHF512ExportWrapper(nn.Module):
    """Flat Core ML/Core AI contract: [B,T*8,512,512] -> RGB/confidence."""

    def __init__(self, model: MiohNativeHF512, *, clamp: bool = True) -> None:
        super().__init__()
        self.model = model
        self.clamp = clamp

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expected_channels = self.model.config.input_frames * NATIVE_HF_FRAME_CHANNELS
        if frames.ndim != 4 or frames.shape[1] != expected_channels:
            raise ValueError(
                f"flat Native-HF input must have {expected_channels} channels"
            )
        values = frames.reshape(
            frames.shape[0],
            self.model.config.input_frames,
            NATIVE_HF_FRAME_CHANNELS,
            frames.shape[-2],
            frames.shape[-1],
        )
        restored, confidence = self.model(values)
        if self.clamp:
            restored = restored.clamp(0, 1)
        return restored.flatten(1, 2), confidence.flatten(1, 2)


class FrozenBasicVSRPP256Guide(nn.Module):
    """Training-only adapter that builds the three global-base channels."""

    def __init__(self, backbone: nn.Module, *, global_size: int = 256) -> None:
        super().__init__()
        if global_size < 256:
            raise ValueError("BasicVSR++ global guide must be at least 256px")
        self.backbone = backbone.eval()
        self.backbone.requires_grad_(False)
        self.global_size = int(global_size)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, *, use_ema: bool = True
    ) -> "FrozenBasicVSRPP256Guide":
        backbone = BasicVSRPlusPlusNet(
            mid_channels=64,
            num_blocks=15,
            max_residue_magnitude=10,
            spynet_pretrained=None,
        )
        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError("BasicVSR++ checkpoint must contain a state dictionary")
        state = payload.get("state_dict", payload)
        if not isinstance(state, dict):
            raise TypeError("BasicVSR++ state_dict is invalid")
        preferred = "generator_ema." if use_ema else "generator."
        selected = {
            key.removeprefix(preferred): value
            for key, value in state.items()
            if key.startswith(preferred)
        }
        if not selected and use_ema:
            selected = {
                key.removeprefix("generator."): value
                for key, value in state.items()
                if key.startswith("generator.")
            }
        if not selected:
            raise ValueError("checkpoint contains no BasicVSR++ generator weights")
        backbone.load_state_dict(selected, strict=True)
        return cls(backbone)

    def train(self, mode: bool = True) -> "FrozenBasicVSRPP256Guide":
        # The guide remains deterministic even while the refiner is training.
        super().train(False)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, native_values: torch.Tensor) -> torch.Tensor:
        if native_values.ndim != 5 or native_values.shape[2] < 5:
            raise ValueError("native guide input must be [B,9,>=5,H,W]")
        batch, frames, _channels, height, width = native_values.shape
        rgb = native_values[:, :, :3]
        low = F.interpolate(
            rgb.reshape(batch * frames, 3, height, width),
            size=(self.global_size, self.global_size),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, 3, self.global_size, self.global_size)
        base_low = self.backbone(low)
        base = F.interpolate(
            base_low.reshape(batch * frames, 3, self.global_size, self.global_size),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, frames, 3, height, width)
        return torch.cat((native_values[:, :, :5], base), dim=2)


def initialize_frozen_analytic_alignment(model: MiohNativeHF512) -> None:
    """Apply the measured, unbiased alignment calibration in-place."""

    alignment = model.decoder.alignment
    with torch.no_grad():
        for correlation in (
            alignment.corr_8,
            alignment.corr_4,
            alignment.detail_corr_8,
            alignment.detail_corr_4,
            alignment.corr_2,
        ):
            correlation.offset_bias.zero_()
            desired = 0.02
            fraction = (
                (desired - correlation.minimum_temperature)
                / (
                    correlation.maximum_temperature
                    - correlation.minimum_temperature
                )
            )
            correlation.raw_temperature.copy_(
                torch.logit(correlation.raw_temperature.new_tensor(fraction))
            )
        alignment.phase_bias.zero_()
        phase_fraction = (0.02 - 0.01) / (0.25 - 0.01)
        alignment.phase_temperature.copy_(
            torch.logit(alignment.phase_temperature.new_tensor(phase_fraction))
        )


def build_mioh_native_hf512(
    config: NativeHF512Config | None = None,
) -> MiohNativeHF512:
    """Build the reproducible frozen-alignment model without consuming RNG."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(NATIVE_HF_MODEL_INITIALIZATION_SEED)
        model = MiohNativeHF512(config or NativeHF512Config())
    initialize_frozen_analytic_alignment(model)
    return model


def native_hf_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
