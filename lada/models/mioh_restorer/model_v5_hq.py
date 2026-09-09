# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Quality-first MiohRestorer V5-HQ.

V5-HQ deliberately separates the quality ceiling from the ANE-first V5-S
deployment model.  Its backbone keeps BasicVSR++'s learned optical flow,
sub-pixel grid sampling, second-order deformable convolution and four recurrent
propagation passes.  A full-resolution, flow-residual deformable-attention
refiner then restores only the requested mosaic ROI.

The graph is intended for Core AI with the custom Metal grid-sample and DCNv2
kernels in ``scripts/apple/basicvsrpp_coreai_kernels.py``.  It is not required
to stay on the Neural Engine and it never silently substitutes fixed shifts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from lada.models.basicvsrpp.mmagic.basicvsr_plusplus_net import (
    BasicVSRPlusPlusNet,
)
from lada.models.basicvsrpp.mmagic.flow_warp import flow_warp

from .model_v5 import QUALITY_OUTPUT_INDICES


def feather_texture_compositor_mask(
    effective_mask: torch.Tensor,
    compositor_mask: torch.Tensor,
    *,
    radius: int,
) -> torch.Tensor:
    """Keep texture inside the real ROI with a short, bounded soft edge."""

    if effective_mask.shape != compositor_mask.shape or effective_mask.ndim != 5:
        raise ValueError("texture masks must have matching [B,T,1,H,W] shapes")
    if radius < 0:
        raise ValueError("texture mask feather radius cannot be negative")
    effective = effective_mask.clamp(0, 1)
    if radius:
        batch, frames, channels, height, width = effective.shape
        flat = effective.reshape(batch * frames, channels, height, width)
        kernel = 2 * radius + 1
        feathered = F.avg_pool2d(flat, kernel, stride=1, padding=radius)
        effective = torch.maximum(effective, feathered.reshape_as(effective))
    return torch.minimum(effective, compositor_mask.clamp(0, 1))


@dataclass(frozen=True)
class MiohRestorerV5HQConfig:
    """Fixed export contract for one V5-HQ specialization."""

    input_frames: int = 9
    output_indices: tuple[int, ...] = QUALITY_OUTPUT_INDICES
    backbone_channels: int = 64
    backbone_blocks: int = 15
    detail_channels: int = 64
    attention_channels: int = 24
    attention_radius: int = 1
    maximum_residual_offset: float = 4.0
    raw_temporal_candidates: bool = False
    raw_temporal_encoder_channels: int = 0
    raw_temporal_nearest_warp: bool = False
    raw_temporal_zero_input: bool = False

    def validate(self) -> None:
        if self.input_frames < 3:
            raise ValueError("V5-HQ needs at least three input frames")
        if not self.output_indices:
            raise ValueError("V5-HQ needs at least one output frame")
        if any(index < 0 or index >= self.input_frames for index in self.output_indices):
            raise ValueError("V5-HQ output index is outside the input window")
        if self.backbone_channels <= 0 or self.backbone_blocks <= 0:
            raise ValueError("V5-HQ backbone dimensions must be positive")
        if self.detail_channels <= 0 or self.attention_channels <= 0:
            raise ValueError("V5-HQ detail dimensions must be positive")
        if self.attention_radius <= 0:
            raise ValueError("V5-HQ attention radius must be positive")
        if self.maximum_residual_offset <= 0:
            raise ValueError("V5-HQ residual offset limit must be positive")
        if self.raw_temporal_encoder_channels < 0:
            raise ValueError("raw temporal encoder channels cannot be negative")
        if self.raw_temporal_encoder_channels and not self.raw_temporal_candidates:
            raise ValueError(
                "raw temporal encoder requires raw temporal candidates"
            )
        if self.raw_temporal_nearest_warp and not self.raw_temporal_candidates:
            raise ValueError("nearest raw warp requires raw temporal candidates")
        if self.raw_temporal_zero_input and not self.raw_temporal_candidates:
            raise ValueError("zeroed raw input requires raw temporal candidates")
        if self.raw_temporal_candidates and any(
            index - self.attention_radius < 0
            or index + self.attention_radius >= self.input_frames
            for index in self.output_indices
        ):
            raise ValueError(
                "raw temporal candidates require a complete window around every output"
            )


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.body(values)


class FlowResidualDeformableAttention(nn.Module):
    """Full-resolution attention with learned offsets around optical flow.

    The base sampling location comes from SPyNet.  A learned bounded residual
    offset corrects the flow, and ``flow_warp`` performs true bilinear dynamic
    sampling.  Candidate weights are content-dependent and include an explicit
    visibility gate.  This is deformable attention rather than a fixed shift
    approximation.
    """

    def __init__(self, config: MiohRestorerV5HQConfig) -> None:
        super().__init__()
        channels = config.detail_channels
        attention = config.attention_channels
        self.maximum_residual_offset = config.maximum_residual_offset
        self.query = nn.Conv2d(channels, attention, 1)
        self.key = nn.Conv2d(channels, attention, 1)
        self.offset = nn.Sequential(
            nn.Conv2d(channels * 3 + 2, channels, 3, padding=1),
            nn.SiLU(),
            _ResidualBlock(channels),
            nn.Conv2d(channels, 2, 3, padding=1),
        )
        self.visibility = nn.Sequential(
            nn.Conv2d(channels * 3 + 2, channels // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels // 2, 1, 3, padding=1),
        )
        nn.init.zeros_(self.offset[-1].weight)
        nn.init.zeros_(self.offset[-1].bias)

    @staticmethod
    def _normalize(values: torch.Tensor) -> torch.Tensor:
        return F.normalize(values.float(), dim=1).to(values.dtype)

    def forward(
        self,
        reference: torch.Tensor,
        candidates: torch.Tensor,
        base_flows: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Align and fuse K candidate features.

        Args:
            reference: ``[B,C,H,W]``.
            candidates: ``[B,K,C,H,W]``; candidate zero may be the reference.
            base_flows: ``[B,K,2,H,W]`` mapping reference pixels to candidates.
        """

        if candidates.ndim != 5 or base_flows.ndim != 5:
            raise ValueError("V5-HQ attention expects batched candidate windows")
        batch, count, channels, height, width = candidates.shape
        if reference.shape != (batch, channels, height, width):
            raise ValueError("V5-HQ reference and candidates do not match")
        if base_flows.shape != (batch, count, 2, height, width):
            raise ValueError("V5-HQ flow shape does not match candidates")

        repeated_reference = reference[:, None].expand(-1, count, -1, -1, -1)
        pair_values = torch.cat(
            (
                repeated_reference,
                candidates,
                torch.abs(repeated_reference - candidates),
                base_flows,
            ),
            dim=2,
        ).reshape(batch * count, channels * 3 + 2, height, width)
        residual = self.maximum_residual_offset * torch.tanh(self.offset(pair_values))
        visibility = torch.sigmoid(self.visibility(pair_values))
        total_flow = base_flows.reshape(batch * count, 2, height, width) + residual
        aligned = flow_warp(
            candidates.reshape(batch * count, channels, height, width),
            total_flow.permute(0, 2, 3, 1),
            padding_mode="border",
        ).reshape(batch, count, channels, height, width)

        query = self._normalize(self.query(reference))
        key = self._normalize(
            self.key(aligned.reshape(batch * count, channels, height, width))
        ).reshape(batch, count, -1, height, width)
        logits = (query[:, None] * key).sum(dim=2)
        logits = logits + torch.log(
            visibility.reshape(batch, count, height, width).clamp_min(1e-6)
        )
        weights = torch.softmax(logits, dim=1)
        fused = (weights[:, :, None] * aligned).sum(dim=1)
        return fused, weights, total_flow.reshape(batch, count, 2, height, width)


class MiohRestorerV5HQ(nn.Module):
    """BasicVSR++ propagation plus V5 full-resolution ROI refinement."""

    def __init__(
        self,
        config: MiohRestorerV5HQConfig | None = None,
        *,
        backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config or MiohRestorerV5HQConfig()
        self.config.validate()
        self.backbone = backbone or BasicVSRPlusPlusNet(
            mid_channels=self.config.backbone_channels,
            num_blocks=self.config.backbone_blocks,
            max_residue_magnitude=10,
            spynet_pretrained=None,
        )
        channels = self.config.detail_channels
        self.detail_encoder = nn.Sequential(
            nn.Conv2d(5, channels, 3, padding=1),
            nn.SiLU(),
            _ResidualBlock(channels),
            _ResidualBlock(channels),
        )
        self.deformable_attention = FlowResidualDeformableAttention(self.config)
        if self.config.raw_temporal_encoder_channels:
            raw_channels = self.config.raw_temporal_encoder_channels
            self.raw_temporal_encoder: nn.Module | None = nn.Sequential(
                nn.Conv2d(3, raw_channels, 3, padding=1),
                nn.SiLU(),
                _ResidualBlock(raw_channels),
                _ResidualBlock(raw_channels),
            )
        else:
            raw_channels = 3
            self.raw_temporal_encoder = None
        raw_temporal_channels = (
            (2 * self.config.attention_radius + 1) * raw_channels * 2
            if self.config.raw_temporal_candidates
            else 0
        )
        head_channels = channels * 3 + 8 + raw_temporal_channels
        self.fusion = nn.Sequential(
            nn.Conv2d(head_channels, channels * 2, 3, padding=1),
            nn.SiLU(),
            _ResidualBlock(channels * 2),
            _ResidualBlock(channels * 2),
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.SiLU(),
        )
        self.base_head = nn.Conv2d(channels, 3, 3, padding=1)
        self.texture_head = nn.Sequential(
            _ResidualBlock(channels),
            _ResidualBlock(channels),
            nn.Conv2d(channels, 3, 3, padding=1),
        )
        self.confidence_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels // 2, 1, 3, padding=1),
        )
        # The added V5 branch starts as an exact BasicVSR++ ROI wrapper.
        nn.init.zeros_(self.base_head.weight)
        nn.init.zeros_(self.base_head.bias)
        nn.init.zeros_(self.texture_head[-1].weight)
        nn.init.zeros_(self.texture_head[-1].bias)
        nn.init.zeros_(self.confidence_head[-1].weight)
        nn.init.zeros_(self.confidence_head[-1].bias)

    @property
    def spynet(self) -> nn.Module:
        value = getattr(self.backbone, "spynet", None)
        if value is None:
            raise TypeError("V5-HQ backbone must expose SPyNet")
        return value

    def load_basicvsrpp_checkpoint(self, checkpoint: str | Path) -> None:
        """Initialize only the recurrent backbone from a Lada v1.2 checkpoint."""

        payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise TypeError("BasicVSR++ checkpoint must contain a state dictionary")
        raw = payload.get("state_dict", payload)
        if not isinstance(raw, dict):
            raise TypeError("BasicVSR++ checkpoint state_dict is invalid")
        # Dedicated checkpoints retain both the live generator and its EMA.
        # Prefer the EMA because that is the state exported by mioh. Generic
        # v1.2 checkpoints only have the historical ``generator.`` prefix.
        prefix = (
            "generator_ema."
            if any(str(key).startswith("generator_ema.") for key in raw)
            else "generator."
        )
        state = {
            str(key).removeprefix(prefix): value
            for key, value in raw.items()
            if str(key).startswith(prefix)
        }
        if not state:
            raise ValueError("checkpoint contains no generator weights")
        self.backbone.load_state_dict(state, strict=True)

    def _candidate_indices(self, reference: int) -> tuple[int, ...]:
        start = max(0, reference - self.config.attention_radius)
        end = min(self.config.input_frames, reference + self.config.attention_radius + 1)
        indices = list(range(start, end))
        # Put the identity candidate first so a cold visibility head has a
        # stable no-motion choice without hard-coding its final weight.
        indices.remove(reference)
        return (reference, *indices)

    def _direct_flows(
        self,
        restored: torch.Tensor,
        reference_indices: Sequence[int],
        candidate_sets: Sequence[Sequence[int]],
    ) -> list[torch.Tensor]:
        pair_references: list[torch.Tensor] = []
        pair_candidates: list[torch.Tensor] = []
        pair_counts: list[int] = []
        for reference, candidates in zip(reference_indices, candidate_sets, strict=True):
            nonidentity = [index for index in candidates if index != reference]
            pair_counts.append(len(nonidentity))
            for index in nonidentity:
                pair_references.append(restored[:, reference])
                pair_candidates.append(restored[:, index])
        if pair_references:
            reference_batch = torch.cat(pair_references, dim=0)
            candidate_batch = torch.cat(pair_candidates, dim=0)
            estimated = self.spynet(reference_batch, candidate_batch)
        else:
            estimated = restored.new_empty(0, 2, *restored.shape[-2:])

        result: list[torch.Tensor] = []
        cursor = 0
        batch = restored.shape[0]
        for count in pair_counts:
            zero = restored.new_zeros(batch, 1, 2, *restored.shape[-2:])
            if count:
                values = estimated[cursor * batch : (cursor + count) * batch]
                # Pairs were appended candidate-major, so restore B,K order.
                values = values.reshape(count, batch, 2, *restored.shape[-2:]).permute(1, 0, 2, 3, 4)
                result.append(torch.cat((zero, values), dim=1))
            else:
                result.append(zero)
            cursor += count
        return result

    @staticmethod
    def _aligned_raw_candidates(
        raw_features: torch.Tensor,
        candidates: Sequence[int],
        flows: torch.Tensor,
        *,
        interpolation: str = "bilinear",
    ) -> torch.Tensor:
        """Warp un-restored candidate pixels/features with the attention flow."""

        raw = raw_features[:, list(candidates)]
        batch, count, channels, height, width = raw.shape
        if flows.shape != (batch, count, 2, height, width):
            raise ValueError("raw temporal candidate flow shape is invalid")
        return flow_warp(
            raw.reshape(batch * count, channels, height, width),
            flows.reshape(batch * count, 2, height, width).permute(0, 2, 3, 1),
            interpolation=interpolation,
            padding_mode="border",
        ).reshape(batch, count, channels, height, width)

    def forward_components(
        self,
        values: torch.Tensor,
        *,
        texture_compositor_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 5 or values.shape[1] != self.config.input_frames:
            raise ValueError(
                f"V5-HQ expects [B,{self.config.input_frames},5,H,W]"
            )
        if values.shape[2] != 5:
            raise ValueError("V5-HQ input channels must be RGB, mask, reliability")
        rgb = values[:, :, :3]
        restored_all = self.backbone(rgb)
        batch, frames, _channels, height, width = restored_all.shape
        detail_inputs = torch.cat((restored_all, values[:, :, 3:5]), dim=2)
        detail = self.detail_encoder(
            detail_inputs.reshape(batch * frames, 5, height, width)
        ).reshape(batch, frames, self.config.detail_channels, height, width)
        raw_features = rgb
        if self.raw_temporal_encoder is not None:
            raw_channels = self.config.raw_temporal_encoder_channels
            raw_features = self.raw_temporal_encoder(
                rgb.reshape(batch * frames, 3, height, width)
            ).reshape(batch, frames, raw_channels, height, width)

        references = self.config.output_indices
        if texture_compositor_mask is not None:
            expected_mask_shape = (
                batch,
                len(references),
                1,
                height,
                width,
            )
            if tuple(texture_compositor_mask.shape) != expected_mask_shape:
                raise ValueError(
                    "texture compositor mask must be "
                    f"{expected_mask_shape}, got {tuple(texture_compositor_mask.shape)}"
                )
            texture_compositor_mask = texture_compositor_mask.clamp(0, 1)
        candidate_sets = [self._candidate_indices(index) for index in references]
        flows = self._direct_flows(restored_all, references, candidate_sets)
        outputs = []
        confidences = []
        bases = []
        textures = []
        for output_offset, (
            reference,
            candidates,
            candidate_flows,
        ) in enumerate(zip(references, candidate_sets, flows, strict=True)):
            reference_feature = detail[:, reference]
            candidate_features = detail[:, list(candidates)]
            fused, _weights, total_flow = self.deformable_attention(
                reference_feature, candidate_features, candidate_flows
            )
            source = rgb[:, reference]
            basic = restored_all[:, reference]
            mask = values[:, reference, 3:4].clamp(0, 1)
            reliability = values[:, reference, 4:5].clamp(0, 1)
            fusion_inputs = [
                reference_feature,
                fused,
                torch.abs(reference_feature - fused),
                source,
                basic,
                mask,
                reliability,
            ]
            if self.config.raw_temporal_candidates:
                aligned_raw = self._aligned_raw_candidates(
                    raw_features,
                    candidates,
                    total_flow,
                    interpolation=(
                        "nearest"
                        if self.config.raw_temporal_nearest_warp
                        else "bilinear"
                    ),
                )
                raw_reference = raw_features[:, reference]
                raw_difference = torch.abs(
                    aligned_raw - raw_reference[:, None]
                )
                if self.config.raw_temporal_zero_input:
                    aligned_raw = torch.zeros_like(aligned_raw)
                    raw_difference = torch.zeros_like(raw_difference)
                fusion_inputs.extend(
                    (
                        aligned_raw.flatten(1, 2),
                        raw_difference.flatten(1, 2),
                    )
                )
            features = self.fusion(torch.cat(fusion_inputs, dim=1))
            base = basic - source + self.base_head(features)
            texture = self.texture_head(features)
            confidence = torch.sigmoid(self.confidence_head(features)) * reliability
            texture_mask = (
                mask
                if texture_compositor_mask is None
                else texture_compositor_mask[:, output_offset]
            )
            output = (
                source
                + mask * base
                + texture_mask * confidence * texture
            )
            outputs.append(output)
            confidences.append(confidence)
            bases.append(base)
            textures.append(texture)
        return (
            torch.stack(outputs, dim=1),
            torch.stack(confidences, dim=1),
            torch.stack(bases, dim=1),
            torch.stack(textures, dim=1),
        )

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        restored, confidence, _base, _texture = self.forward_components(values)
        return restored, confidence


class MiohRestorerV5HQExportWrapper(nn.Module):
    """Flat FP16 Core AI interface."""

    def __init__(self, model: MiohRestorerV5HQ, *, clamp: bool = True) -> None:
        super().__init__()
        self.model = model
        self.clamp = clamp

    def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        restored, confidence = self.model(frames)
        if self.clamp:
            restored = restored.clamp(0, 1)
        return restored.flatten(1, 2), confidence.flatten(1, 2)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
