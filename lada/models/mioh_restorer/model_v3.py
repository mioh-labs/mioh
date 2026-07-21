# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Core AI compatible second-order video restoration network.

The model mirrors BasicVSR++'s four propagation branches and second-order
temporal state without relying on optical-flow warping, grid sampling or
deformable convolution.  Motion alignment is expressed as a soft mixture of a
fixed bank of padded feature shifts, which keeps the inference graph limited
to convolution, slicing, concatenation, softmax and pointwise arithmetic.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from .model import ResidualBlock


def _make_offsets(radius: int, dilation: int) -> tuple[tuple[int, int], ...]:
    if radius < 0:
        raise ValueError("alignment radius must not be negative")
    if dilation <= 0:
        raise ValueError("alignment dilation must be positive")
    return tuple(
        (vertical * dilation, horizontal * dilation)
        for vertical in range(-radius, radius + 1)
        for horizontal in range(-radius, radius + 1)
    )


class CoreAIShiftAlignment(nn.Module):
    """Align grouped features using a learned mixture of static shifts.

    BasicVSR++ predicts separate deformable offsets for multiple channel
    groups.  Dynamic sampling is not portable to Core AI, so this module uses
    the closest static-graph equivalent: every group predicts a distribution
    over a fixed bank of padded shifts.  All shifts are ordinary slices and
    the dynamic part is limited to convolution, softmax and multiplication.
    """

    def __init__(
        self,
        channels: int,
        *,
        radius: int = 1,
        dilation: int = 2,
        key_channels: int = 16,
        groups: int = 1,
        temperature: float = 1.0,
        apply_confidence: bool = True,
        predict_confidence: bool = True,
    ) -> None:
        super().__init__()
        if channels <= 0 or key_channels <= 0 or groups <= 0:
            raise ValueError("alignment channels must be positive")
        if channels % groups:
            raise ValueError("alignment channels must be divisible by groups")
        if key_channels % groups:
            raise ValueError("alignment key channels must be divisible by groups")
        if temperature <= 0:
            raise ValueError("alignment temperature must be positive")
        self.channels = channels
        self.radius = radius
        self.dilation = dilation
        self.key_channels = key_channels
        self.groups = groups
        self.temperature = temperature
        self.apply_confidence = apply_confidence
        self.predict_confidence = predict_confidence
        self.channels_per_group = channels // groups
        self.key_channels_per_group = key_channels // groups
        self.offsets = _make_offsets(radius, dilation)
        self.padding = radius * dilation
        self.query = nn.Conv2d(channels, key_channels, 1)
        self.key = nn.Conv2d(channels, key_channels, 1)
        self.score_predictor = nn.Sequential(
            nn.Conv2d(channels * 2, key_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(key_channels, groups * len(self.offsets), 1),
        )
        self.confidence_predictor: nn.Sequential | None = None
        if predict_confidence:
            self.confidence_predictor = nn.Sequential(
                nn.Conv2d(channels * 2, key_channels, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(key_channels, groups, 1),
            )
            nn.init.zeros_(self.confidence_predictor[-1].weight)
            nn.init.constant_(self.confidence_predictor[-1].bias, -1.38629436)
        self.offset_bias = nn.Parameter(
            torch.zeros(1, groups, len(self.offsets), 1, 1)
        )
        center = self.offsets.index((0, 0))
        with torch.no_grad():
            self.offset_bias[:, :, center].fill_(2.0)

    @staticmethod
    def _shift(
        values: torch.Tensor,
        vertical: int,
        horizontal: int,
        padding: int,
    ) -> torch.Tensor:
        height, width = values.shape[-2:]
        if padding == 0:
            return values
        padded = F.pad(values, (padding, padding, padding, padding))
        top = padding + vertical
        left = padding + horizontal
        return padded[..., top : top + height, left : left + width]

    def alignment_weights(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        if reference.shape != candidate.shape:
            raise ValueError("alignment tensors must have identical shapes")
        batch, _, height, width = reference.shape
        query = self.query(reference).reshape(
            batch,
            self.groups,
            self.key_channels_per_group,
            height,
            width,
        )
        key = self.key(candidate)
        predicted_scores = self.score_predictor(
            torch.cat((reference, candidate), dim=1)
        ).reshape(
            batch,
            self.groups,
            len(self.offsets),
            height,
            width,
        )
        scale = self.key_channels_per_group**-0.5
        scores: list[torch.Tensor] = []
        for vertical, horizontal in self.offsets:
            shifted_key = self._shift(
                key, vertical, horizontal, self.padding
            ).reshape(
                batch,
                self.groups,
                self.key_channels_per_group,
                height,
                width,
            )
            scores.append(
                (query * shifted_key).sum(dim=2, keepdim=True) * scale
            )
        return torch.softmax(
            (
                torch.cat(scores, dim=2)
                + predicted_scores
                + self.offset_bias
            )
            / self.temperature,
            dim=2,
        )

    def forward_with_weights(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if reference.shape != candidate.shape:
            raise ValueError("alignment tensors must have identical shapes")
        batch, _, height, width = reference.shape
        weights = self.alignment_weights(reference, candidate)
        if self.confidence_predictor is None:
            confidence = candidate.new_ones(
                batch * self.groups, 1, height, width
            )
        else:
            confidence = torch.sigmoid(
                self.confidence_predictor(
                    torch.cat((reference, candidate), dim=1)
                )
            ).reshape(batch * self.groups, 1, height, width)
        shifted_groups: list[torch.Tensor] = []
        for index, (vertical, horizontal) in enumerate(self.offsets):
            shifted = self._shift(
                candidate, vertical, horizontal, self.padding
            ).reshape(
                batch * self.groups,
                self.channels_per_group,
                height,
                width,
            )
            shifted_groups.append(shifted)
        grouped_weights = weights.reshape(
            batch * self.groups, 1, len(self.offsets), height, width
        )
        if self.training:
            # Avoid materializing the complete candidate bank during
            # backpropagation. Inference/export uses the faster stacked graph.
            aligned = candidate.new_zeros(
                batch * self.groups,
                self.channels_per_group,
                height,
                width,
            )
            for index, shifted in enumerate(shifted_groups):
                aligned = (
                    aligned
                    + shifted * grouped_weights[:, :, index]
                )
            if self.apply_confidence:
                aligned = aligned * confidence
            return (
                aligned.reshape(batch, self.channels, height, width),
                weights,
                confidence.reshape(batch, self.groups, 1, height, width),
            )
        # Flatten batch and groups before stacking so the deployable graph
        # never exceeds Core ML's rank-5 tensor limit.
        candidates = torch.stack(shifted_groups, dim=2)
        aligned = (candidates * grouped_weights).sum(dim=2)
        if self.apply_confidence:
            aligned = aligned * confidence
        return (
            aligned.reshape(batch, self.channels, height, width),
            weights,
            confidence.reshape(batch, self.groups, 1, height, width),
        )

    def forward(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        aligned, _weights, _confidence = self.forward_with_weights(
            reference, candidate
        )
        return aligned


class HierarchicalCoreAIShiftAlignment(nn.Module):
    """Coarse-to-fine static alignment with balanced shift stages.

    Dilations ``(9, 3, 1)`` and radius one can express every integer offset
    from -13 through +13 on each feature axis.  MiohRestorerV3 operates at
    quarter resolution, so this covers +/-52 input pixels while evaluating
    only three nine-way shift banks instead of one 27x27 bank.
    """

    def __init__(
        self,
        channels: int,
        *,
        dilations: Sequence[int],
        key_channels: int,
        groups: int,
        temperature: float,
    ) -> None:
        super().__init__()
        if not dilations:
            raise ValueError("hierarchical alignment needs at least one stage")
        if any(dilation <= 0 for dilation in dilations):
            raise ValueError("hierarchical alignment dilations must be positive")
        self.channels = channels
        self.dilations = tuple(int(item) for item in dilations)
        self.groups = groups
        self.stages = nn.ModuleList(
            CoreAIShiftAlignment(
                channels,
                radius=1,
                dilation=dilation,
                key_channels=key_channels,
                groups=groups,
                temperature=temperature,
                apply_confidence=False,
                predict_confidence=index == len(self.dilations) - 1,
            )
            for index, dilation in enumerate(self.dilations)
        )

    @property
    def stage_offsets(self) -> tuple[tuple[tuple[int, int], ...], ...]:
        return tuple(stage.offsets for stage in self.stages)

    @property
    def maximum_offset(self) -> int:
        return sum(self.dilations)

    def forward_with_weights(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
        aligned = candidate
        weights: list[torch.Tensor] = []
        confidence: torch.Tensor | None = None
        for stage in self.stages:
            aligned, stage_weights, confidence = stage.forward_with_weights(
                reference, aligned
            )
            weights.append(stage_weights)
        if confidence is None:
            raise RuntimeError("hierarchical alignment did not run a stage")
        batch, _, height, width = reference.shape
        grouped_confidence = confidence.reshape(
            batch * self.groups, 1, height, width
        )
        aligned = (
            aligned.reshape(
                batch * self.groups,
                self.channels // self.groups,
                height,
                width,
            )
            * grouped_confidence
        ).reshape(batch, self.channels, height, width)
        return aligned, tuple(weights), confidence

    def forward(
        self,
        reference: torch.Tensor,
        candidate: torch.Tensor,
    ) -> torch.Tensor:
        aligned, _weights, _confidence = self.forward_with_weights(
            reference, candidate
        )
        return aligned


class SecondOrderShiftPropagation(nn.Module):
    """One BasicVSR++-style propagation branch with static-shift alignment."""

    def __init__(
        self,
        channels: int,
        *,
        prior_branches: int,
        num_blocks: int,
        alignment_radius: int,
        first_order_dilation: int,
        second_order_dilation: int,
        alignment_key_channels: int,
        alignment_groups: int,
        hierarchical_alignment_dilations: Sequence[int] = (),
        alignment_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if prior_branches < 0 or num_blocks < 0:
            raise ValueError("branch counts must not be negative")
        self.prior_branches = prior_branches
        self.gradient_checkpointing = False
        self.input_projection = nn.Sequential(
            nn.Conv2d(
                channels * (1 + prior_branches),
                channels,
                3,
                padding=1,
            ),
            nn.ReLU(),
        )
        if hierarchical_alignment_dilations:
            self.first_order_alignment = HierarchicalCoreAIShiftAlignment(
                channels,
                dilations=hierarchical_alignment_dilations,
                key_channels=alignment_key_channels,
                groups=alignment_groups,
                temperature=alignment_temperature,
            )
            self.second_order_alignment = HierarchicalCoreAIShiftAlignment(
                channels,
                dilations=hierarchical_alignment_dilations,
                key_channels=alignment_key_channels,
                groups=alignment_groups,
                temperature=alignment_temperature,
            )
        else:
            self.first_order_alignment = CoreAIShiftAlignment(
                channels,
                radius=alignment_radius,
                dilation=first_order_dilation,
                key_channels=alignment_key_channels,
                groups=alignment_groups,
                temperature=alignment_temperature,
            )
            self.second_order_alignment = CoreAIShiftAlignment(
                channels,
                radius=alignment_radius,
                dilation=second_order_dilation,
                key_channels=alignment_key_channels,
                groups=alignment_groups,
                temperature=alignment_temperature,
            )
        self.alignment_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.ReLU(),
        )
        self.backbone = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1),
            nn.ReLU(),
            *(ResidualBlock(channels) for _ in range(num_blocks)),
        )

    def forward(
        self,
        spatial: Sequence[torch.Tensor],
        prior: Sequence[Sequence[torch.Tensor]],
        *,
        reverse: bool,
        capture_calls: frozenset[int] | None = None,
        diagnostics: list[dict[str, object]] | None = None,
    ) -> list[torch.Tensor]:
        if len(prior) != self.prior_branches:
            raise ValueError("unexpected number of prior propagation branches")
        indices = list(range(len(spatial)))
        if reverse:
            indices.reverse()
        state_n1 = torch.zeros_like(spatial[0])
        state_n2 = torch.zeros_like(spatial[0])
        produced: list[torch.Tensor | None] = [None] * len(spatial)
        for sequence_index, frame_index in enumerate(indices):
            call_index = sequence_index - 1
            capture = (
                diagnostics is not None
                and capture_calls is not None
                and call_index in capture_calls
            )
            current = self.input_projection(
                torch.cat(
                    [spatial[frame_index]]
                    + [branch[frame_index] for branch in prior],
                    dim=1,
                )
            )
            if sequence_index:
                if capture:
                    aligned_n1, first_weights, first_confidence = (
                        self.first_order_alignment.forward_with_weights(
                            current, state_n1
                        )
                    )
                else:
                    aligned_n1 = (
                        activation_checkpoint(
                            self.first_order_alignment,
                            current,
                            state_n1,
                            use_reentrant=False,
                        )
                        if self.training and self.gradient_checkpointing
                        else self.first_order_alignment(current, state_n1)
                    )
                    first_weights = None
                    first_confidence = None
            else:
                aligned_n1 = torch.zeros_like(current)
                first_weights = None
                first_confidence = None
            if sequence_index > 1:
                if capture:
                    aligned_n2, second_weights, second_confidence = (
                        self.second_order_alignment.forward_with_weights(
                            current, state_n2
                        )
                    )
                else:
                    aligned_n2 = (
                        activation_checkpoint(
                            self.second_order_alignment,
                            current,
                            state_n2,
                            use_reentrant=False,
                        )
                        if self.training and self.gradient_checkpointing
                        else self.second_order_alignment(current, state_n2)
                    )
                    second_weights = None
                    second_confidence = None
            else:
                aligned_n2 = torch.zeros_like(current)
                second_weights = None
                second_confidence = None
            if sequence_index:
                fusion_input = torch.cat((aligned_n1, aligned_n2), dim=1)
                aligned = (
                    activation_checkpoint(
                        self.alignment_fusion,
                        fusion_input,
                        use_reentrant=False,
                    )
                    if self.training and self.gradient_checkpointing
                    else self.alignment_fusion(fusion_input)
                )
            else:
                aligned = torch.zeros_like(current)
            if capture:
                item: dict[str, object] = {
                    "call_index": call_index,
                    "frame_index": frame_index,
                    "aligned": aligned,
                }
                if first_weights is not None:
                    item["first_weights"] = first_weights
                if first_confidence is not None:
                    item["first_confidence"] = first_confidence
                if second_weights is not None:
                    item["second_weights"] = second_weights
                if second_confidence is not None:
                    item["second_confidence"] = second_confidence
                diagnostics.append(item)
            backbone_input = torch.cat((current, aligned_n1, aligned_n2), dim=1)
            residual = (
                activation_checkpoint(
                    self.backbone,
                    backbone_input,
                    use_reentrant=False,
                )
                if self.training and self.gradient_checkpointing
                else self.backbone(backbone_input)
            )
            propagated = aligned + residual
            produced[frame_index] = propagated
            state_n2 = state_n1
            state_n1 = propagated
        if any(item is None for item in produced):
            raise RuntimeError("propagation did not produce every frame")
        return [item for item in produced if item is not None]


class DirectionDecoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.up1 = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.up2 = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.output_head = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = F.relu(F.pixel_shuffle(self.up1(features), 2))
        features = F.relu(F.pixel_shuffle(self.up2(features), 2))
        return torch.tanh(self.output_head(features))


class MiohRestorerV3(nn.Module):
    """Fixed-window BasicVSR++ student built from Core AI friendly ops."""

    BRANCHES = (
        ("backward_1", True),
        ("forward_1", False),
        ("backward_2", True),
        ("forward_2", False),
    )
    DEFAULT_WINDOW_FRAMES = 24
    DEFAULT_CHANNELS = 64
    DEFAULT_BLOCKS = 7
    ARCHITECTURE_REVISION = 2
    HIERARCHICAL_ARCHITECTURE_REVISION = 3
    SUPPORTED_ARCHITECTURE_REVISIONS = frozenset((2, 3))

    def __init__(
        self,
        *,
        window_frames: int = DEFAULT_WINDOW_FRAMES,
        channels: int = DEFAULT_CHANNELS,
        num_blocks: int = DEFAULT_BLOCKS,
        encoder_blocks: int = 5,
        reconstruction_blocks: int = 5,
        alignment_radius: int = 1,
        first_order_dilation: int = 2,
        second_order_dilation: int = 4,
        alignment_key_channels: int = 16,
        alignment_groups: int = 1,
        hierarchical_alignment_dilations: Sequence[int] = (),
        alignment_temperature: float = 1.0,
        detail_scale: float = 0.25,
    ) -> None:
        super().__init__()
        if window_frames <= 1:
            raise ValueError("window_frames must be greater than one")
        if channels <= 0 or min(num_blocks, encoder_blocks, reconstruction_blocks) < 0:
            raise ValueError("model sizes must not be negative")
        if detail_scale <= 0:
            raise ValueError("detail_scale must be positive")
        if alignment_temperature <= 0:
            raise ValueError("alignment_temperature must be positive")
        self.window_frames = window_frames
        self.channels = channels
        self.num_blocks = num_blocks
        self.encoder_blocks = encoder_blocks
        self.reconstruction_blocks = reconstruction_blocks
        self.alignment_radius = alignment_radius
        self.first_order_dilation = first_order_dilation
        self.second_order_dilation = second_order_dilation
        self.alignment_key_channels = alignment_key_channels
        self.alignment_groups = alignment_groups
        self.hierarchical_alignment_dilations = tuple(
            int(item) for item in hierarchical_alignment_dilations
        )
        if any(item <= 0 for item in self.hierarchical_alignment_dilations):
            raise ValueError("hierarchical alignment dilations must be positive")
        self.alignment_temperature = alignment_temperature
        self.architecture_revision = (
            self.HIERARCHICAL_ARCHITECTURE_REVISION
            if self.hierarchical_alignment_dilations
            else self.ARCHITECTURE_REVISION
        )
        self.detail_scale = detail_scale
        self.gradient_checkpointing = False

        self.encoder = nn.Sequential(
            nn.Conv2d(4, channels, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.encoder_refinement = nn.Sequential(
            *(ResidualBlock(channels) for _ in range(encoder_blocks))
        )
        self.propagation = nn.ModuleDict()
        for branch_index, (name, _reverse) in enumerate(self.BRANCHES):
            self.propagation[name] = SecondOrderShiftPropagation(
                channels,
                prior_branches=branch_index,
                num_blocks=num_blocks,
                alignment_radius=alignment_radius,
                first_order_dilation=first_order_dilation,
                second_order_dilation=second_order_dilation,
                alignment_key_channels=alignment_key_channels,
                alignment_groups=alignment_groups,
                hierarchical_alignment_dilations=(
                    self.hierarchical_alignment_dilations
                ),
                alignment_temperature=alignment_temperature,
            )
        self.direction_decoder = DirectionDecoder(channels)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(channels * 5, channels, 3, padding=1),
            nn.ReLU(),
            *(ResidualBlock(channels) for _ in range(reconstruction_blocks)),
        )
        self.up1 = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.up2 = nn.Conv2d(channels, channels * 4, 3, padding=1)
        self.output_head = nn.Conv2d(channels, 3, 3, padding=1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)

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
            raise ValueError("frame dimensions must be divisible by four")

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        """Trade additional training compute for substantially lower memory."""

        self.gradient_checkpointing = enabled
        for propagation in self.propagation.values():
            propagation.gradient_checkpointing = enabled

    def _encode(
        self, frames: torch.Tensor, masks: torch.Tensor
    ) -> list[torch.Tensor]:
        encoded: list[torch.Tensor] = []
        for frame_index in range(self.window_frames):
            features = self.encoder(
                torch.cat(
                    (frames[:, frame_index], masks[:, frame_index]), dim=1
                )
            )
            encoded.append(
                activation_checkpoint(
                    self.encoder_refinement,
                    features,
                    use_reentrant=False,
                )
                if self.training and self.gradient_checkpointing
                else self.encoder_refinement(features)
            )
        return encoded

    def propagation_features(
        self, frames: torch.Tensor, masks: torch.Tensor
    ) -> dict[str, list[torch.Tensor]]:
        features, _diagnostics = self._propagation_features(frames, masks)
        return features

    def _propagation_features(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
        *,
        capture_branch: str | None = None,
        capture_calls: frozenset[int] | None = None,
    ) -> tuple[
        dict[str, list[torch.Tensor]],
        list[dict[str, object]],
    ]:
        self._validate_inputs(frames, masks)
        features: dict[str, list[torch.Tensor]] = {
            "spatial": self._encode(frames, masks)
        }
        diagnostics: list[dict[str, object]] = []
        prior: list[list[torch.Tensor]] = []
        for name, reverse in self.BRANCHES:
            branch = self.propagation[name](
                features["spatial"],
                prior,
                reverse=reverse,
                capture_calls=capture_calls if name == capture_branch else None,
                diagnostics=diagnostics if name == capture_branch else None,
            )
            features[name] = branch
            prior.append(branch)
        return features, diagnostics

    def _decode_direction(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
        features: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        outputs = []
        for frame_index, item in enumerate(features):
            residual = self.direction_decoder(item) * self.detail_scale
            outputs.append(
                torch.clamp(
                    frames[:, frame_index]
                    + residual * masks[:, frame_index],
                    0.0,
                    1.0,
                )
            )
        return torch.stack(outputs, dim=1)

    def forward_with_directions(
        self, frames: torch.Tensor, masks: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.propagation_features(frames, masks)
        restored = self._restore_from_features(frames, masks, features)
        forward = self._decode_direction(
            frames, masks, features["forward_2"]
        )
        backward = self._decode_direction(
            frames, masks, features["backward_2"]
        )
        return restored, forward, backward

    def forward_with_distillation(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
        *,
        capture_branch: str,
        capture_calls: frozenset[int],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, object]],
    ]:
        """Run training forward while retaining a bounded alignment sample."""

        if capture_branch not in dict(self.BRANCHES):
            raise ValueError(f"unknown propagation branch: {capture_branch}")
        features, diagnostics = self._propagation_features(
            frames,
            masks,
            capture_branch=capture_branch,
            capture_calls=capture_calls,
        )
        restored = self._restore_from_features(frames, masks, features)
        forward = self._decode_direction(
            frames, masks, features["forward_2"]
        )
        backward = self._decode_direction(
            frames, masks, features["backward_2"]
        )
        return restored, forward, backward, diagnostics

    def _restore_from_features(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
        features: dict[str, list[torch.Tensor]],
    ) -> torch.Tensor:
        outputs: list[torch.Tensor] = []
        for frame_index in range(self.window_frames):
            combined = torch.cat(
                [features["spatial"][frame_index]]
                + [features[name][frame_index] for name, _ in self.BRANCHES],
                dim=1,
            )
            item = (
                activation_checkpoint(
                    self.reconstruction,
                    combined,
                    use_reentrant=False,
                )
                if self.training and self.gradient_checkpointing
                else self.reconstruction(combined)
            )
            item = F.relu(F.pixel_shuffle(self.up1(item), 2))
            item = F.relu(F.pixel_shuffle(self.up2(item), 2))
            residual = torch.tanh(self.output_head(item)) * self.detail_scale
            outputs.append(
                torch.clamp(
                    frames[:, frame_index]
                    + residual * masks[:, frame_index],
                    0.0,
                    1.0,
                )
            )
        return torch.stack(outputs, dim=1)

    def forward(self, frames: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        features = self.propagation_features(frames, masks)
        return self._restore_from_features(frames, masks, features)

    def initialize_from_v2(self, model: nn.Module) -> None:
        """Reuse only V2 features whose meaning is unchanged in V3.

        V2 directional heads decode a different propagation representation.
        Copying them makes an otherwise identity-initialized V3 emit strong,
        unrelated residuals before its new propagation branches have learned.
        """
        branch = getattr(model, "forward_branch", None)
        if branch is None or getattr(branch, "channels", None) != self.channels:
            raise ValueError("V2 channels do not match V3 channels")
        self.encoder.load_state_dict(branch.encoder.state_dict(), strict=True)
        source_blocks = list(branch.refinement.children())
        target_blocks = list(self.encoder_refinement.children())
        for target, source in zip(target_blocks, source_blocks, strict=False):
            target.load_state_dict(source.state_dict(), strict=True)

    def initialize_from_v3_state_dict(
        self, source_state: dict[str, torch.Tensor]
    ) -> tuple[int, int]:
        """Reuse V3 features while deliberately relearning alignment.

        The hierarchical stages do not have a one-to-one equivalent in the
        original V3.  Encoder, propagation backbones, fusion, reconstruction
        and output heads are copied when their names and shapes match; every
        alignment parameter starts fresh.
        """

        target_state = self.state_dict()
        transferable = {
            name: value
            for name, value in source_state.items()
            if name in target_state
            and target_state[name].shape == value.shape
            and ".first_order_alignment." not in name
            and ".second_order_alignment." not in name
        }
        self.load_state_dict(transferable, strict=False)
        return len(transferable), len(target_state) - len(transferable)
