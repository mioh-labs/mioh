# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Non-invasive activation analysis for the PyTorch BasicVSR++ teacher.

The analyzer uses forward hooks only.  It does not replace the deformable
convolution implementation or alter any tensor passed through the network.
This makes it suitable for studying the exact teacher used for distillation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


class StreamingTensorStats:
    """Accumulate scalar moments and a bounded sample for percentiles."""

    def __init__(self, *, sample_capacity: int = 32768):
        self.count = 0
        self.total = 0.0
        self.total_squared = 0.0
        self.total_absolute = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.sample_capacity = sample_capacity
        self._samples: list[np.ndarray] = []
        self._sample_count = 0

    def update(self, value: torch.Tensor) -> None:
        value = value.detach().float()
        if value.numel() == 0:
            return
        self.count += value.numel()
        # MPS does not implement float64 reductions.  Accumulate Python
        # float64 totals from float32 device reductions instead.
        self.total += value.sum().item()
        self.total_squared += value.square().sum().item()
        self.total_absolute += value.abs().sum().item()
        self.minimum = min(self.minimum, value.amin().item())
        self.maximum = max(self.maximum, value.amax().item())

        remaining = self.sample_capacity - self._sample_count
        if remaining <= 0:
            return
        flat = value.reshape(-1)
        take = min(remaining, 1024, flat.numel())
        if take < flat.numel():
            indices = torch.linspace(
                0, flat.numel() - 1, take, device=flat.device
            ).long()
            flat = flat.index_select(0, indices)
        sample = flat.cpu().numpy().astype(np.float32, copy=False)
        self._samples.append(sample)
        self._sample_count += sample.size

    def as_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {"count": 0}
        mean = self.total / self.count
        variance = max(0.0, self.total_squared / self.count - mean * mean)
        samples = (
            np.concatenate(self._samples)
            if self._samples
            else np.empty((0,), dtype=np.float32)
        )
        result: dict[str, float | int | None] = {
            "count": self.count,
            "mean": mean,
            "std": variance**0.5,
            "abs_mean": self.total_absolute / self.count,
            "min": self.minimum,
            "max": self.maximum,
        }
        for percentile in (1, 5, 25, 50, 75, 95, 99):
            result[f"p{percentile}"] = (
                float(np.percentile(samples, percentile)) if samples.size else None
            )
        return result


@dataclass
class _AlignmentState:
    raw_outputs: list[torch.Tensor] = field(default_factory=list)
    calls: int = 0
    total_calls: int = 0


@dataclass(frozen=True)
class AlignmentCapturePolicy:
    """Select a bounded subset of alignment calls for tensor consumers.

    ``call_stride`` and ``max_calls_per_branch`` apply independently to every
    branch in every clip (``begin_clip`` resets the call index). Statistics are
    still accumulated for every call; this policy bounds only tensors retained
    in memory or sent to ``activation_callback``.
    """

    branches: frozenset[str] | None = None
    call_stride: int = 1
    max_calls_per_branch: int = 2
    channels: int = 8
    spatial_size: int = 16

    def __post_init__(self) -> None:
        if self.call_stride < 1:
            raise ValueError("call_stride must be at least 1")
        if self.max_calls_per_branch < 0:
            raise ValueError("max_calls_per_branch cannot be negative")
        if self.channels < 1 or self.spatial_size < 1:
            raise ValueError("channels and spatial_size must be positive")

    def selects(self, branch: str, call_index: int) -> bool:
        return (
            (self.branches is None or branch in self.branches)
            and call_index % self.call_stride == 0
            and call_index // self.call_stride < self.max_calls_per_branch
        )


@dataclass(frozen=True)
class AlignmentActivation:
    """Live tensors from one selected teacher alignment call.

    Tensors are detached but remain on their inference device.  A training
    callback can consume them immediately without a CPU round-trip.  The
    analyzer itself stores only cropped CPU samples controlled by the policy.
    """

    branch: str
    call_index: int
    offset_residual: torch.Tensor
    offset: torch.Tensor
    mask: torch.Tensor
    aligned_output: torch.Tensor | None


class BasicVSRPPActivationAnalyzer:
    """Capture SPyNet and deformable-alignment activations with hooks.

    ``model`` is expected to expose the normal BasicVSR++ ``spynet`` module
    and ``deform_align`` ModuleDict.  Keeping this duck-typed also makes the
    hook mechanics straightforward to test without loading a checkpoint.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        sample_capacity: int = 32768,
        activation_samples_per_branch: int = 2,
        sample_channels: int = 8,
        sample_spatial_size: int = 16,
        capture_policy: AlignmentCapturePolicy | None = None,
        activation_callback: Callable[[AlignmentActivation], None] | None = None,
        collect_statistics: bool = True,
    ):
        self.model = model
        self.sample_capacity = sample_capacity
        self.capture_policy = capture_policy or AlignmentCapturePolicy(
            max_calls_per_branch=activation_samples_per_branch,
            channels=sample_channels,
            spatial_size=sample_spatial_size,
        )
        self.activation_callback = activation_callback
        self.collect_statistics = collect_statistics
        self.stats: dict[str, StreamingTensorStats] = {}
        self.group_stats: dict[str, dict[int, dict[str, StreamingTensorStats]]] = (
            defaultdict(dict)
        )
        self.shift_counts: dict[str, dict[int, Counter[tuple[int, int]]]] = (
            defaultdict(lambda: defaultdict(Counter))
        )
        self.shift_mask_weights: dict[
            str, dict[int, Counter[tuple[int, int]]]
        ] = defaultdict(lambda: defaultdict(Counter))
        self.sampling_position_counts: dict[
            str, dict[int, Counter[tuple[int, int]]]
        ] = defaultdict(lambda: defaultdict(Counter))
        self.sampling_position_mask_weights: dict[
            str, dict[int, Counter[tuple[int, int]]]
        ] = defaultdict(lambda: defaultdict(Counter))
        self.sampling_position_contribution_weights: dict[
            str, dict[int, Counter[tuple[int, int]]]
        ] = defaultdict(lambda: defaultdict(Counter))
        self.samples: dict[str, np.ndarray] = {}
        self._handles: list[Any] = []
        self._alignment_states: dict[str, _AlignmentState] = {}
        self._spynet_calls = 0
        self.clip_count = 0
        self.frame_count = 0
        self._register_hooks()

    def _stat(self, name: str) -> StreamingTensorStats:
        if name not in self.stats:
            self.stats[name] = StreamingTensorStats(
                sample_capacity=self.sample_capacity
            )
        return self.stats[name]

    def _register_hooks(self) -> None:
        spynet = getattr(self.model, "spynet", None)
        deform_align = getattr(self.model, "deform_align", None)
        if spynet is None or deform_align is None:
            raise TypeError("model must expose spynet and deform_align modules")
        if self.collect_statistics:
            self._handles.append(spynet.register_forward_hook(self._capture_spynet))
        for branch, alignment in deform_align.items():
            state = _AlignmentState()
            self._alignment_states[branch] = state
            offset_head = alignment.conv_offset[-1]
            self._handles.append(
                offset_head.register_forward_hook(self._raw_offset_hook(branch))
            )
            self._handles.append(
                alignment.register_forward_hook(self._alignment_hook(branch))
            )

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __enter__(self) -> "BasicVSRPPActivationAnalyzer":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def begin_clip(self, frames: int) -> None:
        self.clip_count += 1
        self.frame_count += frames
        self._spynet_calls = 0
        for state in self._alignment_states.values():
            state.calls = 0
            state.raw_outputs.clear()

    def abort_clip(self, frames: int) -> None:
        """Exclude a clip whose forward pass failed from report totals."""
        self.clip_count = max(0, self.clip_count - 1)
        self.frame_count = max(0, self.frame_count - frames)

    def _capture_spynet(
        self, _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any
    ) -> None:
        if not isinstance(output, torch.Tensor):
            return
        # compute_flow invokes SPyNet for backward and then forward flow.
        direction = "backward" if self._spynet_calls % 2 == 0 else "forward"
        self._spynet_calls += 1
        flow = output.detach().float()
        self._stat(f"flow.{direction}.x").update(flow[:, 0])
        self._stat(f"flow.{direction}.y").update(flow[:, 1])
        self._stat(f"flow.{direction}.magnitude").update(
            torch.linalg.vector_norm(flow, dim=1)
        )
        self._save_sample(f"flow_{direction}", flow)

    def _raw_offset_hook(self, branch: str):
        def hook(
            _module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any
        ) -> None:
            if isinstance(output, torch.Tensor):
                self._alignment_states[branch].raw_outputs.append(output.detach())

        return hook

    def _alignment_hook(self, branch: str):
        def hook(
            module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
        ) -> None:
            state = self._alignment_states[branch]
            if not state.raw_outputs or len(inputs) < 4:
                return
            raw = state.raw_outputs.pop(0)
            selected = self.capture_policy.selects(branch, state.calls)
            if not self.collect_statistics and not selected:
                state.calls += 1
                state.total_calls += 1
                return
            flow_1 = inputs[2].detach()
            flow_2 = inputs[3].detach()
            o1, o2, mask_logits = torch.chunk(raw, 3, dim=1)
            residual = float(module.max_residue_magnitude) * torch.tanh(
                torch.cat((o1, o2), dim=1)
            )
            residual_1, residual_2 = torch.chunk(residual, 2, dim=1)
            offset_1 = residual_1 + flow_1.flip(1).repeat(
                1, residual_1.size(1) // 2, 1, 1
            )
            offset_2 = residual_2 + flow_2.flip(1).repeat(
                1, residual_2.size(1) // 2, 1, 1
            )
            offset = torch.cat((offset_1, offset_2), dim=1)
            mask = torch.sigmoid(mask_logits)

            if self.collect_statistics:
                prefix = f"alignment.{branch}"
                self._stat(f"{prefix}.offset_logits_o1").update(o1)
                self._stat(f"{prefix}.offset_logits_o2").update(o2)
                self._stat(f"{prefix}.mask_logits").update(mask_logits)
                self._stat(f"{prefix}.offset_residual").update(residual)
                self._stat(f"{prefix}.offset_final").update(offset)
                self._stat(f"{prefix}.mask").update(mask)
                if isinstance(output, torch.Tensor):
                    self._stat(f"{prefix}.aligned_output").update(output)

                self._capture_group_statistics(
                    branch=branch,
                    residual=residual,
                    offset=offset,
                    mask=mask,
                    deform_groups=int(module.deform_groups),
                    kernel_size=getattr(module, "kernel_size", 3),
                    dilation=getattr(module, "dilation", 1),
                    padding=getattr(module, "padding", 1),
                    convolution_weight=getattr(module, "weight", None),
                )
            sample_key = f"alignment_{branch}_{state.calls:04d}"
            if selected:
                activation = AlignmentActivation(
                    branch=branch,
                    call_index=state.calls,
                    offset_residual=residual.detach(),
                    offset=offset.detach(),
                    mask=mask.detach(),
                    aligned_output=(
                        output.detach() if isinstance(output, torch.Tensor) else None
                    ),
                )
                if self.activation_callback is not None:
                    self.activation_callback(activation)
                if self.collect_statistics:
                    self._save_sample(f"{sample_key}_offset", offset, force=True)
                    self._save_sample(f"{sample_key}_mask", mask, force=True)
                    if isinstance(output, torch.Tensor):
                        self._save_sample(
                            f"{sample_key}_aligned", output.detach(), force=True
                        )
            state.calls += 1
            state.total_calls += 1

        return hook

    def _capture_group_statistics(
        self,
        *,
        branch: str,
        residual: torch.Tensor,
        offset: torch.Tensor,
        mask: torch.Tensor,
        deform_groups: int,
        kernel_size: int | tuple[int, int],
        dilation: int | tuple[int, int],
        padding: int | tuple[int, int],
        convolution_weight: torch.Tensor | None,
    ) -> None:
        batch, offset_channels, height, width = offset.shape
        kernel_points = offset_channels // (2 * deform_groups)
        if kernel_points <= 0 or offset_channels != deform_groups * kernel_points * 2:
            return
        vectors = offset.reshape(
            batch, deform_groups, kernel_points, 2, height, width
        )
        residual_vectors = residual.reshape_as(vectors)
        masks = mask.reshape(batch, deform_groups, kernel_points, height, width)
        kernel_height, kernel_width = (
            (kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else kernel_size
        )
        dilation_y, dilation_x = (
            (dilation, dilation) if isinstance(dilation, int) else dilation
        )
        padding_y, padding_x = (
            (padding, padding) if isinstance(padding, int) else padding
        )
        if kernel_height * kernel_width != kernel_points:
            return
        base_positions = torch.tensor(
            [
                (
                    kernel_y * dilation_y - padding_y,
                    kernel_x * dilation_x - padding_x,
                )
                for kernel_y in range(kernel_height)
                for kernel_x in range(kernel_width)
            ],
            device=offset.device,
            dtype=offset.dtype,
        )
        sampling_positions = vectors + base_positions.view(
            1, 1, kernel_points, 2, 1, 1
        )
        kernel_contribution = torch.ones(
            deform_groups, kernel_points, device=offset.device, dtype=offset.dtype
        )
        if isinstance(convolution_weight, torch.Tensor):
            weight = convolution_weight.detach().float()
            channels_per_group = weight.shape[1] // deform_groups
            if channels_per_group > 0:
                contributions = []
                for group in range(deform_groups):
                    channel_start = group * channels_per_group
                    channel_end = (
                        weight.shape[1]
                        if group == deform_groups - 1
                        else channel_start + channels_per_group
                    )
                    contributions.append(
                        weight[:, channel_start:channel_end]
                        .abs()
                        .mean(dim=(0, 1))
                        .reshape(-1)
                    )
                kernel_contribution = torch.stack(contributions)
        for group in range(deform_groups):
            if group not in self.group_stats[branch]:
                self.group_stats[branch][group] = {
                    "residual_magnitude": StreamingTensorStats(
                        sample_capacity=min(self.sample_capacity, 4096)
                    ),
                    "final_magnitude": StreamingTensorStats(
                        sample_capacity=min(self.sample_capacity, 4096)
                    ),
                    "mask": StreamingTensorStats(
                        sample_capacity=min(self.sample_capacity, 4096)
                    ),
                    "offset_x": StreamingTensorStats(
                        sample_capacity=min(self.sample_capacity, 4096)
                    ),
                    "offset_y": StreamingTensorStats(
                        sample_capacity=min(self.sample_capacity, 4096)
                    ),
                    "sampling_position_x": StreamingTensorStats(
                        sample_capacity=min(self.sample_capacity, 4096)
                    ),
                    "sampling_position_y": StreamingTensorStats(
                        sample_capacity=min(self.sample_capacity, 4096)
                    ),
                }
            group_stats = self.group_stats[branch][group]
            group_stats["residual_magnitude"].update(
                torch.linalg.vector_norm(residual_vectors[:, group], dim=2)
            )
            group_stats["final_magnitude"].update(
                torch.linalg.vector_norm(vectors[:, group], dim=2)
            )
            group_stats["mask"].update(masks[:, group])
            group_stats["offset_y"].update(vectors[:, group, :, 0])
            group_stats["offset_x"].update(vectors[:, group, :, 1])
            group_stats["sampling_position_y"].update(
                sampling_positions[:, group, :, 0]
            )
            group_stats["sampling_position_x"].update(
                sampling_positions[:, group, :, 1]
            )

            # The rounded residual vectors reveal which fixed shifts can best
            # approximate the learned deformable sampler.  Subsample before
            # transferring to CPU so long clips remain practical.
            vec = residual_vectors[:, group].permute(0, 1, 3, 4, 2).reshape(-1, 2)
            weights = masks[:, group].reshape(-1)
            take = min(4096, vec.shape[0])
            if take < vec.shape[0]:
                indices = torch.linspace(
                    0, vec.shape[0] - 1, take, device=vec.device
                ).long()
                vec = vec.index_select(0, indices)
                weights = weights.index_select(0, indices)
            rounded = torch.round(vec).to(torch.int16).cpu().numpy()
            mask_weights = weights.cpu().numpy()
            counts = self.shift_counts[branch][group]
            weighted = self.shift_mask_weights[branch][group]
            for (y, x), weight in zip(rounded.tolist(), mask_weights.tolist()):
                key = (int(x), int(y))
                counts[key] += 1
                weighted[key] += float(weight)

            position_vec = (
                sampling_positions[:, group]
                .permute(0, 1, 3, 4, 2)
                .reshape(-1, 2)
            )
            position_masks = masks[:, group].reshape(-1)
            contribution = (
                kernel_contribution[group]
                .view(1, kernel_points, 1, 1)
                .expand(batch, kernel_points, height, width)
                .reshape(-1)
            )
            take = min(4096, position_vec.shape[0])
            if take < position_vec.shape[0]:
                indices = torch.linspace(
                    0, position_vec.shape[0] - 1, take, device=position_vec.device
                ).long()
                position_vec = position_vec.index_select(0, indices)
                position_masks = position_masks.index_select(0, indices)
                contribution = contribution.index_select(0, indices)
            rounded_positions = (
                torch.round(position_vec).to(torch.int16).cpu().numpy()
            )
            position_masks_np = position_masks.cpu().numpy()
            contribution_np = contribution.cpu().numpy()
            position_counts = self.sampling_position_counts[branch][group]
            position_mask_weights = self.sampling_position_mask_weights[branch][group]
            position_contribution_weights = (
                self.sampling_position_contribution_weights[branch][group]
            )
            for (y, x), mask_weight, kernel_weight in zip(
                rounded_positions.tolist(),
                position_masks_np.tolist(),
                contribution_np.tolist(),
            ):
                key = (int(x), int(y))
                position_counts[key] += 1
                position_mask_weights[key] += float(mask_weight)
                position_contribution_weights[key] += float(mask_weight * kernel_weight)

    def _save_sample(
        self, name: str, tensor: torch.Tensor, *, force: bool = False
    ) -> None:
        if not force and name in self.samples:
            return
        tensor = tensor.detach().float()
        if tensor.ndim < 4:
            return
        tensor = tensor[:1, : self.capture_policy.channels]
        height, width = tensor.shape[-2:]
        size = min(self.capture_policy.spatial_size, height, width)
        top = (height - size) // 2
        left = (width - size) // 2
        tensor = tensor[..., top : top + size, left : left + size]
        self.samples[name] = tensor.cpu().numpy().astype(np.float16)

    def report(self, *, top_shifts: int = 12) -> dict[str, Any]:
        groups: dict[str, dict[str, Any]] = {}
        for branch, by_group in sorted(self.group_stats.items()):
            groups[branch] = {}
            for group, metrics in sorted(by_group.items()):
                counts = self.shift_counts[branch][group]
                weighted = self.shift_mask_weights[branch][group]
                position_counts = self.sampling_position_counts[branch][group]
                position_mask_weights = self.sampling_position_mask_weights[branch][group]
                position_contribution_weights = (
                    self.sampling_position_contribution_weights[branch][group]
                )
                groups[branch][str(group)] = {
                    "metrics": {
                        name: stat.as_dict() for name, stat in metrics.items()
                    },
                    "top_rounded_residual_shifts": [
                        {
                            "x": shift[0],
                            "y": shift[1],
                            "count": count,
                            "mask_weight": float(weighted[shift]),
                        }
                        for shift, count in counts.most_common(top_shifts)
                    ],
                    "recommended_sampling_codebook": [
                        {
                            "x": position[0],
                            "y": position[1],
                            "count": position_counts[position],
                            "mask_weight": float(position_mask_weights[position]),
                            "mask_x_kernel_weight": float(weight),
                        }
                        for position, weight in position_contribution_weights.most_common(
                            top_shifts
                        )
                    ],
                }
        return {
            "clips": self.clip_count,
            "frames": self.frame_count,
            "alignment_calls": {
                branch: state.total_calls
                for branch, state in sorted(self._alignment_states.items())
            },
            "metrics": {
                name: stat.as_dict() for name, stat in sorted(self.stats.items())
            },
            "alignment_groups": groups,
            "activation_sample_keys": sorted(self.samples),
        }

    def save_samples(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.samples)
