# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Analytic motion supervision for the Native-HF alignment pyramid."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch.nn import functional as F

from .model_native_hf import MiohNativeHF512


NATIVE_HF_ALIGNMENT_SCALES = (8, 4, 2, 1)
NATIVE_HF_MAXIMUM_TRANSLATION = 23.0


def _translate(
    values: torch.Tensor,
    vertical: torch.Tensor,
    horizontal: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    if values.ndim != 4:
        raise ValueError("translation values must be BCHW")
    if vertical.shape != (values.shape[0],) or horizontal.shape != vertical.shape:
        raise ValueError("translation offsets do not match the batch")
    batch, _channels, height, width = values.shape
    theta = values.new_zeros(batch, 2, 3)
    theta[:, 0, 0] = 1
    theta[:, 1, 1] = 1
    theta[:, 0, 2] = -2.0 * horizontal / max(width - 1, 1)
    theta[:, 1, 2] = -2.0 * vertical / max(height - 1, 1)
    grid = F.affine_grid(theta, values.shape, align_corners=True)
    return F.grid_sample(
        values,
        grid,
        mode=mode,
        padding_mode="zeros",
        align_corners=True,
    )


def make_known_motion_window(
    values: torch.Tensor,
    *,
    maximum_translation: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Translate one guided centre frame into a known temporal trajectory.

    The returned motion is the ``(vertical, horizontal)`` displacement applied
    to target frame zero relative to the unshifted centre reference.  Aligning
    target zero back to the centre therefore requires ``-motion``.

    Synthetic reliability represents geometric validity, not the reliability
    of the source sample: it is one wherever translated content exists and
    zero in padding introduced by the translation.
    """

    if values.ndim != 5 or values.shape[2] != 8:
        raise ValueError("Native-HF known motion requires [B,T,8,H,W]")
    if values.shape[1] < 3 or values.shape[1] % 2 == 0:
        raise ValueError("Native-HF known motion requires an odd frame count")
    if not 0 < maximum_translation <= NATIVE_HF_MAXIMUM_TRANSLATION:
        raise ValueError("Native-HF motion exceeds its +/-23px alignment reach")
    center = values.shape[1] // 2
    source = values[:, center]
    source_validity = torch.ones_like(source[:, 4:5])
    batch = values.shape[0]
    steps = max(1, int(round(maximum_translation * 4)))
    vertical = torch.randint(-steps, steps + 1, (batch,), device=values.device)
    horizontal = torch.randint(-steps, steps + 1, (batch,), device=values.device)
    vertical = vertical.to(values.dtype) / 4
    horizontal = horizontal.to(values.dtype) / 4
    too_small = vertical.abs() + horizontal.abs() < 1
    horizontal = torch.where(
        too_small, horizontal.new_full((), 1.0), horizontal
    )

    frames: list[torch.Tensor] = []
    for index in range(values.shape[1]):
        factor = (center - index) / center
        frames.append(
            torch.cat(
                (
                    _translate(
                        source[:, :3],
                        vertical * factor,
                        horizontal * factor,
                        mode="bilinear",
                    ),
                    _translate(
                        source[:, 3:4],
                        vertical * factor,
                        horizontal * factor,
                        mode="nearest",
                    ),
                    _translate(
                        source_validity,
                        vertical * factor,
                        horizontal * factor,
                        mode="nearest",
                    ),
                    _translate(
                        source[:, 5:8],
                        vertical * factor,
                        horizontal * factor,
                        mode="bilinear",
                    ),
                ),
                dim=1,
            )
        )
    return torch.stack(frames, dim=1), torch.stack((vertical, horizontal), dim=1)


def alignment_offsets_and_scales(
    model: MiohNativeHF512,
) -> tuple[tuple[Sequence[tuple[int, int]], ...], tuple[int, ...]]:
    alignment = model.decoder.alignment
    offsets = (
        alignment.offsets_8,
        alignment.offsets_4,
        alignment.offsets_2,
        alignment.phase_bank.offsets,
    )
    return offsets, NATIVE_HF_ALIGNMENT_SCALES


def _alignment_reach(
    offsets: Sequence[Sequence[tuple[int, int]]],
    scales: Sequence[int],
) -> tuple[float, float]:
    """Return the decomposable vertical/horizontal reach in source pixels."""

    if len(offsets) != len(scales) or not offsets:
        raise ValueError("alignment offsets and scales do not match")
    if any(scale <= 0 for scale in scales):
        raise ValueError("alignment scales must be positive")
    vertical = 0.0
    horizontal = 0.0
    for stage_offsets, scale in zip(offsets, scales, strict=True):
        if not stage_offsets:
            raise ValueError("alignment stage has no offsets")
        vertical += max(abs(offset[0]) for offset in stage_offsets) * scale
        horizontal += max(abs(offset[1]) for offset in stage_offsets) * scale
    return vertical, horizontal


def _interior_crop(
    values: torch.Tensor, *, margin_y: int, margin_x: int
) -> torch.Tensor:
    """Remove borders that can contain zeros from any legal shift candidate."""

    if margin_y < 0 or margin_x < 0:
        raise ValueError("interior margins must be non-negative")
    height, width = values.shape[-2:]
    if height <= margin_y * 2 or width <= margin_x * 2:
        raise ValueError("known-motion sample has no padding-free interior")
    vertical = slice(margin_y, height - margin_y) if margin_y else slice(None)
    horizontal = slice(margin_x, width - margin_x) if margin_x else slice(None)
    return values[..., vertical, horizontal]


def _percentile_95(values: torch.Tensor) -> torch.Tensor:
    """Linear-interpolated p95 without relying on accelerator quantile support."""

    flat = values.detach().float().reshape(-1)
    if not flat.numel():
        raise ValueError("cannot summarize an empty endpoint-error tensor")
    ordered = flat.sort().values
    position = 0.95 * (ordered.numel() - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def hierarchical_motion_targets(
    motion: torch.Tensor,
    offsets: Sequence[Sequence[tuple[int, int]]],
    scales: Sequence[int],
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Greedily decompose inverse target motion over the four shift banks.

    ``motion`` and every offset use ``(vertical, horizontal)`` order.  Motion
    describes target-zero relative to the centre reference, while the returned
    distributions describe the inverse shifts that align that target.
    """

    if motion.ndim != 2 or motion.shape[1] != 2:
        raise ValueError("known motion must have shape [B,2]")
    if not bool(torch.isfinite(motion).all()):
        raise ValueError("known motion must be finite")
    reach_y, reach_x = _alignment_reach(offsets, scales)
    if bool(
        torch.any(motion[:, 0].abs() > reach_y + 1e-6)
        or torch.any(motion[:, 1].abs() > reach_x + 1e-6)
    ):
        raise ValueError("known motion exceeds the alignment pyramid reach")
    remaining = -motion.float()
    targets: list[torch.Tensor] = []
    for stage_offsets, scale in zip(offsets, scales, strict=True):
        candidates = torch.tensor(
            stage_offsets, device=motion.device, dtype=torch.float32
        ) * float(scale)
        distance = (remaining[:, None] - candidates[None]).square().sum(dim=2)
        if scale == 1:
            # Exact bilinear interpolation over the four neighbouring integer
            # shifts.  A radial softmax assigns non-zero mass to unrelated
            # candidates even for an integer displacement and directly
            # teaches high-frequency blur.
            vertical = candidates[:, 0][None]
            horizontal = candidates[:, 1][None]
            distribution = (
                (1.0 - (remaining[:, :1] - vertical).abs()).clamp_min(0.0)
                * (1.0 - (remaining[:, 1:] - horizontal).abs()).clamp_min(0.0)
            )
            distribution = distribution / distribution.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            selected = (distribution[..., None] * candidates[None]).sum(dim=1)
        else:
            index = distance.argmin(dim=1)
            distribution = F.one_hot(index, len(stage_offsets)).float()
            selected = candidates[index]
        remaining = remaining - selected
        targets.append(distribution.to(dtype=dtype))
    return tuple(targets)


def known_motion_alignment_loss(
    model: MiohNativeHF512,
    values: torch.Tensor,
    *,
    maximum_translation: float = 20.0,
    teacher_forcing: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if values.ndim != 5 or values.shape[1] != model.config.input_frames:
        raise ValueError("known-motion window does not match the model frame count")
    offsets, scales = alignment_offsets_and_scales(model)
    reach_y, reach_x = _alignment_reach(offsets, scales)
    if maximum_translation > min(reach_y, reach_x):
        raise ValueError("requested motion exceeds this model's alignment reach")
    synthetic, motion = make_known_motion_window(
        values, maximum_translation=maximum_translation
    )
    reference_index = synthetic.shape[1] // 2
    targets = hierarchical_motion_targets(
        motion, offsets, scales, dtype=values.dtype
    )
    # Operational validation must run the complete student chain end to end.
    # Teacher forcing remains available only for per-stage calibration
    # diagnostics; it must never be used for a promotion decision because it
    # conceals upstream candidate errors.
    aligned, weights = model.alignment_diagnostics(
        synthetic,
        reference_index=reference_index,
        target_index=0,
        teacher_weights=targets if teacher_forcing else None,
    )
    losses: list[torch.Tensor] = []
    accuracies: list[torch.Tensor] = []
    expected_shifts: list[torch.Tensor] = []
    target_shifts: list[torch.Tensor] = []
    for student, target, stage_offsets, scale in zip(
        weights, targets, offsets, scales, strict=True
    ):
        height, width = student.shape[-2:]
        margin_y = max(1, height // 8)
        margin_x = max(1, width // 8)
        interior_student = student[
            ..., margin_y : height - margin_y, margin_x : width - margin_x
        ]
        if interior_student.numel() == 0:
            interior_student = student
        losses.append(
            -(
                target[..., None, None]
                * interior_student.clamp_min(1e-7).log()
            )
            .sum(dim=1)
            .mean()
        )
        # Spatial pooling is diagnostic-only.  Supervising log(mean(p)) would
        # allow a few correct pixels to conceal wrong distributions elsewhere.
        pooled = interior_student.mean(dim=(-2, -1)).clamp_min(1e-7)
        accuracies.append(
            (pooled.argmax(dim=1) == target.argmax(dim=1)).float().mean()
        )
        candidates = torch.tensor(
            stage_offsets, device=values.device, dtype=torch.float32
        ) * float(scale)
        if scale == 1:
            operational = pooled.float()
        else:
            operational = F.one_hot(
                pooled.argmax(dim=1), len(stage_offsets)
            ).float()
        expected_shifts.append(
            (operational[..., None] * candidates[None]).sum(dim=1)
        )
        target_shifts.append(
            (target.float()[..., None] * candidates[None]).sum(dim=1)
        )

    expected = torch.stack(expected_shifts).sum(dim=0)
    expected_target = torch.stack(target_shifts).sum(dim=0)
    inverse_motion = -motion.float()
    endpoint_error = torch.linalg.vector_norm(expected - inverse_motion, dim=1)
    target_decomposition_error = torch.linalg.vector_norm(
        expected_target - inverse_motion, dim=1
    )

    # Alignment gates live on the half-resolution plane.  Cropping by the
    # complete pyramid reach excludes padding from every legal shift mixture,
    # so reliability=1 and occlusion=0 are reachable gate targets.
    margin_y = int(math.ceil(reach_y / 2.0))
    margin_x = int(math.ceil(reach_x / 2.0))
    reliability = _interior_crop(
        aligned.reliability, margin_y=margin_y, margin_x=margin_x
    ).clamp_min(1e-6)
    occlusion = _interior_crop(
        aligned.occlusion, margin_y=margin_y, margin_x=margin_x
    )
    entropy = _interior_crop(
        aligned.entropy, margin_y=margin_y, margin_x=margin_x
    )
    occlusion_free = (1.0 - occlusion).clamp_min(1e-6)
    # The pure-translation bootstrap contains no true occlusion examples, so
    # its gate statistics are diagnostics only.  Pair-gate training starts in
    # the later GT stages where missing/invalid temporal evidence exists.
    loss = torch.stack(
        [
            value / math.log(float(len(stage_offsets)))
            for value, stage_offsets in zip(losses, offsets, strict=True)
        ]
    ).mean()
    stats: dict[str, torch.Tensor] = {
        "total": loss.detach(),
        "known_motion": loss.detach(),
        "known_motion_teacher_forcing": loss.new_tensor(
            float(teacher_forcing)
        ).detach(),
        "known_motion_accuracy": torch.stack(accuracies).mean().detach(),
        "known_motion_epe": endpoint_error.mean().detach(),
        "known_motion_epe_p95": _percentile_95(endpoint_error),
        "known_motion_within_1px": (endpoint_error <= 1.0).float().mean().detach(),
        "known_motion_target_decomposition_epe": (
            target_decomposition_error.mean().detach()
        ),
        "known_reliability": reliability.mean().detach(),
        "known_occlusion": occlusion.mean().detach(),
        "known_entropy": entropy.mean().detach(),
    }
    for index, accuracy in enumerate(accuracies):
        stats[f"known_stage_{index}_accuracy"] = accuracy.detach()
    return loss, stats
