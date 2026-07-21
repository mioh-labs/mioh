# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Memory-bounded intermediate distillation for MiohRestorerV3."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.nn import functional as F


def teacher_shift_distribution(
    offset: torch.Tensor,
    mask: torch.Tensor,
    shifts: Sequence[tuple[int, int]],
    *,
    source: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Project BasicVSR++ DCN offsets onto a fixed shift bank.

    BasicVSR++ aligns two 64-channel temporal sources in one deformable
    convolution.  With 16 deform groups, the first eight groups address the
    first-order source and the final eight address the second-order source.
    Its regular 3x3 kernel footprint is intentionally not added here: V3's
    alignment-fusion convolution already supplies that static footprint.  The
    target therefore describes only the dynamic displacement.
    """

    if source not in (0, 1):
        raise ValueError("source must be 0 or 1")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if offset.ndim != 4 or mask.ndim != 4:
        raise ValueError("teacher offset and mask must be BCHW tensors")
    if not shifts:
        raise ValueError("at least one fixed shift is required")
    offset = offset.float()
    mask = mask.float()
    batch, mask_channels, height, width = mask.shape
    kernel_points = 9
    if mask_channels % kernel_points:
        raise ValueError("unexpected teacher mask channel count")
    deform_groups = mask_channels // kernel_points
    if deform_groups % 2:
        raise ValueError("teacher deform groups must split into two sources")
    if offset.shape != (
        batch,
        deform_groups * kernel_points * 2,
        height,
        width,
    ):
        raise ValueError("teacher offset and mask shapes do not match")

    source_groups = deform_groups // 2
    group_start = source * source_groups
    group_end = group_start + source_groups
    vectors = offset.reshape(
        batch, deform_groups, kernel_points, 2, height, width
    )[:, group_start:group_end]
    modulation = mask.reshape(
        batch, deform_groups, kernel_points, height, width
    )[:, group_start:group_end]
    shift_tensor = offset.new_tensor(shifts).reshape(
        1, 1, 1, len(shifts), 2, 1, 1
    )
    squared_distance = (
        vectors.unsqueeze(3) - shift_tensor
    ).square().sum(dim=4)
    assignment = torch.softmax(-squared_distance / temperature, dim=3)
    distribution = (
        assignment * modulation.unsqueeze(3)
    ).sum(dim=2)
    return distribution / distribution.sum(dim=2, keepdim=True).clamp_min(1e-6)


def teacher_hierarchical_shift_distributions(
    offset: torch.Tensor,
    mask: torch.Tensor,
    stage_shifts: Sequence[Sequence[tuple[int, int]]],
    *,
    source: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, ...]:
    """Decompose teacher offsets into coarse-to-fine shift targets.

    Every teacher DCN vector is projected onto the first shift bank.  The
    expected selected shift is subtracted and the residual is projected onto
    the next bank.  With 9/3/1 dilations this is a soft balanced-ternary
    decomposition that gives every hierarchy stage a direct supervision
    signal instead of asking the final restored image to discover motion by
    itself.
    """

    if source not in (0, 1):
        raise ValueError("source must be 0 or 1")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if offset.ndim != 4 or mask.ndim != 4:
        raise ValueError("teacher offset and mask must be BCHW tensors")
    if not stage_shifts or any(not shifts for shifts in stage_shifts):
        raise ValueError("every hierarchy stage needs fixed shifts")

    offset = offset.float()
    modulation_mask = mask.float()
    batch, mask_channels, height, width = modulation_mask.shape
    kernel_points = 9
    if mask_channels % kernel_points:
        raise ValueError("unexpected teacher mask channel count")
    deform_groups = mask_channels // kernel_points
    if deform_groups % 2:
        raise ValueError("teacher deform groups must split into two sources")
    if offset.shape != (
        batch,
        deform_groups * kernel_points * 2,
        height,
        width,
    ):
        raise ValueError("teacher offset and mask shapes do not match")

    source_groups = deform_groups // 2
    start = source * source_groups
    residual = offset.reshape(
        batch, deform_groups, kernel_points, 2, height, width
    )[:, start : start + source_groups]
    modulation = modulation_mask.reshape(
        batch, deform_groups, kernel_points, height, width
    )[:, start : start + source_groups]
    targets: list[torch.Tensor] = []
    for shifts in stage_shifts:
        shift_tensor = offset.new_tensor(shifts).reshape(
            1, 1, 1, len(shifts), 2, 1, 1
        )
        squared_distance = (
            residual.unsqueeze(3) - shift_tensor
        ).square().sum(dim=4)
        assignment = torch.softmax(
            -squared_distance / temperature, dim=3
        )
        distribution = (
            assignment * modulation.unsqueeze(3)
        ).sum(dim=2)
        targets.append(
            distribution
            / distribution.sum(dim=2, keepdim=True).clamp_min(1e-6)
        )
        expected_shift = (
            assignment.unsqueeze(4) * shift_tensor
        ).sum(dim=3)
        residual = residual - expected_shift
    return tuple(targets)


def teacher_source_confidence(
    mask: torch.Tensor,
    *,
    source: int,
) -> torch.Tensor:
    """Reduce the teacher's nine DCN modulation masks per source group."""

    if source not in (0, 1):
        raise ValueError("source must be 0 or 1")
    if mask.ndim != 4 or mask.shape[1] % 9:
        raise ValueError("unexpected teacher mask shape")
    batch, mask_channels, height, width = mask.shape
    deform_groups = mask_channels // 9
    if deform_groups % 2:
        raise ValueError("teacher deform groups must split into two sources")
    source_groups = deform_groups // 2
    start = source * source_groups
    return mask.float().reshape(
        batch, deform_groups, 9, height, width
    )[:, start : start + source_groups].mean(dim=2, keepdim=True)


def roi_alignment_kl_loss(
    student_weights: torch.Tensor,
    teacher_distribution: torch.Tensor,
    roi_mask: torch.Tensor,
) -> torch.Tensor:
    """KL divergence between fixed-shift probabilities inside the ROI."""

    if student_weights.shape != teacher_distribution.shape:
        raise ValueError("student and teacher shift distributions must match")
    if roi_mask.ndim != 4 or roi_mask.shape[:2] != (
        student_weights.shape[0],
        1,
    ):
        raise ValueError("ROI mask must have shape [B,1,H,W]")
    roi = F.interpolate(
        roi_mask.float(),
        size=student_weights.shape[-2:],
        mode="nearest",
    )
    # Include a narrow context ring because temporal alignment crosses the
    # moving ROI boundary even when restoration itself remains mask limited.
    roi = F.max_pool2d(roi, kernel_size=5, stride=1, padding=2)
    target = teacher_distribution.float().clamp_min(1e-6)
    student = student_weights.float().clamp_min(1e-6)
    divergence = (
        target * (target.log() - student.log())
    ).sum(dim=2).mean(dim=1, keepdim=True)
    return (divergence * roi).sum() / roi.sum().clamp_min(1.0)


def roi_confidence_loss(
    student_confidence: torch.Tensor,
    teacher_confidence: torch.Tensor,
    roi_mask: torch.Tensor,
) -> torch.Tensor:
    """Match the teacher's ability to suppress unreliable alignment."""

    if student_confidence.shape != teacher_confidence.shape:
        raise ValueError("student and teacher confidence tensors must match")
    roi = F.interpolate(
        roi_mask.float(),
        size=student_confidence.shape[-2:],
        mode="nearest",
    )
    roi = F.max_pool2d(roi, kernel_size=5, stride=1, padding=2)
    difference = torch.sqrt(
        (student_confidence.float() - teacher_confidence.float()).square()
        + 1e-6
    ).mean(dim=1)
    return (difference * roi).sum() / roi.sum().clamp_min(1.0)


def roi_feature_energy_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    roi_mask: torch.Tensor,
) -> torch.Tensor:
    """Channel-count-independent attention transfer for aligned features."""

    if student.ndim != 4 or teacher.ndim != 4:
        raise ValueError("alignment features must be BCHW tensors")
    if student.shape[0] != teacher.shape[0] or student.shape[-2:] != teacher.shape[-2:]:
        raise ValueError("student and teacher feature geometry must match")
    student_energy = student.float().square().mean(dim=1, keepdim=True)
    teacher_energy = teacher.float().square().mean(dim=1, keepdim=True)
    student_energy = student_energy / student_energy.square().mean(
        dim=(-2, -1), keepdim=True
    ).sqrt().clamp_min(1e-6)
    teacher_energy = teacher_energy / teacher_energy.square().mean(
        dim=(-2, -1), keepdim=True
    ).sqrt().clamp_min(1e-6)
    roi = F.interpolate(
        roi_mask.float(), size=student.shape[-2:], mode="nearest"
    )
    roi = F.max_pool2d(roi, kernel_size=5, stride=1, padding=2)
    difference = torch.sqrt(
        (student_energy - teacher_energy).square() + 1e-6
    )
    return (difference * roi).sum() / roi.sum().clamp_min(1.0)
