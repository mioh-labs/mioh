# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Exact synthetic motion for V4 alignment pre-training.

This module deliberately starts from one frame.  Applying independent shifts
to frames from a real video would leave the unknown natural motion in the
result, so the injected translation would not be an exact alignment target.

Positions use input-pixel ``(dy, dx)`` coordinates.  A positive position moves
content down/right.  Therefore a target at ``p_target`` is aligned to a
reference at ``p_reference`` by the shift ``p_reference - p_target``.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


NUM_FRAMES = 9
OUTPUT_INDICES = tuple(range(2, 7))
LOCAL_CONTEXT_RADIUS = 2
MOTION_QUANTUM = 8
MAXIMUM_LOCAL_DISPLACEMENT = 40
_POSITION_CANDIDATES = tuple(
    range(
        -MAXIMUM_LOCAL_DISPLACEMENT,
        MAXIMUM_LOCAL_DISPLACEMENT + MOTION_QUANTUM,
        MOTION_QUANTUM,
    )
)


def _resolve_generator(
    *,
    seed: int | None,
    generator: torch.Generator | None,
) -> torch.Generator | None:
    if seed is not None and generator is not None:
        raise ValueError("seed and generator are mutually exclusive")
    if seed is None:
        return generator
    result = torch.Generator(device="cpu")
    result.manual_seed(seed)
    return result


def _sample_axis(generator: torch.Generator | None) -> torch.Tensor:
    """Sample a moderately smooth quantized path with exact local bounds."""

    candidates = torch.tensor(_POSITION_CANDIDATES, dtype=torch.int64)
    positions = torch.zeros(NUM_FRAMES, dtype=torch.int64)
    for frame_index in range(1, NUM_FRAMES):
        allowed = (candidates - positions[frame_index - 1]).abs() <= (
            MAXIMUM_LOCAL_DISPLACEMENT
        )
        if frame_index >= 2:
            allowed &= (candidates - positions[frame_index - 2]).abs() <= (
                MAXIMUM_LOCAL_DISPLACEMENT
            )

        # Prefer continuous camera/object motion while retaining a small
        # uniform component that exposes the +/-40-pixel edge of the bank.
        distance = (candidates - positions[frame_index - 1]).abs().float()
        weights = 0.05 + torch.exp(-distance / 16.0)
        weights *= allowed.float()
        selected = torch.multinomial(weights, 1, generator=generator)
        positions[frame_index] = candidates[selected]
    return positions


def sample_motion_positions(
    batch_size: int,
    *,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return quantized positions with shape ``[B,9,2]``.

    Every pair used by a V4 local five-frame context differs by at most 40
    input pixels on either axis.  Sampling is performed on CPU so a seeded CPU
    generator remains usable even when the eventual training device is MPS.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    generator = _resolve_generator(seed=seed, generator=generator)
    positions = torch.stack(
        [
            torch.stack((_sample_axis(generator), _sample_axis(generator)), dim=1)
            for _ in range(batch_size)
        ]
    )
    validate_motion_positions(positions, batch_size=batch_size)
    return positions.to(device=device) if device is not None else positions


def validate_motion_positions(
    positions: torch.Tensor,
    *,
    batch_size: int | None = None,
) -> None:
    """Validate externally supplied V4 synthetic positions."""

    expected_batch = positions.shape[0] if batch_size is None else batch_size
    if positions.shape != (expected_batch, NUM_FRAMES, 2):
        raise ValueError("positions must have shape [B,9,2]")
    if positions.is_floating_point() and not torch.equal(
        positions, positions.round()
    ):
        raise ValueError("positions must contain integer input-pixel offsets")
    integral = positions.to(dtype=torch.int64)
    if not torch.equal(integral.remainder(MOTION_QUANTUM), torch.zeros_like(integral)):
        raise ValueError("positions must be quantized to eight input pixels")
    if int(integral.abs().max()) > MAXIMUM_LOCAL_DISPLACEMENT:
        raise ValueError("absolute synthetic positions must not exceed 40 pixels")
    for reference_index in OUTPUT_INDICES:
        context = integral[
            :,
            reference_index - LOCAL_CONTEXT_RADIUS : reference_index
            + LOCAL_CONTEXT_RADIUS
            + 1,
        ]
        displacement = context - integral[:, reference_index : reference_index + 1]
        if int(displacement.abs().max()) > MAXIMUM_LOCAL_DISPLACEMENT:
            raise ValueError("a V4 local context exceeds the +/-40-pixel reach")


def shift_without_wrap(
    values: torch.Tensor,
    vertical: int,
    horizontal: int,
) -> torch.Tensor:
    """Move BCHW content using zero padding and slicing, never wraparound."""

    if values.ndim != 4:
        raise ValueError("shift input must have shape [B,C,H,W]")
    if vertical == 0 and horizontal == 0:
        return values
    height, width = values.shape[-2:]
    left = max(horizontal, 0)
    right = max(-horizontal, 0)
    top = max(vertical, 0)
    bottom = max(-vertical, 0)
    padded = F.pad(values, (left, right, top, bottom))
    return padded[..., bottom : bottom + height, right : right + width]


def make_synthetic_motion_sequence(
    anchor_values: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Translate one RGB+mask frame into an exact nine-frame sequence.

    Args:
        anchor_values: Tensor ``[B,4,H,W]``.  Channel three is the ROI mask.
        positions: Optional explicit ``[B,9,2]`` positions in input pixels.
        seed/generator: Reproducible sampling controls when positions are not
            supplied.  They are mutually exclusive.

    Returns:
        ``(values, positions, validity)`` with shapes ``[B,9,4,H,W]``,
        ``[B,9,2]`` and ``[B,9,1,H,W]``.  Validity marks pixels originating
        from the anchor rather than zero padding.
    """

    if anchor_values.ndim != 4 or anchor_values.shape[1] != 4:
        raise ValueError("anchor_values must have shape [B,4,H,W]")
    if seed is not None and generator is not None:
        raise ValueError("seed and generator are mutually exclusive")
    batch_size = anchor_values.shape[0]
    if positions is None:
        positions = sample_motion_positions(
            batch_size,
            seed=seed,
            generator=generator,
            device=anchor_values.device,
        )
    else:
        validate_motion_positions(positions, batch_size=batch_size)
        positions = positions.to(device=anchor_values.device, dtype=torch.int64)

    validity_source = torch.ones_like(anchor_values[:, :1])
    value_frames: list[torch.Tensor] = []
    validity_frames: list[torch.Tensor] = []
    # Different examples may have different offsets; keep this training-only
    # helper simple and explicit rather than introducing grid_sample.
    for frame_index in range(NUM_FRAMES):
        batch_values: list[torch.Tensor] = []
        batch_validity: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            vertical, horizontal = (
                int(item)
                for item in positions[batch_index, frame_index].detach().cpu()
            )
            batch_values.append(
                shift_without_wrap(
                    anchor_values[batch_index : batch_index + 1],
                    vertical,
                    horizontal,
                )
            )
            batch_validity.append(
                shift_without_wrap(
                    validity_source[batch_index : batch_index + 1],
                    vertical,
                    horizontal,
                )
            )
        value_frames.append(torch.cat(batch_values, dim=0))
        validity_frames.append(torch.cat(batch_validity, dim=0))
    return (
        torch.stack(value_frames, dim=1),
        positions,
        torch.stack(validity_frames, dim=1),
    )


def alignment_displacement(
    positions: torch.Tensor,
    reference_index: int,
    target_index: int,
) -> torch.Tensor:
    """Shift required to align a synthetic target to its reference."""

    validate_motion_positions(positions)
    if not 0 <= reference_index < NUM_FRAMES or not 0 <= target_index < NUM_FRAMES:
        raise IndexError("frame index is outside the nine-frame sequence")
    return positions[:, reference_index] - positions[:, target_index]
