from __future__ import annotations

import pytest
import torch

from lada.models.mioh_restorer.synthetic_motion_v4 import (
    MAXIMUM_LOCAL_DISPLACEMENT,
    MOTION_QUANTUM,
    OUTPUT_INDICES,
    alignment_displacement,
    make_synthetic_motion_sequence,
    sample_motion_positions,
    shift_without_wrap,
    validate_motion_positions,
)


def test_seed_reproduces_positions_and_different_seed_changes_them() -> None:
    first = sample_motion_positions(4, seed=19)
    second = sample_motion_positions(4, seed=19)
    different = sample_motion_positions(4, seed=20)

    assert torch.equal(first, second)
    assert not torch.equal(first, different)


def test_generator_state_advances_but_is_reproducible() -> None:
    first_generator = torch.Generator().manual_seed(7)
    first = sample_motion_positions(2, generator=first_generator)
    second = sample_motion_positions(2, generator=first_generator)

    replay_generator = torch.Generator().manual_seed(7)
    assert torch.equal(first, sample_motion_positions(2, generator=replay_generator))
    assert torch.equal(second, sample_motion_positions(2, generator=replay_generator))


def test_positions_are_quantized_and_fit_every_local_context() -> None:
    positions = sample_motion_positions(32, seed=123)

    assert torch.all(positions.remainder(MOTION_QUANTUM) == 0)
    for reference_index in OUTPUT_INDICES:
        context = positions[:, reference_index - 2 : reference_index + 3]
        difference = context - positions[:, reference_index : reference_index + 1]
        assert int(difference.abs().max()) <= MAXIMUM_LOCAL_DISPLACEMENT


def test_positive_position_moves_content_down_and_right() -> None:
    anchor = torch.zeros(1, 4, 17, 17)
    anchor[0, 0, 2, 3] = 1.0
    anchor[0, 3, 2, 3] = 1.0
    positions = torch.zeros(1, 9, 2, dtype=torch.int64)
    positions[:, 1] = torch.tensor((8, 8))

    values, actual_positions, validity = make_synthetic_motion_sequence(
        anchor, positions=positions
    )

    assert torch.equal(actual_positions, positions)
    assert values[0, 1, 0, 10, 11] == 1.0
    assert values[0, 1, 0, 2, 3] == 0.0
    assert validity[0, 1, 0, :8].count_nonzero() == 0
    assert validity[0, 1, 0, :, :8].count_nonzero() == 0


def test_alignment_displacement_has_the_shift2d_sign() -> None:
    anchor = torch.zeros(1, 4, 25, 25)
    anchor[0, 0, 12, 12] = 1.0
    positions = torch.zeros(1, 9, 2, dtype=torch.int64)
    positions[:, 2] = torch.tensor((8, 0))
    positions[:, 4] = torch.tensor((-8, 8))
    values, _, _ = make_synthetic_motion_sequence(anchor, positions=positions)

    displacement = alignment_displacement(positions, 2, 4)
    aligned = shift_without_wrap(
        values[:, 4], int(displacement[0, 0]), int(displacement[0, 1])
    )

    assert displacement.tolist() == [[16, -8]]
    assert torch.equal(aligned[:, :, 1:-1, 1:-1], values[:, 2, :, 1:-1, 1:-1])


def test_shift_does_not_wrap_at_edges() -> None:
    values = torch.zeros(1, 1, 8, 8)
    values[0, 0, 0, 0] = 1.0

    shifted = shift_without_wrap(values, -8, -8)

    assert shifted.count_nonzero() == 0


def test_invalid_positions_are_rejected() -> None:
    positions = torch.zeros(1, 9, 2, dtype=torch.int64)
    positions[:, 3, 0] = 7
    with pytest.raises(ValueError, match="quantized"):
        validate_motion_positions(positions)

    positions[:, 3, 0] = 40
    positions[:, 5, 0] = -40
    with pytest.raises(ValueError, match="local context"):
        validate_motion_positions(positions)


def test_seed_and_generator_cannot_both_be_used() -> None:
    anchor = torch.zeros(1, 4, 16, 16)
    with pytest.raises(ValueError, match="mutually exclusive"):
        make_synthetic_motion_sequence(
            anchor,
            seed=1,
            generator=torch.Generator().manual_seed(1),
        )
