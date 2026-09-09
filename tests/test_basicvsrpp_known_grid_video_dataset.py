# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch

from lada.models.basicvsrpp.known_grid_video_dataset import (
    AlternatingKnownGridMosaicVideoDataset,
)
from lada.models.basicvsrpp.mmagic.data_sample import DataSample


class _DummyDataset:
    def __init__(self, length: int, *, exact: bool) -> None:
        self.length = length
        self.exact = exact

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        sample = DataSample(
            gt_img=torch.zeros((26, 3, 8, 8)),
            mask=torch.ones((26, 1, 8, 8)),
        )
        if self.exact:
            sample.mosaic_phase = torch.full((26, 2), index, dtype=torch.int64)
            sample.mosaic_block_size = torch.tensor(8, dtype=torch.int64)
            sample.mosaic_observation_weight = torch.tensor(1.0)
        return {
            'inputs': torch.full((26, 3, 8, 8), index, dtype=torch.uint8),
            'data_samples': sample,
        }


def test_alternating_dataset_balances_generic_and_exact_samples():
    dataset = AlternatingKnownGridMosaicVideoDataset(
        _DummyDataset(3, exact=False),
        _DummyDataset(2, exact=True),
    )

    assert len(dataset) == 6
    generic = dataset[4]['data_samples']
    exact = dataset[5]['data_samples']
    assert float(generic.mosaic_observation_weight) == 0.0
    assert int(generic.mosaic_block_size) == 2
    assert torch.count_nonzero(generic.mosaic_phase) == 0
    assert float(exact.mosaic_observation_weight) == 1.0
    assert int(exact.mosaic_block_size) == 8
    assert torch.all(exact.mosaic_phase == 0)


def test_generic_branch_metadata_has_one_phase_per_frame():
    dataset = AlternatingKnownGridMosaicVideoDataset(
        _DummyDataset(1, exact=False),
        _DummyDataset(1, exact=True),
    )
    sample = dataset[0]['data_samples']

    assert sample.mosaic_phase.shape == (26, 2)
    assert sample.mosaic_observation_weight.dtype == torch.float32
