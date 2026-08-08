# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import pytest
import torch

from lada.models.basicvsrpp.mmagic.registry import METRICS
from lada.models.basicvsrpp.mmagic.mosaic_consistency_error import (
    ROIMosaicConsistencyError,
    known_grid_mosaic_consistency_error,
)
from lada.models.basicvsrpp.mmagic.roi_laplacian_error import (
    ROILaplacianError,
    roi_laplacian_error,
)
from lada.models.basicvsrpp.recoverable_hf_dataset import (
    phase_block_average_mosaic,
)


def test_roi_laplacian_error_is_zero_for_identical_images() -> None:
    image = torch.rand(3, 7, 7)
    mask = torch.ones(1, 7, 7)

    assert roi_laplacian_error(image, image.clone(), mask) == 0.0


def test_roi_laplacian_error_ignores_changes_outside_roi() -> None:
    target = torch.zeros(3, 7, 7)
    prediction = target.clone()
    prediction[:, 1, 1] = 1.0
    mask = torch.zeros(1, 7, 7)
    mask[:, 5, 5] = 1.0

    assert roi_laplacian_error(target, prediction, mask) == 0.0


def test_roi_laplacian_error_integrates_with_sample_wise_evaluator() -> None:
    metric = METRICS.build(dict(type='ROILaplacianError'))
    assert isinstance(metric, ROILaplacianError)

    target = torch.zeros(3, 7, 7)
    prediction = target.clone()
    prediction[:, 3, 3] = 1.0
    mask = torch.zeros(1, 7, 7)
    mask[:, 3, 3] = 255.0
    metric.process(
        data_batch=[],
        data_samples=[
            {
                'gt_img': target,
                'mask': mask,
                'output': {'pred_img': prediction},
            }
        ],
    )

    assert metric.compute_metrics(metric.results) == {
        'ROILaplacianError': pytest.approx(4.0)
    }


def test_roi_laplacian_error_rejects_empty_roi() -> None:
    with pytest.raises(ValueError, match='positive pixel'):
        roi_laplacian_error(
            torch.zeros(3, 5, 5),
            torch.zeros(3, 5, 5),
            torch.zeros(1, 5, 5),
        )


def test_known_grid_mosaic_consistency_error_is_zero_for_exact_observation():
    block_size = 6
    phase = (2, 4)
    rng = torch.Generator().manual_seed(20260808)
    prediction_uint8 = torch.randint(
        0, 256, (1, 3, 32, 34), generator=rng, dtype=torch.uint8
    )
    image = prediction_uint8[0].permute(1, 2, 0).numpy()
    observation = phase_block_average_mosaic(
        image, block_size=block_size, phase=phase
    )
    value = known_grid_mosaic_consistency_error(
        prediction_uint8.float(),
        torch.from_numpy(observation.transpose(2, 0, 1)).unsqueeze(0).float(),
        torch.ones(1, 1, 32, 34),
        torch.tensor([phase]),
        torch.tensor(block_size),
        torch.tensor(1.0),
    )

    assert value < 1e-9


def test_known_grid_mosaic_consistency_error_detects_mean_violation():
    prediction = torch.full((1, 3, 12, 12), 100.0)
    observation = prediction.clone()
    prediction[:, :, 0:6, 0:6] += 4.0

    value = known_grid_mosaic_consistency_error(
        prediction,
        observation,
        torch.ones(1, 1, 12, 12),
        torch.tensor([(0, 0)]),
        torch.tensor(6),
        1.0,
    )

    assert value > 0.0


def test_roi_mosaic_consistency_metric_uses_input_and_metadata():
    metric = METRICS.build(dict(type='ROIMosaicConsistencyError'))
    assert isinstance(metric, ROIMosaicConsistencyError)

    prediction = torch.full((1, 3, 12, 12), 100.0)
    observation = prediction.clone()
    prediction[:, :, 0:6, 0:6] += 4.0
    metric.process(
        data_batch=[],
        data_samples=[
            {
                'input': observation,
                'mask': torch.ones(1, 1, 12, 12),
                'mosaic_phase': torch.tensor([(0, 0)]),
                'mosaic_block_size': torch.tensor(6),
                'mosaic_observation_weight': torch.tensor(1.0),
                'output': {'pred_img': prediction},
            }
        ],
    )

    result = metric.compute_metrics(metric.results)
    assert result['ROIMosaicConsistencyError'] > 0.0
