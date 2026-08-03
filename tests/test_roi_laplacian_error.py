# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import pytest
import torch

from lada.models.basicvsrpp.mmagic.registry import METRICS
from lada.models.basicvsrpp.mmagic.roi_laplacian_error import (
    ROILaplacianError,
    roi_laplacian_error,
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
