# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import math

import pytest
import torch

from lada.models.basicvsrpp.mmagic.roi_psnr import roi_psnr


def test_roi_psnr_ignores_outside_error_and_uses_inside_error():
    target = torch.zeros(3, 5, 5)
    prediction = target.clone()
    prediction[:, 0, 0] = 255
    mask = torch.zeros(1, 5, 5)
    mask[:, 2, 2] = 1
    assert math.isinf(roi_psnr(target, prediction, mask))

    prediction[:, 2, 2] = 25.5
    assert roi_psnr(target, prediction, mask) == pytest.approx(20.0)


def test_roi_psnr_rejects_empty_mask():
    with pytest.raises(ValueError, match='positive pixel'):
        roi_psnr(
            torch.zeros(3, 5, 5),
            torch.zeros(3, 5, 5),
            torch.zeros(1, 5, 5),
        )
