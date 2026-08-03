# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""ROI-only PSNR used as a fidelity guard for detail fine-tuning."""

from typing import Optional

import numpy as np
import torch

from .base_sample_wise_metric import BaseSampleWiseMetric
from .registry import METRICS


def roi_psnr(gt, pred, mask, *, data_range: float = 255.0) -> float:
    gt_tensor = torch.as_tensor(gt, dtype=torch.float64)
    pred_tensor = torch.as_tensor(pred, dtype=torch.float64)
    mask_tensor = torch.as_tensor(mask)
    if gt_tensor.shape != pred_tensor.shape:
        raise ValueError('GT and prediction shapes must match')
    if gt_tensor.ndim != 3:
        raise ValueError('ROI PSNR expects CHW images')
    if mask_tensor.ndim == 2:
        mask_tensor = mask_tensor.unsqueeze(0)
    if mask_tensor.shape[-2:] != gt_tensor.shape[-2:]:
        raise ValueError('ROI mask spatial shape must match the images')
    if mask_tensor.shape[0] == 1:
        mask_tensor = mask_tensor.expand(gt_tensor.shape[0], -1, -1)
    elif mask_tensor.shape[0] != gt_tensor.shape[0]:
        raise ValueError('ROI mask must have one channel or match the image')
    roi = mask_tensor != 0
    if not bool(roi.any()):
        raise ValueError('ROI mask must contain at least one positive pixel')
    mse = (gt_tensor[roi] - pred_tensor[roi]).square().mean().item()
    if mse == 0:
        return float('inf')
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


@METRICS.register_module()
class ROIPSNR(BaseSampleWiseMetric):
    """Peak signal-to-noise ratio restricted to the restoration ROI."""

    metric = 'ROIPSNR'

    def __init__(
        self,
        gt_key: str = 'gt_img',
        pred_key: str = 'pred_img',
        mask_key: str = 'mask',
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
        data_range: float = 255.0,
    ) -> None:
        super().__init__(
            gt_key=gt_key,
            pred_key=pred_key,
            mask_key=mask_key,
            collect_device=collect_device,
            prefix=prefix,
        )
        self.data_range = float(data_range)

    def process_image(self, gt, pred, mask) -> float:
        return roi_psnr(gt, pred, mask, data_range=self.data_range)
