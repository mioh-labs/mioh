# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Mask-aware Laplacian error for restoration validation."""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .base_sample_wise_metric import BaseSampleWiseMetric
from .registry import METRICS


def _as_chw_tensor(
    value: torch.Tensor | np.ndarray,
    *,
    input_order: str,
    name: str,
) -> torch.Tensor:
    tensor = (
        value.detach()
        if isinstance(value, torch.Tensor)
        else torch.as_tensor(value)
    )
    if tensor.ndim == 2:
        return tensor.unsqueeze(0).to(dtype=torch.float32)
    if tensor.ndim != 3:
        raise ValueError(f'{name} must be a 2D or 3D image, got {tuple(tensor.shape)}')

    if input_order == 'HWC':
        tensor = tensor.permute(2, 0, 1)
    elif input_order != 'CHW':
        raise ValueError('input_order must be either "CHW" or "HWC"')
    return tensor.to(dtype=torch.float32)


def roi_laplacian_error(
    gt: torch.Tensor | np.ndarray,
    pred: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    *,
    input_order: str = 'CHW',
) -> float:
    """Return mean absolute Laplacian error inside a non-empty ROI.

    The four-neighbour Laplacian matches the kernel used by
    :class:`ROIHighFrequencyLoss`. The result is expressed in the same value
    scale as the input images, and lower values are better.
    """

    gt_tensor = _as_chw_tensor(gt, input_order=input_order, name='gt')
    pred_tensor = _as_chw_tensor(pred, input_order=input_order, name='pred')
    if gt_tensor.shape != pred_tensor.shape:
        raise ValueError(
            'GT and prediction shapes must match, got '
            f'{tuple(gt_tensor.shape)} and {tuple(pred_tensor.shape)}'
        )

    mask_tensor = _as_chw_tensor(mask, input_order=input_order, name='mask')
    if mask_tensor.shape[-2:] != gt_tensor.shape[-2:]:
        raise ValueError(
            'ROI mask spatial shape must match the images, got '
            f'{tuple(mask_tensor.shape[-2:])} and {tuple(gt_tensor.shape[-2:])}'
        )
    if mask_tensor.shape[0] not in (1, gt_tensor.shape[0]):
        raise ValueError(
            'ROI mask must have one channel or match the image channels, got '
            f'{mask_tensor.shape[0]} and {gt_tensor.shape[0]}'
        )

    device = gt_tensor.device
    pred_tensor = pred_tensor.to(device)
    mask_tensor = mask_tensor.to(device)
    roi = mask_tensor > 0
    if mask_tensor.shape[0] != gt_tensor.shape[0]:
        roi = roi.expand(gt_tensor.shape[0], -1, -1)
    if not bool(roi.any()):
        raise ValueError('ROI mask must contain at least one positive pixel')

    kernel = gt_tensor.new_tensor(
        ((0.0, 1.0, 0.0), (1.0, -4.0, 1.0), (0.0, 1.0, 0.0))
    ).reshape(1, 1, 3, 3)
    channels = gt_tensor.shape[0]
    filters = kernel.expand(channels, 1, 3, 3)
    gt_laplacian = F.conv2d(
        gt_tensor.unsqueeze(0), filters, padding=1, groups=channels
    )[0]
    pred_laplacian = F.conv2d(
        pred_tensor.unsqueeze(0), filters, padding=1, groups=channels
    )[0]
    return float(torch.abs(pred_laplacian - gt_laplacian)[roi].mean().cpu())


@METRICS.register_module()
class ROILaplacianError(BaseSampleWiseMetric):
    """Mean absolute Laplacian error inside the restoration ROI.

    This is a sample-wise, lower-is-better metric. Sequences are evaluated one
    frame at a time and averaged by :class:`BaseSampleWiseMetric`.
    """

    metric = 'ROILaplacianError'

    def __init__(
        self,
        gt_key: str = 'gt_img',
        pred_key: str = 'pred_img',
        mask_key: str = 'mask',
        collect_device: str = 'cpu',
        prefix: Optional[str] = None,
        input_order: str = 'CHW',
    ) -> None:
        super().__init__(
            gt_key=gt_key,
            pred_key=pred_key,
            mask_key=mask_key,
            collect_device=collect_device,
            prefix=prefix,
        )
        if input_order not in ('CHW', 'HWC'):
            raise ValueError('input_order must be either "CHW" or "HWC"')
        self.input_order = input_order

    def process_image(self, gt, pred, mask) -> float:
        return roi_laplacian_error(
            gt,
            pred,
            mask,
            input_order=self.input_order,
        )
