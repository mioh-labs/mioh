# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Known-grid remosaic consistency metric for recoverable HF validation."""

from __future__ import annotations

from typing import Sequence

import torch
from mmengine.evaluator import BaseMetric
from mmengine.model import is_model_wrapper

from .registry import METRICS
from .roi_loss import _known_phase_block_average


def _as_tchw_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = value.detach() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.to(device='cpu', dtype=torch.float32)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f'{name} must have shape [T,C,H,W] or [C,H,W]')
    if tensor.shape[1] not in (1, 3):
        raise ValueError(f'{name} must be channel-first, got {tuple(tensor.shape)}')
    if float(tensor.detach().max().cpu()) > 2.0:
        tensor = tensor / 255.0
    return tensor


def _as_t1hw_mask(value: torch.Tensor, *, name: str) -> torch.Tensor:
    tensor = _as_tchw_tensor(value, name=name)
    if tensor.shape[1] != 1:
        tensor = tensor.max(dim=1, keepdim=True).values
    return tensor.clamp(0.0, 1.0)


def known_grid_mosaic_consistency_error(
    prediction: torch.Tensor,
    observation: torch.Tensor,
    mask: torch.Tensor,
    phases: torch.Tensor,
    block_size: torch.Tensor | int,
    observation_weight: torch.Tensor | float = 1.0,
    *,
    full_mask_threshold: float = 0.999,
    dead_zone: float = 0.5 / 255.0,
) -> float:
    """Return mean remosaic residual in complete known-grid ROI cells.

    Lower is better.  The value is in normalized RGB units after subtracting a
    half-LSB dead zone, so an exactly matching uint8 synthetic observation is
    effectively zero.  Partial cells at crop borders and soft-mask boundaries
    are excluded to avoid penalizing measurements whose source pixels are not
    available in the crop.
    """

    if not 0.0 < full_mask_threshold <= 1.0:
        raise ValueError('full-mask threshold must be in (0, 1]')
    if dead_zone < 0:
        raise ValueError('dead zone cannot be negative')

    pred = _as_tchw_tensor(prediction, name='prediction')
    obs = _as_tchw_tensor(observation, name='observation')
    alpha = _as_t1hw_mask(mask, name='mask')
    if pred.shape != obs.shape:
        raise ValueError(
            f'prediction and observation shapes must match: {tuple(pred.shape)} '
            f'vs {tuple(obs.shape)}'
        )
    if alpha.shape[0] != pred.shape[0] or alpha.shape[-2:] != pred.shape[-2:]:
        raise ValueError('mask shape must match the prediction sequence')

    phase_tensor = torch.as_tensor(phases).detach().cpu().reshape(-1, 2)
    if phase_tensor.shape[0] != pred.shape[0]:
        raise ValueError('mosaic phases must provide one [x,y] phase per frame')
    block = int(torch.as_tensor(block_size).detach().cpu().reshape(-1)[0])
    weight = float(torch.as_tensor(observation_weight).detach().cpu().reshape(-1)[0])
    if weight <= 0:
        return 0.0

    weighted_sum = pred.new_tensor(0.0)
    denominator = pred.new_tensor(0.0)
    for frame_index in range(pred.shape[0]):
        phase = tuple(int(v) for v in phase_tensor[frame_index].tolist())
        measurement, complete_crop_cells = _known_phase_block_average(
            pred[frame_index], block_size=block, phase=phase
        )
        hard_roi = (alpha[frame_index] >= full_mask_threshold).to(dtype=pred.dtype)
        roi_coverage, _ = _known_phase_block_average(
            hard_roi, block_size=block, phase=phase
        )
        complete_roi_cells = (roi_coverage >= 1.0 - 1e-6).to(dtype=pred.dtype)
        valid = complete_crop_cells * complete_roi_cells
        residual = (measurement - obs[frame_index]).abs() - dead_zone
        residual = residual.clamp_min(0.0)
        valid = valid.expand_as(residual)
        weighted_sum = weighted_sum + (residual * valid).sum()
        denominator = denominator + valid.sum()

    if float(denominator.detach().cpu()) <= 0.0:
        return 0.0
    return float((weighted_sum / denominator).detach().cpu())


@METRICS.register_module()
class ROIMosaicConsistencyError(BaseMetric):
    """Mean known-grid remosaic residual for recoverable HF samples."""

    default_prefix = None
    SAMPLER_MODE = 'normal'
    sample_model = 'orig'

    def __init__(
        self,
        collect_device: str = 'cpu',
        prefix: str | None = None,
        full_mask_threshold: float = 0.999,
        dead_zone: float = 0.5 / 255.0,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.full_mask_threshold = float(full_mask_threshold)
        self.dead_zone = float(dead_zone)

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]) -> None:
        for data in data_samples:
            prediction = data['output']['pred_img']
            observation = data['input']
            value = known_grid_mosaic_consistency_error(
                prediction,
                observation,
                data['mask'],
                data['mosaic_phase'],
                data['mosaic_block_size'],
                data.get('mosaic_observation_weight', 1.0),
                full_mask_threshold=self.full_mask_threshold,
                dead_zone=self.dead_zone,
            )
            self.results.append({'ROIMosaicConsistencyError': value})

    def compute_metrics(self, results: list[dict]) -> dict[str, float]:
        if not results:
            return {'ROIMosaicConsistencyError': 0.0}
        values = [item['ROIMosaicConsistencyError'] for item in results]
        return {'ROIMosaicConsistencyError': float(sum(values) / len(values))}

    def prepare(self, module, dataloader) -> None:
        self.size = len(dataloader.dataset)
        if is_model_wrapper(module):
            module = module.module
        self.data_preprocessor = module.data_preprocessor

    def evaluate(self) -> dict:
        assert hasattr(self, 'size'), (
            'Cannot find \'size\', please make sure \'self.prepare\' is '
            'called correctly.'
        )
        return super().evaluate(self.size)
