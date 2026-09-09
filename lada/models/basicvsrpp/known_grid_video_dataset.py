# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Exact known-grid 26-frame data for BasicVSR++.

The manifest freezes which source clips belong to each split.  Samples are
generated on demand from the existing clean video and mask assets, so no large
duplicate frame store is required.  All spatial augmentation is applied before
the square-average mosaic.  Consequently every returned observation has an
exact block size and grid phase that can be used by
``KnownGridMosaicConsistencyLoss``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import lada.utils.video_utils as video_utils
from lada.datasetcreation.restoration_dataset_metadata import (
    RestorationDatasetMetadataV2,
)
from lada.models.basicvsrpp.mmagic.data_sample import DataSample
from lada.models.basicvsrpp.mmagic.registry import DATASETS
from lada.models.basicvsrpp.mosaic_video_dataset import (
    _as_single_channel_mask,
    _native_roi_crop,
    _rotate_mask,
)
from lada.models.basicvsrpp.recoverable_hf_dataset import (
    phase_block_average_mosaic,
)
from lada.utils.image_utils import repad_image


MANIFEST_VERSION = 1


def _read_manifest(path: Path) -> list[dict]:
    entries: list[dict] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get('version') != MANIFEST_VERSION:
                raise ValueError(
                    f'{path}:{line_number}: unsupported known-grid manifest version'
                )
            metadata_path = Path(entry['metadata_path'])
            if not metadata_path.is_file():
                raise FileNotFoundError(metadata_path)
            entries.append(entry)
    if not entries:
        raise ValueError(f'known-grid manifest is empty: {path}')
    return entries


def _rgb_uint8_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(
                np.ascontiguousarray(frame.transpose(2, 0, 1))
            )
            for frame in frames
        ],
        dim=0,
    )


def _mask_tensor(frames: list[np.ndarray]) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(
                np.ascontiguousarray(_as_single_channel_mask(frame))
            ).float().unsqueeze(0) / 255.0
            for frame in frames
        ],
        dim=0,
    ).clamp_(0.0, 1.0)


@DATASETS.register_module()
class KnownGridMosaicVideoDataset(Dataset):
    """Generate exact square-average mosaics from fixed source manifests."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        num_frame: int = 26,
        lq_size: int = 256,
        training: bool = True,
        use_hflip: bool = True,
        time_reverse: bool = True,
        rotation_probability: float = 0.15,
        minimum_block_size: int = 6,
        maximum_block_size: int = 12,
        seed: int = 0,
    ) -> None:
        self.manifest = Path(manifest)
        self.entries = _read_manifest(self.manifest)
        self.num_frame = int(num_frame)
        self.lq_size = int(lq_size)
        self.training = bool(training)
        self.use_hflip = bool(use_hflip)
        self.time_reverse = bool(time_reverse)
        self.rotation_probability = float(rotation_probability)
        self.minimum_block_size = int(minimum_block_size)
        self.maximum_block_size = int(maximum_block_size)
        self.seed = int(seed)
        if self.num_frame < 2:
            raise ValueError('known-grid video data requires at least two frames')
        if self.lq_size < 64:
            raise ValueError('known-grid crop size must be at least 64 pixels')
        if not 0.0 <= self.rotation_probability <= 1.0:
            raise ValueError('rotation probability must be in [0, 1]')
        if (
            self.minimum_block_size <= 1
            or self.maximum_block_size < self.minimum_block_size
        ):
            raise ValueError('invalid known-grid block-size range')
        for entry in self.entries:
            if int(entry['frames_count']) < self.num_frame:
                raise ValueError(
                    f"{entry['metadata_path']} has fewer than {self.num_frame} frames"
                )

    def __len__(self) -> int:
        return len(self.entries)

    def _rng(self, index: int):
        if self.training:
            return random
        return random.Random(self.seed + index * 1_000_003)

    @staticmethod
    def _resolve_asset(metadata_path: Path, relative_path: str) -> Path:
        path = (metadata_path.parent / relative_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | DataSample]:
        entry = self.entries[index]
        metadata_path = Path(entry['metadata_path'])
        metadata = RestorationDatasetMetadataV2.from_json_file(metadata_path)
        rng = self._rng(index)

        maximum_start = metadata.frames_count - self.num_frame
        start = rng.randint(0, maximum_start) if self.training else maximum_start // 2
        end = start + self.num_frame
        pads = metadata.pad[start:end]

        target_path = self._resolve_asset(
            metadata_path, metadata.relative_nsfw_video_path
        )
        mask_path = self._resolve_asset(
            metadata_path, metadata.relative_mask_video_path
        )
        targets = video_utils.read_video_frames(
            str(target_path), float32=False, start_idx=start, end_idx=end
        )
        masks = video_utils.read_video_frames(
            str(mask_path),
            float32=False,
            start_idx=start,
            end_idx=end,
            binary_frames=True,
        )
        if len(targets) != self.num_frame or len(masks) != self.num_frame:
            raise RuntimeError(
                f'decoded {len(targets)} target and {len(masks)} mask frames; '
                f'expected {self.num_frame}'
            )

        # Restore the tracked crops to a common scene coordinate system, then
        # take one ROI-anchored native crop shared by the complete sequence.
        targets = repad_image(targets, pads, mode='zero')
        masks = repad_image(masks, pads, mode='zero')
        targets, _, masks = _native_roi_crop(
            targets, targets, masks, self.lq_size, rng
        )

        if self.training and self.time_reverse and rng.random() < 0.5:
            targets.reverse()
            masks.reverse()
        if self.training and self.use_hflip and rng.random() < 0.5:
            targets = [np.ascontiguousarray(np.fliplr(frame)) for frame in targets]
            masks = [np.ascontiguousarray(np.fliplr(mask)) for mask in masks]
        if self.training and rng.random() < self.rotation_probability:
            degrees = rng.choice((-2, -1, 1, 2))
            targets = [
                cv2.warpAffine(
                    frame,
                    cv2.getRotationMatrix2D(
                        (frame.shape[1] / 2, frame.shape[0] / 2), degrees, 1
                    ),
                    (frame.shape[1], frame.shape[0]),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
                for frame in targets
            ]
            masks = [_rotate_mask(mask, degrees) for mask in masks]

        block_size = rng.randint(
            self.minimum_block_size, self.maximum_block_size
        )
        phase = (rng.randrange(block_size), rng.randrange(block_size))
        observations: list[np.ndarray] = []
        hard_masks: list[np.ndarray] = []
        for target, mask in zip(targets, masks, strict=True):
            hard_mask = (
                _as_single_channel_mask(mask) >= 128
            ).astype(np.uint8) * 255
            mosaic = phase_block_average_mosaic(
                target, block_size=block_size, phase=phase
            )
            alpha = (hard_mask.astype(np.float32) / 255.0)[..., None]
            observation = np.rint(
                target.astype(np.float32) * (1.0 - alpha)
                + mosaic.astype(np.float32) * alpha
            )
            observations.append(
                np.clip(observation, 0, 255).astype(np.uint8)
            )
            hard_masks.append(hard_mask)

        input_tensor = _rgb_uint8_tensor(observations)
        target_tensor = _rgb_uint8_tensor(targets)
        mask_tensor = _mask_tensor(hard_masks)
        phases = torch.tensor(
            [phase] * self.num_frame, dtype=torch.int64
        )
        data_sample = DataSample(gt_img=target_tensor, mask=mask_tensor)
        data_sample.mosaic_phase = phases
        data_sample.mosaic_block_size = torch.tensor(
            block_size, dtype=torch.int64
        )
        data_sample.mosaic_observation_weight = torch.tensor(
            1.0, dtype=torch.float32
        )
        data_sample.set_metainfo(
            {
                'img_channel_order': 'RGB',
                'img_color_type': 'color',
                'gt_channel_order': 'RGB',
                'gt_color_type': 'color',
                'gt_path': str(target_path),
                'sample_idx': index,
                'source_metadata_path': str(metadata_path),
                'source_start_frame': start,
                'num_input_frames': self.num_frame,
                'num_output_frames': self.num_frame,
                'known_grid_block_size': block_size,
                'known_grid_phase': phase,
            }
        )
        return {'inputs': input_tensor, 'data_samples': data_sample}


@DATASETS.register_module()
class AlternatingKnownGridMosaicVideoDataset(Dataset):
    """Balance broad synthetic mosaics with exact known-grid observations.

    Generic samples retain midpoint, rectangular, feathered and degraded
    mosaics, but opt out of the exact forward loss.  Every adjacent dataset
    index alternates branches, yielding a deterministic 50/50 branch balance
    even though the outer infinite sampler shuffles indices.
    """

    def __init__(self, generic_dataset, known_grid_dataset) -> None:
        self.generic_dataset = (
            DATASETS.build(generic_dataset)
            if isinstance(generic_dataset, dict)
            else generic_dataset
        )
        self.known_grid_dataset = (
            DATASETS.build(known_grid_dataset)
            if isinstance(known_grid_dataset, dict)
            else known_grid_dataset
        )
        if len(self.generic_dataset) == 0 or len(self.known_grid_dataset) == 0:
            raise ValueError('alternating known-grid datasets cannot be empty')
        self.branch_length = max(
            len(self.generic_dataset), len(self.known_grid_dataset)
        )

    def __len__(self) -> int:
        return self.branch_length * 2

    @staticmethod
    def _disable_exact_forward_loss(sample: dict) -> dict:
        data_sample = sample['data_samples']
        frames = int(data_sample.gt_img.shape[0])
        data_sample.mosaic_phase = torch.zeros(
            (frames, 2), dtype=torch.int64
        )
        data_sample.mosaic_block_size = torch.tensor(2, dtype=torch.int64)
        data_sample.mosaic_observation_weight = torch.tensor(
            0.0, dtype=torch.float32
        )
        return sample

    def __getitem__(self, index: int):
        branch_index = index // 2
        if index % 2 == 0:
            sample = self.generic_dataset[
                branch_index % len(self.generic_dataset)
            ]
            return self._disable_exact_forward_loss(sample)
        return self.known_grid_dataset[
            branch_index % len(self.known_grid_dataset)
        ]
