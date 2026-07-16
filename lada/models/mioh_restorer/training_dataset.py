# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Dataset adapter for training MiohRestorer from Lada restoration metadata."""

from __future__ import annotations

import random
import warnings
from contextlib import contextmanager
from json import JSONDecodeError
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms as torchvision_transforms

from lada.datasetcreation.restoration_dataset_metadata import (
    RestorationDatasetMetadataV2,
)
from lada.utils import image_utils, transforms as lada_transforms, video_utils
from lada.utils.image_utils import (
    pad_image_by_pad,
    repad_image,
    scale_pad,
    unpad_image,
)
from lada.utils.mosaic_utils import addmosaic_base, get_random_parameters_by_block_size


def create_degradation_pipeline(image_size: int):
    return torchvision_transforms.Compose(
        [
            lada_transforms.ResizeFrames(image_size),
            lada_transforms.VideoCompression(
                p=0.9,
                codecs=["libx264", "libx265", "libvpx-vp9", "mpeg2video"],
                codec_probs=[0.3, 0.3, 0.3, 0.1],
                crf_ranges={"libx264": (16, 28), "libx265": (20, 36)},
                bitrate_ranges={
                    "libvpx-vp9": (6_000, 16_000),
                    "mpeg2video": (18_000, 40_000),
                },
            ),
            lada_transforms.GaussianBlur(sigma_range=[1.0, 4.0], p=0.3),
            lada_transforms.GaussianNoise(snr=50, p=0.2),
            lada_transforms.VideoCompression(
                p=0.15,
                codecs=["libx264"],
                codec_probs=[1.0],
                crf_ranges={"libx264": (24, 28)},
                bitrate_ranges={},
            ),
        ]
    )


class MiohRestorationDataset(Dataset):
    """Produces paired mosaic/clean sequences and the exact affected masks."""

    def __init__(
        self,
        metadata_roots: list[Path],
        *,
        sequence_frames: int,
        image_size: int = 256,
        degrade: bool = True,
        horizontal_flip: bool = True,
        deterministic: bool = False,
        limit: int | None = None,
    ) -> None:
        if not metadata_roots:
            raise ValueError("at least one metadata root is required")
        if sequence_frames <= 0:
            raise ValueError("sequence_frames must be positive")
        if image_size <= 0:
            raise ValueError("image_size must be positive")
        self.sequence_frames = sequence_frames
        self.image_size = image_size
        self.degrade = degrade
        self.horizontal_flip = horizontal_flip
        self.deterministic = deterministic
        self.samples: list[tuple[Path, RestorationDatasetMetadataV2]] = []
        skipped_metadata = 0

        for root in metadata_roots:
            root = Path(root)
            if not root.is_dir():
                raise FileNotFoundError(f"metadata directory not found: {root}")
            for metadata_path in sorted(root.glob("*.json")):
                if metadata_path.name.startswith("._"):
                    skipped_metadata += 1
                    continue
                try:
                    metadata = RestorationDatasetMetadataV2.from_json_file(metadata_path)
                except (OSError, UnicodeDecodeError, JSONDecodeError, KeyError, TypeError):
                    skipped_metadata += 1
                    continue
                if metadata.frames_count >= sequence_frames:
                    self.samples.append((metadata_path, metadata))
        if skipped_metadata:
            warnings.warn(
                f"skipped {skipped_metadata} invalid metadata files",
                RuntimeWarning,
                stacklevel=2,
            )
        if limit is not None:
            if limit <= 0:
                raise ValueError("limit must be positive")
            self.samples = self.samples[:limit]
        if not self.samples:
            raise ValueError(
                f"no clips with at least {sequence_frames} frames found in metadata roots"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _rng(self, index: int) -> random.Random:
        return random.Random(index) if self.deterministic else random

    @staticmethod
    @contextmanager
    def _temporary_seed(seed: int):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)

    def _degrade_inputs(
        self,
        inputs: list[np.ndarray],
    ) -> list[np.ndarray]:
        if not self.degrade:
            return inputs
        return create_degradation_pipeline(self.image_size)(inputs)

    @staticmethod
    def _read_exact_frames(
        path: Path,
        *,
        start: int,
        end: int,
        binary: bool = False,
    ) -> list[np.ndarray]:
        frames = video_utils.read_video_frames(
            str(path),
            float32=False,
            start_idx=start,
            end_idx=end,
            binary_frames=binary,
        )
        if len(frames) != end - start:
            raise RuntimeError(
                f"decoded {len(frames)} frames from {path}; expected {end - start}"
            )
        return frames

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        if self.deterministic:
            with self._temporary_seed(index):
                return self._load_item(index)
        return self._load_item(index)

    def _load_item(self, index: int) -> dict[str, torch.Tensor | str]:
        metadata_path, metadata = self.samples[index]
        rng = self._rng(index)
        max_start = metadata.frames_count - self.sequence_frames
        start = rng.randint(0, max_start) if max_start else 0
        end = start + self.sequence_frames
        pads = metadata.pad[start:end]
        metadata_root = metadata_path.parent
        target_path = metadata_root / metadata.relative_nsfw_video_path
        mask_path = metadata_root / metadata.relative_mask_video_path

        targets = self._read_exact_frames(target_path, start=start, end=end)
        source_masks = self._read_exact_frames(
            mask_path,
            start=start,
            end=end,
            binary=True,
        )
        source_height, source_width = targets[0].shape[:2]
        scaled_pads = [
            scale_pad(pad, source_height / self.image_size, source_width / self.image_size)
            for pad in pads
        ]
        mosaic_size, mosaic_mode, rectangle_ratio, feather_size = (
            get_random_parameters_by_block_size(
                metadata.base_mosaic_block_size.mosaic_size_v1_normal,
                randomize_size=True,
                repeatable_random=False,
            )
        )

        inputs: list[np.ndarray] = []
        mosaic_masks: list[np.ndarray] = []
        for target, source_mask, pad in zip(targets, source_masks, pads, strict=True):
            mosaic, mosaic_mask = addmosaic_base(
                unpad_image(target, pad),
                unpad_image(source_mask, pad),
                mosaic_size,
                model=mosaic_mode,
                rect_ratio=rectangle_ratio,
                feather=feather_size,
            )
            inputs.append(pad_image_by_pad(mosaic, pad))
            mosaic_masks.append(pad_image_by_pad(mosaic_mask, pad))

        inputs = self._degrade_inputs(inputs)

        targets = video_utils.resize_video_frames(targets, self.image_size)
        inputs = video_utils.resize_video_frames(inputs, self.image_size)
        resized_masks = []
        for mask in mosaic_masks:
            resized = cv2.resize(
                mask,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
            resized_masks.append(
                resized[..., None] if resized.ndim == 2 else resized[..., :1]
            )
        mosaic_masks = resized_masks

        inputs = repad_image(inputs, scaled_pads, mode="zero")
        targets = repad_image(targets, scaled_pads, mode="zero")
        mosaic_masks = repad_image(mosaic_masks, scaled_pads, mode="zero")

        if self.horizontal_flip and rng.random() < 0.5:
            inputs = [np.ascontiguousarray(np.fliplr(frame)) for frame in inputs]
            targets = [np.ascontiguousarray(np.fliplr(frame)) for frame in targets]
            mosaic_masks = [
                np.ascontiguousarray(np.fliplr(mask)) for mask in mosaic_masks
            ]

        input_tensors = image_utils.img2tensor(inputs, bgr2rgb=True, float32=True)
        target_tensors = image_utils.img2tensor(targets, bgr2rgb=True, float32=True)
        mask_tensors = [
            torch.from_numpy(mask.transpose(2, 0, 1)).float().div_(255.0)
            for mask in mosaic_masks
        ]
        return {
            "inputs": torch.stack(input_tensors),
            "targets": torch.stack(target_tensors),
            "masks": torch.stack(mask_tensors).clamp_(0.0, 1.0),
            "name": metadata.name,
        }
