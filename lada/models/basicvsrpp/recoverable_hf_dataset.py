# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Recoverable native high-frequency data for BasicVSR++.

The recoverability manifests describe nine-frame, native 512 pixel windows.
This loader synthesizes an exact block-average mosaic on the decoded source
frame *before* either crop is taken, then extracts a native 512 crop followed
by one shared, ROI-anchored native 256 crop.  No spatial resampling is
performed anywhere in the dataset.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from lada.models.basicvsrpp.mmagic.data_sample import DataSample
from lada.models.basicvsrpp.mmagic.registry import DATASETS
from lada.models.mioh_restorer.native_dataset_v5 import (
    V5NativeManifestEntry,
    crop_native_frame,
    decode_native_frames,
    read_v5_native_manifest,
)


NUM_FRAMES = 9
SOURCE_CROP_SIZE = 512
FINAL_CROP_SIZE = 256


def phase_block_average_mosaic(
    image: np.ndarray,
    *,
    block_size: int,
    phase: tuple[int, int],
) -> np.ndarray:
    """Return an exact uint8 block-average mosaic with a known grid phase.

    ``phase`` gives the x/y coordinate of a block boundary modulo
    ``block_size``.  Replicating only at the real decoded-frame boundary makes
    partial cells well-defined without inventing a boundary at either native
    crop edge.
    """

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("phase mosaic expects an HWC RGB image")
    if block_size <= 1:
        raise ValueError("mosaic block size must exceed one pixel")

    phase_x, phase_y = (int(value) % block_size for value in phase)
    pad_left = (-phase_x) % block_size
    pad_top = (-phase_y) % block_size
    height, width = image.shape[:2]
    pad_right = (-(width + pad_left)) % block_size
    pad_bottom = (-(height + pad_top)) % block_size
    padded = np.pad(
        image.astype(np.float32),
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="edge",
    )
    padded_height, padded_width = padded.shape[:2]
    cells = padded.reshape(
        padded_height // block_size,
        block_size,
        padded_width // block_size,
        block_size,
        3,
    ).mean(axis=(1, 3))
    expanded = np.repeat(
        np.repeat(cells, block_size, axis=0), block_size, axis=1
    )
    result = expanded[pad_top : pad_top + height, pad_left : pad_left + width]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def _rgb_uint8_tensor(frames: Sequence[np.ndarray]) -> torch.Tensor:
    """Stack RGB HWC arrays in the raw 0..255 BasicVSR++ data contract."""

    return torch.stack(
        [
            torch.from_numpy(
                np.ascontiguousarray(frame.transpose(2, 0, 1))
            )
            for frame in frames
        ],
        dim=0,
    )


def _single_channel_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        mask = np.max(mask, axis=2)
    if mask.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {mask.shape}")
    return np.ascontiguousarray(mask)


def _mask_tensor(mask: np.ndarray) -> torch.Tensor:
    value = torch.from_numpy(np.ascontiguousarray(mask)).float()
    if value.numel() and float(value.max()) > 1.0:
        value.div_(255.0)
    return value.unsqueeze(0)


def _crop_at(
    frame: np.ndarray,
    *,
    top: int,
    left: int,
    size: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    if top < 0 or left < 0 or top + size > height or left + size > width:
        raise ValueError(
            f"native crop ({left}, {top}, {size}) exceeds {width}x{height}"
        )
    return np.ascontiguousarray(frame[top : top + size, left : left + size])


def _shared_roi_crop_offset(
    masks: Sequence[np.ndarray],
    *,
    rng,
    training: bool,
) -> tuple[int, int]:
    """Choose one even (top, left) crop shared by all nine frames."""

    union = np.zeros((SOURCE_CROP_SIZE, SOURCE_CROP_SIZE), dtype=np.uint8)
    for mask in masks:
        if mask.shape != union.shape:
            raise ValueError(
                f"expected a 512x512 mask crop, got {mask.shape}"
            )
        union = np.maximum(union, mask)
    ys, xs = np.where(union > 0)
    if len(ys):
        if training:
            selected = rng.randrange(len(ys))
            anchor_y = int(ys[selected])
            anchor_x = int(xs[selected])
            jitter = FINAL_CROP_SIZE // 8
            anchor_y += rng.randint(-jitter, jitter)
            anchor_x += rng.randint(-jitter, jitter)
        else:
            # +1 represents the geometric centre of an inclusive bbox.  A
            # full 0..511 union therefore selects the exact 128 centre crop.
            anchor_y = (int(ys.min()) + int(ys.max()) + 1) // 2
            anchor_x = (int(xs.min()) + int(xs.max()) + 1) // 2
    else:
        anchor_y = SOURCE_CROP_SIZE // 2
        anchor_x = SOURCE_CROP_SIZE // 2

    maximum = SOURCE_CROP_SIZE - FINAL_CROP_SIZE
    top = max(0, min(maximum, anchor_y - FINAL_CROP_SIZE // 2))
    left = max(0, min(maximum, anchor_x - FINAL_CROP_SIZE // 2))
    # Preserve a stable phase for any later pixel-unshuffle path.
    top -= top % 2
    left -= left % 2
    return top, left


@DATASETS.register_module()
class RecoverableHFMosaicVideoDataset(Dataset):
    """Nine-frame native 512-to-256 curriculum for BasicVSR++.

    Training samples draw a uniform 6--12 pixel block size and a random known
    phase, then optionally apply horizontal flip and temporal reversal.
    Validation samples disable both augmentations and derive all random values
    from ``seed`` and the manifest index, making repeated evaluation byte
    identical.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        training: bool = True,
        use_hflip: bool = True,
        time_reverse: bool = True,
        minimum_block_size: int = 6,
        maximum_block_size: int = 12,
        seed: int = 0,
    ) -> None:
        self.manifest = Path(manifest)
        self.entries = read_v5_native_manifest(self.manifest)
        if minimum_block_size <= 1 or maximum_block_size < minimum_block_size:
            raise ValueError("invalid recoverable HF block-size range")
        if any(entry.bucket != SOURCE_CROP_SIZE for entry in self.entries):
            raise ValueError(
                "recoverable HF manifests must contain native 512 crops"
            )
        self.training = bool(training)
        self.use_hflip = bool(use_hflip)
        self.time_reverse = bool(time_reverse)
        self.minimum_block_size = int(minimum_block_size)
        self.maximum_block_size = int(maximum_block_size)
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _decode(path: Path, start: int, pixel_format: str) -> list[np.ndarray]:
        frames = decode_native_frames(
            path,
            start=start,
            count=NUM_FRAMES,
            pixel_format=pixel_format,
        )
        if len(frames) != NUM_FRAMES:
            raise RuntimeError(
                f"decoded {len(frames)} frames from {path}; expected nine"
            )
        return frames

    def _rng_for_index(self, index: int):
        # PyTorch seeds Python's module RNG independently in every worker.
        # Retain that evolving stream for training, while validation must be
        # invariant to worker count, access order, and previous samples.
        if self.training:
            return random
        return random.Random(self.seed + index * 1_000_003)

    @staticmethod
    def _final_phase(
        *,
        source_phase: tuple[int, int],
        source_origin: tuple[int, int],
        final_offset: tuple[int, int],
        block_size: int,
        hflip: bool,
    ) -> tuple[int, int]:
        final_left, final_top = final_offset
        final_x = source_origin[0] + final_left
        final_y = source_origin[1] + final_top
        phase_x = (source_phase[0] - final_x) % block_size
        phase_y = (source_phase[1] - final_y) % block_size
        if hflip:
            # A boundary at p in [0, W) becomes a left boundary at W-p.
            phase_x = (FINAL_CROP_SIZE - phase_x) % block_size
        return phase_x, phase_y

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | DataSample]:
        entry: V5NativeManifestEntry = self.entries[index]
        rng = self._rng_for_index(index)
        source_frames = self._decode(
            entry.target_video, entry.start_frame, "rgb24"
        )
        source_masks = self._decode(
            entry.mask_video, entry.start_frame, "gray"
        )
        origins = list(entry.origins)
        reliability = list(entry.mask_reliability)

        reversed_time = (
            self.training and self.time_reverse and rng.random() < 0.5
        )
        if reversed_time:
            source_frames.reverse()
            source_masks.reverse()
            origins.reverse()
            reliability.reverse()

        block_size = rng.randint(
            self.minimum_block_size, self.maximum_block_size
        )
        source_phase = (
            rng.randrange(block_size),
            rng.randrange(block_size),
        )
        hflip = self.training and self.use_hflip and rng.random() < 0.5

        target_512_frames: list[np.ndarray] = []
        mosaic_512_frames: list[np.ndarray] = []
        mask_512_frames: list[np.ndarray] = []
        for source, source_mask, origin in zip(
            source_frames, source_masks, origins, strict=True
        ):
            # Generate the degradation in full source coordinates.  Cropping
            # first would replicate the 512/256 edge and silently change the
            # block means used as supervision.
            full_mosaic = phase_block_average_mosaic(
                source,
                block_size=block_size,
                phase=source_phase,
            )
            target_512_frames.append(
                crop_native_frame(
                    source, origin=origin, size=SOURCE_CROP_SIZE
                )
            )
            mosaic_512_frames.append(
                crop_native_frame(
                    full_mosaic, origin=origin, size=SOURCE_CROP_SIZE
                )
            )
            mask_512_frames.append(
                crop_native_frame(
                    _single_channel_mask(source_mask),
                    origin=origin,
                    size=SOURCE_CROP_SIZE,
                    mask=True,
                )
            )

        final_top, final_left = _shared_roi_crop_offset(
            mask_512_frames,
            rng=rng,
            training=self.training,
        )
        targets: list[np.ndarray] = []
        inputs: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        phases: list[tuple[int, int]] = []
        for target_512, mosaic_512, mask_512, origin in zip(
            target_512_frames,
            mosaic_512_frames,
            mask_512_frames,
            origins,
            strict=True,
        ):
            target = _crop_at(
                target_512,
                top=final_top,
                left=final_left,
                size=FINAL_CROP_SIZE,
            )
            mosaic = _crop_at(
                mosaic_512,
                top=final_top,
                left=final_left,
                size=FINAL_CROP_SIZE,
            )
            mask = _crop_at(
                mask_512,
                top=final_top,
                left=final_left,
                size=FINAL_CROP_SIZE,
            )

            alpha = mask.astype(np.float32)[..., None]
            if alpha.size and float(alpha.max()) > 1.0:
                alpha /= 255.0
            observation = (
                target.astype(np.float32) * (1.0 - alpha)
                + mosaic.astype(np.float32) * alpha
            )
            observation = np.clip(np.rint(observation), 0, 255).astype(np.uint8)

            if hflip:
                target = np.ascontiguousarray(np.fliplr(target))
                observation = np.ascontiguousarray(np.fliplr(observation))
                mask = np.ascontiguousarray(np.fliplr(mask))
            targets.append(target)
            inputs.append(observation)
            masks.append(mask)
            phases.append(
                self._final_phase(
                    source_phase=source_phase,
                    source_origin=origin,
                    final_offset=(final_left, final_top),
                    block_size=block_size,
                    hflip=hflip,
                )
            )

        input_tensor = _rgb_uint8_tensor(inputs)
        target_tensor = _rgb_uint8_tensor(targets)
        mask_tensor = torch.stack(
            [_mask_tensor(mask) for mask in masks],
            dim=0,
        ).clamp_(0.0, 1.0)

        data_sample = DataSample(gt_img=target_tensor, mask=mask_tensor)
        data_sample.mosaic_phase = torch.tensor(phases, dtype=torch.int64)
        data_sample.mosaic_block_size = torch.tensor(
            block_size, dtype=torch.int64
        )
        # Explicit opt-in prevents exact observation consistency from being
        # applied accidentally if a future curriculum mixes approximate,
        # resized, compressed, or otherwise non-invertible degradations.
        data_sample.mosaic_observation_weight = torch.tensor(
            1.0, dtype=torch.float32
        )
        data_sample.mask_reliability = torch.tensor(
            reliability, dtype=torch.float32
        )
        data_sample.set_metainfo(
            {
                "img_channel_order": "RGB",
                "img_color_type": "color",
                "gt_channel_order": "RGB",
                "gt_color_type": "color",
                "gt_path": str(entry.target_video),
                "sample_idx": index,
                "num_input_frames": NUM_FRAMES,
                "num_output_frames": NUM_FRAMES,
                "name": entry.name,
                "source_video_id": entry.source_video_id,
                "native_source_crop_size": SOURCE_CROP_SIZE,
                "native_final_crop_size": FINAL_CROP_SIZE,
                "native_final_crop_offset": (final_left, final_top),
                "time_reversed": reversed_time,
                "hflip": hflip,
            }
        )
        return {"inputs": input_tensor, "data_samples": data_sample}
