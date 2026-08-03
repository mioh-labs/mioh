# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Known-phase native mosaic data for the Native-HF prototype."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from lada.utils.mask_utils import stabilize_temporal_mask_tensor

from .model_native_hf import NATIVE_HF_GUIDE_FRAMES
from .native_dataset_v5 import (
    V5NativeManifestEntry,
    _rgb_tensor,
    create_native_degradation_pipeline,
    crop_native_frame,
    decode_native_frames,
    read_v5_native_manifest,
)


def phase_block_average_mosaic(
    image: np.ndarray,
    *,
    block_size: int,
    phase: tuple[int, int],
) -> np.ndarray:
    """Apply exact block averaging on a grid with a known x/y phase.

    ``phase`` is the coordinate of a block boundary modulo ``block_size`` in
    crop coordinates.  Edge replication makes the forward operator defined
    even when a crop starts in the middle of a block.
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
    cropped = expanded[pad_top : pad_top + height, pad_left : pad_left + width]
    return np.clip(np.rint(cropped), 0, 255).astype(np.uint8)


def composite_known_mosaic(
    target: np.ndarray,
    mask: np.ndarray,
    *,
    block_size: int,
    phase: tuple[int, int],
) -> np.ndarray:
    mosaic = phase_block_average_mosaic(
        target, block_size=block_size, phase=phase
    ).astype(np.float32)
    alpha = mask.astype(np.float32)
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    if alpha.size and float(alpha.max()) > 1.0:
        alpha /= 255.0
    result = target.astype(np.float32) * (1.0 - alpha) + mosaic * alpha
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def recentered_origins(
    entry: V5NativeManifestEntry, *, native_size: int
) -> tuple[tuple[int, int], ...]:
    """Expand an old smaller native crop without any image resampling."""

    if native_size < entry.bucket:
        raise ValueError("Native-HF cannot shrink a manifest crop")
    adjustment = (native_size - entry.bucket) // 2
    if (native_size - entry.bucket) % 2:
        raise ValueError("manifest and Native-HF sizes must share an even centre")
    return tuple((x - adjustment, y - adjustment) for x, y in entry.origins)


class MiohNativeHF512Dataset(Dataset):
    """Return nine native frames plus exact mosaic forward-model metadata."""

    def __init__(
        self,
        manifest: Path,
        *,
        native_size: int = 512,
        output_indices: tuple[int, ...] = (4,),
        degrade: bool = True,
        time_reverse: bool = True,
        deterministic: bool = False,
        minimum_block_size: int = 6,
        maximum_block_size: int = 48,
        block_size_sampling: str = "manifest",
        seed: int = 0,
    ) -> None:
        self.entries = read_v5_native_manifest(Path(manifest))
        if native_size <= 0 or native_size % 8:
            raise ValueError("Native-HF size must be positive and divisible by eight")
        if not output_indices or any(
            index < 0 or index >= NATIVE_HF_GUIDE_FRAMES for index in output_indices
        ):
            raise ValueError("invalid Native-HF target indices")
        if minimum_block_size <= 1 or maximum_block_size < minimum_block_size:
            raise ValueError("invalid Native-HF block-size range")
        if block_size_sampling not in ("manifest", "uniform"):
            raise ValueError("invalid Native-HF block-size sampling mode")
        if any(entry.bucket > native_size for entry in self.entries):
            raise ValueError("manifest contains crops larger than Native-HF size")
        self.native_size = int(native_size)
        self.output_indices = output_indices
        self.degrade = bool(degrade)
        self.time_reverse = bool(time_reverse)
        self.deterministic = bool(deterministic)
        self.minimum_block_size = int(minimum_block_size)
        self.maximum_block_size = int(maximum_block_size)
        self.block_size_sampling = block_size_sampling
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.entries)

    @staticmethod
    def _decode(path: Path, start: int, pixel_format: str) -> list[np.ndarray]:
        frames = decode_native_frames(
            path,
            start=start,
            count=NATIVE_HF_GUIDE_FRAMES,
            pixel_format=pixel_format,
        )
        if len(frames) != NATIVE_HF_GUIDE_FRAMES:
            raise RuntimeError(
                f"decoded {len(frames)} frames from {path}; expected nine"
            )
        return frames

    @staticmethod
    def _mask_tensor(masks: Sequence[np.ndarray]) -> torch.Tensor:
        tensor = torch.stack(
            [
                torch.from_numpy(np.ascontiguousarray(mask)).float().div_(255.0).unsqueeze(0)
                for mask in masks
            ]
        ).clamp_(0.0, 1.0)
        return stabilize_temporal_mask_tensor(tensor)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        entry = self.entries[index]
        sample_seed = (
            index
            if self.deterministic
            else self.seed + self.epoch * 1_000_003 + index
        )
        rng = random.Random(sample_seed)
        source_frames = self._decode(entry.target_video, entry.start_frame, "rgb24")
        source_masks = self._decode(entry.mask_video, entry.start_frame, "gray")
        origins = recentered_origins(entry, native_size=self.native_size)
        reliability = list(entry.mask_reliability)
        if self.time_reverse and rng.random() < 0.5:
            source_frames.reverse()
            source_masks.reverse()
            origins = tuple(reversed(origins))
            reliability.reverse()

        targets = [
            crop_native_frame(frame, origin=origin, size=self.native_size)
            for frame, origin in zip(source_frames, origins, strict=True)
        ]
        masks = [
            crop_native_frame(
                frame[..., 0] if frame.ndim == 3 else frame,
                origin=origin,
                size=self.native_size,
                mask=True,
            )
            for frame, origin in zip(source_masks, origins, strict=True)
        ]

        mask_tensor = self._mask_tensor(masks)
        stabilized_masks = [
            np.clip(np.rint(value[0].numpy() * 255.0), 0, 255).astype(np.uint8)
            for value in mask_tensor
        ]
        if self.block_size_sampling == "uniform":
            # Stage-one curriculum: do not inherit the representative set's
            # overwhelmingly large-block distribution.  Uniform sampling in
            # the explicitly bounded range first teaches a recoverable native
            # HF mapping; later stages return to the deployment distribution.
            block_size = rng.randint(
                self.minimum_block_size, self.maximum_block_size
            )
        else:
            base_block = max(
                self.minimum_block_size,
                min(self.maximum_block_size, int(round(entry.mosaic_block_size))),
            )
            block_size = max(
                self.minimum_block_size,
                min(
                    self.maximum_block_size,
                    int(round(base_block * rng.uniform(0.75, 1.25))),
                ),
            )
        global_phase = (rng.randrange(block_size), rng.randrange(block_size))
        phases = tuple(
            (
                (global_phase[0] - origin[0]) % block_size,
                (global_phase[1] - origin[1]) % block_size,
            )
            for origin in origins
        )
        # Build the block grid before cropping so boundary cells use real
        # neighbouring source pixels instead of replicated crop-edge pixels.
        full_mosaics = [
            phase_block_average_mosaic(
                frame,
                block_size=block_size,
                phase=global_phase,
            )
            for frame in source_frames
        ]
        mosaic_crops = [
            crop_native_frame(frame, origin=origin, size=self.native_size)
            for frame, origin in zip(full_mosaics, origins, strict=True)
        ]
        observations = []
        for target, mosaic, mask in zip(
            targets, mosaic_crops, stabilized_masks, strict=True
        ):
            alpha = mask.astype(np.float32)[..., None] / 255.0
            composited = target.astype(np.float32) * (1.0 - alpha) + mosaic * alpha
            observations.append(
                np.clip(np.rint(composited), 0, 255).astype(np.uint8)
            )
        model_inputs = observations
        profile = "clean"
        if self.degrade:
            profile = rng.choices(
                ("clean", "mild", "full"), weights=(0.35, 0.50, 0.15), k=1
            )[0]
            # Existing degradation transforms draw from NumPy's module-level
            # RNG.  Scope that state to (seed, epoch, sample) so a resumed run
            # reproduces the exact augmentation without replaying old batches.
            numpy_state = np.random.get_state()
            try:
                np.random.seed(sample_seed % (2**32))
                model_inputs = create_native_degradation_pipeline(profile)(
                    model_inputs
                )
            finally:
                np.random.set_state(numpy_state)

        input_rgb = _rgb_tensor(model_inputs)
        target_rgb = _rgb_tensor(targets)
        observation_rgb = _rgb_tensor(observations)
        reliability_tensor = torch.tensor(
            reliability, dtype=torch.float32
        ).reshape(-1, 1, 1, 1).expand_as(mask_tensor)
        native_values = torch.cat(
            (input_rgb, mask_tensor, reliability_tensor), dim=1
        )
        selected = list(self.output_indices)
        return {
            "native_inputs": native_values,
            "targets": target_rgb[selected],
            "masks": mask_tensor[selected],
            "mosaic_observations": observation_rgb[selected],
            "mosaic_phases": torch.tensor(
                [phases[position] for position in selected], dtype=torch.int64
            ),
            "mosaic_block_size": torch.tensor(block_size, dtype=torch.int64),
            "observation_weight": torch.tensor(
                1.0 if profile == "clean" else 0.0, dtype=torch.float32
            ),
            "degradation_profile": profile,
            "bucket": self.native_size,
            "name": entry.name,
            "source_video_id": entry.source_video_id,
        }
