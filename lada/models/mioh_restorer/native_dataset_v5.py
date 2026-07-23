# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Native-resolution manifest dataset for MiohRestorer V5.

V5 deliberately does not reuse ``MiohRestorationDataset`` because that loader
always resizes every sample to one training size.  This module only crops and
pads source pixels.  A 256 bucket therefore always contains a native 256x256
crop, never an enlarged or downscaled image.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import cv2
import numpy as np
import torch
import av
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import transforms as torchvision_transforms

from lada.utils import transforms as lada_transforms
from lada.utils.mask_utils import stabilize_temporal_mask_tensor
from lada.utils.mosaic_utils import addmosaic_base, get_random_parameters_by_block_size

from .model_v5 import FRAME_CHANNELS, NUM_INPUT_FRAMES, V5_BUCKETS


def decode_native_frames(
    path: Path,
    *,
    start: int = 0,
    count: int | None = None,
    pixel_format: str = "rgb24",
) -> list[np.ndarray]:
    """Decode exact source frames with no spatial conversion or resizing.

    PyAV is used here because OpenCV can spin for minutes at EOF on some FFV1
    mask clips.  The manifest clips are short, so decoding from frame zero is
    deterministic and avoids timestamp/keyframe ambiguity.
    """

    frames: list[np.ndarray] = []
    with av.open(str(path), mode="r") as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index < start:
                continue
            frames.append(frame.to_ndarray(format=pixel_format))
            if count is not None and len(frames) >= count:
                break
    return frames


@dataclass(frozen=True)
class V5NativeManifestEntry:
    name: str
    target_video: Path
    mask_video: Path
    start_frame: int
    bucket: int
    origins: tuple[tuple[int, int], ...]
    mask_reliability: tuple[float, ...]
    mosaic_block_size: float
    source_video_id: str

    @classmethod
    def from_dict(cls, value: dict[str, object], root: Path) -> "V5NativeManifestEntry":
        def resolve(path_value: object) -> Path:
            path = Path(str(path_value))
            return path if path.is_absolute() else root / path

        origins = tuple((int(pair[0]), int(pair[1])) for pair in value["origins"])  # type: ignore[index]
        reliability = tuple(float(item) for item in value["mask_reliability"])  # type: ignore[arg-type]
        entry = cls(
            name=str(value["name"]),
            target_video=resolve(value["target_video"]),
            mask_video=resolve(value["mask_video"]),
            start_frame=int(value["start_frame"]),
            bucket=int(value["bucket"]),
            origins=origins,
            mask_reliability=reliability,
            mosaic_block_size=float(value["mosaic_block_size"]),
            source_video_id=str(value["source_video_id"]),
        )
        entry.validate()
        return entry

    def validate(self) -> None:
        if self.bucket not in V5_BUCKETS:
            raise ValueError(f"unsupported V5 bucket: {self.bucket}")
        if self.start_frame < 0:
            raise ValueError("start_frame must be non-negative")
        if len(self.origins) != NUM_INPUT_FRAMES:
            raise ValueError("a V5 manifest entry requires nine crop origins")
        if len(self.mask_reliability) != NUM_INPUT_FRAMES:
            raise ValueError("a V5 manifest entry requires nine reliability values")
        if any(x % 2 or y % 2 for x, y in self.origins):
            raise ValueError("V5 crop origins must use even source pixels")
        if any(value < 0 or value > 1 for value in self.mask_reliability):
            raise ValueError("mask reliability must be in [0, 1]")
        if self.mosaic_block_size <= 0:
            raise ValueError("mosaic block size must be positive")


def read_v5_native_manifest(path: Path) -> list[V5NativeManifestEntry]:
    entries: list[V5NativeManifestEntry] = []
    root = path.parent
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                entries.append(V5NativeManifestEntry.from_dict(value, root))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid V5 manifest line {line_number}: {error}") from error
    if not entries:
        raise ValueError(f"V5 manifest contains no samples: {path}")
    return entries


def crop_native_frame(
    frame: np.ndarray,
    *,
    origin: tuple[int, int],
    size: int,
    mask: bool = False,
) -> np.ndarray:
    """Crop without resampling; replicate RGB edges and zero-pad masks."""

    if size not in V5_BUCKETS:
        raise ValueError(f"unsupported V5 bucket: {size}")
    x, y = origin
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("cannot crop an empty native frame")
    if mask:
        cropped = np.zeros((size, size, *frame.shape[2:]), dtype=frame.dtype)
        source_x0 = max(x, 0)
        source_y0 = max(y, 0)
        source_x1 = min(x + size, width)
        source_y1 = min(y + size, height)
        if source_x1 > source_x0 and source_y1 > source_y0:
            destination_x0 = source_x0 - x
            destination_y0 = source_y0 - y
            destination_x1 = destination_x0 + source_x1 - source_x0
            destination_y1 = destination_y0 + source_y1 - source_y0
            cropped[destination_y0:destination_y1, destination_x0:destination_x1] = (
                frame[source_y0:source_y1, source_x0:source_x1]
            )
    else:
        # Clipped integer index grids are exactly BORDER_REPLICATE, including
        # crops whose requested rectangle lies entirely outside the frame.
        horizontal = np.clip(np.arange(x, x + size), 0, width - 1)
        vertical = np.clip(np.arange(y, y + size), 0, height - 1)
        cropped = frame[vertical[:, None], horizontal[None, :]]
    if cropped.shape[:2] != (size, size):
        raise RuntimeError(
            f"native crop produced {cropped.shape[:2]}; expected {(size, size)}"
        )
    return np.ascontiguousarray(cropped)


def create_native_degradation_pipeline(profile: str):
    """Build the existing degradation mix without a resize operation."""

    if profile == "clean":
        return torchvision_transforms.Compose([])
    if profile == "mild":
        return torchvision_transforms.Compose(
            [
                lada_transforms.VideoCompression(
                    p=0.65,
                    codecs=["libx264", "libx265", "libvpx-vp9"],
                    codec_probs=[0.45, 0.35, 0.20],
                    crf_ranges={"libx264": (14, 22), "libx265": (18, 28)},
                    bitrate_ranges={"libvpx-vp9": (10_000, 22_000)},
                ),
                lada_transforms.GaussianBlur(sigma_range=[0.5, 1.5], p=0.15),
                lada_transforms.GaussianNoise(snr=60, p=0.10),
            ]
        )
    if profile != "full":
        raise ValueError(f"unknown degradation profile: {profile}")
    return torchvision_transforms.Compose(
        [
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
        ]
    )


def _rgb_tensor(frames: Sequence[np.ndarray]) -> torch.Tensor:
    return torch.stack(
        [
            torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))
            .float()
            .div_(255.0)
            for frame in frames
        ]
    )


class MiohRestorerV5NativeDataset(Dataset):
    """Read exact native windows described by a V5 JSONL manifest."""

    def __init__(
        self,
        manifest: Path,
        *,
        output_indices: tuple[int, ...] = (4,),
        degrade: bool = True,
        horizontal_flip: bool = True,
        time_reverse: bool = True,
        deterministic: bool = False,
    ) -> None:
        self.entries = read_v5_native_manifest(Path(manifest))
        if not output_indices or any(index < 0 or index >= NUM_INPUT_FRAMES for index in output_indices):
            raise ValueError("invalid V5 output indices")
        self.output_indices = output_indices
        self.degrade = degrade
        self.horizontal_flip = horizontal_flip
        self.time_reverse = time_reverse
        self.deterministic = deterministic

    def __len__(self) -> int:
        return len(self.entries)

    def bucket_for_index(self, index: int) -> int:
        return self.entries[index].bucket

    @staticmethod
    def _read_frames(path: Path, start: int, pixel_format: str) -> list[np.ndarray]:
        frames = decode_native_frames(
            path,
            start=start,
            count=NUM_INPUT_FRAMES,
            pixel_format=pixel_format,
        )
        if len(frames) != NUM_INPUT_FRAMES:
            raise RuntimeError(f"decoded {len(frames)} frames from {path}; expected 9")
        return frames

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        entry = self.entries[index]
        rng = random.Random(index) if self.deterministic else random
        targets = self._read_frames(entry.target_video, entry.start_frame, "rgb24")
        masks = self._read_frames(entry.mask_video, entry.start_frame, "gray")
        targets = [
            crop_native_frame(frame, origin=origin, size=entry.bucket)
            for frame, origin in zip(targets, entry.origins, strict=True)
        ]
        native_masks = []
        for frame, origin in zip(masks, entry.origins, strict=True):
            if frame.ndim == 3:
                frame = frame[..., 0]
            native_masks.append(
                crop_native_frame(frame, origin=origin, size=entry.bucket, mask=True)
            )

        if self.deterministic:
            # The mosaic utility uses the module-level Python and NumPy RNGs,
            # including one direct ``random.uniform`` call for feathering.
            # Preserve both states so validation sample N is byte-identical on
            # every pass without perturbing training augmentation randomness.
            python_state = random.getstate()
            numpy_state = np.random.get_state()
            random.seed(index)
            np.random.seed(index % (2**32))
        try:
            mosaic_size, mosaic_mode, rectangle_ratio, feather_size = (
                get_random_parameters_by_block_size(
                    entry.mosaic_block_size,
                    randomize_size=True,
                    repeatable_random=False,
                )
            )
        finally:
            if self.deterministic:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
        inputs: list[np.ndarray] = []
        affected_masks: list[np.ndarray] = []
        for target, source_mask in zip(targets, native_masks, strict=True):
            if source_mask.ndim == 2:
                source_mask = source_mask[..., None]
            mosaic, affected = addmosaic_base(
                target,
                source_mask,
                mosaic_size,
                model=mosaic_mode,
                rect_ratio=rectangle_ratio,
                feather=feather_size,
            )
            inputs.append(mosaic)
            affected_masks.append(affected[..., 0] if affected.ndim == 3 else affected)

        if self.degrade:
            profile = rng.choices(("clean", "mild", "full"), weights=(0.20, 0.50, 0.30), k=1)[0]
            inputs = create_native_degradation_pipeline(profile)(inputs)
        if self.time_reverse and rng.random() < 0.5:
            inputs.reverse()
            targets.reverse()
            affected_masks.reverse()
            reliability = tuple(reversed(entry.mask_reliability))
        else:
            reliability = entry.mask_reliability
        if self.horizontal_flip and rng.random() < 0.5:
            inputs = [np.ascontiguousarray(np.fliplr(value)) for value in inputs]
            targets = [np.ascontiguousarray(np.fliplr(value)) for value in targets]
            affected_masks = [np.ascontiguousarray(np.fliplr(value)) for value in affected_masks]

        input_rgb = _rgb_tensor(inputs)
        target_rgb = _rgb_tensor(targets)
        mask_tensor = torch.stack(
            [torch.from_numpy(value).float().div_(255.0).unsqueeze(0) for value in affected_masks]
        ).clamp_(0.0, 1.0)
        # Match the production V5 compositor: retain neighbouring detections
        # softly and include a small guard band. The added clean context ring
        # has an identity target, teaching the model a stable transition
        # instead of a frame-varying hard edge.
        mask_tensor = stabilize_temporal_mask_tensor(mask_tensor)
        reliability_tensor = torch.tensor(reliability, dtype=torch.float32).view(-1, 1, 1, 1)
        reliability_tensor = reliability_tensor.expand_as(mask_tensor)
        values = torch.cat((input_rgb, mask_tensor, reliability_tensor), dim=1)
        if values.shape[1] != FRAME_CHANNELS:
            raise RuntimeError("invalid V5 frame channel count")
        return {
            "inputs": values,
            "targets": target_rgb[list(self.output_indices)],
            "masks": mask_tensor[list(self.output_indices)],
            "bucket": entry.bucket,
            "name": entry.name,
            "source_video_id": entry.source_video_id,
        }


class V5BucketBatchSampler(Sampler[list[int]]):
    """Yield homogeneous native-size batches without resizing any sample."""

    def __init__(
        self,
        dataset: MiohRestorerV5NativeDataset,
        *,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        by_bucket: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.dataset)):
            by_bucket[self.dataset.bucket_for_index(index)].append(index)
        batches: list[list[int]] = []
        for indices in by_bucket.values():
            if self.shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        counts: dict[int, int] = defaultdict(int)
        for index in range(len(self.dataset)):
            counts[self.dataset.bucket_for_index(index)] += 1
        if self.drop_last:
            return sum(value // self.batch_size for value in counts.values())
        return sum((value + self.batch_size - 1) // self.batch_size for value in counts.values())
