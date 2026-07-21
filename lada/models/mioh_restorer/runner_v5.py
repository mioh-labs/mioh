# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Native ROI selection and cut-safe streaming utilities for V5."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .model_v4 import shift2d
from .model_v5 import CENTER_INDEX, NUM_INPUT_FRAMES, V5_BUCKETS


def round_to_even(value: float) -> int:
    """Round a coordinate to the nearest even source pixel."""

    return int(round(float(value) / 2.0) * 2)


def select_v5_bucket(
    roi_width: float,
    roi_height: float,
    *,
    context_fraction: float = 0.30,
) -> int:
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI dimensions must be positive")
    if context_fraction < 0:
        raise ValueError("context fraction must be non-negative")
    required = max(roi_width, roi_height) * (1.0 + 2.0 * context_fraction)
    for bucket in V5_BUCKETS:
        if required <= bucket:
            return bucket
    return V5_BUCKETS[-1]


def required_v5_crop_size(
    roi_width: float,
    roi_height: float,
    *,
    context_fraction: float = 0.30,
) -> float:
    """Return the native-pixel square extent required by an ROI.

    Unlike :func:`select_v5_bucket`, this does not clamp at 512 and is useful
    when building a training manifest: windows that do not fit a single native
    bucket can be rejected instead of silently clipping their supervision.
    """

    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI dimensions must be positive")
    if context_fraction < 0:
        raise ValueError("context fraction must be non-negative")
    return max(roi_width, roi_height) * (1.0 + 2.0 * context_fraction)


def smooth_even_centers(
    centers: list[tuple[float, float] | None],
) -> tuple[tuple[int, int], ...]:
    """Interpolate missing detections and smooth a crop trajectory.

    The returned coordinates are quantized to even source pixels so the four
    PixelUnshuffle phases remain stable from frame to frame.  No image
    resampling is performed.
    """

    if not centers or all(value is None for value in centers):
        raise ValueError("at least one detected centre is required")
    valid = [index for index, value in enumerate(centers) if value is not None]
    interpolated: list[tuple[float, float]] = []
    for index, value in enumerate(centers):
        if value is not None:
            interpolated.append(value)
            continue
        left = max((candidate for candidate in valid if candidate < index), default=None)
        right = min((candidate for candidate in valid if candidate > index), default=None)
        if left is None:
            assert right is not None
            interpolated.append(centers[right])  # type: ignore[arg-type]
        elif right is None:
            interpolated.append(centers[left])  # type: ignore[arg-type]
        else:
            fraction = (index - left) / (right - left)
            left_value = centers[left]
            right_value = centers[right]
            assert left_value is not None and right_value is not None
            interpolated.append(
                (
                    left_value[0] + (right_value[0] - left_value[0]) * fraction,
                    left_value[1] + (right_value[1] - left_value[1]) * fraction,
                )
            )
    weights = (1.0, 2.0, 3.0, 2.0, 1.0)
    smoothed: list[tuple[int, int]] = []
    for index in range(len(interpolated)):
        total = 0.0
        horizontal = 0.0
        vertical = 0.0
        for offset, weight in zip(range(-2, 3), weights, strict=True):
            source = min(max(index + offset, 0), len(interpolated) - 1)
            horizontal += interpolated[source][0] * weight
            vertical += interpolated[source][1] * weight
            total += weight
        smoothed.append(
            (round_to_even(horizontal / total), round_to_even(vertical / total))
        )
    return tuple(smoothed)


def native_tile_offsets(
    roi_width: float,
    roi_height: float,
    *,
    bucket: int,
    context_fraction: float = 0.30,
    overlap: int = 64,
) -> tuple[tuple[int, int], ...]:
    """Return even top-left offsets relative to a tracked ROI centre.

    Small regions produce one centred tile.  Regions larger than the maximum
    native bucket produce an overlapping 512 grid instead of being resized or
    discarded.  These offsets can be added to each frame's smoothed centre so
    every grid tile follows the tracked subject.
    """

    if bucket not in V5_BUCKETS:
        raise ValueError(f"unsupported V5 bucket: {bucket}")
    if not 0 <= overlap < bucket:
        raise ValueError("tile overlap must be smaller than the bucket")
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI dimensions must be positive")
    width = roi_width * (1.0 + 2.0 * context_fraction)
    height = roi_height * (1.0 + 2.0 * context_fraction)

    def positions(extent: float) -> tuple[int, ...]:
        if extent <= bucket:
            return (round_to_even(-bucket / 2),)
        count = math.ceil((extent - bucket) / (bucket - overlap)) + 1
        left = -extent / 2
        right = extent / 2 - bucket
        return tuple(
            round_to_even(left + (right - left) * index / (count - 1))
            for index in range(count)
        )

    return tuple((x, y) for y in positions(height) for x in positions(width))


@dataclass(frozen=True)
class NativeCrop:
    """Even-origin square crop plus source-edge padding."""

    x: int
    y: int
    size: int
    pad_left: int
    pad_right: int
    pad_top: int
    pad_bottom: int


def native_crop_for_center(
    center_x: float,
    center_y: float,
    *,
    size: int,
    source_width: int,
    source_height: int,
) -> NativeCrop:
    if size not in V5_BUCKETS:
        raise ValueError(f"unsupported V5 bucket: {size}")
    if source_width <= 0 or source_height <= 0:
        raise ValueError("source dimensions must be positive")
    x = round_to_even(center_x - size / 2)
    y = round_to_even(center_y - size / 2)
    source_right = x + size
    source_bottom = y + size
    return NativeCrop(
        x=max(x, 0),
        y=max(y, 0),
        size=size,
        pad_left=max(-x, 0),
        pad_right=max(source_right - source_width, 0),
        pad_top=max(-y, 0),
        pad_bottom=max(source_bottom - source_height, 0),
    )


class V5BucketHysteresis:
    """Expand immediately; contract only after sustained spare capacity."""

    def __init__(self, initial_bucket: int, *, contraction_frames: int = 18) -> None:
        if initial_bucket not in V5_BUCKETS:
            raise ValueError("invalid initial V5 bucket")
        if contraction_frames < NUM_INPUT_FRAMES:
            raise ValueError("contraction delay must cover at least one window")
        self.bucket = initial_bucket
        self.contraction_frames = int(contraction_frames)
        self._smaller_candidate: int | None = None
        self._smaller_frames = 0

    def update(
        self,
        required_width: float,
        required_height: float,
        *,
        at_window_boundary: bool,
        context_fraction: float = 0.30,
    ) -> int:
        requested = select_v5_bucket(
            required_width,
            required_height,
            context_fraction=context_fraction,
        )
        if requested > self.bucket:
            self._smaller_candidate = None
            self._smaller_frames = 0
            if at_window_boundary:
                self.bucket = requested
            return self.bucket
        if requested == self.bucket:
            self._smaller_candidate = None
            self._smaller_frames = 0
            return self.bucket
        if requested != self._smaller_candidate:
            self._smaller_candidate = requested
            self._smaller_frames = 1
        else:
            self._smaller_frames += 1
        if (
            at_window_boundary
            and self._smaller_frames >= self.contraction_frames
        ):
            self.bucket = requested
            self._smaller_candidate = None
            self._smaller_frames = 0
        return self.bucket


def cut_safe_window_indices(
    center: int,
    *,
    frame_count: int,
    cut_starts: tuple[int, ...] = (),
) -> tuple[int, ...]:
    """Return nine indices without crossing a scene boundary.

    ``cut_starts`` contains the first frame of each new scene. Missing frames
    at either edge replicate the nearest valid frame.
    """

    if frame_count <= 0 or center < 0 or center >= frame_count:
        raise ValueError("invalid video frame range")
    boundaries = (0,) + tuple(
        sorted({value for value in cut_starts if 0 < value < frame_count})
    ) + (frame_count,)
    segment_start = 0
    segment_end = frame_count
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if start <= center < end:
            segment_start, segment_end = start, end
            break
    radius = NUM_INPUT_FRAMES // 2
    return tuple(
        min(max(center + offset, segment_start), segment_end - 1)
        for offset in range(-radius, radius + 1)
    )


def repair_isolated_mask_misses(
    masks: torch.Tensor,
    direct_reliability: torch.Tensor,
    crop_origins: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill missed masks from nearest direct neighbours in source coordinates.

    This deliberately avoids a raw temporal union. Direct masks remain
    untouched; only unreliable frames are repaired and marked with reduced
    reliability.
    """

    if masks.ndim != 5 or masks.shape[2] != 1:
        raise ValueError("masks must be [B,T,1,H,W]")
    if direct_reliability.shape != masks.shape[:2] + (1, 1, 1):
        raise ValueError("direct reliability must be [B,T,1,1,1]")
    if crop_origins.shape != masks.shape[:2] + (2,):
        raise ValueError("crop origins must be [B,T,2]")
    repaired = masks.clone()
    reliability = direct_reliability.expand_as(masks).clone()
    batch, frames = masks.shape[:2]
    for batch_index in range(batch):
        valid = [
            index
            for index in range(frames)
            if float(direct_reliability[batch_index, index].item()) >= 0.5
        ]
        for index in range(frames):
            if index in valid or not valid:
                continue
            nearest = min(valid, key=lambda candidate: abs(candidate - index))
            source_x, source_y = crop_origins[batch_index, nearest]
            target_x, target_y = crop_origins[batch_index, index]
            horizontal = int(round(float(source_x - target_x)))
            vertical = int(round(float(source_y - target_y)))
            repaired[batch_index : batch_index + 1, index] = shift2d(
                masks[batch_index : batch_index + 1, nearest],
                vertical,
                horizontal,
            )
            reliability[batch_index : batch_index + 1, index].fill_(0.5)
    return repaired.clamp_(0.0, 1.0), reliability.clamp_(0.0, 1.0)


class MiohRestorerV5StreamingRunner:
    """Cut-safe center-output runner used before the feature-cache backend."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def restore(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor,
        mask_reliability: torch.Tensor,
        *,
        cut_starts: tuple[int, ...] = (),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must be [B,T,3,H,W]")
        expected = frames.shape[:2] + (1,) + frames.shape[-2:]
        if masks.shape != expected or mask_reliability.shape != expected:
            raise ValueError("V5 masks and reliability must match video frames")
        outputs: list[torch.Tensor] = []
        confidences: list[torch.Tensor] = []
        for center in range(frames.shape[1]):
            indices = cut_safe_window_indices(
                center,
                frame_count=frames.shape[1],
                cut_starts=cut_starts,
            )
            values = torch.stack(
                [
                    torch.cat(
                        (
                            frames[:, index],
                            masks[:, index],
                            mask_reliability[:, index],
                        ),
                        dim=1,
                    )
                    for index in indices
                ],
                dim=1,
            )
            restored, confidence = self.model(values)
            if restored.shape[1] != 1:
                raise ValueError("streaming V5 runner requires a center-output model")
            outputs.append(restored[:, 0])
            confidences.append(confidence[:, 0])
        return torch.stack(outputs, dim=1), torch.stack(confidences, dim=1)
