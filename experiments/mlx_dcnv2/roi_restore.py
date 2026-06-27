"""ROI crop/restore/composite helpers for the LADA MLX port."""

from __future__ import annotations

from collections.abc import Callable

import mlx.core as mx
import numpy as np

from .sequence import lada_sequence_forward


BBox = tuple[int, int, int, int]
FrameRange = tuple[int, int]


def bbox_from_masks(
    masks: np.ndarray,
    *,
    expansion_ratio: float = 0.0,
    align_multiple: int = 32,
) -> BBox | None:
    """Return `(x0, y0, x1, y1)` covering all non-zero mask pixels."""

    if masks.ndim != 3:
        raise ValueError(f"masks must be THW, got {masks.shape}")
    _, height, width = masks.shape
    _, ys, xs = np.nonzero(masks > 0)
    if len(xs) == 0:
        return None

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    box_w = x1 - x0
    box_h = y1 - y0
    pad_x = int(np.ceil(box_w * expansion_ratio))
    pad_y = int(np.ceil(box_h * expansion_ratio))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(width, x1 + pad_x)
    y1 = min(height, y1 + pad_y)

    if align_multiple > 1:
        x0 = _floor_to_multiple(x0, align_multiple)
        y0 = _floor_to_multiple(y0, align_multiple)
        x1 = min(width, _ceil_to_multiple(x1, align_multiple))
        y1 = min(height, _ceil_to_multiple(y1, align_multiple))
    return x0, y0, x1, y1


def roi_area_from_masks(
    masks: np.ndarray,
    *,
    expansion_ratio: float = 0.0,
    align_multiple: int = 1,
) -> int:
    bbox = bbox_from_masks(masks, expansion_ratio=expansion_ratio, align_multiple=align_multiple)
    if bbox is None:
        return 0
    x0, y0, x1, y1 = bbox
    return max(0, x1 - x0) * max(0, y1 - y0)


def split_ranges_by_max_roi_area(
    masks: np.ndarray,
    *,
    max_roi_area: int | None,
    expansion_ratio: float = 0.0,
    align_multiple: int = 1,
) -> list[FrameRange]:
    """Split a THW mask window so each subwindow's union ROI stays bounded."""

    if masks.ndim != 3:
        raise ValueError(f"masks must be THW, got {masks.shape}")
    frame_count = masks.shape[0]
    if frame_count == 0:
        return []
    if max_roi_area is None or max_roi_area <= 0:
        return [(0, frame_count)]

    ranges: list[FrameRange] = []
    start = 0
    while start < frame_count:
        end = start + 1
        while end < frame_count:
            candidate_area = roi_area_from_masks(
                masks[start : end + 1],
                expansion_ratio=expansion_ratio,
                align_multiple=align_multiple,
            )
            if candidate_area > max_roi_area and roi_area_from_masks(masks[start:end]) > 0:
                break
            if candidate_area > max_roi_area and end > start:
                break
            end += 1
        ranges.append((start, end))
        start = end
    return _merge_single_frame_ranges(ranges)


def _merge_single_frame_ranges(ranges: list[FrameRange]) -> list[FrameRange]:
    merged: list[FrameRange] = []
    idx = 0
    while idx < len(ranges):
        start, end = ranges[idx]
        if end - start != 1:
            merged.append((start, end))
            idx += 1
            continue
        if merged:
            prev_start, _ = merged.pop()
            merged.append((prev_start, end))
            idx += 1
            continue
        if idx + 1 < len(ranges):
            _, next_end = ranges[idx + 1]
            merged.append((start, next_end))
            idx += 2
            continue
        merged.append((start, end))
        idx += 1
    return merged


def restore_masked_roi_sequence(
    frames: mx.array,
    masks: mx.array,
    restore_roi: Callable[[mx.array], mx.array],
    *,
    expansion_ratio: float = 0.0,
    align_multiple: int = 32,
) -> mx.array:
    """Restore one masked ROI sequence and composite it back into frames.

    `frames` is `TCHW`, `masks` is `THW`, both float arrays. The restore
    callable receives the cropped `TCHW` ROI and must return the same shape.
    """

    masks_np = np.array(masks)
    bbox = bbox_from_masks(masks_np, expansion_ratio=expansion_ratio, align_multiple=align_multiple)
    if bbox is None:
        return frames
    x0, y0, x1, y1 = bbox
    roi_frames = frames[:, :, y0:y1, x0:x1]
    restored_roi = restore_roi(roi_frames)
    roi_mask = masks[:, y0:y1, x0:x1]
    roi_mask = mx.expand_dims(roi_mask, axis=1)

    before_y = frames[:, :, :y0, :]
    after_y = frames[:, :, y1:, :]
    middle = frames[:, :, y0:y1, :]
    before_x = middle[:, :, :, :x0]
    after_x = middle[:, :, :, x1:]
    original_roi = middle[:, :, :, x0:x1]
    composited_roi = restored_roi * roi_mask + original_roi * (1.0 - roi_mask)
    middle = mx.concatenate([before_x, composited_roi, after_x], axis=3)
    return mx.concatenate([before_y, middle, after_y], axis=2)


def restore_masked_roi_sequence_with_lada(
    frames: mx.array,
    masks: mx.array,
    bundle: dict[str, object],
    *,
    sequence_forward: Callable[[mx.array, dict[str, object]], mx.array] = lada_sequence_forward,
    expansion_ratio: float = 0.0,
    align_multiple: int = 32,
) -> mx.array:
    """Restore masked TCHW frames through the LADA MLX sequence path."""

    def restore_roi(roi_frames: mx.array) -> mx.array:
        batch_roi = mx.expand_dims(roi_frames, axis=0)
        restored = sequence_forward(batch_roi, bundle)
        return restored[0]

    return restore_masked_roi_sequence(
        frames,
        masks,
        restore_roi,
        expansion_ratio=expansion_ratio,
        align_multiple=align_multiple,
    )


def _floor_to_multiple(value: int, multiple: int) -> int:
    return (value // multiple) * multiple


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
