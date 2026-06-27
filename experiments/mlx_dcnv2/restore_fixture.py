"""Fixture runners for the MLX LADA restore path."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import time

import cv2
import mlx.core as mx
import numpy as np

from .media_io import (
    _bgr_hwc_uint8_to_tchw_float,
    _tchw_float_to_bgr_hwc_uint8,
    mux_audio_from_source_video,
    read_image_sequence_tchw,
    read_video_tchw,
    write_image_sequence_tchw,
    write_video_tchw,
)
from .roi_restore import restore_masked_roi_sequence_with_lada
from .roi_restore import split_ranges_by_max_roi_area
from .sequence import lada_sequence_forward


def restore_image_sequence_with_masks(
    frame_paths: list[str | Path],
    mask_paths: list[str | Path],
    output_dir: str | Path,
    *,
    bundle: dict[str, object],
    sequence_forward: Callable[[mx.array, dict[str, object]], mx.array] = lada_sequence_forward,
    expansion_ratio: float = 0.0,
    align_multiple: int = 32,
) -> list[Path]:
    """Restore an image sequence with matching grayscale masks."""

    frames = read_image_sequence_tchw(frame_paths)
    masks = _read_mask_sequence(mask_paths)
    if frames.shape[0] != masks.shape[0]:
        raise ValueError(f"frame/mask count mismatch: {frames.shape[0]} != {masks.shape[0]}")
    restored = restore_masked_roi_sequence_with_lada(
        mx.array(frames),
        mx.array(masks),
        bundle,
        sequence_forward=sequence_forward,
        expansion_ratio=expansion_ratio,
        align_multiple=align_multiple,
    )
    return write_image_sequence_tchw(np.array(restored), output_dir, prefix="restored")


def restore_video_with_masks(
    input_video: str | Path,
    mask_paths: list[str | Path],
    output_video: str | Path,
    *,
    bundle: dict[str, object],
    sequence_forward: Callable[[mx.array, dict[str, object]], mx.array] = lada_sequence_forward,
    expansion_ratio: float = 0.0,
    align_multiple: int = 32,
    copy_audio: bool = False,
) -> Path:
    """Restore a short video fixture with matching grayscale mask images."""

    frames, metadata = read_video_tchw(input_video)
    masks = _read_mask_sequence(mask_paths)
    if frames.shape[0] != masks.shape[0]:
        raise ValueError(f"frame/mask count mismatch: {frames.shape[0]} != {masks.shape[0]}")
    restored = restore_masked_roi_sequence_with_lada(
        mx.array(frames),
        mx.array(masks),
        bundle,
        sequence_forward=sequence_forward,
        expansion_ratio=expansion_ratio,
        align_multiple=align_multiple,
    )
    output_video = Path(output_video)
    write_video_tchw(np.array(restored), output_video, fps=int(metadata["fps"]))
    if copy_audio:
        mux_audio_from_source_video(input_video, output_video)
    return output_video


def restore_video_with_mask_windows(
    input_video: str | Path,
    mask_paths: list[str | Path],
    output_video: str | Path,
    *,
    bundle: dict[str, object],
    sequence_forward: Callable[[mx.array, dict[str, object]], mx.array] = lada_sequence_forward,
    window_size: int = 15,
    overlap: int = 4,
    expansion_ratio: float = 0.0,
    align_multiple: int = 32,
    progress_callback: Callable[[int], None] | None = None,
    timing_callback: Callable[[dict[str, float | int]], None] | None = None,
    copy_audio: bool = False,
    max_restore_roi_area: int | None = None,
) -> Path:
    """Restore a video in temporal windows without loading the whole clip.

    The final `overlap` frames of each non-final window are carried into the
    next window as temporal context, but are written only once.
    """

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if overlap < 0 or overlap >= window_size:
        raise ValueError("overlap must be >= 0 and smaller than window_size")

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise ValueError(f"failed to open video: {input_video}")

    fps = int(round(capture.get(cv2.CAP_PROP_FPS) or 0))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if expected_frames and len(mask_paths) < expected_frames:
        capture.release()
        raise ValueError(f"frame/mask count mismatch: {expected_frames} video frames > {len(mask_paths)} masks")

    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError(f"failed to open video writer: {output_video}")

    carry_frames: list[np.ndarray] = []
    window_start = 0
    wrote_frames = 0
    try:
        while True:
            window_frames = list(carry_frames)
            hit_eof = False
            while len(window_frames) < window_size:
                ok, frame = capture.read()
                if not ok:
                    hit_eof = True
                    break
                window_frames.append(_bgr_hwc_uint8_to_tchw_float(frame))

            if not window_frames:
                break

            mask_end = window_start + len(window_frames)
            is_final_window = hit_eof or (expected_frames > 0 and mask_end >= expected_frames)
            if mask_end > len(mask_paths):
                raise ValueError(f"frame/mask count mismatch at frame {mask_end}: only {len(mask_paths)} masks")

            masks = _read_mask_sequence(mask_paths[window_start:mask_end])
            restore_started = time.perf_counter()
            restored_np = _restore_window_with_optional_roi_splits(
                np.stack(window_frames, axis=0),
                masks,
                bundle=bundle,
                sequence_forward=sequence_forward,
                expansion_ratio=expansion_ratio,
                align_multiple=align_multiple,
                max_restore_roi_area=max_restore_roi_area,
            )
            restore_seconds = time.perf_counter() - restore_started

            write_count = len(window_frames) if is_final_window else len(window_frames) - overlap
            for frame in restored_np[:write_count]:
                writer.write(_tchw_float_to_bgr_hwc_uint8(frame))
                wrote_frames += 1
            if progress_callback is not None:
                progress_callback(wrote_frames)
            if timing_callback is not None:
                timing_callback(
                    {
                        "window_start": window_start,
                        "window_frames": len(window_frames),
                        "written_frames": wrote_frames,
                        "restore_seconds": restore_seconds,
                        "effective_fps": len(window_frames) / restore_seconds if restore_seconds > 0 else 0.0,
                    }
                )

            if is_final_window:
                break

            carry_frames = window_frames[write_count:]
            window_start += write_count
            if not carry_frames and window_start >= len(mask_paths):
                break
    finally:
        capture.release()
        writer.release()

    if wrote_frames == 0:
        raise ValueError(f"no frames restored from video: {input_video}")
    if copy_audio:
        mux_audio_from_source_video(input_video, output_video)
    return output_video


def _restore_window_with_optional_roi_splits(
    frames: np.ndarray,
    masks: np.ndarray,
    *,
    bundle: dict[str, object],
    sequence_forward: Callable[[mx.array, dict[str, object]], mx.array],
    expansion_ratio: float,
    align_multiple: int,
    max_restore_roi_area: int | None,
) -> np.ndarray:
    ranges = split_ranges_by_max_roi_area(
        masks,
        max_roi_area=max_restore_roi_area,
        expansion_ratio=expansion_ratio,
        align_multiple=align_multiple,
    )
    if len(ranges) == 1 and ranges[0] == (0, frames.shape[0]):
        restored = restore_masked_roi_sequence_with_lada(
            mx.array(frames),
            mx.array(masks),
            bundle,
            sequence_forward=sequence_forward,
            expansion_ratio=expansion_ratio,
            align_multiple=align_multiple,
        )
        return np.array(restored)

    restored_parts = []
    for start, end in ranges:
        restored = restore_masked_roi_sequence_with_lada(
            mx.array(frames[start:end]),
            mx.array(masks[start:end]),
            bundle,
            sequence_forward=sequence_forward,
            expansion_ratio=expansion_ratio,
            align_multiple=align_multiple,
        )
        restored_parts.append(np.array(restored))
    return np.concatenate(restored_parts, axis=0)


def _read_mask_sequence(mask_paths: list[str | Path]) -> np.ndarray:
    masks = []
    for path in mask_paths:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"failed to read mask: {path}")
        masks.append(mask.astype(np.float32) / 255.0)
    if not masks:
        raise ValueError("no mask paths provided")
    return np.stack(masks, axis=0)
