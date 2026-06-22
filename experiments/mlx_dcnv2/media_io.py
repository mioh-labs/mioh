"""Small media IO helpers for MLX LADA restore fixtures."""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np


def read_image_sequence_tchw(paths: list[str | Path]) -> np.ndarray:
    frames = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read image: {path}")
        frames.append(_bgr_hwc_uint8_to_tchw_float(image))
    if not frames:
        raise ValueError("no image paths provided")
    return np.stack(frames, axis=0)


def write_image_sequence_tchw(frames: np.ndarray, output_dir: str | Path, *, prefix: str = "frame") -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []
    for idx, frame in enumerate(frames):
        path = output_dir / f"{prefix}_{idx:04d}.png"
        if not cv2.imwrite(str(path), _tchw_float_to_bgr_hwc_uint8(frame)):
            raise ValueError(f"failed to write image: {path}")
        output_paths.append(path)
    return output_paths


def read_video_tchw(path: str | Path, *, max_frames: int | None = None) -> tuple[np.ndarray, dict[str, int | float]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"failed to open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        if max_frames is not None and len(frames) >= max_frames:
            break
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(_bgr_hwc_uint8_to_tchw_float(frame))
    capture.release()
    if not frames:
        raise ValueError(f"no frames read from video: {path}")
    return np.stack(frames, axis=0), {"fps": int(round(fps)), "width": width, "height": height}


def write_video_tchw(frames: np.ndarray, path: str | Path, *, fps: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frames.ndim != 4:
        raise ValueError(f"frames must be TCHW, got {frames.shape}")
    _, _, height, width = frames.shape
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"failed to open video writer: {path}")
    for frame in frames:
        writer.write(_tchw_float_to_bgr_hwc_uint8(frame))
    writer.release()


def mux_audio_from_source_video(
    source_video: str | Path,
    restored_video: str | Path,
    output_video: str | Path | None = None,
    *,
    ffmpeg: str = "ffmpeg",
    run: Callable[..., object] = subprocess.run,
    replace: Callable[[str | Path, str | Path], object] = os.replace,
) -> Path:
    """Copy restored video frames and optional source audio into one MP4."""

    source_video = Path(source_video)
    restored_video = Path(restored_video)
    output_video = Path(output_video) if output_video is not None else restored_video
    output_video.parent.mkdir(parents=True, exist_ok=True)

    if output_video == restored_video:
        with tempfile.NamedTemporaryFile(
            prefix=f"{restored_video.stem}_mux_",
            suffix=restored_video.suffix,
            dir=restored_video.parent,
            delete=False,
        ) as temp:
            mux_path = Path(temp.name)
    else:
        mux_path = output_video

    command = [
        ffmpeg,
        "-y",
        "-i",
        str(restored_video),
        "-i",
        str(source_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(mux_path),
    ]
    run(command, check=True)
    if mux_path != output_video:
        replace(mux_path, output_video)
    return output_video


def _bgr_hwc_uint8_to_tchw_float(image: np.ndarray) -> np.ndarray:
    return np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))


def _tchw_float_to_bgr_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    frame = np.clip(frame, 0.0, 1.0)
    return np.transpose((frame * 255.0).round().astype(np.uint8), (1, 2, 0))
