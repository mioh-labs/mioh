#!/usr/bin/env python3
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Measure the shipped BasicVSR++ checkpoint across input-space degradations.

All comparisons score the *same central frame*.  Mosaic block sizes refer to
the 256px model input, not to pixels in the original full-resolution video.
The source must be clean: a pre-existing mosaic is not ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config

from lada.models.basicvsrpp import register_all_modules
from lada.models.basicvsrpp.mmagic.registry import MODELS
from lada.models.basicvsrpp.recoverable_hf_dataset import (
    phase_block_average_mosaic,
)


DEFAULT_CONFIG = Path(
    "configs/basicvsrpp/mosaic_restoration_generic_stage2.19_clean_detail_recovery.py"
)
DEFAULT_CHECKPOINT = Path(
    "model_weights/basicvsrpp-v1.2-detail-recovery-30000-ema.pth"
)


def positive_int_list(value: str) -> list[int]:
    try:
        result = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item < 1 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must be unique positive integers")
    return result


def parse_crop(value: str) -> tuple[int, int, int, int]:
    try:
        items = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("crop must be x,y,width,height") from error
    if len(items) != 4:
        raise argparse.ArgumentTypeError("crop must be x,y,width,height")
    x, y, width, height = items
    if x < 0 or y < 0 or width < 1 or height < 1:
        raise argparse.ArgumentTypeError("invalid crop coordinates")
    return x, y, width, height


def nonnegative_int_list(value: str) -> list[int]:
    if value.strip().lower() == "none":
        return []
    try:
        result = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item < 0 for item in result) or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must be unique nonnegative integers")
    return result


def parse_phase(value: str) -> tuple[int, int]:
    try:
        phase = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("phase must be x,y") from error
    if len(phase) != 2 or any(item < 0 for item in phase):
        raise argparse.ArgumentTypeError("phase must be two nonnegative integers")
    return phase


def read_clean_frames(
    video: Path, *, start: int, count: int, crop: tuple[int, int, int, int] | None
) -> np.ndarray:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open clean source video: {video}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for index in range(count):
            ok, bgr = capture.read()
            if not ok:
                raise RuntimeError(
                    f"source ended at frame {start + index}; need {count} frames"
                )
            height, width = bgr.shape[:2]
            if crop is None:
                size = min(width, height)
                x, y = (width - size) // 2, (height - size) // 2
                crop_box = (x, y, size, size)
            else:
                crop_box = crop
            x, y, crop_width, crop_height = crop_box
            if crop_width != crop_height:
                raise ValueError("clean-source crop must be square to avoid aspect distortion")
            if x + crop_width > width or y + crop_height > height:
                raise ValueError(f"crop {crop_box} exceeds source frame {width}x{height}")
            selected = bgr[y : y + crop_height, x : x + crop_width]
            resized = cv2.resize(selected, (256, 256), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        return np.stack(frames)
    finally:
        capture.release()


def mosaic_clip(
    clean: np.ndarray, block: int, phase: tuple[int, int], roi: np.ndarray
) -> np.ndarray:
    # The phase is constant across the clip, as in the stage2.19 dataset.
    frames = []
    for frame in clean:
        mosaic = phase_block_average_mosaic(frame, block_size=block, phase=phase)
        observation = frame.copy()
        observation[roi] = mosaic[roi]
        frames.append(observation)
    return np.stack(frames)


def h264_roundtrip(frames: np.ndarray, crf: int, ffmpeg: str) -> np.ndarray:
    if not 0 <= crf <= 51:
        raise ValueError("H.264 CRF must be between 0 and 51")
    if frames.dtype != np.uint8 or frames.shape[1:] != (256, 256, 3):
        raise ValueError("H.264 input must be uint8 RGB frames at 256x256")
    with tempfile.TemporaryDirectory(prefix="mioh-basicvsrpp-grid-") as temporary:
        encoded = Path(temporary) / "compressed.mp4"
        encode = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "256x256",
             "-r", "24", "-i", "pipe:0", "-an", "-c:v", "libx264",
             "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
             str(encoded)],
            input=frames.tobytes(), capture_output=True, check=False,
        )
        if encode.returncode:
            raise RuntimeError(f"H.264 encoding failed: {encode.stderr.decode(errors='replace')}")
        decode = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(encoded),
             "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
            capture_output=True, check=False,
        )
        if decode.returncode:
            raise RuntimeError(f"H.264 decoding failed: {decode.stderr.decode(errors='replace')}")
        expected = frames.size
        decoded = np.frombuffer(decode.stdout, dtype=np.uint8)
        if decoded.size != expected:
            raise RuntimeError(f"H.264 roundtrip returned {decoded.size} bytes, expected {expected}")
        return decoded.reshape(frames.shape).copy()


def window_for(count: int, maximum: int) -> tuple[int, int]:
    if count < 1 or count > maximum:
        raise ValueError("window length exceeds the decoded clip")
    center = maximum // 2
    start = center - count // 2
    return start, center - start


def score(prediction: np.ndarray, target: np.ndarray, roi: np.ndarray) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.shape != (256, 256, 3):
        raise ValueError("metric inputs must be matching 256x256 RGB images")
    if roi.shape != (256, 256) or not roi.any():
        raise ValueError("ROI mask must have at least one pixel")
    reference = target.astype(np.float32) / 255.0
    estimate = prediction.astype(np.float32) / 255.0
    error = estimate[roi] - reference[roi]
    mse = float(np.mean(error * error))
    # Same Gaussian high-pass definition as evaluate-multiframe-mosaic-oracle.py.
    reference_hf = reference - cv2.GaussianBlur(reference, (0, 0), 1.2)
    estimate_hf = estimate - cv2.GaussianBlur(estimate, (0, 0), 1.2)
    reference_values = reference_hf[roi]
    estimate_values = estimate_hf[roi]
    reference_values -= reference_values.mean(axis=0, keepdims=True)
    estimate_values -= estimate_values.mean(axis=0, keepdims=True)
    reference_energy = float(np.mean(reference_values * reference_values))
    estimate_energy = float(np.mean(estimate_values * estimate_values))
    covariance = float(np.mean(reference_values * estimate_values))
    correlation = covariance / max(math.sqrt(reference_energy * estimate_energy), 1e-12)
    amplitude = math.sqrt(estimate_energy / max(reference_energy, 1e-12))
    return {
        "roi_psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
        "hf_correlation": correlation,
        "hf_amplitude_ratio": amplitude,
        "hf_corr_times_amp": correlation * amplitude,
    }


def load_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    register_all_modules()
    config = Config.fromfile(str(config_path))
    # The checkpoint is a trusted local project artifact, never a URL.
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has no state_dict")
    prefix = "generator_ema."
    selected = {
        key[len(prefix):]: value for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not selected:
        raise ValueError("checkpoint has no generator_ema state")
    model = MODELS.build(config.model.generator)
    model.load_state_dict(selected, strict=True)
    return model.eval().requires_grad_(False).to(device)


def predict_center(model, frames: np.ndarray, center: int, device: torch.device) -> np.ndarray:
    # The Python BasicVSR++ graph needs at least two frames for optical flow.
    # For N=1, repeat the one observation twice: one distinct observed frame.
    model_frames = frames if len(frames) > 1 else np.repeat(frames, 2, axis=0)
    tensor = torch.from_numpy(np.ascontiguousarray(model_frames)).permute(0, 3, 1, 2)
    tensor = tensor.unsqueeze(0).to(device=device, dtype=torch.float32).div_(255)
    with torch.inference_mode():
        output = model(tensor)[0, center].clamp_(0, 1).mul_(255)
    return output.permute(1, 2, 0).round().to(torch.uint8).cpu().numpy()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-video", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--crop", type=parse_crop, help="source x,y,width,height; default center square")
    parser.add_argument("--roi", type=parse_crop, default=(64, 64, 128, 128),
                        help="256px input x,y,width,height; default central 128x128")
    parser.add_argument("--blocks", type=positive_int_list, default=[8, 16, 32])
    parser.add_argument("--frames", type=positive_int_list, default=[1, 9, 26, 48])
    parser.add_argument("--crfs", type=nonnegative_int_list, default=[18, 28],
                        help="H.264 CRFs or 'none'; no-additional-compression is always included")
    parser.add_argument("--phase", type=parse_phase,
                        default=(0, 0), help="fixed grid x,y phase in input pixels")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--trust-checkpoint", action="store_true")
    args = parser.parse_args()
    if not args.trust_checkpoint:
        parser.error("loading a project checkpoint requires --trust-checkpoint")
    if args.start_frame < 0 or any(block <= 1 for block in args.blocks):
        parser.error("start-frame must be nonnegative and blocks must exceed 1")
    if any(not 0 <= crf <= 51 for crf in args.crfs):
        parser.error("CRFs must be in 0...51")
    video = args.clean_video.expanduser().resolve()
    config = args.config.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    for path in (video, config, checkpoint):
        if not path.is_file():
            parser.error(f"input file does not exist: {path}")
    if output.exists():
        parser.error(f"output directory already exists: {output}")
    maximum = max(args.frames)
    clean = read_clean_frames(video, start=args.start_frame, count=maximum, crop=args.crop)
    x, y, width, height = args.roi
    if x + width > 256 or y + height > 256:
        parser.error("ROI must fit within the 256px model input")
    roi = np.zeros((256, 256), dtype=bool)
    roi[y:y + height, x:x + width] = True
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available()
        else "cpu" if args.device == "auto" else args.device
    )
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available")
    model = load_model(config, checkpoint, device)
    anchor = maximum // 2
    target = clean[anchor]
    rows = []
    for block in args.blocks:
        mosaicked = mosaic_clip(clean, block, args.phase, roi)
        for compression, observations in [
            ("none", mosaicked),
            *((f"h264-crf{crf}", h264_roundtrip(mosaicked, crf, args.ffmpeg))
              for crf in args.crfs),
        ]:
            for count in args.frames:
                start, center = window_for(count, maximum)
                window = observations[start:start + count]
                prediction = predict_center(model, window, center, device)
                baseline = score(observations[anchor], target, roi)
                restored = score(prediction, target, roi)
                row = {
                    "block_px_at_model_input": block,
                    "observed_frames": count,
                    "compression": compression,
                    "baseline_roi_psnr_db": baseline["roi_psnr_db"],
                    "baseline_hf_corr_times_amp": baseline["hf_corr_times_amp"],
                    **restored,
                    "roi_psnr_gain_db": restored["roi_psnr_db"] - baseline["roi_psnr_db"],
                    "hf_corr_times_amp_gain": (
                        restored["hf_corr_times_amp"] - baseline["hf_corr_times_amp"]
                    ),
                }
                rows.append(row)
                print(
                    f"block={block:2d} N={count:2d} {compression:11s} "
                    f"PSNR={row['roi_psnr_db']:.2f}dB "
                    f"HF corr×amp={row['hf_corr_times_amp']:.4f}",
                    flush=True,
                )
                if device.type == "mps":
                    torch.mps.empty_cache()
    output.mkdir(parents=True)
    report = {
        "format_version": 1,
        "note": "One center frame per condition; not a temporal flicker or real-mosaic test",
        "source": str(video),
        "source_start_frame": args.start_frame,
        "source_crop": args.crop,
        "roi_at_model_input": [x, y, width, height],
        "phase_at_model_input": args.phase,
        "model_input_size": 256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "config": str(config),
        "device": str(device),
        "single_frame_policy": "repeat once for Python BasicVSR++ optical flow",
        "rows": rows,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {output / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("LADA_DEFORM_CONV_BACKEND", "mps_deform_conv")
    raise SystemExit(main())
