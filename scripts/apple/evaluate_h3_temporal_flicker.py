#!/usr/bin/env python3
"""Measure global and localized luminance flicker in an H3 output movie."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


REGIONS = {
    "full": (0.0, 0.0, 1.0, 1.0),
    "left_background": (0.02, 0.12, 0.27, 0.88),
    "upper_background": (0.27, 0.02, 0.73, 0.30),
    "right_background": (0.73, 0.12, 0.98, 0.88),
    "center_subject": (0.30, 0.18, 0.70, 0.92),
}


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    left = window // 2
    right = window - 1 - left
    padded = np.pad(values, (left, right), mode="reflect")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def load_region_means(path: Path) -> tuple[float, dict[str, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    series: dict[str, list[float]] = {name: [] for name in REGIONS}
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        height, width = frame.shape[:2]
        # OpenCV decodes BGR in nominal 8-bit display range.  This metric is
        # deliberately evaluated in displayed BT.709-like luminance units.
        b, g, r = cv2.split(frame.astype(np.float32))
        luma = 0.0722 * b + 0.7152 * g + 0.2126 * r
        for name, (x0, y0, x1, y1) in REGIONS.items():
            xa, xb = round(x0 * width), round(x1 * width)
            ya, yb = round(y0 * height), round(y1 * height)
            series[name].append(float(luma[ya:yb, xa:xb].mean()))
    capture.release()
    if not series["full"]:
        raise RuntimeError(f"video contains no frames: {path}")
    return fps, {name: np.asarray(values) for name, values in series.items()}


def evaluate(path: Path, window: int, start_fraction: float) -> dict[str, object]:
    fps, series = load_region_means(path)
    frame_count = len(series["full"])
    start = min(frame_count - 2, max(0, round(frame_count * start_fraction)))
    residuals = {
        name: values - moving_average(values, window)
        for name, values in series.items()
    }
    full_residual = residuals["full"]
    metrics: dict[str, object] = {}
    for name, values in series.items():
        segment = values[start:]
        residual = residuals[name][start:]
        delta = np.diff(segment)
        local = residual - full_residual[start:]
        metrics[name] = {
            "mean_luma": float(segment.mean()),
            "frame_delta_rms": float(np.sqrt(np.mean(delta * delta))),
            "detrended_rms": float(np.sqrt(np.mean(residual * residual))),
            "localized_detrended_rms": float(np.sqrt(np.mean(local * local))),
            "detrended_peak": float(np.max(np.abs(residual))),
            "correlation_with_full": (
                1.0
                if name == "full"
                else float(np.corrcoef(residual, full_residual[start:])[0, 1])
            ),
        }
    return {
        "path": str(path.resolve()),
        "fps": fps,
        "frames": frame_count,
        "start_frame": start,
        "detrend_window": window,
        "regions": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--window", type=int, default=9)
    parser.add_argument("--start-fraction", type=float, default=0.5)
    args = parser.parse_args()
    if args.window < 3 or args.window % 2 == 0:
        parser.error("--window must be an odd integer >= 3")
    results = [
        evaluate(path, args.window, args.start_fraction) for path in args.videos
    ]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
