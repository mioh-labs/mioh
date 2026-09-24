#!/usr/bin/env python3
"""Extract and verify aligned 2K/4K frame pairs for a local PiperSR pilot.

Frames and reports remain in an ignored build directory; source videos are read
only. A matching catalog number alone is not enough to establish alignment.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def probe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate:format=duration",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def extract(path: Path, seconds: float, output: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{seconds:.3f}", "-i", str(path),
         "-frames:v", "1", "-y", str(output)],
        check=True,
    )


def similarity(lr_path: Path, hr_path: Path) -> tuple[float, float, float]:
    with Image.open(lr_path) as image:
        lr = np.asarray(image.convert("RGB").resize((240, 135)), dtype=np.float32)
    with Image.open(hr_path) as image:
        hr = np.asarray(image.convert("RGB").resize((240, 135)), dtype=np.float32)
    correlation = float(np.corrcoef(lr.reshape(-1), hr.reshape(-1))[0, 1])
    error = float(np.abs(lr - hr).mean())
    contrast = float(hr.std())
    return correlation, error, contrast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hr-video", type=Path, required=True)
    parser.add_argument("--lr-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=float, default=300)
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--min-correlation", type=float, default=0.98)
    args = parser.parse_args()
    hr_info = probe(args.hr_video)
    lr_info = probe(args.lr_video)
    hr_stream = hr_info["streams"][0]
    lr_stream = lr_info["streams"][0]
    if (hr_stream["width"], hr_stream["height"]) != (
        lr_stream["width"] * 2, lr_stream["height"] * 2
    ):
        raise ValueError("videos must have an exact 2:1 spatial ratio")
    duration = min(float(hr_info["format"]["duration"]), float(lr_info["format"]["duration"]))
    args.output.mkdir(parents=True, exist_ok=True)
    lr_directory = args.output / "lr"
    hr_directory = args.output / "hr"
    lr_directory.mkdir(exist_ok=True)
    hr_directory.mkdir(exist_ok=True)
    pairs = []
    for index in range(args.count):
        seconds = args.start + index * args.interval
        if seconds >= duration - 1:
            break
        name = f"frame-{index:03d}-{seconds:08.3f}.png"
        lr_path = lr_directory / name
        hr_path = hr_directory / name
        if not lr_path.is_file():
            extract(args.lr_video, seconds, lr_path)
        if not hr_path.is_file():
            extract(args.hr_video, seconds, hr_path)
        correlation, mae, contrast = similarity(lr_path, hr_path)
        accepted = bool(correlation >= args.min_correlation and contrast >= 8.0)
        if not accepted:
            lr_path.unlink(missing_ok=True)
            hr_path.unlink(missing_ok=True)
        record = {
            "time": seconds, "correlation": correlation,
            "mae_255": mae, "contrast": contrast, "accepted": accepted,
        }
        pairs.append(record)
        print(json.dumps(record), flush=True)
    report = {
        "hr_video": str(args.hr_video), "lr_video": str(args.lr_video),
        "hr_probe": hr_info, "lr_probe": lr_info,
        "pairs": pairs, "accepted": sum(item["accepted"] for item in pairs),
    }
    (args.output / "alignment-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    if report["accepted"] < 6:
        raise ValueError("fewer than six matching frames; do not train this pair")


if __name__ == "__main__":
    main()
