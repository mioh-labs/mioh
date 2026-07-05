# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

"""Compare v4-fast Core ML predictions against the PyTorch model.

Runs both engines through the Ultralytics predict wrapper on the same
images (or frames extracted from a video) and checks detection count,
class ids, confidences, boxes, and mask areas against parity thresholds.
A count mismatch dumps both prediction lists instead of averaging, so
divergence stays visible.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Prediction:
    cls: int
    conf: float
    xyxy: list[float]
    mask_area: int | None


@dataclass
class FrameComparison:
    image: str
    pt_count: int
    coreml_count: int
    passed: bool
    details: list[dict]


def max_box_abs_diff(a: Iterable[float], b: Iterable[float]) -> float:
    return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def relative_area_diff(a: int | None, b: int | None) -> float:
    if a is None or b is None:
        return 1.0
    if a == 0 and b == 0:
        return 0.0
    return abs(a - b) / max(a, b)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate v4-fast Core ML against PyTorch")
    parser.add_argument("--pt-model", type=Path, default=Path("model_weights/lada_mosaic_detection_model_v4_fast.pt"))
    parser.add_argument("--coreml-model", type=Path, default=Path("build/apple/coreml/lada_mosaic_detection_model_v4_fast.mlpackage"))
    parser.add_argument("--images", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--frame-times", default="00:45:00")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--box-threshold", type=float, default=4.0)
    parser.add_argument("--conf-threshold", type=float, default=0.02)
    parser.add_argument("--mask-area-threshold", type=float, default=0.10)
    parser.add_argument("--json", type=Path)
    return parser.parse_args(argv)


def extract_frames(video: Path, frame_times: str, output_dir: Path) -> list[Path]:
    frames = []
    for ts in [value.strip() for value in frame_times.split(",") if value.strip()]:
        out = output_dir / f"{ts.replace(':', '')}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-ss", ts, "-i", str(video), "-frames:v", "1", str(out)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        frames.append(out)
    return frames


def collect_images(images_dir: Path) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in images_dir.iterdir() if path.suffix.lower() in suffixes)


def predictions_for(model_path: Path, image: Path, imgsz: int, conf: float, iou: float) -> list[Prediction]:
    from ultralytics import YOLO

    model = YOLO(str(model_path), task="segment")
    result = model.predict(str(image), imgsz=imgsz, conf=conf, iou=iou, verbose=False)[0]
    boxes = result.boxes
    masks = result.masks
    if boxes is None or len(boxes) == 0:
        return []
    mask_areas = [None] * len(boxes)
    if masks is not None:
        mask_areas = [int(value) for value in masks.data.sum(dim=(1, 2)).cpu().numpy().tolist()]
    return [
        Prediction(
            cls=int(boxes.cls[i].item()),
            conf=float(boxes.conf[i].item()),
            xyxy=[float(v) for v in boxes.xyxy[i].cpu().numpy().tolist()],
            mask_area=mask_areas[i],
        )
        for i in range(len(boxes))
    ]


def compare_predictions(pt: list[Prediction], coreml: list[Prediction], box_threshold: float, conf_threshold: float, mask_area_threshold: float) -> tuple[bool, list[dict]]:
    if len(pt) != len(coreml):
        return False, [{"reason": "count_mismatch", "pt": [asdict(p) for p in pt], "coreml": [asdict(p) for p in coreml]}]

    details = []
    passed = True
    for idx, (a, b) in enumerate(zip(pt, coreml)):
        box_diff = max_box_abs_diff(a.xyxy, b.xyxy)
        conf_diff = abs(a.conf - b.conf)
        area_diff = relative_area_diff(a.mask_area, b.mask_area)
        item_passed = a.cls == b.cls and box_diff <= box_threshold and conf_diff <= conf_threshold and area_diff <= mask_area_threshold
        passed = passed and item_passed
        details.append({
            "index": idx,
            "passed": item_passed,
            "pt": asdict(a),
            "coreml": asdict(b),
            "box_diff": box_diff,
            "conf_diff": conf_diff,
            "mask_area_diff": area_diff,
        })
    return passed, details


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if bool(args.images) == bool(args.video):
        raise SystemExit("Specify exactly one of --images or --video")

    with tempfile.TemporaryDirectory(prefix="lada-v4-fast-coreml-") as tmp:
        images = collect_images(args.images) if args.images else extract_frames(args.video, args.frame_times, Path(tmp))
        comparisons = []
        for image in images:
            pt = predictions_for(args.pt_model, image, args.imgsz, args.conf, args.iou)
            coreml = predictions_for(args.coreml_model, image, args.imgsz, args.conf, args.iou)
            passed, details = compare_predictions(pt, coreml, args.box_threshold, args.conf_threshold, args.mask_area_threshold)
            comparisons.append(FrameComparison(str(image), len(pt), len(coreml), passed, details))

    payload = {
        "passed": all(item.passed for item in comparisons),
        "frames": [asdict(item) for item in comparisons],
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text + "\n")
    print(text)
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
