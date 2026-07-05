# v4-fast Core ML Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible export and validation tooling for `model_weights/lada_mosaic_detection_model_v4_fast.pt` so its Core ML detector can be compared against the current PyTorch/Ultralytics detector.

**Architecture:** Keep this phase as tooling only. Add scripts under `scripts/apple/`, docs under `docs/apple/`, and unit tests that validate argument parsing and metadata helpers without requiring slow Core ML export in normal test runs.

**Tech Stack:** Python 3.13, Ultralytics YOLO, coremltools, ffmpeg, unittest, existing LADA model weights.

---

## File Structure

- Create `scripts/apple/export_v4_fast_coreml.py`: importable export module with CLI entrypoint.
- Create `scripts/apple/validate_v4_fast_coreml.py`: importable validation module with CLI entrypoint.
- Create `scripts/apple/__init__.py`: package marker for test imports.
- Create `docs/apple/v4-fast-coreml-postprocess.md`: Swift/native postprocess contract.
- Create `tests/test_v4_fast_coreml_export.py`: unit tests for export arguments, model metadata extraction, and parity helper math.

Generated model artifacts belong in `build/apple/coreml/` and should not be committed unless explicitly requested.

## Task 1: Export Script

**Files:**
- Create: `scripts/apple/__init__.py`
- Create: `scripts/apple/export_v4_fast_coreml.py`
- Test: `tests/test_v4_fast_coreml_export.py`

- [x] **Step 1: Write failing tests for defaults and export options**

Add tests that import the export module and verify defaults without running Core ML conversion:

```python
from pathlib import Path
import unittest

from scripts.apple import export_v4_fast_coreml as export_mod


class V4FastCoreMLExportTests(unittest.TestCase):
    def test_default_export_paths(self):
        args = export_mod.parse_args([])
        self.assertEqual(args.model, Path("model_weights/lada_mosaic_detection_model_v4_fast.pt"))
        self.assertEqual(args.output_dir, Path("build/apple/coreml"))
        self.assertEqual(args.imgsz, 640)

    def test_export_options_are_raw_segmentation_outputs(self):
        opts = export_mod.build_export_options(imgsz=640)
        self.assertEqual(opts["format"], "coreml")
        self.assertEqual(opts["imgsz"], 640)
        self.assertFalse(opts["half"])
        self.assertFalse(opts["nms"])
        self.assertTrue(opts["simplify"])
```

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
/Users/okatti/.pyenv/versions/lada/bin/python -m unittest tests.test_v4_fast_coreml_export
```

Expected: FAIL because `scripts.apple.export_v4_fast_coreml` does not exist.

- [x] **Step 3: Implement minimal export module**

Create `scripts/apple/__init__.py` as an empty file.

Create `scripts/apple/export_v4_fast_coreml.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_MODEL = Path("model_weights/lada_mosaic_detection_model_v4_fast.pt")
DEFAULT_OUTPUT_DIR = Path("build/apple/coreml")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LADA v4-fast YOLO segmentation model to Core ML")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def build_export_options(imgsz: int) -> dict[str, Any]:
    return {
        "format": "coreml",
        "imgsz": imgsz,
        "half": False,
        "nms": False,
        "simplify": True,
    }


def export_model(model_path: Path, output_dir: Path, imgsz: int, allow_overwrite: bool = False) -> Path:
    from ultralytics import YOLO

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    expected_output = output_dir / f"{model_path.stem}.mlpackage"
    if expected_output.exists() and not allow_overwrite:
        return expected_output

    model = YOLO(str(model_path))
    if model.task != "segment":
        raise ValueError(f"Expected segment model, got {model.task!r}")

    exported = Path(model.export(**build_export_options(imgsz)))
    final_path = output_dir / exported.name
    if exported.resolve() != final_path.resolve():
        if final_path.exists() and allow_overwrite:
            import shutil
            shutil.rmtree(final_path)
        exported.replace(final_path)
    return final_path


def describe_coreml_model(model_path: Path) -> dict[str, Any]:
    import coremltools as ct

    mlmodel = ct.models.MLModel(str(model_path), compute_units=ct.ComputeUnit.ALL)
    spec = mlmodel.get_spec()
    return {
        "type": spec.WhichOneof("Type"),
        "inputs": [feature.name for feature in spec.description.input],
        "outputs": [
            {
                "name": feature.name,
                "shape": list(feature.type.multiArrayType.shape),
                "dataType": feature.type.multiArrayType.dataType,
            }
            for feature in spec.description.output
        ],
        "metadata": dict(spec.description.metadata.userDefined),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    exported = export_model(args.model, args.output_dir, args.imgsz, args.allow_overwrite)
    print(exported)
    print(json.dumps(describe_coreml_model(exported), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 4: Run tests to verify pass**

Run:

```bash
/Users/okatti/.pyenv/versions/lada/bin/python -m unittest tests.test_v4_fast_coreml_export
```

Expected: PASS.

- [x] **Step 5: Run export smoke manually**

Run:

```bash
/Users/okatti/.pyenv/versions/lada/bin/python scripts/apple/export_v4_fast_coreml.py --allow-overwrite
```

Expected: creates `build/apple/coreml/lada_mosaic_detection_model_v4_fast.mlpackage` and prints metadata with input `image` and two multi-array outputs.

- [x] **Step 6: Commit**

```bash
git add scripts/apple/__init__.py scripts/apple/export_v4_fast_coreml.py tests/test_v4_fast_coreml_export.py
git commit -m "Add v4-fast Core ML export script"
```

## Task 2: Validation Harness

**Files:**
- Modify: `scripts/apple/validate_v4_fast_coreml.py`
- Modify: `tests/test_v4_fast_coreml_export.py`

- [x] **Step 1: Write failing tests for comparison helpers**

Extend `tests/test_v4_fast_coreml_export.py`:

```python
from scripts.apple import validate_v4_fast_coreml as validate_mod


class V4FastCoreMLValidationTests(unittest.TestCase):
    def test_box_difference_uses_max_abs_delta(self):
        self.assertEqual(
            validate_mod.max_box_abs_diff([1, 2, 3, 4], [1, 5, 2, 4]),
            3,
        )

    def test_relative_area_difference_handles_zero(self):
        self.assertEqual(validate_mod.relative_area_diff(0, 0), 0.0)
        self.assertEqual(validate_mod.relative_area_diff(10, 5), 0.5)
```

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
/Users/okatti/.pyenv/versions/lada/bin/python -m unittest tests.test_v4_fast_coreml_export
```

Expected: FAIL because `validate_v4_fast_coreml` does not exist.

- [x] **Step 3: Implement validation module**

Create `scripts/apple/validate_v4_fast_coreml.py`:

```python
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
```

- [x] **Step 4: Run tests to verify pass**

Run:

```bash
/Users/okatti/.pyenv/versions/lada/bin/python -m unittest tests.test_v4_fast_coreml_export
```

Expected: PASS.

- [x] **Step 5: Run validation smoke with known frame**

If `build/apple/coreml/lada_mosaic_detection_model_v4_fast.mlpackage` exists, run:

```bash
/Users/okatti/.pyenv/versions/lada/bin/python scripts/apple/validate_v4_fast_coreml.py \
  --video /Volumes/Firewire_HD3/lada/MIDA-176.mp4 \
  --frame-times 00:45:00 \
  --json build/apple/coreml/v4-fast-validation.json
```

Expected: JSON summary. It should pass on the known positive frame observed during design.

- [x] **Step 6: Commit**

```bash
git add scripts/apple/validate_v4_fast_coreml.py tests/test_v4_fast_coreml_export.py
git commit -m "Add v4-fast Core ML validation harness"
```

## Task 3: Postprocess Contract Documentation

**Files:**
- Create: `docs/apple/v4-fast-coreml-postprocess.md`

- [x] **Step 1: Draft postprocess contract**

Create `docs/apple/v4-fast-coreml-postprocess.md` with:

```markdown
# v4-fast Core ML Postprocess Contract

## Model

`lada_mosaic_detection_model_v4_fast.mlpackage` is exported from `model_weights/lada_mosaic_detection_model_v4_fast.pt`.

## Core ML Interface

- Input: `image`, RGB, `640x640`
- Output candidates: shape `1 x 38 x 8400`
- Output prototypes: shape `1 x 32 x 160 x 160`

## Classes

- `0`: `mosaic_nsfw`
- `1`: `mosaic_sfw_head`

## Native Postprocess Requirements

1. Apply YOLO letterbox preprocessing to `640x640`, stride `32`.
2. Normalize RGB pixels to `0...1`.
3. Decode candidate tensor into boxes, class scores, and 32 mask coefficients.
4. Apply confidence threshold, default `0.25`.
5. Apply class-aware NMS, default IoU `0.7`.
6. Multiply selected mask coefficients by the prototype tensor.
7. Sigmoid masks if required by parity tests.
8. Crop masks to selected boxes.
9. Upsample masks to the `640x640` network image.
10. Undo letterbox padding and scale boxes/masks to original image coordinates.
11. Drop zero-area masks.

## Parity Notes

The native implementation must be validated against `scripts/apple/validate_v4_fast_coreml.py` before being used by a macOS/iOS runtime.
```

- [x] **Step 2: Cross-check against current Python implementation**

Review:

```bash
sed -n '1,180p' lada/models/yolo/yolo11_segmentation_model.py
```

Confirm the doc references the same preprocess/postprocess stages:

- `LetterBox`
- normalization by `255.0`
- `non_max_suppression`
- `ops.process_mask`
- `ops.scale_boxes`

- [x] **Step 3: Commit**

```bash
git add docs/apple/v4-fast-coreml-postprocess.md
git commit -m "Document v4-fast Core ML postprocess contract"
```

## Task 4: Final Verification

**Files:**
- Verify only.

- [x] **Step 1: Run unit tests**

Run:

```bash
/Users/okatti/.pyenv/versions/lada/bin/python -m unittest tests.test_v4_fast_coreml_export
```

Expected: PASS.

- [x] **Step 2: Run export smoke**

Run:

```bash
/Users/okatti/.pyenv/versions/lada/bin/python scripts/apple/export_v4_fast_coreml.py --allow-overwrite
```

Expected: `.mlpackage` exists in `build/apple/coreml/`.

- [x] **Step 3: Run validation smoke**

Run:

```bash
/Users/okatti/.pyenv/versions/lada/bin/python scripts/apple/validate_v4_fast_coreml.py \
  --video /Volumes/Firewire_HD3/lada/MIDA-176.mp4 \
  --frame-times 00:45:00 \
  --json build/apple/coreml/v4-fast-validation.json
```

Expected: exit `0` with `"passed": true`. If the external video is unavailable, report that the validation smoke was skipped and run the unit tests plus export smoke.

- [x] **Step 4: Check git status**

Run:

```bash
git status --short
```

Expected: only generated ignored/untracked build artifacts, or clean if build artifacts are ignored/removed.

- [x] **Step 5: Report result**

Report:

- Exported model path
- Unit test result
- Validation smoke result
- Any parity mismatches
