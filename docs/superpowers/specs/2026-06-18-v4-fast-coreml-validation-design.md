# v4-fast Core ML Validation Design

## Goal

Build a focused validation path for `model_weights/lada_mosaic_detection_model_v4_fast.pt` so the LADA mosaic detector can be moved toward macOS/iOS native execution with measurable parity against the current PyTorch implementation.

This phase validates detection only. It does not port restoration, BasicVSR++, UI, or full video processing.

## Current Facts

The `v4-fast` model is a YOLO11 segmentation model with two classes:

- `0: mosaic_nsfw`
- `1: mosaic_sfw_head`

The model exports successfully to Core ML as an `mlProgram` using `ultralytics` and `coremltools`.

Observed exported Core ML interface:

- Input: `image`, RGB, `640x640`
- Output: `var_1324`, shape `1 x 38 x 8400`, float32
- Output: `var_1362`, shape `1 x 32 x 160 x 160`, float32

The raw outputs are still YOLO segmentation outputs. Native macOS/iOS code must implement postprocessing equivalent to LADA's PyTorch/Ultralytics path before the output can be used by the rest of the pipeline.

## Scope

In scope:

- Add a reproducible export script for `v4-fast.pt` to Core ML `.mlpackage`.
- Add a Python validation harness comparing PyTorch and Core ML predictions.
- Document the raw-output postprocess contract needed by Swift/Core ML.
- Define pass/fail thresholds for detector parity.
- Keep generated `.mlpackage` artifacts out of git unless explicitly requested.

Out of scope:

- `v4-accurate.pt`
- BasicVSR++ / restoration model conversion
- `mps-deform-conv`
- Swift application code
- AVFoundation or VideoToolbox integration
- Full video batch processing

## Proposed Files

Create:

- `scripts/apple/export-v4-fast-coreml.py`
- `scripts/apple/validate-v4-fast-coreml.py`
- `docs/apple/v4-fast-coreml-postprocess.md`
- `tests/test_v4_fast_coreml_export.py`

No production runtime path should change in this phase.

## Export Design

The export script will:

1. Load `model_weights/lada_mosaic_detection_model_v4_fast.pt` with `ultralytics.YOLO`.
2. Assert `model.task == "segment"`.
3. Export with:
   - `format="coreml"`
   - `imgsz=640`
   - `half=False`
   - `nms=False`
   - `simplify=True`
4. Write to an explicit output directory, defaulting to `build/apple/coreml/`.
5. Print the exported model path and Core ML input/output metadata.

`nms=False` is intentional. It preserves raw YOLO outputs so Swift can use one deterministic postprocess implementation that matches LADA, instead of relying on exporter-specific postprocessing.

## Validation Design

The validation harness will compare:

- Detection count
- Class IDs
- Confidence values
- Box coordinates
- Mask dimensions
- Mask area

Input sources:

- A user-specified image directory or video path.
- If a video is supplied, the harness extracts frames to a temporary directory using `ffmpeg`.

Default thresholds:

- Box max absolute coordinate difference: `<= 4.0 px`
- Confidence absolute difference: `<= 0.02`
- Class IDs: exact match
- Detection count: exact match for detections above `conf=0.25`
- Mask area relative difference: `<= 10%`

If detection counts differ, the harness should print the full prediction list for both engines and mark the sample as failed. This is important because prior testing showed one frame where PyTorch returned two overlapping masks and Core ML returned one. That needs visibility rather than silent averaging.

## Postprocess Contract

The Swift implementation must replicate the LADA/Ultralytics path represented by `Yolo11SegmentationModel.postprocess()`:

1. Input image is letterboxed to `640x640` using stride `32`.
2. Pixel values are normalized from `0...255` to `0...1`.
3. Model output tensor `1 x 38 x 8400` is decoded into candidate boxes, class scores, and mask coefficients.
4. NMS is applied with the same confidence and IoU thresholds as LADA.
5. Mask coefficients are multiplied with proto output `1 x 32 x 160 x 160`.
6. Masks are cropped to boxes, upsampled, and mapped back through the inverse letterbox transform.
7. Boxes are scaled back to the original image dimensions.
8. Empty or zero-area masks are discarded.

The postprocess document should include enough tensor layout detail for a Swift engineer to implement this without reading Python internals.

## Acceptance Criteria

This phase is complete when:

- The export script creates a valid `.mlpackage` from `v4-fast.pt`.
- The validation harness runs PyTorch and Core ML inference on at least one positive detection sample.
- The harness prints a machine-readable summary.
- At least one known positive frame passes the default parity thresholds.
- The postprocess contract document describes every transform from Core ML raw outputs to LADA-style boxes and masks.
- Unit tests cover script argument parsing and metadata extraction without requiring Core ML export during normal test runs.

## Risks

Core ML export depends on `coremltools` compatibility with the installed PyTorch version. Current local testing emits a warning because `torch 2.12.0` is newer than the latest version officially tested by `coremltools`, but export and inference succeeded in the local validation.

Ultralytics may apply slightly different postprocessing for Core ML models than for PyTorch models when used through its Python wrapper. The native Swift path should not rely on Ultralytics behavior; it should consume raw outputs directly.

The exported model output names, currently `var_1324` and `var_1362`, are generated names. Native code should identify outputs by shape and metadata where possible, or the export script should optionally rename outputs in a later phase.

## Recommended Next Step

Write the implementation plan for the four files listed above, then implement in small commits:

1. Export script
2. Metadata/unit tests
3. Validation harness
4. Postprocess specification
