# Project Memory

## What This Project Is

This workspace is a local Lada source tree. Lada restores pixelated/mosaic regions in videos and provides both a CLI (`lada-cli`) and a GTK/Libadwaita GUI (`lada`).

This copy appears to be focused on Apple/macOS acceleration work, especially MPS, Core ML, MLX experiments, ROI enhancers, and parallel video export behavior.

## Current Local Themes

- macOS/MPS support and memory cleanup for long video processing.
- Core ML detection and ROI enhancer export/runtime support.
- MewZoom, Real-ESRGAN, and SwinIR ROI enhancer support.
- RealESRGAN general x4v3 Core ML ROI enhancer lane.
- SwinIR real-world x4 Core ML ROI enhancer lane.
- Parallel video segmentation/export using `process_video_parallel.py`.
- MLX/DCNv2 experiments and test coverage.
- MLX propagation-warp bridge is currently defaulted off.
- Translation maintenance under `translations/`.

## Current Baseline

Treat the current Git HEAD as the latest known-good project state.

- Baseline date: 2026-07-08 JST
- Baseline commit: `0be8a04 Add project memory notes; ignore large local model weights`
- Worktree status at capture: clean
- Includes Claude-side work up to this point.

## Claude-Side Work Summary

Recent Claude-authored/co-authored work visible in Git:

- `93bfbfb Add realesr-general-x4v3 Core ML ROI enhancer lane`
  - Added `scripts/apple/export_srvgg_coreml.py`.
  - Vendors SRVGGNetCompact inside the export script, avoiding `basicsr`.
  - Exports `realesr-general-x4v3.pth` to `realesr-general-x4v3_256.mlpackage`.
  - Registers the enhancer as `realesr-general-x4v3-coreml`.
  - Adds `lada.prefer_pre_resize` metadata support so Core ML enhancers can opt into pre-resize application without being hard-coded to MewZoom.
- `153067c Default the MLX propagation-warp bridge off`
  - Changes `LADA_BASICVSRPP_MLX_PROPAGATION_WARP` default from on to off.
  - Rationale recorded in commit: MLX 0.31.2 requires MPS -> CPU -> MLX -> CPU -> MPS round trips and was measured slower than pure Torch/MPS on M5 Pro.
  - To re-enable: `LADA_BASICVSRPP_MLX_PROPAGATION_WARP=1`.
- `0be8a04 Add project memory notes; ignore large local model weights`
  - Adds this project memory file.
  - Ignores generated/downloaded local model artifacts, including MewZoom 512, RealESRGAN x2/x4/Core ML, SRVGG x4v3, and detection v2 weights.

The SRVGG/x4v3 and SwinIR lanes have focused tests for registered-name resolution and Core ML ROI enhancer metadata parsing.

Current SwinIR lane:

- `scripts/apple/export_swinir_coreml.py` exports official SwinIR real-world x4 checkpoints to Core ML.
- It expects a local checkout of `JingyunLiang/SwinIR` via `--swinir-repo-dir` instead of vendoring the full architecture into Lada.
- Registered enhancer names: `swinir-x4-coreml` and `swinir-real-x4-coreml`.
- CLI enhancer selector: `--restore-roi-enhancer swinir`.
- Core ML metadata uses `lada.enhancer=swinir` and `lada.prefer_pre_resize=1`.

## Important Entry Points

- `README.md`: upstream-style user overview and install notes.
- `docs/macOS_install.md`: macOS developer setup, MPS usage, Core ML export, ROI enhancer notes.
- `process_video_parallel.py`: large local parallel processing script.
- `PROCESS_VIDEO_PARALLEL_README.md`: Japanese notes for the MPS-oriented parallel script changes.
- `pyproject.toml`: dependencies, extras, command entry points.
- `lada/cli/main.py`: CLI entry point target.
- `lada/gui/main.py`: GUI entry point target.
- `scripts/apple/`: Core ML export and validation scripts.
- `scripts/apple/export_srvgg_coreml.py`: exports compact Real-ESRGAN/SRVGG `realesr-general-x4v3` to Core ML.
- `scripts/apple/export_swinir_coreml.py`: exports official SwinIR real-world x4 checkpoints to Core ML.
- `tests/`: regression coverage for Core ML, MPS, MLX, restoration, and parallel export logic.

## Recent Git Context

Recent commits suggest this branch has been working on:

- project memory notes and ignoring large local model weights,
- SwinIR real-world x4 Core ML ROI enhancer support,
- defaulting the MLX propagation-warp bridge off,
- RealESRGAN general x4v3 Core ML ROI enhancer support,
- ignoring generated MewZoom Core ML packages,
- fixed-count video segmentation,
- MewZoom ROI and BasicVSR chunk controls,
- Core ML dependencies and vendored MewZoom architecture,
- enhancer help text updates for multiple backends.

## Local Setup Hints

For macOS development, the documented path is:

```bash
uv venv
source .venv/bin/activate
uv sync --extra cpu
```

Core ML support is installed with:

```bash
uv sync --extra cpu --extra apple-coreml
```

GUI support is installed with:

```bash
uv sync --extra cpu --extra gui
```

MPS can be checked with:

```bash
uv run --no-project python -c "import torch; print(torch.backends.mps.is_available())"
```

## Useful Test Targets

- `pytest tests/test_coreml_segmentation_model.py`
- `pytest tests/test_coreml_roi_enhancer.py`
- `pytest tests/test_process_video_parallel_segment_count.py`
- `pytest tests/test_process_video_parallel_resume.py`
- `pytest tests/test_restore_sharpen.py`

Run broader tests when touching shared model dispatch, restoration, or export behavior.

## Notes For Future Work

- Avoid reverting unrelated local changes; this workspace may contain user work.
- Prefer existing backend selection and model registration patterns before adding new switches.
- Apple-specific code paths are split across `scripts/apple/`, model helpers, docs, and tests, so changes often need all four touched together.
- Generated `.mlpackage` outputs are large artifacts and should generally stay out of git.
- When adding a new Core ML ROI enhancer, include `lada.enhancer`, `lada.scale`, `lada.imgsz`, and optionally `lada.prefer_pre_resize` metadata.
