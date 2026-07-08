# Project Memory

## What This Project Is

This workspace is a local Lada source tree. Lada restores pixelated/mosaic regions in videos and provides both a CLI (`lada-cli`) and a GTK/Libadwaita GUI (`lada`).

This copy appears to be focused on Apple/macOS acceleration work, especially MPS, Core ML, MLX experiments, ROI enhancers, and parallel video export behavior.

## Current Local Themes

- macOS/MPS support and memory cleanup for long video processing.
- Core ML detection and ROI enhancer export/runtime support.
- MewZoom and Real-ESRGAN ROI enhancer support.
- Parallel video segmentation/export using `process_video_parallel.py`.
- MLX/DCNv2 experiments and test coverage.
- Translation maintenance under `translations/`.

## Important Entry Points

- `README.md`: upstream-style user overview and install notes.
- `docs/macOS_install.md`: macOS developer setup, MPS usage, Core ML export, ROI enhancer notes.
- `process_video_parallel.py`: large local parallel processing script.
- `PROCESS_VIDEO_PARALLEL_README.md`: Japanese notes for the MPS-oriented parallel script changes.
- `pyproject.toml`: dependencies, extras, command entry points.
- `lada/cli/main.py`: CLI entry point target.
- `lada/gui/main.py`: GUI entry point target.
- `scripts/apple/`: Core ML export and validation scripts.
- `tests/`: regression coverage for Core ML, MPS, MLX, restoration, and parallel export logic.

## Recent Git Context

Recent commits suggest this branch has been working on:

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
