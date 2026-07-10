thread_id: 019d3345-3aee-7a21-a8ba-0da876f45b3e
updated_at: 2026-07-01T22:07:59+00:00
rollout_path: /Users/okatti/.codex/sessions/2026/03/28/rollout-2026-03-28T16-07-48-019d3345-3aee-7a21-a8ba-0da876f45b3e.jsonl
cwd: /Users/okatti/Documents/lada
git_branch: main

# The rollout added Apple-Silicon/ROI-enhancement support, fixed a single-file output-path bug in `process_video_parallel.py`, installed Real-ESRGAN/BasicSR via the repo patch helper, and clarified the ROI-enhancer tuning flags.

Rollout context: Working in `/Users/okatti/Documents/lada` and later mirroring changes into `/Users/okatti/Documents/lada_git`. The thread started as an Apple/MPS/CoreML integration effort, then moved into ROI enhancement flags, a parallel-wrapper bugfix, dependency installation, and final parameter explanation.

## Task 1: Add Apple/MPS/CoreML integration and supporting plumbing

Outcome: success

Preference signals:
- When the user said 「対応するように一気に進めてください」「一気に進めてください」, they wanted the agent to continue through implementation and validation instead of pausing after design.
- When the user later said 「process_video_parallel.pyに反映してね。」, that indicated they expect new CLI behavior to be wired through all entrypoints, not only `lada-cli`.

Key steps:
- The repo was inspected and the relevant YOLO/MPS/deform-conv paths were identified.
- A plan doc was written in `docs/superpowers/plans/2026-03-28-mps-coreml-integration.md`.
- `unittest` red/green tests were added for backend selection and deform-conv dispatch, then minimal helpers were implemented.
- A CoreML detect-backend path was added alongside the existing Ultralytics/PyTorch segmentation model path.
- `pyproject.toml` gained an `apple` extra with `coremltools` and `mps-deform-conv`.
- The work was later mirrored into `lada_git` and pushed to `origin`.

Failures and how to do differently:
- The CoreML Hugging Face model turned out to be `task=detect`, not segmentation, so the integration had to use a synthesized-mask compatibility mode rather than assume native masks.
- `process_video_parallel.py` did not automatically inherit the new `lada-cli` flag, so it had to be patched separately.
- The environment lacked ML dependencies during initial validation, so the agent relied on dependency-free tests and `py_compile` for verification.

Reusable knowledge:
- `riddhimanrana/yolo11n-coreml` exposes CoreML outputs `coordinates` and `confidence` and is detection-only; downstream code must synthesize masks from boxes if it wants to reuse the existing ROI pipeline.
- `mps-deform-conv` can be used as an MPS-only fallback for deformable convolution, with torchvision as the non-MPS fallback.
- `process_video_parallel.py` has separate argparse/runtime plumbing, so it must be updated explicitly when adding CLI options to `lada-cli`.

References:
- `docs/superpowers/plans/2026-03-28-mps-coreml-integration.md`
- `lada/models/yolo/backend_selection.py`
- `lada/models/yolo/yolo11_coreml_model.py`
- `lada/models/basicvsrpp/deformconv.py`
- `lada/cli/main.py`
- `process_video_parallel.py`
- Verification commands that passed: `python3 -m unittest tests/test_detection_backend_selection.py tests/test_deform_conv_dispatch.py`, `python3 -m py_compile ...`, `git diff --check`

## Task 2: Add restore smoothing and wire it through CLI and parallel wrapper

Outcome: success

Preference signals:
- The user repeatedly asked to keep going and then later requested specific follow-through like 「コミットした上で」 and to reflect changes in the parallel wrapper, showing they want implementation plus repository hygiene, not just a patch suggestion.
- The user corrected the help text request from 「--help2」 to 「--helpです。」, indicating they care about the actual help output and want argument docs to be accurate.

Key steps:
- Added `apply_restore_smoothing()` to `lada/restorationpipeline/frame_restorer.py`.
- Applied smoothing after texture/detail/sharpen processing, only within the restored ROI mask.
- Added `--restore-smooth-strength` to `lada-cli` and `process_video_parallel.py`.
- Added/updated tests to cover smoothing behavior, CLI parsing, and parallel command propagation.
- Verified with unittest, `py_compile`, and `git diff --check`.
- Committed in both repos and pushed `lada_git` to Codeberg.

Failures and how to do differently:
- The first attempt to add the CLI flag to `process_video_parallel.py` needed follow-up because that file has its own parser and worker config; the correct fix was to thread the option through `WorkerRuntimeConfig`, command builders, parser, and validation.
- The `tests.test_process_video_parallel_shutdown` run emits an existing SIGINT message on success; that is expected noise, not a failure.

Reusable knowledge:
- `--restore-smooth-strength` is a post-processing softening control; it is applied after texture/detail/sharpen effects and before compositing.
- Good starting values are in the `0.10–0.25` range.
- `process_video_parallel.py` needs explicit output-path and option propagation for every new LADA feature.

References:
- `lada/restorationpipeline/frame_restorer.py`
- `lada/cli/main.py`
- `process_video_parallel.py`
- `tests/test_restore_sharpen.py`
- `tests/test_process_video_parallel_output_path.py`
- Passed checks: `python -m unittest tests.test_restore_sharpen`, `python -m unittest tests.test_process_video_parallel_shutdown`, `python -m py_compile ...`, `git diff --check`

## Task 3: Install Real-ESRGAN and BasicSR via the repo patch helper

Outcome: success

Preference signals:
- When the user said 「realesrganとbasicsrのインストールして。apply_lada_patches.pyにある」, they wanted the repo’s own patch/install path used rather than a manual dependency recipe.

Key steps:
- Confirmed `basicsr` and `realesrgan` were not installed in the current venv.
- Ran `apply_lada_patches.py --install-roi-enhancer-deps --skip-downloads`.
- Verified `basicsr`, `realesrgan`, `facexlib`, and `gfpgan` imports and ran `pip check`.
- The script patched BasicSR’s Python 3.13 setup and the `torchvision.transforms.functional_tensor` compatibility issue.

Failures and how to do differently:
- `apply_lada_patches.py` emits some expected patch-failure messages for unrelated patches when the corresponding files are absent; those were not blockers for the ROI-enhancer install.

Reusable knowledge:
- On this Mac/venv, the patch helper successfully installs and verifies the ROI-enhancer stack.
- `pip check` reported no broken requirements after installation.

References:
- Command used: `/Users/okatti/.pyenv/versions/lada/bin/python apply_lada_patches.py --install-roi-enhancer-deps --skip-downloads`
- Verified imports: `basicsr 1.4.2`, `realesrgan 0.3.0`, `facexlib 0.3.0`, `gfpgan 1.3.8`
- `pip check` output: `No broken requirements found.`

## Task 4: Explain ROI-enhancer tuning flags

Outcome: success

Preference signals:
- The user asked directly for explanation of the Real-ESRGAN flags, so short operational parameter explanations are useful in this workflow.

Key steps:
- Explained `--restore-roi-enhancer-scale`, `--restore-roi-enhancer-strength`, and `--restore-roi-enhancer-tile` in terms of ROI-only processing, blending strength, and memory/performance tradeoffs.

Reusable knowledge:
- A practical starting point is `scale 2`, `strength 0.20`, `tile 128`.
- Lower `strength` and lower `tile` are the main levers for MPS stability.

References:
- Suggested command fragment: `--restore-roi-enhancer realesrgan --restore-roi-enhancer-model-path /Users/okatti/Documents/lada/model_weights/RealESRGAN_x2plus.pth --restore-roi-enhancer-scale 2 --restore-roi-enhancer-strength 0.25 --restore-roi-enhancer-tile 128`

