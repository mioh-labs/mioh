# Task Group: LADA Apple Silicon restore plumbing, CoreML/ROI enhancement, and dependency patching
scope: Use when extending the Apple Silicon path in LADA, wiring restore options through `process_video_parallel.py`, fixing single-file wrapper behavior, or making Real-ESRGAN/BasicSR setup reproducible on this Mac.
applies_to: cwd=/Users/okatti/Documents/lada; reuse_rule=safe for the same LADA checkout family and Apple/MPS workflow, but confirm live branch state, installed deps, and current runtime errors before reusing code-level guidance

## Task 1: Add Apple/MPS and CoreML support plus masked high-resolution ROI enhancement, partial

### rollout_summary_files

- rollout_summaries/2026-03-28T07-07-48-3Rp4-lada_apple_coreml_smoothing_realesrgan_output_path_fixes.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/sessions/2026/03/28/rollout-2026-03-28T16-07-48-019d3345-3aee-7a21-a8ba-0da876f45b3e.jsonl, updated_at=2026-07-01T22:07:59+00:00, thread_id=019d3345-3aee-7a21-a8ba-0da876f45b3e, added CoreML detect-backend wiring, `mps-deform-conv`, and explicit parallel-wrapper plumbing)
- rollout_summaries/2026-06-19T09-08-13-XfHH-lada_mps_coreml_roi_enhancement_and_patch_helper.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/archived_sessions/rollout-2026-06-19T18-08-14-019edf23-5028-71e3-80f7-062957277119.jsonl, updated_at=2026-06-28T04:12:36+00:00, thread_id=019edf23-5028-71e3-80f7-062957277119, added CoreML backend wiring, `mps-deform-conv`, ROI-only restore upscale, and `process_video_parallel.py` plumbing)

### keywords

- CoreML, mps-deform-conv, process_video_parallel.py, --mosaic-detection-backend coreml, --restore-effect-upscale, empty_lookahead, frame_restorer.py, mask-scoped ROI enhancement, riddhimanrana/yolo11n-coreml, yolo11_coreml_model.py, backend_selection.py

## Task 2: Add restore smoothing option and propagate it through the CLI/parallel wrapper, success

### rollout_summary_files

- rollout_summaries/2026-03-28T07-07-48-3Rp4-lada_apple_coreml_smoothing_realesrgan_output_path_fixes.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/sessions/2026/03/28/rollout-2026-03-28T16-07-48-019d3345-3aee-7a21-a8ba-0da876f45b3e.jsonl, updated_at=2026-07-01T22:07:59+00:00, thread_id=019d3345-3aee-7a21-a8ba-0da876f45b3e, added `--restore-smooth-strength` and verified help/plumbing across `lada-cli` and `process_video_parallel.py`)

### keywords

- restore-smooth-strength, frame_restorer.py, lada/cli/main.py, process_video_parallel.py, WorkerRuntimeConfig.restore_smooth_strength, tests/test_restore_sharpen.py, unittest, py_compile, git diff --check

## Task 3: Fix `process_video_parallel.py` single-file output path handling, success

### rollout_summary_files

- rollout_summaries/2026-03-28T07-07-48-3Rp4-lada_apple_coreml_smoothing_realesrgan_output_path_fixes.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/sessions/2026/03/28/rollout-2026-03-28T16-07-48-019d3345-3aee-7a21-a8ba-0da876f45b3e.jsonl, updated_at=2026-07-01T22:07:59+00:00, thread_id=019d3345-3aee-7a21-a8ba-0da876f45b3e, normalized directory-like or extensionless single-file outputs before ffmpeg)

### keywords

- resolve_single_output_path, ffmpeg output format error, process_video_parallel.py, tests/test_process_video_parallel_output_path.py, MIDV-024-UC.mp4, Unable to choose an output format

## Task 4: Add and use the Real-ESRGAN dependency helper in `apply_lada_patches.py`, success

### rollout_summary_files

- rollout_summaries/2026-03-28T07-07-48-3Rp4-lada_apple_coreml_smoothing_realesrgan_output_path_fixes.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/sessions/2026/03/28/rollout-2026-03-28T16-07-48-019d3345-3aee-7a21-a8ba-0da876f45b3e.jsonl, updated_at=2026-07-01T22:07:59+00:00, thread_id=019d3345-3aee-7a21-a8ba-0da876f45b3e, ran the helper in the live venv and verified package/import health)
- rollout_summaries/2026-06-19T09-08-13-XfHH-lada_mps_coreml_roi_enhancement_and_patch_helper.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/archived_sessions/rollout-2026-06-19T18-08-14-019edf23-5028-71e3-80f7-062957277119.jsonl, updated_at=2026-06-28T04:12:36+00:00, thread_id=019edf23-5028-71e3-80f7-062957277119, added `--install-roi-enhancer-deps` and the BasicSR compatibility patches)

### keywords

- apply_lada_patches.py, --install-roi-enhancer-deps, BasicSR, Real-ESRGAN, KeyError: '__version__', torchvision.transforms.functional_tensor, basicsr/realesrgan import OK, pip check

## Task 5: Explain ROI-enhancer tuning flags, success

### rollout_summary_files

- rollout_summaries/2026-03-28T07-07-48-3Rp4-lada_apple_coreml_smoothing_realesrgan_output_path_fixes.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/sessions/2026/03/28/rollout-2026-03-28T16-07-48-019d3345-3aee-7a21-a8ba-0da876f45b3e.jsonl, updated_at=2026-07-01T22:07:59+00:00, thread_id=019d3345-3aee-7a21-a8ba-0da876f45b3e, explained practical `scale`, `strength`, and `tile` starting points for ROI-only Real-ESRGAN use)

### keywords

- restore-roi-enhancer-scale, restore-roi-enhancer-strength, restore-roi-enhancer-tile, RealESRGAN_x2plus.pth, tile 128, strength 0.20, scale 2, MPS stability

## User preferences

- when the user said `対応するように一気に進めてください` / `一気に進めてください。`, continue through implementation and validation instead of stopping after design or partial plumbing [Task 1][Task 2]
- when the user explicitly requested `process_video_parallel.pyに反映してね。` and later checked `--help`, propagate CLI changes across every entrypoint and verify they appear in help output too [Task 1][Task 2]
- when the user asked whether skip logic still works after the high-resolution path, preserve and verify shortcuts such as empty-lookahead rather than assuming expensive restore features are harmless [Task 1]
- when the user said `realesrganとbasicsrのインストールして。apply_lada_patches.pyにある`, prefer the repo-supported patch helper over ad hoc install instructions for environment fixes [Task 4]
- when the user later asked for the flag meanings directly, respond with short operational guidance for `scale`, `strength`, and `tile` instead of only restating implementation details [Task 5]

## Reusable knowledge

- `process_video_parallel.py` has its own argparse, worker-runtime plumbing, and output-path semantics; adding a flag or behavior in `lada-cli` does not wire the parallel wrapper automatically [Task 1][Task 2][Task 3]
- the linked Hugging Face CoreML artifact `riddhimanrana/yolo11n-coreml` is `task=detect`, with `coordinates` and `confidence` outputs, so the downstream path needs a synthesized-mask compatibility mode rather than segmentation assumptions [Task 1]
- `mps-deform-conv` can act as an MPS-only replacement for `torchvision.ops.deform_conv2d`, with fallback to torchvision when the MPS path is unavailable [Task 1]
- the empty-lookahead optimization in `lada/restorationpipeline/mosaic_detector.py` happens before restore work, so expensive ROI enhancement still skips `skip_empty_range` windows [Task 1]
- `--restore-effect-upscale` is a multiplier, so `3` means 3x working resolution and roughly 9x pixel work inside the masked ROI [Task 1]
- the restore post-processing order became `texture -> detail -> sharpen -> smooth` after `apply_restore_smoothing()` was wired into `frame_restorer.py`; `--restore-smooth-strength` is most useful around `0.10-0.25` [Task 2]
- for this repo, `python -m unittest`, `python -m py_compile`, and `git diff --check` were reliable verification steps for CLI/plumbing changes [Task 2]
- when the input is a single file, a directory-like or extensionless `--output` should be normalized to `<stem>-UC<suffix>` before reaching ffmpeg [Task 3]
- `apply_lada_patches.py --install-roi-enhancer-deps --skip-downloads` is the supported install path in this environment; it patches BasicSR for Python 3.13 and current torchvision, verifies `basicsr/realesrgan import OK`, and should be followed by `python -m pip check` [Task 4]
- a plain dependency declaration for `realesrgan==0.3.0` was not enough on this Python 3.13/macOS stack because BasicSR first failed on `KeyError: '__version__'` and then on `torchvision.transforms.functional_tensor` compatibility [Task 4]
- for ROI-only Real-ESRGAN on this Mac, practical starting values are `--restore-roi-enhancer-scale 2`, `--restore-roi-enhancer-strength 0.20-0.25`, and `--restore-roi-enhancer-tile 128`; lower `strength` or smaller `tile` are the first stability levers on MPS [Task 5]

## Failures and how to do differently

- symptom: a new Apple/restore option works in `lada-cli` but is rejected or ignored in `process_video_parallel.py`; cause: only one parser/call path was updated; fix: verify the library entrypoint, wrapper parser, worker config, command builder, and `--help` output together [Task 1][Task 2]
- symptom: ffmpeg aborts with `Unable to choose an output format ... use a standard extension`; cause: the final single-file output path was directory-like or extensionless; fix: normalize the output path before concat/final encode and cover directory, extensionless, and explicit-file cases with tests [Task 3]
- symptom: a plain Real-ESRGAN dependency install still fails in a fresh environment; cause: BasicSR breaks first on metadata/build (`KeyError: '__version__'`) and then on torchvision API drift; fix: use the patch-helper path and keep real import verification plus `pip check` in the install flow [Task 4]
- symptom: publish looks incomplete even though local code and tests passed; cause: Codeberg returned HTTP 503 on push from the mirror repo; fix: treat this as an external publish blocker, keep the local mirror state clear, and report the remote outage explicitly [Task 1]

# Task Group: LADA mosaic restoration dataset creation, fine-tuning, and source-data hygiene
scope: Use when working on `create-mosaic-restoration-dataset.py`, BasicVSR++ fine-tuning, dataset-quality triage, FC2 source preparation, processed-source cleanup, or MPS training behavior in LADA.
applies_to: cwd=/Users/okatti/Documents/lada; reuse_rule=safe for the same LADA checkout family and local dataset/training workflow, but confirm live source paths, model-weight layout, and current PyTorch/MPS behavior before reusing commands or thresholds

## Task 1: Create mosaic-restoration datasets and judge extraction quality, success

### rollout_summary_files

- rollout_summaries/2026-06-28T07-38-06-fFx9-lada_mosaic_restoration_mps_data_prep_and_filtering.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/archived_sessions/rollout-2026-06-28T16-38-06-019f0d2a-09a3-7602-8859-973866636373.jsonl, updated_at=2026-06-29T09:04:01+00:00, thread_id=019f0d2a-09a3-7602-8859-973866636373, extraction logic, source-path correction, and filter-path debugging on Apple Silicon)

### keywords

- LADA, create-mosaic-restoration-dataset.py, dataset creation, stride-length, DOVER.pth, NudeNet, 640m.pt, 320n.onnx, crop_unscaled_img, crop_unscaled_mask, crop_unscaled_meta, /Volumes/Firewire_HD3/movies/FC2/, dataset_filtered

## Task 2: Fine-tune the existing restoration model with Apple Silicon MPS, success

### rollout_summary_files

- rollout_summaries/2026-06-28T07-38-06-fFx9-lada_mosaic_restoration_mps_data_prep_and_filtering.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/archived_sessions/rollout-2026-06-28T16-38-06-019f0d2a-09a3-7602-8859-973866636373.jsonl, updated_at=2026-06-29T09:04:01+00:00, thread_id=019f0d2a-09a3-7602-8859-973866636373, MPS fine-tuning, `mps_deform_conv`, and fallback-bottleneck findings)

### keywords

- train-mosaic-restoration-basicvsrpp.py, BasicVSR++, mps_deform_conv, LADA_DEFORM_CONV_BACKEND, grid_sampler_2d_backward, model_generic_v1.2_full.pth, MMEngine, metadata_root_dir, PYTORCH_ENABLE_MPS_FALLBACK

## Task 3: Build local helper scripts and sanitize FC2 source names, success

### rollout_summary_files

- rollout_summaries/2026-06-28T07-38-06-fFx9-lada_mosaic_restoration_mps_data_prep_and_filtering.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/archived_sessions/rollout-2026-06-28T16-38-06-019f0d2a-09a3-7602-8859-973866636373.jsonl, updated_at=2026-06-29T09:04:01+00:00, thread_id=019f0d2a-09a3-7602-8859-973866636373, local helper scripts plus ASCII-safe FC2 renaming)

### keywords

- scripts/local/project_hd_finetune, 00_split_existing_dataset.py, 01_extract_clips.sh, 02_create_datasets.sh, 03_train_finetune.sh, fc2_rename_log_20260628.tsv, ASCII renaming, FC2, source hygiene

## Task 4: Tune DOVER quality thresholds from the observed corpus, success

### rollout_summary_files

- rollout_summaries/2026-06-28T07-38-06-fFx9-lada_mosaic_restoration_mps_data_prep_and_filtering.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/archived_sessions/rollout-2026-06-28T16-38-06-019f0d2a-09a3-7602-8859-973866636373.jsonl, updated_at=2026-06-29T09:04:01+00:00, thread_id=019f0d2a-09a3-7602-8859-973866636373, DOVER score distribution and threshold calibration)

### keywords

- DOVER, min-video-quality, 0.25, 0.30, 0.35, video_quality, quality_score.overall, dataset_filtered, median 0.358, below_0.30

## Task 5: Retire processed FC2 source videos with `done_processing.txt`, success

### rollout_summary_files

- rollout_summaries/2026-06-28T07-38-06-fFx9-lada_mosaic_restoration_mps_data_prep_and_filtering.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/archived_sessions/rollout-2026-06-28T16-38-06-019f0d2a-09a3-7602-8859-973866636373.jsonl, updated_at=2026-06-29T09:04:01+00:00, thread_id=019f0d2a-09a3-7602-8859-973866636373, processed-source move rules and exact file-state verification)

### keywords

- done_processing.txt, FC2_processed, processed-file cleanup, fc2_processed_move_log_20260629.tsv, /Volumes/Firewire_HD3/movies/FC2_processed/, exact file-state answers

## User preferences

- when the user kept asking for the concrete command, `本データーのコマンド出して` and `で、コマンドは？`, give runnable commands rather than stopping at conceptual guidance [Task 1][Task 2]
- when the agent proposed rough extraction, the user pushed back with `切り出し方があまり言いように思えないんだけど` and `抽出しているのがあまりよいデーターとは思えない` -> treat extraction strategy as something to evaluate for training usefulness, not just to make runnable [Task 1]
- when the user corrected the media root with `動画はここから /Volumes/Firewire_HD3/movies/FC2/`, future FC2 dataset commands should default to that folder unless they redirect the source again [Task 1][Task 3]
- when the user asked `画質の0.25という値はどうなの？`, justify filtering thresholds with the observed score distribution instead of presenting arbitrary defaults [Task 4]
- when the user rejected a local layout fix with `symlinkはいやだ、コピーして`, prefer real copies over symlinks for stable local setup unless they explicitly ask for a symlink [Task 1][Task 2][Task 3]
- when the user asked to remove spaces and 2-byte characters from FC2 material names, favor ASCII-safe filenames for local source hygiene in this workflow [Task 3]
- when the user asked `処理の終了した素材を違うフォルダに退避してください｡` and then checked one exact filename, separate only the processed source files and be ready to answer specific file-state questions precisely [Task 5]

## Reusable knowledge

- `create-mosaic-restoration-dataset.py` is a candidate-collection tool, not a "good data only" selector: contiguous NSFW detections become scenes, then `scene-min-length`, `stride-length`, `scene-max-length`, `scene-max-memory`, and optional quality/watermark/NudeNet/censor filters decide whether a scene is saved [Task 1]
- `--stride-length` limits how close accepted scenes can be; it does not mean the script only decodes every N seconds because the NSFW tracker still walks the video frame-by-frame [Task 1]
- for this repo, quality filtering requires the expected local path layout; copying repo-root `model_weights` into `lada/model_weights` is what made `VideoQualityEvaluator` find `lada/model_weights/3rd_party/DOVER.pth` without using a symlink [Task 1][Task 4]
- the repo docs already frame dataset creation as a small-subset-first workflow, and human cleanup of `crop_unscaled_img` is still expected even after optional filters run [Task 1]
- for fine-tuning, `model_weights/lada_mosaic_restoration_model_generic_v1.2_full.pth` is the useful checkpoint because it is a full MMEngine checkpoint with `state_dict`, `optimizer`, and `meta.iter = 52000`; `v1.2.pth` is not the right starting point for this path [Task 2]
- `LADA_DEFORM_CONV_BACKEND=mps_deform_conv` roughly halved training time versus the torchvision deform-conv MPS fallback path in this environment [Task 2]
- `grid_sample` forward works on MPS here, but `aten::grid_sampler_2d_backward` still falls back to CPU on this PyTorch build, so MPS training remains partially CPU-bound even after the deform-conv speedup [Task 2]
- the smoke-test training command that worked used `PYTORCH_ENABLE_MPS_FALLBACK=1`, `--load-from model_weights/lada_mosaic_restoration_model_generic_v1.2_full.pth`, and `--cfg-options` overrides for `train_dataloader.dataset.metadata_root_dir` and `val_dataloader.dataset.metadata_root_dir` [Task 2]
- the local helper flow included `scripts/local/project_hd_finetune/00_split_existing_dataset.py`, `01_extract_clips.sh`, `02_create_datasets.sh`, and `03_train_finetune.sh`, while the FC2 source folder for this workflow was verified as `/Volumes/Firewire_HD3/movies/FC2/` [Task 3]
- on the observed `dataset_filtered` sample, DOVER scores were `min 0.252`, `median 0.358`, `max 0.591`; `0.25` was very permissive, while `0.30` was a more meaningful starting point for filtering this dataset [Task 4]
- `nudenet>=3.4.2` installs `NudeDetector` and `320n.onnx`, but LADA's current `NudeNetNsfwDetector` code path still expects a local `640m.pt` via `Yolo(args.nudenet_nsfw_model_path)`, so package installation alone does not enable the current detector path [Task 1][Task 4]
- `done_processing.txt` under the dataset output is the reliable list for which source files are fully processed and safe to move into `/Volumes/Firewire_HD3/movies/FC2_processed/`; extracted metadata alone is not enough evidence [Task 5]

## Failures and how to do differently

- symptom: the extraction command runs against the wrong input tree or produces nothing useful; cause: the source path was assumed instead of verified; fix: confirm the real media root first and default FC2 work to `/Volumes/Firewire_HD3/movies/FC2/` unless the user points elsewhere [Task 1][Task 3]
- symptom: a "good data" extraction still yields weak training material; cause: the extractor was treated as a final selector instead of a candidate generator; fix: manually inspect the outputs, justify threshold choices from the actual score distribution, and tighten filtering only after checking the observed sample [Task 1][Task 4]
- symptom: optional filtering fails before extraction can finish; cause: the environment is missing or mismatching expected assets such as `DOVER.pth`, watermark weights, or the LADA-specific NudeNet model path; fix: validate the model files and repo-relative paths up front before enabling quality, watermark, NudeNet, or censor filters [Task 1][Task 4]
- symptom: enabling `nudenet>=3.4.2` still does not activate the current NSFW detector path; cause: the package ships a different interface (`320n.onnx` + `NudeDetector`) than the code currently calls (`640m.pt` through `Yolo(...)`); fix: either adapt the code path or supply the exact model/interface the current detector expects [Task 1][Task 4]
- symptom: MPS training is still unexpectedly slow after enabling fallback; cause: `grid_sampler_2d_backward` remains CPU fallback and the first run may also still use the slower deform-conv path; fix: enable `LADA_DEFORM_CONV_BACKEND=mps_deform_conv`, treat the first run as a pipeline check, and do not mistake fallback warnings for fully native GPU execution [Task 2]
- symptom: processed-file cleanup moves or answers about the wrong source video; cause: the move list was inferred from partial extraction artifacts instead of `done_processing.txt`; fix: drive cleanup from `done_processing.txt` and confirm exact file presence for any filename the user asks about [Task 5]

# Task Group: LADA VR video viewer design entrypoints and brainstorming posture
scope: Use when the user wants a VR video viewer or adjacent viewer UX in LADA and the work is still at the design-entrypoint stage rather than settled implementation.
applies_to: cwd=/Users/okatti/Documents/lada; reuse_rule=safe for the same LADA checkout family when routing early viewer-design work, but treat architecture and UI decisions as unsettled until a later implementation/spec run confirms them

## Task 1: Start a VR video viewer brainstorming pass, aborted before design output

### rollout_summary_files

- rollout_summaries/2026-06-30T07-12-04-hLbI-lada_vr_video_viewer_brainstorm_aborted.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/sessions/2026/06/30/rollout-2026-06-30T16-12-04-019f175e-eb99-7710-a65f-5e9d237b598a.jsonl, updated_at=2026-06-30T07:12:45+00:00, thread_id=019f175e-eb99-7710-a65f-5e9d237b598a, aborted during repo scan before any design proposal or implementation)

### keywords

- VR video viewer, LADA, brainstorming, lada/gui/watch, watch_view.py, timeline.py, gstreamer_pipeline_manager.py, seek_preview_popover.py, process_video_parallel.py, aborted turn

## User preferences

- when the user said `vr動画viewerを作りたいです`, treat it as a feature-design request and start with scope/requirements clarification before implementation [Task 1]

## Reusable knowledge

- LADA already has viewer-adjacent code under `lada/gui/watch/`, especially `watch_view.py`, `timeline.py`, `gstreamer_pipeline_manager.py`, `seek_preview_popover.py`, and matching `.ui` files, so these are the first routing handles for future viewer work [Task 1]
- `process_video_parallel.py` and the existing `tests/` suite are nearby integration points if the viewer later needs to align with current video-processing or validation patterns [Task 1]
- this rollout saw a dirty tree with local edits in `scripts/dataset_creation/create-mosaic-restoration-dataset.py` and many untracked experiment/model-weight paths, so future viewer work should avoid assuming a clean checkout [Task 1]

## Failures and how to do differently

- symptom: a viewer-feature request stalls before producing anything reusable; cause: the turn ended before one focused clarification question or a design proposal was delivered; fix: keep the same lightweight repo scan, then ask one focused clarifying question sooner [Task 1]
- symptom: future viewer work accidentally treats this rollout as architecture approval; cause: the thread was aborted before any design decision was made; fix: reuse it only for routing and scoping, not as proof of an accepted implementation direction [Task 1]

# Task Group: gbuc_modern remote M1 track evaluation workflow
scope: Use when migrating or checking the gbuc_modern remote track-evaluation worker that runs on a separate Mac, especially current LaunchAgent state, model/env requirements, and safe cutover sequencing to a new machine.
applies_to: cwd=/Users/okatti/Documents/gbuc_modern; reuse_rule=safe for the same gbuc_modern upload and remote-evaluation workflow, but re-check the live LaunchAgent env, webhook URL/signature settings, DB state, and current evaluator script before reuse

## Task 1: Inventory the live evaluation stack for migration planning, uncertain

### rollout_summary_files

- rollout_summaries/2026-04-28T01-17-03-m3Qj-gbuc_modern_track_evaluation_migration_new_pc.md (cwd=/Users/okatti/Documents/gbuc_modern, rollout_path=/Users/okatti/.codex/sessions/2026/04/28/rollout-2026-04-28T10-17-03-019dd1a9-421f-7db3-b15c-fd7be0055ae7.jsonl, updated_at=2026-07-04T05:32:50+00:00, thread_id=019dd1a9-421f-7db3-b15c-fd7be0055ae7, live LaunchAgent/model/env inspection captured before migration)

### keywords

- LaunchAgent, com.gbuc.track-evaluation-webhook, llama-server, gemma-4-12b-it-qat-q4_0.gguf, mmproj-gemma-4-12b-it-qat-q4_0.gguf, TRACK_EVALUATION_WEBHOOK_URL, TRACK_EVALUATION_WEBHOOK_SECRET, TRACK_EVALUATION_LLM_URL, 8788, 18080

## Task 2: Draft the new-PC cutover sequence for the worker, uncertain

### rollout_summary_files

- rollout_summaries/2026-04-28T01-17-03-m3Qj-gbuc_modern_track_evaluation_migration_new_pc.md (cwd=/Users/okatti/Documents/gbuc_modern, rollout_path=/Users/okatti/.codex/sessions/2026/04/28/rollout-2026-04-28T10-17-03-019dd1a9-421f-7db3-b15c-fd7be0055ae7.jsonl, updated_at=2026-07-04T05:32:50+00:00, thread_id=019dd1a9-421f-7db3-b15c-fd7be0055ae7, safe copy-validate-cutover order for the replacement machine)

### keywords

- pyenv, gbuc-ai-eval-3.12, gbuc_rsync_ed25519, /Volumes/Firewire_HD3/gbuc_ai_eval, TRACK_EVALUATION_WEBHOOK_URL, .env, cutover, rollback, test track, new PC IP

## User preferences

- when the user asked `いまの楽曲評価システムを新しいPCに移行したいです。どうすればよい？`, default to a concrete migration recipe with explicit copy, reinstall, validation, and cutover steps instead of a high-level redesign [Task 1][Task 2]
- the migration answer was expected to be grounded in the current live setup, so inspect the active LaunchAgent, model paths, ports, and env vars before proposing the move [Task 1]
- the rollout preserved the old worker until the new one is validated, which is the right future default for similar service migrations on this stack [Task 2]

## Reusable knowledge

- the live worker is a LaunchAgent named `com.gbuc.track-evaluation-webhook` running `scripts/track_evaluation_webhook_server.py` under `~/pyenv/versions/gbuc-ai-eval-3.12/bin/python` [Task 1]
- the current worker env that matters for migration is `TRACK_EVALUATION_WEBHOOK_URL=http://192.168.1.3:8788/track-evaluations/run`, `TRACK_EVALUATION_WEBHOOK_SECRET`, `TRACK_EVALUATION_SCRIPT=/Users/okatti/Documents/gbuc_modern/scripts/evaluate_track_essentia_musicnn_llm.py`, `TRACK_EVALUATION_AUDIO_LOCAL_DIR=/Volumes/Firewire_HD3/gbuc_ai_eval`, `TRACK_EVALUATION_RSYNC_SSH_KEY=/Users/okatti/.ssh/gbuc_rsync_ed25519`, and `TRACK_EVALUATION_LLM_URL=http://127.0.0.1:18080/v1/chat/completions` [Task 1]
- the live model server is `llama-server` on port `18080`, using `~/llama_models/gemma-4-12b-it-qat-q4_0.gguf` plus `~/llama_models/mmproj-gemma-4-12b-it-qat-q4_0.gguf` [Task 1]
- the migration units are: the `gbuc_modern` repo, `~/llama_models`, the LaunchAgent plist/env vars, the `pyenv 3.12.3` + `gbuc-ai-eval-3.12` runtime, and the rsync SSH key [Task 1][Task 2]
- the main server still points at the current worker through `TRACK_EVALUATION_WEBHOOK_URL` in its `.env`, so cutover means updating that URL after the new worker passes a test track [Task 2]
- `/Volumes/Firewire_HD3/gbuc_ai_eval` is a large local cache, not irreplaceable data; it can be recreated on the new PC and repopulated by rsync as needed [Task 1][Task 2]

## Failures and how to do differently

- symptom: a migration answer sounds complete but nothing was actually moved; cause: the rollout only captured the live state and draft sequence; fix: treat this block as a source-of-truth checklist, not as proof that migration already happened [Task 1][Task 2]
- symptom: the final cutover plan still has a hole; cause: the new PC IP address and its actual package/disk layout were not yet verified; fix: confirm those target-machine facts before editing the main server `.env` or disabling the old worker [Task 2]
- symptom: migration causes downtime or leaves no rollback path; cause: the old worker was shut down before the replacement processed a real test track; fix: keep the old machine running until the new one successfully handles one end-to-end evaluation [Task 2]

# Task Group: Mac workstation local model sizing, LAN enumeration, and hardware spot checks
scope: Use when the user asks about comfortable local LLM use on this Mac, wants the current visible LAN IPs, or asks whether a specific MacBook Pro networking accessory exists.
applies_to: cwd=/Users/okatti/Documents/Server and Mac-wide local operations; reuse_rule=safe for this same workstation and nearby Mac-wide questions, but re-check live machine specs, current subnet, and current product availability before reusing details

## Task 1: Recommend a comfortable Ornith setup for this M1 Mac, success

### rollout_summary_files

- rollout_summaries/2026-07-04T02-07-53-jm5S-ornith_m1_lan_arp_dual_ethernet_hub.md (cwd=/Users/okatti/Documents/Server, rollout_path=/Users/okatti/.codex/sessions/2026/07/04/rollout-2026-07-04T11-07-53-019f2ae1-dec8-7bb1-aa5e-4a7f9ececf6c.jsonl, updated_at=2026-07-04T11:53:36+00:00, thread_id=019f2ae1-dec8-7bb1-aa5e-4a7f9ececf6c, local machine sizing check plus practical Ornith guidance)

### keywords

- Ornith, M1, 16GB, Ollama, LM Studio, arm64, macOS 26.6, comfortable local model, ornith:9b, ornith:35b

## Task 2: Enumerate the visible `192.168.11.*` LAN IPs, success

### rollout_summary_files

- rollout_summaries/2026-07-04T02-07-53-jm5S-ornith_m1_lan_arp_dual_ethernet_hub.md (cwd=/Users/okatti/Documents/Server, rollout_path=/Users/okatti/.codex/sessions/2026/07/04/rollout-2026-07-04T11-07-53-019f2ae1-dec8-7bb1-aa5e-4a7f9ececf6c.jsonl, updated_at=2026-07-04T11:53:36+00:00, thread_id=019f2ae1-dec8-7bb1-aa5e-4a7f9ececf6c, route/ifconfig/ARP path for the current subnet)

### keywords

- 192.168.11.*, arp -a, route -n get, ifconfig, en0, visible IPs, LAN enumeration, ports vs IP addresses, 192.168.11.6, 192.168.11.90

## Task 3: Check whether MacBook Pro dual-Ethernet hub/dock options exist, uncertain

### rollout_summary_files

- rollout_summaries/2026-07-04T02-07-53-jm5S-ornith_m1_lan_arp_dual_ethernet_hub.md (cwd=/Users/okatti/Documents/Server, rollout_path=/Users/okatti/.codex/sessions/2026/07/04/rollout-2026-07-04T11-07-53-019f2ae1-dec8-7bb1-aa5e-4a7f9ececf6c.jsonl, updated_at=2026-07-04T11:53:36+00:00, thread_id=019f2ae1-dec8-7bb1-aa5e-4a7f9ececf6c, example-based answer for dual-Ethernet docks and workarounds)

### keywords

- MacBook Pro, dual Ethernet, USB-C hub, Thunderbolt dock, OWC Thunderbolt 5 Dual 10GbE Network Dock, 2.5GbE, 10GbE, buyable examples

## User preferences

- when the user asked `あたらしいAIモデルornithをM1 macで快適に使用する方法`, they wanted a comfort/latency-focused local-model recommendation, not just model background or benchmark trivia [Task 1]
- when the user corrected `ぽーとじゃなくって、IPアドレス`, they signaled that network answers need to match the exact requested artifact and should restate that target before scanning if there is any ambiguity [Task 2]
- when the user asked `macbook pro用ethernetあだぷたが2つついているUSBハブはある？`, they were looking for concrete hardware form factors and practical examples, not a general explanation of Ethernet standards [Task 3]

## Reusable knowledge

- this workstation is `arm64` macOS `26.6` with `16 GB` RAM (`sysctl -n hw.memsize` -> `17179869184`), so `ornith:9b` is the realistic local default while `ornith:35b` is likely to swap heavily and feel poor [Task 1]
- the rollout used Ollama sizing (`ornith:9b` about `5.6GB`, `ornith:35b` about `21GB`) plus prior local-model experience that LM Studio had been the more workable path here when Ollama felt slow on larger models [Task 1]
- on this Mac, `nmap` was absent but `arp`, `nc`, `dns-sd`, and `lsof` were present, and the active interface for the `192.168.11.*` subnet was `en0` [Task 2]
- the reliable one-liner for visible subnet hosts here was `arp -a | awk '/192\\.168\\.11\\./ {match($0,/192\\.168\\.11\\.[0-9]+/); if (RSTART) print substr($0,RSTART,RLENGTH)}' | sort -t. -k1,1n -k2,2n -k3,3n -k4,4n | uniq | grep -v '^192\\.168\\.11\\.255$'` [Task 2]
- `ifconfig` showed the Mac itself on `192.168.11.6` and `192.168.11.90`, so those should not be mistaken for separate remote hosts during quick LAN checks [Task 2]
- dual-Ethernet MacBook Pro solutions are usually Thunderbolt/USB-C docks rather than tiny bus-powered hubs; a normal USB-C hub plus a second USB Ethernet adapter is the practical fallback [Task 3]

## Failures and how to do differently

- symptom: a network lookup answer gets corrected immediately; cause: the scan targeted ports instead of IP addresses; fix: restate whether the user wants ports, IPs, or hosts before gathering output if the noun is ambiguous [Task 2]
- symptom: the ARP list looks noisy or duplicated; cause: `arp -a` includes repeated/interface-specific entries and broadcast addresses; fix: sort, `uniq`, and drop `192.168.11.255` before replying [Task 2]
- symptom: a hardware answer sounds more certain than the evidence supports; cause: it was based on product examples rather than region-specific stock or purchase verification; fix: label it as example-based and clarify compactness, link speed, and local availability before deeper shopping guidance [Task 3]

# Task Group: jumbo kakaku_api risk-first code review
scope: Use when the task is to evaluate `/Users/okatti/Documents/jumbo/kakaku_api`, especially to surface operationally risky sync-path issues, identify the real update entrypoints quickly, or preserve concrete review findings without storing secrets.
applies_to: cwd=/Users/okatti/Documents/jumbo; reuse_rule=safe for the same `jumbo/kakaku_api` checkout family and similar PHP sync-path reviews, but confirm the live file set and current config layout before reusing code-level findings

## Task 1: Review kakaku_api folder and surface high-risk operational issues, success

### rollout_summary_files

- rollout_summaries/2026-03-06T06-42-20-iWUk-kakaku_api_review_and_risk_assessment.md (cwd=/Users/okatti/Documents/jumbo, rollout_path=/Users/okatti/.codex/sessions/2026/03/06/rollout-2026-03-06T15-42-20-019cc1e2-04c2-7e52-a3a5-d776ea8baa9c.jsonl, updated_at=2026-06-26T08:32:24+00:00, thread_id=019cc1e2-04c2-7e52-a3a5-d776ea8baa9c, risk-first code review of the API sync path and config handling)

### keywords

- jumbo, kakaku_api, code-review, ApiClient.php, config.php, item_update.php, realtime_update.php, lowprice_update.php, htmlprice_update.php, PriceUpdate, PriceList, JSON decode, XML escaping, htpasswd, CSV sync, OAuth

## User preferences

- when the user asked `kakaku_apiフォルダ、評価して`, they wanted a code review with prioritized operational risk, not a folder summary or architecture tour [Task 1]
- when the task is an evaluation request like `評価して`, surface the concrete breakage paths first and keep the findings auditable with file/line anchors [Task 1]

## Reusable knowledge

- `kakaku_api` is the API-based sync path, distinct from the older `kakaku` / `priceedit.asp` flow; the key execution entrypoints in this folder are `htmlprice_update.php`, `lowprice_update.php`, `realtime_update.php`, and `item_update.php` [Task 1]
- `libs/config.php` centralizes API credentials, DB settings, mail settings, and URL constants, so it is a high-risk operational file even when its literal secret values must not be retained in memory [Task 1]
- `libs/ApiClient.php` is the shared OAuth-signed `PriceList` / `PriceUpdate` client; in this review it had two concrete risk points: XML child values were pre-escaped before `addChild()`, and `json_decode()` results were used without strict `json_last_error()` handling [Task 1]
- `item_update.php` still depends on the management-page CSV download path rather than a pure public API sync, so bad download responses can flow straight into the CSV parser if the response is not validated first [Task 1]
- `realtime_update.php` uses `logs/realtime_update.lock`, splits work by slot, and spawns `htmlprice_update.php`, `lowprice_update.php`, and `post_lowprice.php` via `exec()` in a loop, which is a useful routing handle when evaluating concurrency or lock behavior in this folder [Task 1]

## Failures and how to do differently

- symptom: the first reconnaissance pass produces huge, low-value output from minified UI assets; cause: the search starts too broadly across the whole folder; fix: target `kakaku_api/libs/*.php` and the main update scripts first, then widen only if the sync path remains unclear [Task 1]
- symptom: a durable memory artifact leaks secrets during config review; cause: `.htpasswd` or `config.php` values were copied instead of abstracted; fix: record only that plaintext credentials or sensitive files exist, never the literal contents [Task 1]
- symptom: a review drifts into structure summary and misses the sharpest risks; cause: the evaluation request is treated like repo orientation instead of a code review; fix: prioritize concrete breakage paths such as secret storage, request/response parsing, and unvalidated sync inputs before broader architecture notes [Task 1]

# Task Group: LADA MLX ROI seam removal, progress logging, and memory-scope clarifications
scope: Use when working on the MLX restore path in LADA, especially ROI splitting quality, `[MLX]` progress-line observability, memory-pressure follow-ups, or conceptual questions about frame streaming versus compression-style reuse.
applies_to: cwd=/Users/okatti/Documents/lada; reuse_rule=safe for the same LADA checkout family and MLX restore/logging workflow, but confirm live branch state, current defaults, and the exact user scope before reusing tuning guidance

## Task 1: Remove visible seams from MLX ROI splitting, success

### rollout_summary_files

- rollout_summaries/2026-06-25T08-48-20-VmEg-lada_mlx_roi_seam_fix_progress_logging_memory_followup.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/archived_sessions/rollout-2026-06-25T17-48-20-019efdf7-42d0-7e40-b01c-ee5b93e5efc0.jsonl, updated_at=2026-06-25T09:18:17+00:00, thread_id=019efdf7-42d0-7e40-b01c-ee5b93e5efc0, removed spatial tiling for connected ROI because seams were visibly unacceptable)

### keywords

- LADA, MLX, ROI splitting, seams, split_bbox_by_max_area, roi_restore.py, restore_fixture.py, connected ROI, spatial tiling, python -m unittest

## Task 2: Expose ROI stats in parent MLX progress output, success

### rollout_summary_files

- rollout_summaries/2026-06-25T08-48-20-VmEg-lada_mlx_roi_seam_fix_progress_logging_memory_followup.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/archived_sessions/rollout-2026-06-25T17-48-20-019efdf7-42d0-7e40-b01c-ee5b93e5efc0.jsonl, updated_at=2026-06-25T09:18:17+00:00, thread_id=019efdf7-42d0-7e40-b01c-ee5b93e5efc0, parent `[MLX]` line now carries ROI and memory context)

### keywords

- process_video_parallel.py, format_mlx_progress_line, window timing, roi=, area=max/sum, mem rss, tests/test_process_video_parallel_mlx.py, parent log, progress logging

## Task 3: Clarify that low-memory default tuning is separate from the seam fix, partial

### rollout_summary_files

- rollout_summaries/2026-06-25T08-48-20-VmEg-lada_mlx_roi_seam_fix_progress_logging_memory_followup.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/archived_sessions/rollout-2026-06-25T17-48-20-019efdf7-42d0-7e40-b01c-ee5b93e5efc0.jsonl, updated_at=2026-06-25T09:18:17+00:00, thread_id=019efdf7-42d0-7e40-b01c-ee5b93e5efc0, user rejected lowering MLX defaults while keeping the no-seam behavior)

### keywords

- memory pressure, window=20, overlap 4, max-roi-area=131072, window=15, overlap 3, temporal ROI area 65536, precise scoping, defaults clarification

## Task 4: Explain frame extraction versus compression-style reuse for the MLX pipeline, success

### rollout_summary_files

- rollout_summaries/2026-06-25T08-48-20-VmEg-lada_mlx_roi_seam_fix_progress_logging_memory_followup.md (cwd=/Users/okatti/Documents/lada, rollout_path=/Users/okatti/.codex/archived_sessions/rollout-2026-06-25T17-48-20-019efdf7-42d0-7e40-b01c-ee5b93e5efc0.jsonl, updated_at=2026-06-25T09:18:17+00:00, thread_id=019efdf7-42d0-7e40-b01c-ee5b93e5efc0, conceptual guidance only; no code changes)

### keywords

- frame extraction, per-frame RGB, masks, streaming pipeline, compression-style reuse, mask interpolation, ROI tracking, no disk writes, conceptual guidance

## User preferences

- when the user said `ROI分割はダメですね。分割した境がくっきり分かる。`, prefer visible-quality fixes over keeping spatial ROI tiling for speed or memory once seams are obvious [Task 1]
- when the user followed the seam diagnosis with `切って`, take direct action on the bad path instead of lingering in explanation mode [Task 1]
- when the user corrected scope with `いや、分割やめるのはそのまま。...`, keep reversions and follow-up changes tightly scoped; do not revert an accepted fix while discussing a different tuning problem [Task 1][Task 3]
- when the user pasted differing logs and asked `セグメント数を変えたらこんなに数字が違うのはなぜ？`, prefer logs that explain behavior, not just throughput numbers [Task 2]
- when the user asked `どうしても1枚1枚画像をとりださないとマスクと復元は当てられないのかね？` and `同じことはできないの？`, answer the conceptual question practically without forcing it into a code-change proposal [Task 4]

## Reusable knowledge

- in this MLX restore path, the visible seam came from applying `split_bbox_by_max_area(...)` to one connected ROI and compositing the spatial tiles separately; the no-seam fix was to keep connected ROIs as one tile while still allowing disconnected components to stay separate [Task 1]
- `max_restore_roi_area` is still useful as a time-window guard or logging aid, but using it to spatially cut one connected ROI produces obvious seam artifacts on this path [Task 1]
- the child `window timing` line already carried the useful `roi=` and `area=max/sum` data; the parent `process_video_parallel.py` parser just needed to preserve those fields so one `[MLX]` line can distinguish skipped, small, and large ROI windows [Task 2]
- the formatted parent progress line now includes ROI stats plus memory fields when present, making later throughput debugging materially easier [Task 2]
- current MLX CLI defaults in this rollout remained `window=20`, auto overlap `4` when unspecified, and `max-roi-area=131072`; the user explicitly rejected changing those to `window=15 / overlap 3 / temporal ROI area 65536` as part of this follow-up [Task 3]
- in this environment, `pytest` was unavailable; `python -m unittest` was the reliable verification command for the MLX restore and `process_video_parallel.py` tests [Task 1][Task 2][Task 3]
- the current MLX path already behaves as a streaming frame pipeline rather than a “write every frame to disk” workflow; the more practical compression-like savings are mask interpolation, ROI tracking, windowing, and avoiding debug-image writes, not feeding compressed deltas directly into the restoration model [Task 4]

## Failures and how to do differently

- symptom: a connected ROI restore shows a crisp seam at the split boundary; cause: one connected ROI was spatially tiled and composited tile-by-tile; fix: stop spatially splitting connected ROIs and verify with the focused MLX ROI restore tests [Task 1]
- symptom: MLX throughput numbers look inconsistent and are hard to interpret; cause: the parent progress line dropped the ROI and memory context already emitted by the child timing line; fix: preserve `roi=`, `area=max/sum`, and memory fields in the rendered `[MLX]` output [Task 2]
- symptom: a memory-pressure follow-up accidentally undoes an accepted quality fix; cause: seam removal and default tuning were treated as one decision; fix: keep them as separate tracks and confirm any default-parameter edits explicitly before changing them [Task 1][Task 3]
- symptom: verification instructions default to `pytest` and fail in this repo environment; cause: the available test runner was assumed instead of checked; fix: use `python -m unittest` for this MLX test family unless the environment changes [Task 1][Task 2][Task 3]

# Task Group: Mac-wide operational defaults and production-sync completion criteria
scope: Use when work starts from this Mac and needs the durable default operating assumptions for shared-host access, LaunchAgent triage, bounded mail/web evidence gathering, PDF verification, or deciding whether server-bound local edits are actually complete.
applies_to: cwd=/Users/okatti/Documents/Server and cross-repo operational work from this Mac; reuse_rule=safe as a Mac-wide starting point for recurring operational workflows, but re-check live service labels, paths, schedules, deploy targets, and repo-specific runtime details each run

## Task 1: Analyze repeated work on this Mac and store durable defaults, success

### rollout_summary_files

- extensions/ad_hoc/notes/20260615-173315-mac-repeated-work-defaults.md (cwd=/Users/okatti/Documents/Server and cross-repo operational work, rollout_path=extension-note, updated_at=2026-06-15T17:33:40+09:00, thread_id=none, ad-hoc Mac-wide defaults distilled from existing memories, shell history, and current LaunchAgents)

### keywords

- launchctl, LaunchAgent, /opt/homebrew/bin/rsync, gbuc_rsync_ed25519, root@192.168.1.2, live state first, shared production host, mail ops, Apache, PDF verification, TOORICHO

## Task 2: Add the default that locally edited production scripts must be uploaded to the server, success

### rollout_summary_files

- extensions/ad_hoc/notes/20260615-173541-local-edit-upload-default.md (cwd=/Users/okatti/Documents/Server and cross-repo operational work, rollout_path=extension-note, updated_at=2026-06-15T17:35:51+09:00, thread_id=none, ad-hoc scope guard for local edits that affect server-run files)

### keywords

- production edits, server sync, upload default, server-side verification, root@192.168.1.2, gbuc_rsync_ed25519, /opt/homebrew/bin/rsync, deploy scope guard

## Task 3: Set the default communication language for progress and final updates in this environment, success

### rollout_summary_files

- extensions/ad_hoc/notes/20260628-104025-japanese-progress-updates.md (cwd=/Users/okatti/Documents/Server and cross-repo operational work, rollout_path=extension-note, updated_at=2026-06-28T10:40:25+09:00, thread_id=none, ad-hoc communication default for this environment)

### keywords

- Japanese progress updates, 日本語, status updates, final answers, communication default, cross-repo, this environment

## User preferences

- when the user asked `このMacを俯瞰して見てもらって、何度も同じ作業をしていることがあると思うんだけど、分析した上で、メモリの覚えてデフォルトにしてください。`, proactively look for repeated workflows on this Mac and promote the useful defaults into memory instead of waiting for the same steering again [Task 1]
- when the user said `ローカルでスクリプト編集などの作業をした場合には、本サーバーにも必ずアップするのもデフォルトに`, treat uploading or syncing those changes to the production server as part of the default completion criteria for server-run work, not as an optional follow-up [Task 2]
- when the ad-hoc note said `For future work in this environment, provide progress/status updates in Japanese.` and `Final answers should also prefer Japanese unless the user asks otherwise`, default progress updates and final answers to Japanese in this environment [Task 3]

## Reusable knowledge

- for operational questions on this Mac or the shared server, inspect live state first: current processes, LaunchAgent state, logs, DB rows, Maildir contents, and config files [Task 1]
- `/opt/homebrew/bin/rsync` exists and is the active `rsync` on this Mac; prefer it for LaunchAgents or scripted GBuc workflows that need `rsync` [Task 1][Task 2]
- the shared production-host transport default remains `root@192.168.1.2` with `-i /Users/okatti/.ssh/gbuc_rsync_ed25519 -o IdentitiesOnly=yes`; deployment paths, PM2 labels, and health checks stay project-specific unless the user explicitly asks to persist them [Task 1][Task 2]
- for macOS LaunchAgent troubleshooting on this Mac, start with the plist under `~/Library/LaunchAgents`, then use `launchctl print gui/$(id -u)/<label>`, `launchctl bootout gui/$(id -u) <plist>`, `launchctl bootstrap gui/$(id -u) <plist>`, and `launchctl kickstart -k gui/$(id -u)/<label>` [Task 1]
- for self-hosted mail work and Apache/web-abuse triage, default to date-bounded, server-side evidence gathering across Maildir, amavis temp/quarantine, Dovecot/Postfix/Amavis logs, bounded Apache log windows, targeted `curl` probes, and `apachectl configtest` instead of assumption-based answers or whole-file processing [Task 1]
- for production/server-bound local edits, completion includes sync/upload plus server-side verification; rediscover project-specific deployment targets, service labels, PM2 names, health checks, and restart commands from current repo/config/live state unless the user explicitly asks to persist them [Task 2]
- the communication default in this environment is Japanese for progress/status updates and final answers unless the user or artifact needs another language [Task 3]
- for recurring PDF/report workflows, verify with `pdfinfo`, `pdftotext`, and `pdftoppm` before claiming completion [Task 1]
- Related skill: skills/japanese-pdf-verification/SKILL.md [Task 1]
- for TOORICHO/news drafting automation, keep the existing defaults: search current sources, prioritize official pages, duplicate-check before drafting, stay draft-only unless publication is explicitly authorized, and do not force rights-unclear images [Task 1]

## Failures and how to do differently

- symptom: local script or operational-file edits are reported done after only local verification; cause: the workflow forgot that the file is meant to run on the shared production server; fix: upload or sync the changed file, then verify the server-side file or service state before claiming completion [Task 2]
- symptom: Mac-wide defaults start preserving project-specific deploy targets or runtime labels; cause: shared transport defaults and repo-specific deployment details were merged together; fix: keep the Mac-wide memory at the level of transport, verification posture, and first-step commands, and rediscover project-specific runtime details each run [Task 1][Task 2]

# Task Group: Mac disk cleanup and model-cache forensics
scope: Use when a local model download suddenly consumes tens of gigabytes and the user wants evidence-backed identification, safe cleanup, and space-recovery verification on this Mac.
applies_to: cwd=/Users/okatti/Documents/Server for this rollout, but operationally Mac-wide for local model-cache cleanup; reuse_rule=safe for similar local disk-forensics runs on this Mac, but re-check active model directories before deleting anything outside the confirmed cache paths

## Task 1: Find leftover model cache after a 12B download and remove only the confirmed remnants, success

### rollout_summary_files

- rollout_summaries/2026-06-18T03-49-19-1RTe-huggingface_cache_disk_cleanup_model_remnants.md (cwd=/Users/okatti/Documents/Server, rollout_path=/Users/okatti/.codex/sessions/2026/06/18/rollout-2026-06-18T12-49-19-019ed8d8-fc3f-7760-b342-ab65c00709cc.jsonl, updated_at=2026-06-18T03:53:59+00:00, thread_id=019ed8d8-fc3f-7760-b342-ab65c00709cc, local Hugging Face cache forensics plus verified deletion)

### keywords

- huggingface cache, GemmaMenuChat, pixtral-12b-4bit, gemma-4-12B, df -h, du -sh, rm -rf, /System/Volumes/Data, llama_models, 43G, 45GiB

## User preferences

- when the user asked `12Bのモデルをダウンロードしただけで30GB近く容量が減ったんだけど、どっかにくずファイルは残ってないですか？`, start with evidence-based disk forensics and separate real cache from junk rather than guessing [Task 1]
- when the user later said `全部消去して`, move from identification to direct cleanup once the large cache paths are confirmed and the protected directories have been called out [Task 1]

## Reusable knowledge

- on this Mac, `~/.cache/huggingface` was the dominant space consumer in this model-download cleanup run at `43G`; the apparent `gemma-4-12B` hits were only `0B` lock directories, not the main disk culprit [Task 1]
- the largest confirmed cache directories were `models--n0kovo--llama-joycaption-beta-one-hf-llava-mlx-8Bit` (`8.5G`), `models--mlx-community--pixtral-12b-4bit` (`6.7G`), and `models--unsloth--gemma-4-E2B-it-GGUF` (`6.0G`) under `~/.cache/huggingface/hub/` [Task 1]
- `rm -rf "$HOME/.cache/huggingface" "$HOME/.cache/GemmaMenuChat"` reclaimed about `45GiB` on this machine; after deletion, `/System/Volumes/Data` free space rose from `69Gi` to `114Gi`, while `~/llama_models` remained `17G` and was intentionally preserved [Task 1]
- before deleting anything, split findings into real model blobs, empty lock or metadata directories, and protected active-model directories such as `~/llama_models` [Task 1]

## Failures and how to do differently

- symptom: a search for the requested model name points at tiny lock directories and the real disk user remains unclear; cause: model-name grep alone overweights metadata directories; fix: run `du` on the surrounding cache roots and compare real directory sizes before concluding what to delete [Task 1]
- symptom: home-wide disk scans feel slow and noisy; cause: `du` across large home or Library trees takes time on this Mac; fix: query likely cache roots first such as `~/.cache/huggingface`, then broaden only if they do not explain the space loss [Task 1]
- symptom: cleanup risks deleting part of the active local stack; cause: cache and live model directories are mixed conceptually; fix: treat `~/llama_models` and other known active model roots as protected until the user explicitly wants them removed [Task 1]

# Task Group: booked and booked_api local LLM setup, dashboard changes, and production deployment
scope: Use when work in `booked` or sibling `booked_api` mixes local Gemma 4 setup on this Mac with admin-dashboard changes, multi-repo commits, deployment to the Cafeyu production host, or follow-up model-linkage requests.
applies_to: cwd=/Users/okatti/Documents/booked and sibling /Users/okatti/Documents/booked_api; reuse_rule=safe for these repos and this Mac's local model setup, but re-check current production paths, PM2 labels, and model/config sources before reuse

## Task 1: Set up a local CLI-accessible Gemma 4 path and pivot from Ollama to LM Studio, partial

### rollout_summary_files

- rollout_summaries/2026-04-04T09-34-58-pfeb-booked_llm_setup_calendar_bulk_delete_prod_deploy_tooricho_m.md (cwd=/Users/okatti/Documents/booked, rollout_path=/Users/okatti/.codex/sessions/2026/04/04/rollout-2026-04-04T18-34-58-019d57d8-7acf-7721-8cca-f03f621eddd3.jsonl, updated_at=2026-06-17T09:48:16+00:00, thread_id=019d57d8-7acf-7721-8cca-f03f621eddd3, local Gemma 4 setup ended on LM Studio after Ollama proved too slow)

### keywords

- ollama, LM Studio, lms, gemma4, scripts/lmstudio-chat.sh, scripts/lmstudio-models.sh, 127.0.0.1:11434, 127.0.0.1:1234, missing tensor blk.15.attn_k.weight

## Task 2: Add dashboard calendar bulk-select and bulk delete using the existing delete route, success

### rollout_summary_files

- rollout_summaries/2026-04-04T09-34-58-pfeb-booked_llm_setup_calendar_bulk_delete_prod_deploy_tooricho_m.md (cwd=/Users/okatti/Documents/booked, rollout_path=/Users/okatti/.codex/sessions/2026/04/04/rollout-2026-04-04T18-34-58-019d57d8-7acf-7721-8cca-f03f621eddd3.jsonl, updated_at=2026-06-17T09:48:16+00:00, thread_id=019d57d8-7acf-7721-8cca-f03f621eddd3, dashboard calendar bulk delete implemented and committed)

### keywords

- admin-dashboard.html, dashboardCalendarSelectedDates, bulk delete, DELETE /admin/bookings/:bookingId, Parsed 7 inline scripts, ec5c6e9

## Task 3: Commit all pending changes across booked and booked_api, success

### rollout_summary_files

- rollout_summaries/2026-04-04T09-34-58-pfeb-booked_llm_setup_calendar_bulk_delete_prod_deploy_tooricho_m.md (cwd=/Users/okatti/Documents/booked, rollout_path=/Users/okatti/.codex/sessions/2026/04/04/rollout-2026-04-04T18-34-58-019d57d8-7acf-7721-8cca-f03f621eddd3.jsonl, updated_at=2026-06-17T09:48:16+00:00, thread_id=019d57d8-7acf-7721-8cca-f03f621eddd3, both repos validated and committed cleanly)

### keywords

- booked_api, node --check, node --test utils/__tests__/*.test.js, 36/36 tests, cd97539, ec5c6e9, 全部コミットして

## Task 4: Deploy booked and booked_api to production and narrow shared memory to the SSH connection method, success

### rollout_summary_files

- rollout_summaries/2026-04-04T09-34-58-pfeb-booked_llm_setup_calendar_bulk_delete_prod_deploy_tooricho_m.md (cwd=/Users/okatti/Documents/booked, rollout_path=/Users/okatti/.codex/sessions/2026/04/04/rollout-2026-04-04T18-34-58-019d57d8-7acf-7721-8cca-f03f621eddd3.jsonl, updated_at=2026-06-17T09:48:16+00:00, thread_id=019d57d8-7acf-7721-8cca-f03f621eddd3, production deploy verified and shared-memory scope corrected)

### keywords

- root@192.168.1.2, gbuc_rsync_ed25519, IdentitiesOnly=yes, cafeyu_api, /var/www/html/booked, /var/www/html/booked_api, api.cafeyu.xyz/api/health, 接続鍵はgbuc_modernを見て

## Task 5: Carry forward the request to link TOORICHO AI draft model selection, uncertain

### rollout_summary_files

- rollout_summaries/2026-04-04T09-34-58-pfeb-booked_llm_setup_calendar_bulk_delete_prod_deploy_tooricho_m.md (cwd=/Users/okatti/Documents/booked, rollout_path=/Users/okatti/.codex/sessions/2026/04/04/rollout-2026-04-04T18-34-58-019d57d8-7acf-7721-8cca-f03f621eddd3.jsonl, updated_at=2026-06-17T09:48:16+00:00, thread_id=019d57d8-7acf-7721-8cca-f03f621eddd3, follow-up request only; no implementation evidence)

### keywords

- tooricho, AI draft, model linkage, モデルに連動, aborted follow-up

## User preferences

- when the user moved through the local model setup with `LM studioは使いません`, `1で`, and later `LMstudioだともっと速いと思うな。`, they wanted a fast tool choice followed by execution, and they were willing to pivot once the slower route was proven in practice [Task 1]
- when the user asked `管理画面のダッシュボードのカレンダーで予約が入ってる曜日にチェックボタンを追加して、チェックがある場合には一括で削除ができるようにしてください。`, they expected the concrete UI behavior to be implemented directly, not re-scoped into a backend redesign [Task 2]
- when the user said `正しくコミットしてください` and then `全部コミットして`, they wanted clean commits first, and then all remaining changes across both related repos included once they expanded the scope [Task 2][Task 3]
- when the user asked `本サーバへは？` and then `接続鍵はgbuc_modernを見て`, treat deployment as the next expected step after local completion and inspect the sibling project for the canonical SSH/rsync pattern when they point there [Task 4]
- when the user corrected the memory scope with `配置先とデプロイ手順は、プロジェクト毎に変わるので覚えなくて良いです`, keep only the shared SSH connection method as durable cross-project memory and re-discover deploy paths or PM2 names per project [Task 4]
- when the user said `toorichoのAI下書きで使うモデルに連動させて欲しいです。`, carry that forward as a real follow-up integration request even though the turn ended before implementation [Task 5]

## Reusable knowledge

- `ollama` on this Mac could be upgraded and started successfully, but `gemma4:e4b` was too slow on the M1/16GB machine for the user's latency expectations; LM Studio became the workable local path instead [Task 1]
- LM Studio is installed at `/Applications/LM Studio.app`, its CLI is `/Users/okatti/.lmstudio/bin/lms`, `lms server start` brings up the local API, and the repo now has `scripts/lmstudio-models.sh` plus `scripts/lmstudio-chat.sh` as local wrappers [Task 1]
- in this app pair, the existing delete API was already enough: `DELETE /admin/bookings/:bookingId` in `booked_api/routes/admin.js`, while the dashboard calendar state and bulk-delete flow lived entirely in `admin-dashboard.html` [Task 2]
- the validation path used for the dashboard change was a lightweight Node parse of inline scripts (`Parsed 7 inline scripts`) rather than a full browser run [Task 2]
- `booked_api` validation covered `node --check` on the changed JS files plus `node --test utils/__tests__/*.test.js`, which passed 36/36 tests before commit [Task 3]
- the verified production SSH command shape was `ssh -i /Users/okatti/.ssh/gbuc_rsync_ed25519 -o IdentitiesOnly=yes root@192.168.1.2`; within this project pair the live deploy paths were `/var/www/html/booked` and `/var/www/html/booked_api`, and the API PM2 process was `cafeyu_api` [Task 4]
- the deploy verification path here was `https://cafeyu.xyz/admin-dashboard.html` returning `200 OK` plus `https://api.cafeyu.xyz/api/health` returning `{"status":"ok", ... "environment":"production"}` after restart [Task 4]
- the model-linkage follow-up is still unresolved: future work should first find where TOORICHO stores the AI draft model selection, then wire it to the desired shared source [Task 5]

## Failures and how to do differently

- symptom: a local LLM route looks installed but still is not a good answer for the user; cause: setup success was mistaken for usable latency; fix: check actual first-response speed early and pivot if the route is technically working but too slow [Task 1]
- symptom: a Gemma 4 GGUF route fails with `missing tensor 'blk.15.attn_k.weight'`; cause: the chosen GGUF variant and binary do not match; fix: verify binary/model compatibility before assuming the local `llama.cpp` path is viable [Task 1]
- symptom: a UI patch misses its target lines; cause: the CSS or inline script block was patched by rough location instead of exact context; fix: read the adjacent block first and patch the precise section [Task 2]
- symptom: a commit sweep accidentally pulls in unrelated repo work; cause: `全部` was inferred too broadly before the user expanded scope; fix: keep commits repo-local unless the user explicitly widens the request to both repos [Task 3]
- symptom: SSH to the shared production host fails even though the host is right; cause: the default key or host-key state was wrong; fix: use the explicit GBUC rsync key with `IdentitiesOnly=yes` and fall back to the sibling project's known-good access pattern [Task 4]
- symptom: cross-project memory starts preserving deploy paths or restart commands from one app; cause: task-specific deploy details were promoted into shared defaults; fix: keep only the transport pattern as durable memory unless the user explicitly broadens the scope [Task 4]

# Task Group: TOORICHO Marugame event-news automation and research PDF verification
scope: Use when running or adapting the Marugame event-news drafting workflow, especially for duplicate suppression, draft-only registration, source verification, image-rights decisions, and required Desktop PDF research logs.
applies_to: cwd=/Users/okatti/Documents/tooricho; reuse_rule=safe for the TOORICHO Marugame event/news automation workflow, but re-check live DB rows, source URLs, and image rights on each run

## Task 1: Define the Marugame event-news automation contract and fallback behavior, uncertain

### rollout_summary_files

- rollout_summaries/2026-06-18T22-30-42-tuZV-marugame_event_news_drafting_automation.md (cwd=/Users/okatti/Documents/tooricho, rollout_path=/Users/okatti/.codex/sessions/2026/06/19/rollout-2026-06-19T07-30-42-019edcdb-a27b-7742-87e3-65c43a8c3baf.jsonl, updated_at=2026-06-18T22:30:45+00:00, thread_id=019edcdb-a27b-7742-87e3-65c43a8c3baf, newer instruction-only automation contract with explicit output bundle and automation-memory pointer)
- rollout_summaries/2026-06-10T22-30-12-rwCy-marugame_event_news_drafter_automation.md (cwd=/Users/okatti/Documents/tooricho, rollout_path=/Users/okatti/.codex/sessions/2026/06/11/rollout-2026-06-11T07-30-12-019eb3a8-5011-7193-9b05-ad0bab41161e.jsonl, updated_at=2026-06-10T22:30:16+00:00, thread_id=019eb3a8-5011-7193-9b05-ad0bab41161e, instruction-only contract)
- rollout_summaries/2026-06-08T22-31-04-YdQS-marugame_event_news_drafting_automation_setup.md (cwd=/Users/okatti/Documents/tooricho, rollout_path=/Users/okatti/.codex/sessions/2026/06/09/rollout-2026-06-09T07-31-04-019ea95c-615e-7380-8fa0-7dd6d278a102.jsonl, updated_at=2026-06-08T22:31:08+00:00, thread_id=019ea95c-615e-7380-8fa0-7dd6d278a102, setup-only automation contract with no execution evidence)

### keywords

- marugame-event-news-drafter, automation, automation memory, official sources, 45 days, this weekend, draft-only, event_date, eventDateRaw, HeiseiKakuGo-W5, Desktop PDF, checked sources, image candidates, 未登録の下書き

## Task 2: Search, dedupe, and draft Marugame event news in batch, success

### rollout_summary_files

- rollout_summaries/2026-05-31T22-31-51-P3xZ-marugame_event_news_drafts_20260601.md (cwd=/Users/okatti/Documents/tooricho, rollout_path=/Users/okatti/.codex/sessions/2026/06/01/rollout-2026-06-01T07-31-51-019e802a-3867-7461-89f5-10d11881565f.jsonl, updated_at=2026-05-31T22:37:40+00:00, thread_id=019e802a-3867-7461-89f5-10d11881565f, five draft inserts plus verified Desktop PDF)
- rollout_summaries/2026-06-03T22-30-59-koBI-marugame_event_news_drafts_and_research_pdf_20260604.md (cwd=/Users/okatti/Documents/tooricho, rollout_path=/Users/okatti/.codex/sessions/2026/06/04/rollout-2026-06-04T07-31-00-019e8f9c-847c-7f10-a476-a0e168378442.jsonl, updated_at=2026-06-03T22:37:20+00:00, thread_id=019e8f9c-847c-7f10-a476-a0e168378442, later successful two-draft run with verified PDF)

### keywords

- TOORICHO, createPost, duplicate check, dateRaw, published_at, marugame-marutasu.jp, MIMOCA, KITOKURAS, config/database, HeiseiKakuGo-W5, pdfinfo, pdftotext, pdftoppm

## Task 3: Search current sources, find only duplicates or already-registered private items, and still verify the PDF, success

### rollout_summary_files

- rollout_summaries/2026-06-11T03-47-23-8PRe-marugame_event_news_search_dedupe_pdf_no_new_drafts.md (cwd=/Users/okatti/Documents/tooricho, rollout_path=/Users/okatti/.codex/sessions/2026/06/11/rollout-2026-06-11T12-47-23-019eb4ca-b34e-79b3-a96a-48304a1b87b0.jsonl, updated_at=2026-06-11T04:31:33+00:00, thread_id=019eb4ca-b34e-79b3-a96a-48304a1b87b0, no new drafts but verified PDF and dedupe path confirmed)

### keywords

- duplicate suppression, private status, published_at, dateRaw, official sources, ReportLab, HeiseiKakuGo-W5, pdfinfo, pdftotext, pdftoppm, tooricho_contents, title LIKE, slug LIKE, body_html LIKE

## Task 4: Search current sources, register 2 draft posts with `eventDateRaw`, and verify the Japanese PDF archive, success

### rollout_summary_files

- rollout_summaries/2026-06-17T22-30-43-25S8-marugame_event_news_automation_draft_registration_and_pdf_ar.md (cwd=/Users/okatti/Documents/tooricho, rollout_path=/Users/okatti/.codex/sessions/2026/06/18/rollout-2026-06-18T07-30-44-019ed7b5-4dfa-7272-94a1-d40171a51e30.jsonl, updated_at=2026-06-17T22:37:16+00:00, thread_id=019ed7b5-4dfa-7272-94a1-d40171a51e30, two draft inserts with explicit `eventDateRaw`, duplicate suppression, local uploads, and verified PDF)

### keywords

- eventDateRaw, createPost, marugame2.jp, maroota.net, Gruun Marugame, Mooovi Marugame, tooricho_contents, id 617, id 618, /assets/uploads/, curl 200, HeiseiKakuGo-W5

## Task 5: Search current sources, register 3 Marutasu draft posts, update automation memory, and verify the Japanese PDF archive, success

### rollout_summary_files

- rollout_summaries/2026-06-21T21-35-49-jtLU-tooricho_marugame_event_news_drafts_and_research_pdf.md (cwd=/Users/okatti/Documents/tooricho, rollout_path=/Users/okatti/.codex/sessions/2026/06/22/rollout-2026-06-22T06-35-49-019eec1c-7983-7170-8864-1bc94ea99506.jsonl, updated_at=2026-06-21T21:43:28+00:00, thread_id=019eec1c-7983-7170-8864-1bc94ea99506, three Marutasu drafts plus automation memory update and verified Desktop PDF)

### keywords

- marugame-marutasu, eventDateRaw, createPost, tooricho_contents, id 625, id 626, id 627, /Users/okatti/.codex/automations/automation/memory.md, HeiseiKakuGo-W5, pdfinfo, pdftotext, pdftoppm, Bind parameters must not contain undefined

## Task 6: Search current sources, register one new official draft, and stop on unreadable CID-font PDF render, partial

### rollout_summary_files

- rollout_summaries/2026-06-28T00-52-09-2BD8-tooricho_marugame_event_news_draft_pdf_verification.md (cwd=/Users/okatti/Documents/tooricho, rollout_path=/Users/okatti/.codex/sessions/2026/06/28/rollout-2026-06-28T09-52-09-019f0bb6-5f5f-7672-8911-d7a31edce52e.jsonl, updated_at=2026-06-28T00:59:03+00:00, thread_id=019f0bb6-5f5f-7672-8911-d7a31edce52e, one official draft inserted but PDF verification failed on CID-font rendering)

### keywords

- Marutasu, createPost, eventDateRaw, id 633, /page/43743.html, /assets/uploads/news_20260809_solar_house_1.png, pdfinfo, pdftotext, pdftoppm, HeiseiKakuGo-W5, AppleGothic.ttf, Missing language pack for 'Adobe-Japan1' mapping, No font in show

## User preferences

- when running this automation, the user explicitly said `Do not publish live news automatically.` -> keep the workflow draft-only unless publication is separately authorized [Task 1][Task 2][Task 3]
- when the user later reinforced the source list with `marugame2.jp`, `maroota.net`, and `Gruun Marugame / Mooovi Marugame`, keep those as default discovery sources but still let official or organizer pages decide the final facts [Task 1][Task 4][Task 5]
- when choosing sources, the user said `Prioritize official sources, avoid duplicates and past events.` -> use official, venue, organizer, flyer, or application pages as the source of truth and treat `marugame2.jp` / `maroota.net` as discovery only [Task 1][Task 2][Task 3]
- when setting scope, the user said `Use today through the next 45 days as the main search window, treat 'this weekend' only as a supplemental check` and still consider major local events or soon-closing applications outside that window -> keep the 45-day window as the default, with narrow urgency exceptions [Task 1][Task 2][Task 3][Task 5]
- when drafting body copy, the user asked for `moderately substantial content_html`, `600-1000 Japanese characters`, `h2 section headings`, and a `strong-emphasized event name near the opening` -> avoid template-thin copy and write event-specific article text when the evidence is strong enough [Task 1][Task 2][Task 3]
- when registering event news, the user required the confirmed event start datetime to be stored as `event_date / eventDateRaw` -> always pass the actual event start time rather than relying on `published_at` [Task 1][Task 4][Task 5]
- when handling images, the user required images already uploaded to TOORICHO, user-approved, or clearly permitted for promotional reuse -> do not hotlink or reuse rights-unclear images; leaving the thumbnail blank is acceptable when rights are weak [Task 1][Task 2][Task 3][Task 5]
- when finishing the run, the user required that `Whether or not an item is registered as news` the run must summarize materially relevant findings and save a timestamped PDF on `/Users/okatti/Desktop/` -> always produce and verify the research-log PDF, even when no draft is inserted [Task 1][Task 2][Task 3][Task 5]
- when safe authenticated draft insertion is not available, the user asked for draft-ready JSON labeled `未登録の下書き` -> fall back to explicit JSON output rather than forcing unsafe posting [Task 1]
- when the user invoked `Automation: 丸亀イベントニュース下書き` with `Automation ID: automation`, treat `$CODEX_HOME/automations/automation/memory.md` as part of the required preflight context instead of relying only on old rollout summaries [Task 1][Task 5]
- when the user asked to `Report checked sources, image candidates used or skipped, draft-ready items, skipped candidates, warnings, and the verified Desktop PDF path`, include that full bundle in the PDF log and final report rather than a short recap [Task 5][Task 6]
- when the user provided an automation-memory pointer, leave a compact durable note there so future runs can suppress already-registered candidates before re-drafting [Task 5]

## Reusable knowledge

- `services/adminDbService.createPost({ postType: 'post', status: 'draft', isGlobalNews: true, dateRaw, eventDateRaw, thumbnailPath, slugRaw })` successfully inserts a TOORICHO news draft; verify the resulting row in `tooricho_contents` before calling the run complete [Task 2][Task 4][Task 5]
- for duplicate suppression, the useful early DB checks were `content_type='news' AND status <> 'trash' AND (slug = ? OR title = ?)` before insertion, and `title LIKE ? OR slug LIKE ? OR body_html LIKE ?` when matching looser near-duplicates from prior runs [Task 2][Task 3]
- in this workflow, `dateRaw` is the draft-creation timestamp and `eventDateRaw` must hold the actual event start datetime so TOORICHO’s event-date logic does not group the post under the draft day [Task 2][Task 3][Task 4][Task 5]
- `status=private` should count as already registered for duplicate avoidance, not as a free slot for a new draft [Task 3]
- in the API repo, use `./config/database`, not `./db`, when querying the production DB from scripts run inside the TOORICHO API checkout [Task 2][Task 3][Task 4]
- the newer automation contract explicitly points future runs at `$CODEX_HOME/automations/automation/memory.md` and expects the output bundle to include verified sources, skipped candidates, image candidate decisions, draft-ready items, warnings, and the verified Desktop PDF path [Task 1][Task 5]
- if `$CODEX_HOME` is empty in the shell, the working fallback path for this automation memory is `/Users/okatti/.codex/automations/automation/memory.md` [Task 5]
- saving an official image under `/var/www/html/tooricho/assets/uploads/` and confirming the public URL returns HTTP 200 is a workable rights-safe image path when the source image is clearly reusable; when the image comes from a ticketing page, keep a human-rights-review note before any future publish step [Task 2][Task 4][Task 5]
- duplicated or already-covered candidates recur often in this workflow; the run saved time by checking the live DB before spending effort on long-form drafting, and one later run showed older memory about draft IDs could already be stale because the live DB had moved those rows to `publish` [Task 2][Task 3]
- early live duplicate checks prevent wasted image/download work; in the newer run, a farm-game image was fetched before the item was confirmed as an existing published duplicate, and the latest run still narrowed several attractive candidates only after live DB review, so duplicate suppression should happen before image handling whenever possible [Task 3][Task 4][Task 5][Task 6]
- `pdfinfo`, `pdftotext`, and `pdftoppm` remain the right verification bundle, but the June 28 run showed that a CID-font PDF can exist and still render unreadably; treat `pdftoppm` warnings like `Missing language pack for 'Adobe-Japan1' mapping`, `Unknown font tag`, or `No font in show` as a failed verification until the rendered PNG is visually checked [Task 6]
- `HeiseiKakuGo-W5` is still a preferred requested font, but in this environment it was not sufficient for one June 28 research PDF; `AppleGothic.ttf` was confirmed embeddable with ReportLab `TTFont` and is the validated fallback when CID rendering breaks [Task 6]
- the June 28 official draft run inserted `id 633`, title `夏休みの自由研究にも、マルタスで親子ソーラーハウス工作教室`, `slug marutasu-solar-house-workshop-20260809`, `status='draft'`, and `event_date` corresponding to `2026-08-09 10:00 JST` [Task 6]
- the June 2026 Marutasu run verified draft IDs `625`, `626`, and `627`, each with `status='draft'`, explicit `event_date`, and local `/assets/uploads/` thumbnails that returned HTTP 200 [Task 5]
- Related skill: skills/marugame-event-news-drafting/SKILL.md [Task 1][Task 2][Task 3][Task 4][Task 5][Task 6]
- Related skill: skills/japanese-pdf-verification/SKILL.md [Task 1][Task 2][Task 3][Task 4][Task 5][Task 6]

## Failures and how to do differently

- symptom: the workflow claims the automation contract but there is no evidence of search, draft creation, or PDF output; cause: only the instruction payload was captured; fix: do not treat contract-only rollouts as execution evidence, and explicitly verify source checks, artifact creation, and saved paths before claiming success [Task 1]
- symptom: production DB probing fails even though the query logic is sound; cause: `dotenv` was not loaded, the wrong DB helper import was used, the bind parameters included `undefined`, or the query was run inline with brittle shell quoting; fix: load environment first where needed, sanitize bind values before execution, use `require('./config/database')` inside the API repo, and prefer a properly quoted Node script or file over inline shell SQL [Task 2][Task 3][Task 4][Task 5]
- symptom: the workflow spends time drafting a candidate that later proves to be a duplicate; cause: duplicate checks ran too late or older memory was trusted over the live DB; fix: query `tooricho_contents` early and treat both `publish` and `private` rows as already registered before image handling or long-form drafting [Task 2][Task 3][Task 4][Task 5]
- symptom: a candidate has only local-media or social images with unclear rights; cause: source discovery outran rights verification; fix: keep the draft text-only or skip registration instead of forcing a thumbnail [Task 2][Task 3][Task 4]
- symptom: helper processes linger after DB work completes; cause: a Node process kept the connection open; fix: confirm process exit after insertion and terminate the leftover worker if needed [Task 2]
- symptom: the automation-memory write fails because the target expands to `/automations/...`; cause: `$CODEX_HOME` was empty in the shell; fix: fall back to `/Users/okatti/.codex` before writing the memory file [Task 5]
- symptom: the PDF exists, `pdfinfo` passes, and `pdftotext` extracts text, but the rendered Japanese page is unreadable; cause: CID-font output can still break in Poppler/rendering with `Missing language pack for 'Adobe-Japan1' mapping`, `Unknown font tag`, or `No font in show`; fix: fail the verification, inspect the rendered PNG, and rebuild with an embeddable TrueType Japanese font such as `AppleGothic.ttf` before calling the Desktop PDF complete [Task 6]

# Task Group: shared production server certbot renewal and Apache dependency handling
scope: Use when `certbot renew` or Let's Encrypt validation fails on the shared production host, when `api.cafeyu.xyz` renewal cadence needs adjustment, or when Apache service state may block HTTP-01 validation.
applies_to: cwd=/Users/okatti/Documents/Server; reuse_rule=safe for this same host's certbot/Apache workflow, but re-check the live vhost, service name, renewal config, and timer state before reusing exact commands or schedules

## Task 1: Diagnose why `certbot renew` for `api.cafeyu.xyz` failed, success

### rollout_summary_files

- rollout_summaries/2026-06-17T07-54-48-4sKf-api_cafeyu_certbot_renewal_diagnosis_and_monthly_timer.md (cwd=/Users/okatti/Documents/Server, rollout_path=/Users/okatti/.codex/sessions/2026/06/17/rollout-2026-06-17T16-54-48-019ed493-5f11-7782-9aa5-0324c9c0a1d3.jsonl, updated_at=2026-07-03T10:40:25+00:00, thread_id=019ed493-5f11-7782-9aa5-0324c9c0a1d3, live root-cause diagnosis for the failed `api.cafeyu.xyz` renewal)

### keywords

- certbot renew, api.cafeyu.xyz, letsencrypt, httpd2.service, http-01, webroot, connection refused, /etc/letsencrypt/renewal/api.cafeyu.xyz.conf, ss -ltnp, /var/log/letsencrypt/letsencrypt.log.1

## Task 2: Enable automatic renewal and change the cadence to monthly, success

### rollout_summary_files

- rollout_summaries/2026-06-17T07-54-48-4sKf-api_cafeyu_certbot_renewal_diagnosis_and_monthly_timer.md (cwd=/Users/okatti/Documents/Server, rollout_path=/Users/okatti/.codex/sessions/2026/06/17/rollout-2026-06-17T16-54-48-019ed493-5f11-7782-9aa5-0324c9c0a1d3.jsonl, updated_at=2026-07-03T10:40:25+00:00, thread_id=019ed493-5f11-7782-9aa5-0324c9c0a1d3, enabled the existing timer, added hooks, and validated the monthly override)

### keywords

- certbot-renew.timer, monthly override, OnCalendar=monthly, RandomizedDelaySec=6h, Persistent=true, renewal-hooks, ensure-httpd2-running.sh, reload-httpd2.sh, root crontab, openssl s_client, certbot renew --dry-run

## User preferences

- when the user asked `なぜ？`, they wanted the exact renewal root cause from live logs and listener state, not a config-only guess [Task 1]
- when the user pushed back with `絶伊更新がないのに1日2回って馬鹿らしいでしょ。`, prefer lower-noise maintenance schedules over technically acceptable but wasteful defaults [Task 2]
- when the user said `1ヶ月に一回でできるならそうして`, monthly checks were the explicit preferred cadence for this certificate workflow [Task 2]

## Reusable knowledge

- on this host, `api.cafeyu.xyz` renews through certbot `webroot` using `/var/www/html/booked_api`, and the relevant Apache service name is `httpd2.service` [Task 1]
- the failure signature for this rollout was Let’s Encrypt fetching `http://api.cafeyu.xyz/.well-known/acme-challenge/...` on `219.117.224.111:80` and getting `Connection refused`; the renewal config was otherwise aligned, so Apache downtime was the real blocker [Task 1]
- `certbot renew --cert-name api.cafeyu.xyz` can still print `Certificate not yet due for renewal` after the root cause is fixed because the live cert may already be valid; in this run the lineage was valid through `2026-10-01` [Task 1]
- the host did not use a root `crontab` entry for certbot; the active automation path is `certbot-renew.timer`, and the working monthly override lives at `/etc/systemd/system/certbot-renew.timer.d/monthly.conf` [Task 2]
- the verified safety hooks were `/etc/letsencrypt/renewal-hooks/pre/ensure-httpd2-running.sh` to start/check Apache before renewal and `/etc/letsencrypt/renewal-hooks/deploy/reload-httpd2.sh` to reload after success [Task 2]
- after the override, the timer was `enabled` and `active`, with the next run reported as `2026-08-01 01:05:24 JST`, and the served certificate still matched the renewed lineage [Task 2]

## Failures and how to do differently

- symptom: certbot renewal looks like a webroot or file-layout problem; cause: only config files were inspected; fix: check certbot logs, `systemctl status httpd2`, and live listeners on `:80` / `:443` before changing renewal files [Task 1]
- symptom: HTTP-01 validation fails with `Connection refused`; cause: Apache is stopped even though vhost and webroot settings are correct; fix: restore `httpd2.service`, confirm the challenge path returns `200`, and then rerun renewal verification [Task 1][Task 2]
- symptom: the host starts checking renewals too often for the user's tolerance; cause: the default `certbot-renew.timer` cadence was left in place; fix: override the timer with a host-native monthly drop-in instead of adding duplicate cron automation [Task 2]
- symptom: certbot hook output becomes noisy enough to obscure the meaningful lines; cause: `apachectl configtest` prints `Syntax OK` on every run; fix: redirect the pre-hook configtest output to keep certbot logs readable [Task 2]

# Task Group: Apache vhost additions on the self-hosted production host
scope: Use when adding or correcting live Apache HTTP/HTTPS vhosts on the shared production host, especially topmode-okada.jp-style additions with php-fpm and existing Let’s Encrypt certificates.
applies_to: cwd=/Users/okatti/Documents/Server; reuse_rule=safe for this same host’s Apache vhost workflow, but re-check the exact domain, docroot, certificate path, and active service name before reusing the details

## Task 1: Add Apache HTTP and HTTPS vhosts for topmode-okada.jp, success

### rollout_summary_files

- rollout_summaries/2026-06-16T08-40-28-mthc-apache_topmode_okada_vhost_addition.md (cwd=/Users/okatti/Documents/Server, rollout_path=/Users/okatti/.codex/sessions/2026/06/16/rollout-2026-06-16T17-40-29-019ecf96-d446-7663-942e-94cc4dd2fe6e.jsonl, updated_at=2026-06-16T08:45:42+00:00, thread_id=019ecf96-d446-7663-942e-94cc4dd2fe6e, topmode-okada.jp vhosts added and validated on the live host)

### keywords

- topmode-okada.jp, httpd-vhosts.conf, ServerName, SSL, php8-fpm, httpd2.service, apachectl configtest, curl -Ik, typoでした, 正確にはhttp://topmode-okada.jpです

## User preferences

- when the user corrected the hostname with `typoでした。正確にはhttp://topmode-okada.jpです。`, treat exact domain spelling as critical before editing live Apache config [Task 1]
- when the user asked for both `http://...` and `https://...` settings together, implement matching HTTP and HTTPS vhosts in the same pass unless they narrow the scope [Task 1]

## Reusable knowledge

- on this host, Apache vhost edits belong in `/usr/local/apache2/conf/extra/httpd-vhosts.conf`, the topmode site root already existed at `/var/www/html/topmode`, and the active certificate path was `/etc/letsencrypt/live/topmode-okada.jp-0001/` [Task 1]
- the working PHP handler for the new topmode vhosts used `SetHandler "proxy:unix:/var/run/php8-fpm.sock|fcgi://localhost"`, and the running Apache unit to reload was `httpd2.service`, not `httpd.service` [Task 1]
- the reliable validation path here was `/usr/local/apache2/bin/apachectl configtest` plus targeted `curl -Ik` checks for both `http://topmode-okada.jp/` and `https://topmode-okada.jp/` after reload [Task 1]

## Failures and how to do differently

- symptom: `systemctl reload httpd` fails on this host; cause: the active Apache unit is `httpd2.service`, not `httpd.service`; fix: check the live unit with `systemctl is-active` first and reload `httpd2` when that is the running service [Task 1]
- symptom: a live vhost edit gets applied to the wrong domain; cause: hostname spelling was assumed instead of rechecked; fix: confirm the exact `ServerName` string before editing and validate both schemes with targeted `curl` probes after reload [Task 1]

# Task Group: self-hosted mail maintenance, spam hardening, and Spark sync optimization on the shared mail host
scope: Use when improving server-side spam handling, diagnosing why spam passed the current thresholds, running live Maildir spam triage, running daily mail-learning automation, reducing IMAP sync load, or verifying blocked-spam disposition on the shared mail host.
applies_to: cwd=/Users/okatti/Documents/gbuc_modern and /Users/okatti/Documents/Server for the same shared self-hosted mail host workflow; reuse_rule=safe for the same hosted-mail environment, but validate current mailbox layout, amavis policy, LaunchAgent state, and current spoof patterns before reusing exact paths or counts

## Task 1: Investigate why nightly English spam was arriving and harden the live filtering stack, success

### rollout_summary_files

- rollout_summaries/2026-06-03T19-45-25-HIBR-gbuc_mail_spam_root_cause_and_hardening.md (cwd=/Users/okatti/Documents/gbuc_modern, rollout_path=/Users/okatti/.codex/sessions/2026/06/04/rollout-2026-06-04T04-45-25-019e8f04-ef1b-76b2-8535-5d4ead3ee9c3.jsonl, updated_at=2026-06-03T20:21:33+00:00, thread_id=019e8f04-ef1b-76b2-8535-5d4ead3ee9c3, root-cause analysis plus verified live hardening)

### keywords

- English spam, amavis, SpamAssassin, fail2ban, sender_access, X-Spam-Status, low-score promo spam, open relay, smtpd_sender_restrictions

## Task 2: Learn spam from today’s received `.com` mail and harden SpamAssassin rules, success

### rollout_summary_files

- rollout_summaries/2026-06-04T21-30-18-xJ7z-gbuc_mail_spam_cleanup_and_spark_sync_optimization.md (cwd=/Users/okatti/Documents/gbuc_modern, rollout_path=/Users/okatti/.codex/sessions/2026/06/05/rollout-2026-06-05T06-30-18-019e948b-4f2e-7c43-9090-ad2cbb3214c1.jsonl, updated_at=2026-06-05T21:40:12+00:00, thread_id=019e948b-4f2e-7c43-9090-ad2cbb3214c1, live server spam-learning and rule edits verified)

### keywords

- sa-learn, SpamAssassin, amavis, vusers, /var/spool/virtualmailbox, /etc/mail/spamassassin/local.cf, D_DISCARD, fake Amazon, Walmart Store

## Task 3: Build daily spam-learning automation for all virtual mailboxes and install a LaunchAgent, success

### rollout_summary_files

- rollout_summaries/2026-06-04T21-30-18-xJ7z-gbuc_mail_spam_cleanup_and_spark_sync_optimization.md (cwd=/Users/okatti/Documents/gbuc_modern, rollout_path=/Users/okatti/.codex/sessions/2026/06/05/rollout-2026-06-05T06-30-18-019e948b-4f2e-7c43-9090-ad2cbb3214c1.jsonl, updated_at=2026-06-05T21:40:12+00:00, thread_id=019e948b-4f2e-7c43-9090-ad2cbb3214c1, launchd automation installed and observed running)

### keywords

- gbuc_spam_watch.py, install_gbuc_spam_watch.sh, LaunchAgent, com.okatti.gbuc-spam-watch.plist, processed.jsonl, launchctl print, StartCalendarInterval, 05:30

## Task 4: Archive old INBOX mail and delete legacy folders to reduce Spark sync load, success

### rollout_summary_files

- rollout_summaries/2026-06-04T21-30-18-xJ7z-gbuc_mail_spam_cleanup_and_spark_sync_optimization.md (cwd=/Users/okatti/Documents/gbuc_modern, rollout_path=/Users/okatti/.codex/sessions/2026/06/05/rollout-2026-06-05T06-30-18-019e948b-4f2e-7c43-9090-ad2cbb3214c1.jsonl, updated_at=2026-06-05T21:40:12+00:00, thread_id=019e948b-4f2e-7c43-9090-ad2cbb3214c1, old INBOX mail moved and legacy folders deleted after verification)

### keywords

- doveadm move, savedbefore 30d, Archive, Spark, INBOX_TOTAL, OLD_TOTAL 0, .legacy-*, maildir:/var/spool/virtualmailbox/%d/%n/

## Task 5: Verify whether blocked spam is discarded or quarantined, success

### rollout_summary_files

- rollout_summaries/2026-06-04T21-30-18-xJ7z-gbuc_mail_spam_cleanup_and_spark_sync_optimization.md (cwd=/Users/okatti/Documents/gbuc_modern, rollout_path=/Users/okatti/.codex/sessions/2026/06/05/rollout-2026-06-05T06-30-18-019e948b-4f2e-7c43-9090-ad2cbb3214c1.jsonl, updated_at=2026-06-05T21:40:12+00:00, thread_id=019e948b-4f2e-7c43-9090-ad2cbb3214c1, amavis disposition checked on the host)

### keywords

- blockしたspamは捨ててる, amavisd.conf, final_spam_destiny, D_DISCARD, QUARANTINE_COUNT 0, Blocked SPAM {DiscardedOpenRelay,Quarantined}

## Task 6: Record the effective amavis Bayes DB path for future shared-host learning, success

### rollout_summary_files

- extensions/ad_hoc/notes/20260628-103947-amavis-bayes-spam-learning.md (cwd=/Users/okatti/Documents/Server and /Users/okatti/Documents/gbuc_modern on the shared mail host, rollout_path=extension-note, updated_at=2026-06-28T10:39:47+09:00, thread_id=none, ad-hoc note clarifying the effective Bayes DB for amavis filtering)

### keywords

- /var/spool/amavisd/.spamassassin/bayes, sa-learn --dbpath, sa-learn --sync, amavis:amavis, content_filter smtp-amavis, /root/.spamassassin, [ad-hoc note]

## User preferences

- when the user asked `spam学習してほしい` for mail received `今日0時から今まで`, default to a date-bounded server-side scan instead of a generic mailbox cleanup [Task 1][Task 2]
- when the user expanded scope to `対象を全てのメールアドレスに拡大してください`, cover all server mailboxes unless the user narrows it [Task 2][Task 3]
- when the user said `1日毎でいいです`, daily scheduling is the preferred cadence for this workflow [Task 3]
- when the user wanted to reduce Spark’s `syncing with server` slowdown and asked to keep only `1ヶ月分だけINBOXに残して、あとはアーカイブに移動`, prioritize practical mailbox cleanup with measurable sync impact, not theory [Task 4]
- when the user later said `消去してください` after the legacy-folder check, complete cleanup once the folders are verified empty instead of leaving temporary backups around [Task 4]
- when the user asked `blockしたspamは捨ててる？`, answer from the actual amavis policy and filesystem state, not from assumptions about the word `Quarantined` in logs [Task 5]

## Reusable knowledge

- the nightly English spam was inbound low-score promotional mail that passed the current amavis/SpamAssassin thresholds; it was not evidence of an open relay or an application-side sender bug [Task 1]
- on this host, the effective mail store for the main accounts is `/var/spool/virtualmailbox/%d/%n/`, with Dovecot using `maildir:/var/spool/virtualmailbox/%d/%n/` [Task 1][Task 2][Task 4]
- `sa-learn` against `amavis` or `vusers` via direct file access can fail on permissions; root-mediated access or stdin-fed learning is safer [Task 2]
- SpamAssassin custom rules already live in `/etc/mail/spamassassin/local.cf`; appending narrow relay, lure, and subject rules plus `spamassassin --lint` was the working update path [Task 1][Task 2]
- `amavisd` is configured with `D_DISCARD`, so blocked spam is dropped rather than delivered [Task 1][Task 2][Task 5]
- the daily job is scheduled through `~/Library/LaunchAgents/com.okatti.gbuc-spam-watch.plist`, with the runtime wrapper in `~/Library/Scripts/GbucSpamWatch/run-gbuc-spam-watch.sh` and dedupe state in `~/Library/Application Support/GbucSpamWatch/processed.jsonl` [Task 3]
- the automation’s processed-key design includes recipient, message-id, and path, which matters because amavis temp paths can be reused while the message content changes [Task 3]
- Dovecot and the mailbox layout are compatible with `doveadm search` and `doveadm move` for archive cleanup and spam-adjacent mailbox operations [Task 3][Task 4]
- moving INBOX mail older than 30 days into `Archive` materially reduced synchronization work; on the checked mailbox, INBOX dropped from roughly 35k messages to about 3.2k after cleanup [Task 4]
- Dovecot `mailbox_list_index = yes` was already enabled, so the Spark slowdown was better explained by client-side synchronization work than by server corruption or disabled indexing [Task 4]
- on this host, inbound mail filtering goes through `content_filter = smtp-amavis:[127.0.0.1]:10024`, so `sa-learn` must target `/var/spool/amavisd/.spamassassin/bayes`, not only `/root/.spamassassin` [Task 6]
- after live learning, run `sa-learn --dbpath /var/spool/amavisd/.spamassassin/bayes --sync` and keep `/var/spool/amavisd/.spamassassin/bayes_*` owned by `amavis:amavis` [Task 6]
- Related skill: skills/server-mail-spam-triage/SKILL.md [Task 2]

## Failures and how to do differently

- symptom: the investigation jumps to “open relay” or “app bug” too early; cause: the actual message scores and mail path were not checked first; fix: inspect headers, `X-Spam-Status`, and current threshold rules before diagnosing the source of the spam [Task 1]
- symptom: `sa-learn` fails for `amavis` or `vusers`; cause: those users cannot read the candidate files directly; fix: feed the message body from root or use root-owned access [Task 2]
- symptom: a spam scan misses obvious candidates; cause: the search starts only under a home `Maildir`; fix: include `/var/spool/virtualmailbox`, `/var/spool/amavisd/tmp`, and `/var/spool/amavisd/quarantine` from the start [Task 1][Task 2]
- symptom: new spam rules also match legitimate newsletters; cause: the pattern set is too broad; fix: keep rules narrow and re-score representative ham before widening them [Task 1][Task 2]
- symptom: remote learning transport or inline execution breaks; cause: brittle quoting and stdin handling; fix: switch to a base64 payload over SSH or another simpler transport [Task 3]
- symptom: one archive pass still leaves old messages in INBOX; cause: the 30-day boundary moved during the run; fix: run a second pass and verify `OLD_TOTAL 0` [Task 4]
- symptom: Spark sync lag gets blamed on the server by default; cause: the client’s many `STATUS`, `UID SEARCH`, and `UID FETCH` operations on exit were not checked; fix: inspect Dovecot logs and mailbox counts before assuming a server-side fault [Task 4]
- symptom: logs say `Quarantined` and the result is misreported as recoverable quarantine storage; cause: the label is read without checking the actual filesystem state; fix: confirm `/var/spool/amavisd/quarantine` before saying messages are retained [Task 5]
- symptom: `sa-learn` appears to succeed from root but later spam filtering does not improve; cause: learning updated `/root/.spamassassin` instead of the amavis Bayes DB used by the live filter path; fix: run `sa-learn --dbpath /var/spool/amavisd/.spamassassin/bayes --spam`, follow with `--sync`, and preserve `amavis:amavis` ownership on `bayes_*` [Task 6]

# Task Group: TOORICHO API AI draft required-note and OCR enforcement
scope: Use when changing the TOORICHO AI draft-generation backend around server-enforced draft instructions, OCR-assisted image/flyer reading, or PM2 deployment verification after backend changes.
applies_to: cwd=/Users/okatti/Documents/tooricho_api and related TOORICHO admin draft pipeline work; reuse_rule=safe for the separate TOORICHO API service, but confirm live PM2 behavior and the current prompt builder before reusing exact implementation details

## Task 1: Enforce the required draft note server-side and improve flyer OCR before generation, success

### rollout_summary_files

- rollout_summaries/2026-05-11T01-13-37-K0bj-tooricho_ai_draft_prompt_ocr_note_enforcement.md (cwd=/Users/okatti/Documents/tooricho, rollout_path=/Users/okatti/.codex/sessions/2026/05/11/rollout-2026-05-11T10-13-37-019e1498-c794-77b3-a1ae-9504ab84476d.jsonl, updated_at=2026-06-11T04:35:14+00:00, thread_id=019e1498-c794-77b3-a1ae-9504ab84476d, server-enforced required note plus OCR and rotation support)

### keywords

- REQUIRED_DRAFT_NOTE, ai下書きの指示note, 600-1000文字, h2, strong, services/aiContentService.js, services/materialOcrService.js, Vision, Tesseract, base64 data URL, pm2 restart tooricho-api

## User preferences

- when the user said `ai下書きの指示noteに必須項目として書いてください`, treat the draft-shape instruction as a server-enforced rule, not a UI-only suggestion [Task 1]
- when the user asked to translate and then preserve the same sentence, keep the required note close to the original wording so future agents can recognize the exact contract: `本文HTMLをややしっかりめの分量で作成してください。目安は日本語で600〜1000文字程度です。既存のTOORICHOニュース記事のようにh2の見出しを使い、冒頭付近でイベント名をstrongで強調してください。` [Task 1]

## Reusable knowledge

- in this repo, `services/aiContentService.js` `buildPrompt()` is the effective insertion point for server-side draft rules; putting the required note there makes it independent of browser-side input [Task 1]
- `tests/ai-content-prompt.test.mjs` was the verification path for ensuring the required note is always injected [Task 1]
- OCR support lives in `services/materialOcrService.js`, using Vision on macOS and Tesseract on the Linux production server [Task 1]
- for ticket or flyer images, feeding OCR text plus rotated image variants improved downstream extraction quality more than OCR text alone [Task 1]
- after deploying backend changes, the safe verification path was `pm2 restart tooricho-api --update-env` followed by `curl http://127.0.0.1:3002/api/health` [Task 1]

## Failures and how to do differently

- symptom: `image_url` input fails with `error: cannot make GET request`; cause: the model backend could not fetch the public URL reliably; fix: convert the image to a base64 data URL server-side before sending it to Gemma [Task 1]
- symptom: OCR misses sideways event dates on flyers; cause: only the original orientation was sent; fix: include 90-degree and 270-degree rotated variants alongside the original image [Task 1]
- symptom: a macOS-only OCR approach works locally but not on the server; cause: Vision is not available on the Linux production host; fix: keep the OCR implementation split between Vision and Tesseract [Task 1]

# Task Group: central-city revitalization daily monitoring report automation
scope: Use when running the broad daily news-monitoring workflow for central-city revitalization, shopping-street issues, and subsidy/program updates, including the required Desktop PDF report.
applies_to: cwd=/Users/okatti/Documents/商店街の問題; reuse_rule=safe for this daily monitoring workflow, but search results and local updates are day-specific and must be refreshed each run

## Task 1: Define the daily monitoring report scope, dedupe rules, and fixed output order, uncertain

### rollout_summary_files

- rollout_summaries/2026-06-19T22-00-36-bmqI-central_city_news_monitoring_daily_report_pdf.md (cwd=/Users/okatti/Documents/商店街の問題, rollout_path=/Users/okatti/.codex/sessions/2026/06/20/rollout-2026-06-20T07-00-36-019ee1e6-7111-7542-bc63-0b2f3eb19de8.jsonl, updated_at=2026-06-19T22:00:42+00:00, thread_id=019ee1e6-7111-7542-bc63-0b2f3eb19de8, fresher requirements wording for the separate daily monitor)
- rollout_summaries/2026-06-10T22-01-42-i2ic-daily_center_city_revitalization_news_monitoring_report.md (cwd=/Users/okatti/Documents/商店街の問題, rollout_path=/Users/okatti/.codex/sessions/2026/06/11/rollout-2026-06-11T07-01-42-019eb38e-382f-7462-94bd-8e79ee0e38c0.jsonl, updated_at=2026-06-10T22:01:45+00:00, thread_id=019eb38e-382f-7462-94bd-8e79ee0e38c0, workflow contract for the daily monitor)
- rollout_summaries/2026-06-08T22-01-34-4VU0-automation_3_central_city_revitalization_news_monitoring_spe.md (cwd=/Users/okatti/Documents/商店街の問題, rollout_path=/Users/okatti/.codex/sessions/2026/06/09/rollout-2026-06-09T07-01-34-019ea941-5ef9-7061-8d69-0c311a7e0334.jsonl, updated_at=2026-06-08T22:01:37+00:00, thread_id=019ea941-5ef9-7061-8d69-0c311a7e0334, requirements-only automation specification)

### keywords

- 中心市街地活性化, 商店街再生, ニュース監視, automation-3, 更新理由, 既報除外, 新規性の高い情報は限定的, 丸亀市・香川県, 補助金

## Task 2: Define the PDF export artifact and final-report path requirements, uncertain

### rollout_summary_files

- rollout_summaries/2026-06-19T22-00-36-bmqI-central_city_news_monitoring_daily_report_pdf.md (cwd=/Users/okatti/Documents/商店街の問題, rollout_path=/Users/okatti/.codex/sessions/2026/06/20/rollout-2026-06-20T07-00-36-019ee1e6-7111-7542-bc63-0b2f3eb19de8.jsonl, updated_at=2026-06-19T22:00:42+00:00, thread_id=019ee1e6-7111-7542-bc63-0b2f3eb19de8, explicit PDF export and reporting contract)
- rollout_summaries/2026-06-08T22-01-34-4VU0-automation_3_central_city_revitalization_news_monitoring_spe.md (cwd=/Users/okatti/Documents/商店街の問題, rollout_path=/Users/okatti/.codex/sessions/2026/06/09/rollout-2026-06-09T07-01-34-019ea941-5ef9-7061-8d69-0c311a7e0334.jsonl, updated_at=2026-06-08T22:01:37+00:00, thread_id=019ea941-5ef9-7061-8d69-0c311a7e0334, earlier PDF artifact requirements)

### keywords

- PDF出力, /Users/okatti/Desktop, 中心市街地活性化_ニュース監視_YYYY-MM-DD.pdf, 作成日, 補助金表, 参照リンク, 見出し, ページ番号

## User preferences

- when this workflow comes up, the user corrected: `朝6時の『中心市街地活性化 事例深掘り日次レポート』とは別物` and `1都市深掘りではなく、当日の新規・更新情報の監視` -> treat it as a broad daily monitor, not a city-specific deep dive [Task 1]
- when the user said `一度報告済みのニュース、政策、事例、補助事業、補助金、助成金は原則として重複掲載しない`, future runs should default to diff-based reporting and avoid repeating previously reported items [Task 1]
- when a previously covered item is repeated, the user required `更新理由を1行で明記` -> state the material change explicitly instead of re-listing it as generic news [Task 1]
- when the day is quiet, the user said `新規情報が少ない日は、無理に既報を繰り返さず『新規性の高い情報は限定的』と明記` -> prefer honest scarcity reporting over filler [Task 1]
- when covering local follow-through, the user limited 丸亀市/香川県 to `未報告の新規情報または重要な更新` and otherwise wanted `丸亀市・香川県の新規重要情報は確認されず` in one line -> check them every run but keep the section short when nothing changed [Task 1]
- when the report is finished, the user required `毎回のレポート本文をPDF化` to `/Users/okatti/Desktop` as `中心市街地活性化_ニュース監視_YYYY-MM-DD.pdf`, and said the final report should include the absolute PDF path -> treat the PDF artifact and reported path as part of done [Task 1][Task 2]

## Reusable knowledge

- this automation is for fresh or updated domestic and overseas information on central-city revitalization, shopping-street problems, and subsidy/program changes; it is not the 6am deep-dive report [Task 1]
- `automation-3` is the recurring daily monitor for `cwd=/Users/okatti/Documents/商店街の問題`; earlier evidence also pointed future runs at `$CODEX_HOME/automations/automation-3/memory.md` before drafting [Task 1]
- the monitoring scope includes Japanese shopping-street decline issues, central-city revitalization, walkable policy, vacant-store reuse, tourism linkage, public-transport/parking policy, public-private partnership, redevelopment, social experiments, and city-making corporations/councils; overseas coverage includes old-town, port-town, heritage, high-street, market, creative-district regeneration, 15-minute city, temporary use, and culture-anchored downtown renewal [Task 1]
- the report output order is fixed: `(1) 今日の要点3〜5件, (2) 先進事例/現状分析/課題認識, (3) 新規/更新動向, (4) 新規/更新ありの支援制度・補助事業・補助金, (5) 丸亀市・香川県の新規重要情報の有無, (6) 丸亀市への応用可能性, (7) 参照リンク` [Task 1]
- official-source coverage should start with 中小企業庁, 経済産業省, 国土交通省, 観光庁, 内閣府地方創生, J-Net21, ミラサポplus, 全国商店街振興組合連合会, 全国中心市街地活性化協議会, major municipalities/chambers/city-making corporations, and official overseas case pages such as OECD, UN-Habitat, UNESCO, and Main Street America as needed [Task 1]
- overseas examples should include a short application note for Japanese local shopping streets or Marugame plus a caution about institutional, cultural, and urban-scale differences [Task 1]
- subsidy and program summaries should stay compact but include `制度名`, `対象者`, `補助率/上限額` or support contents, `対象経費/対象活動`, `募集期間/締切`, `申請先`, `公式リンク`, and shop-street use cases [Task 1]
- the report should explicitly say `新規性の高い情報は限定的` when the day is quiet rather than filling the PDF with repeated material [Task 1]
- the PDF must include `作成日`, report body, subsidy table, reference links, readable headings, and page numbers, and the required output path pattern is `/Users/okatti/Desktop/中心市街地活性化_ニュース監視_YYYY-MM-DD.pdf` [Task 2]
- Related skill: skills/japanese-pdf-verification/SKILL.md [Task 2]

## Failures and how to do differently

- symptom: the report starts drifting into a single-city or deep-dive format; cause: it got conflated with the separate morning automation; fix: keep this workflow broad, daily, and update-focused [Task 1]
- symptom: repeated subsidy or policy items keep reappearing without justification; cause: dedupe happened too late or without explicit update triggers; fix: re-report only for deadline extensions, budget changes, opening or closing of recruitment, requirement changes, adoption results, or similar material deltas, and state the update reason in one line [Task 1]
- symptom: the local Marugame/Kagawa section becomes filler when there is no local change; cause: the report tried to force locality on a quiet day; fix: use the requested short negative line instead [Task 1]
- symptom: the workflow is reported as executed without proof; cause: only the instruction payload was preserved; fix: keep execution status uncertain until sources checked, PDF written, and output path confirmed [Task 1][Task 2]
- symptom: the PDF path is reported without artifact proof; cause: the run stopped at text requirements or file-existence only; fix: confirm the file exists and run the verification bundle from `skills/japanese-pdf-verification/SKILL.md` before claiming completion [Task 2]
