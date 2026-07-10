thread_id: 019f0d2a-09a3-7602-8859-973866636373
updated_at: 2026-06-29T09:04:01+00:00
rollout_path: /Users/okatti/.codex/archived_sessions/rollout-2026-06-28T16-38-06-019f0d2a-09a3-7602-8859-973866636373.jsonl
cwd: /Users/okatti/Documents/lada
git_branch: main

# LADA mosaic restoration dataset/training work on macOS, with several pivots from rough auto-extraction to filtered data prep and MPS speedups

Rollout context: The work happened in `/Users/okatti/Documents/lada` on a Mac mini / Apple Silicon environment, using FC2 video sources under `/Volumes/Firewire_HD3/movies/FC2/` and dataset output under `/Volumes/Project_HD/`. The user repeatedly steered the agent toward practical commands and away from vague or overly automatic extraction, and later asked for file renaming and processed-material cleanup.

## Task 1: Understand and run the mosaic restoration dataset creator

Outcome: success

Preference signals:

- The user asked `どうすればよいか、教えて` and later kept narrowing toward runnable commands, indicating they wanted concrete command lines and not just conceptual guidance.
- When the agent proposed a rough extraction flow, the user pushed back with `切り出し方があまり言いように思えないんだけど、もらってるコマンドは一番最善？` and `抽出しているのがあまりよいデーターとは思えない` -> they want the data-prep method evaluated for quality, not blindly accepted.
- When the agent suggested symlinks for convenience, the user said `symlinkはいやだ、コピーして` -> future runs should prefer copying over symlinking when the user is making local data/layout fixes.

Key steps:

- Read `scripts/dataset_creation/create-mosaic-restoration-dataset.py` and the training docs to reconstruct the script’s purpose and defaults.
- Verified the environment: `python` worked, `torch` and `ultralytics` were installed, but `uv` via pyenv shim was not usable from the shell.
- Confirmed the script’s defaults include `--model-device cuda`, `--video-quality-model-device cuda`, `--no-add-watermark-metadata`, `--no-add-nudenet-nsfw-metadata`, and `--no-add-censor-metadata` as the initial fast path; later filtering can be turned on selectively.
- Verified that the script initially failed because `timm` was missing, and later because required weights were missing or in the wrong place.
- Located the real source video under `/Volumes/Firewire_HD3/movies/FC2/` after the original `/Volumes/Project_HD/training` path turned out to be empty.

Failures and how to do differently:

- The early “just try a small clip” approach was only good for debugging, not for quality dataset creation. The user was right that the first cut was too naive.
- The dataset creator does not judge “good training material” by itself; it only gathers scenes that pass detector/filter rules. Human cleanup is still needed.
- DOVER-based quality filtering initially failed because the repo expected the weight under `lada/model_weights/...`; the fix was to copy `model_weights` into `lada/model_weights` instead of using a symlink.
- The NudeNet-related path was confusing because the old `640m.pt` download was HTML/invalid, while `pip install nudenet>=3.4.2` installs a different ONNX-based package and model (`320n.onnx`) that the LADA code did not yet use.

Reusable knowledge:

- The script’s selection logic is mechanical: it tracks NSFW detections frame-by-frame, keeps connected detections as scenes, then filters by scene length, stride, and optional quality/watermark/NudeNet/censor checks.
- `--stride-length` does not mean “skip input frames”; it controls spacing between accepted scenes after detection. The heavy work still processes the video stream.
- `--enable-video-quality-filter` is the main built-in knob for dropping low-quality scenes, but `0.25` is very loose; in the rollout’s real filtered set, a threshold near `0.30` was a more meaningful cutoff.
- For this repo, quality filtering requires `model_weights/3rd_party/DOVER.pth`, and the model path resolution was sensitive to `lada/model_weights` vs repo-root `model_weights`.

References:

- [1] `python scripts/dataset_creation/create-mosaic-restoration-dataset.py --input ... --output-root ...` is the main entrypoint in `docs/training_and_dataset_creation.md`.
- [2] `create-mosaic-restoration-dataset.py` defaults: `--model-device cuda`, `--video-quality-model-device cuda`, `--nudenet-nsfw-model-path model_weights/3rd_party/640m.pt`, `--censor-model-path model_weights/lada_mosaic_detection_model_v2.pt`.
- [3] `NudeNetNsfwDetector` in `lada/datasetcreation/detectors/nudenet_nsfw_detector.py` still expects a `Yolo` wrapper and does not directly use the installed `nudenet` package.
- [4] DOVER path issue fixed by copying `model_weights` into `lada/model_weights` rather than symlinking; after that, `VideoQualityEvaluator(device='mps')` initialized successfully.

## Task 2: Fine-tune the existing restoration model on Apple Silicon, and speed it up with MPS deform-conv

Outcome: success

Preference signals:

- The user asked `今あるrestorationモデルに追加学習させるのは？` and later `追加学習のコマンド出して` -> they want the existing model fine-tuned rather than training from scratch.
- When the agent proposed a training setup, the user kept asking for the exact command and then whether the chosen extraction/data quality was actually good, indicating they want runnable, justified commands rather than theoretical advice.
- When the agent suggested a symlink for `lada/model_weights`, the user rejected it and demanded a copy, so future fixes should avoid symlink-based shortcuts unless the user explicitly accepts them.

Key steps:

- Confirmed `model_weights/lada_mosaic_restoration_model_generic_v1.2_full.pth` is a full MMEngine checkpoint with `state_dict`, `optimizer`, and `meta.iter = 52000`, while `v1.2.pth` is just a state dict and not a good fine-tuning checkpoint.
- Used `configs/basicvsrpp/mosaic_restoration_generic_stage2.py` as the fine-tuning base and `scripts/training/train-mosaic-restoration-basicvsrpp.py` with `--load-from model_weights/lada_mosaic_restoration_model_generic_v1.2_full.pth`.
- Verified that training with `PYTORCH_ENABLE_MPS_FALLBACK=1` works on MPS, but MPS fallback causes `deform_conv2d_backward` and `grid_sampler_2d_backward` CPU fallback warnings.
- Discovered that `mps_deform_conv` was already installed and usable, and that the repo’s `lada/models/basicvsrpp/deformconv.py` dispatches to it when `LADA_DEFORM_CONV_BACKEND=mps_deform_conv` is set.
- Reran training with `LADA_DEFORM_CONV_BACKEND=mps_deform_conv`, which roughly halved iteration time (about 12–14 s/iter to about 6 s/iter).
- Verified via a minimal `torch.grid_sample` test that forward runs on MPS but backward is not implemented and falls back to CPU, so this remains the next bottleneck.

Failures and how to do differently:

- `uv run ...` was not reliable in the shell because the pyenv shim could not find the `uv` binary. Plain `python ...` worked.
- The first “fine-tune on a 3-scene test dataset” run was only a pipeline check; it was not meaningful training data.
- `grid_sample` is not fully MPS-native for training in this environment; don’t assume a warning means the whole op is native on GPU.

Reusable knowledge:

- Fine-tuning command shape that worked:
  - `PYTORCH_ENABLE_MPS_FALLBACK=1 LADA_DEFORM_CONV_BACKEND=mps_deform_conv python scripts/training/train-mosaic-restoration-basicvsrpp.py configs/basicvsrpp/mosaic_restoration_generic_stage2.py --load-from model_weights/lada_mosaic_restoration_model_generic_v1.2_full.pth --work-dir experiments/basicvsrpp/finetune_dataset_test ...`
- The biggest MPS speed win in this rollout came from `mps-deform-conv`; `grid_sample` backward remained CPU fallback.
- `train_cfg.max_iters=1000`, `val_interval=200`, and `batch_size=1` were used for a quick MPS smoke test; a longer run would still be slow because of fallback ops.
- The repo’s fine-tuning configs use `datasets/mosaic_removal_vid/train/crop_unscaled_meta` and `.../val/crop_unscaled_meta` by default; overriding `metadata_root_dir` is the quickest way to point at a custom dataset.

References:

- [1] `model_weights/lada_mosaic_restoration_model_generic_v1.2_full.pth` loaded with `torch.load(..., weights_only=False)` and had `meta.iter = 52000`.
- [2] `lada/models/basicvsrpp/deformconv.py:44-55` dispatches `MPS` inputs to `_mps_deform_conv2d` when `LADA_DEFORM_CONV_BACKEND=mps_deform_conv` is set.
- [3] `torch` test result: `forward device mps:0` but `backward error NotImplementedError: aten::grid_sampler_2d_backward is not currently implemented for the MPS device`.
- [4] Training log example after enabling `mps_deform_conv`: `Iter(train) [ 10/1000] ... time: 6.0668`, roughly half the earlier `12.6883` seconds/iter.

## Task 3: Make local data-prep scripts and adjust the FC2 source workflow

Outcome: success

Preference signals:

- The user explicitly corrected the source location with `動画はここから` and pointed to `/Volumes/Firewire_HD3/movies/FC2/`, so future paths should default there rather than older locations.
- The user asked for safer data handling and later requested `symlinkはいやだ、コピーして`, which implies they prefer direct filesystem copies over symlinks for local pipeline setup.
- The user asked `素材を抽出する基準はなんなんだろ？` and then pushed back on the quality of the extracted data, meaning future commands should separate “candidate extraction” from “final training set” and should not pretend the extractor itself is the quality gate.

Key steps:

- Created local helper scripts under `scripts/local/project_hd_finetune/`:
  - `00_split_existing_dataset.py` to split an existing dataset into train/val via symlinks (later relevant as a setup step, though the user did not ask for symlinks for the FC2 source files themselves).
  - `01_extract_clips.sh`, `02_create_datasets.sh`, and `03_train_finetune.sh` for the FC2 → clips → dataset → fine-tune flow.
- Realized the user wanted a better extraction strategy than a few fixed timestamps, and the rollout showed that a fixed-manifest approach was only useful as a placeholder, not as a good data policy.
- Reworked the filtering flow to use the existing dataset and later to use `dataset_filtered` with stronger filters instead of weak, unfiltered extraction.
- Renamed source files in `/Volumes/Firewire_HD3/movies/FC2/` to remove spaces and non-ASCII characters. Seven files were renamed and a TSV log was written.
- Verified that the renamed source directory had no remaining filenames with spaces or non-ASCII characters.

Failures and how to do differently:

- The initial fixed-interval clip manifest was too crude for “good” training data; the user correctly identified that it was only a rough mechanism.
- The rollout showed that auto-extracted scenes are still just candidates. The correct mental model is “candidate collection” plus human review/cleanup, not “done dataset” straight from the extractor.
- A DOVER/filter run initially failed because `lada/model_weights` did not exist where the code expected; copying the repo-root `model_weights` into `lada/model_weights` resolved the path issue.

Reusable knowledge:

- On this setup, `create-mosaic-restoration-dataset.py` can operate on `/Volumes/Firewire_HD3/movies/FC2` and write into `/Volumes/Project_HD/dataset_filtered`, but it needs the model files and correct local path layout.
- The extracted dataset stores video, mask, and JSON metadata in parallel directories like `crop_unscaled_img`, `crop_unscaled_mask`, and `crop_unscaled_meta`.
- If the user later wants to clean up processed source files, the `done_processing.txt` marker under a dataset output can be used to identify which source files were already processed.
- The user prefers direct confirmation and exact file existence checks before moving/renaming large media files.

References:

- [1] Renamed FC2 filenames to ASCII-safe forms, logged in `scripts/local/project_hd_finetune/fc2_rename_log_20260628.tsv`.
- [2] `scripts/local/project_hd_finetune/fc2_processed_move_log_20260629.tsv` records the later processed-file move operation.
- [3] Source directory after rename: `/Volumes/Firewire_HD3/movies/FC2/`; processed destination: `/Volumes/Firewire_HD3/movies/FC2_processed/`.
- [4] The dataset output structure under `/Volumes/Project_HD/dataset_filtered` showed parallel `crop_unscaled_img`, `crop_unscaled_mask`, and `crop_unscaled_meta` trees with JSON metadata containing `video_quality` scores.

## Task 4: Tune and verify the DOVER quality threshold

Outcome: success

Preference signals:

- The user asked `画質の0.25という値はどうなの？` -> they wanted an evidence-based threshold, not a guessed default.
- The later filtered dataset showed the user was actively judging data quality, so the threshold should be chosen with actual score distribution in mind.

Key steps:

- Read the code path that applies DOVER quality filtering: `scene_processing_options.quality_evaluation.filter` compares `quality_score.overall < min_quality` and skips if below threshold.
- Gathered the actual score distribution from `/Volumes/Project_HD/dataset_filtered`: 43 scenes had overall scores with min around `0.252`, median around `0.358`, max around `0.591`.
- Compared cutoff counts: `below_0.25 = 0`, `below_0.30 = 13`, `below_0.35 = 20`.

Failures and how to do differently:

- `0.25` was too weak to meaningfully filter the observed dataset; it barely removed anything.
- The threshold should not be treated as a universal constant; it should be re-evaluated per source corpus.

Reusable knowledge:

- For this rollout’s actual filtered corpus, `0.30` looked like a more meaningful lower cutoff than `0.25`.
- DOVER in this repo is a rough video-quality filter, not a perfect training-data-quality oracle.

References:

- [1] `scripts/dataset_creation/create-mosaic-restoration-dataset.py:58-63` defines `--min-video-quality` with default `0.1` and the help text about DOVER quality.
- [2] `lada/datasetcreation/nsfw_scene_processor.py:478-480` skips scenes whose `quality_score.overall` is below `min_quality`.
- [3] Observed score distribution on `dataset_filtered`: `min 0.252`, `median 0.358`, `max 0.591`.

## Task 5: Retire processed source videos once extraction is done

Outcome: success

Preference signals:

- The user asked `処理の終了した素材を違うフォルダに退避してください｡` -> they want completed source files separated from remaining source files, not mixed together.
- The user later singled out `FC2PPV_3100741_70_OFF_2_1_30_23_3_4.mp4は退避？` showing they care about precise file-state tracking.

Key steps:

- Read `done_processing.txt` from the dataset output to identify the list of source files that had actually been processed.
- Moved 7 processed files from `/Volumes/Firewire_HD3/movies/FC2/` to `/Volumes/Firewire_HD3/movies/FC2_processed/`.
- Confirmed all 7 files were absent from the source folder and present in the processed folder.
- For `FC2PPV_3100741_70_OFF_2_1_30_23_3_4.mp4`, confirmed it was **not** among the files in `done_processing.txt`, so it was **not** moved in that processed-file cleanup step.

Failures and how to do differently:

- The processed-file cleanup should be driven by the dataset’s `done_processing.txt` marker, not by guessing from the presence of JSON metadata.
- The user’s later check showed that some extracted metadata existed even when the source video was not in the processed list; future cleanup logic should distinguish “extracted some scenes” from “fully processed source file.”

Reusable knowledge:

- `done_processing.txt` under the dataset output is the reliable marker for the processed-source-file move list.
- `FC2PPV_3100741_70_OFF_2_1_30_23_3_4.mp4` remained in the source folder after the cleanup, because it was not marked as done.

References:

- [1] Move log: `scripts/local/project_hd_finetune/fc2_processed_move_log_20260629.tsv`.
- [2] Verified processed destination: `/Volumes/Firewire_HD3/movies/FC2_processed/`.
- [3] Example processed files moved: `FC2PPV-3845406-1.mp4`, `FC2PPV-4057921.mp4`, `fc2ppv-311667011-2_2480-1480-1.mp4`.


