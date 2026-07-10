thread_id: 019efdf7-42d0-7e40-b01c-ee5b93e5efc0
updated_at: 2026-06-25T09:18:17+00:00
rollout_path: /Users/okatti/.codex/archived_sessions/rollout-2026-06-25T17-48-20-019efdf7-42d0-7e40-b01c-ee5b93e5efc0.jsonl
cwd: /Users/okatti/Documents/lada
git_branch: codex/mlx-dcnv2-basicvsrpp

# LADA MLX ROI seam investigation, progress logging, and memory-pressure follow-up

Rollout context: Working in `/Users/okatti/Documents/lada` on the MLX restore path. The user first complained that ROI splitting was bad because the boundary was visible, then later asked for the ROI-related logs to expose more detail, then raised memory pressure concerns and clarified that only the low-memory default proposal should be rejected while keeping the “stop spatial splitting” change. The user also asked a couple of conceptual questions about whether per-image extraction is necessary and whether video-compression-style motion/delta handling could be applied.

## Task 1: Remove visible seams from MLX ROI splitting

Outcome: success

Preference signals:
- The user said `ROI分割はダメですね。分割した境がくっきり分かる。` -> they care more about visible seam quality than about keeping spatial ROI tiling for speed/memory.
- When the assistant proposed reverting the seam-causing behavior, the user said `切って` -> they want direct action on the problematic split path rather than prolonged discussion.
- When the assistant later tried to revert the split-removal change, the user corrected it with `いや、分割やめるのはそのまま。...` -> they want very precise, scoped reversions and do not want unrelated defaults changed.

Key steps:
- Inspected `experiments/mlx_dcnv2/roi_restore.py` and `experiments/mlx_dcnv2/restore_fixture.py`, plus the ROI restore tests.
- Confirmed the seam source was spatial tiling of a single connected ROI: the code split bboxes with `split_bbox_by_max_area(...)` and then composited per tile.
- Wrote a failing test first for the “do not spatially split large connected ROI” behavior, confirmed it failed, then removed the spatial tiling branch and updated the restore fixture to match.
- Updated the README text to say connected ROIs are not spatially tiled because separate tile restores create seams.
- Verified with `python -m unittest tests.test_mlx_dcnv2_roi_restore tests.test_mlx_dcnv2_restore_fixture tests.test_mlx_dcnv2_run_restore_fixture tests.test_process_video_parallel_mlx`.

Failures and how to do differently:
- A later attempt to “fix memory” accidentally started reverting the spatial-split removal; the user immediately corrected that. Future edits should treat the seam fix and any low-memory tuning as separate concerns.
- `pytest` was not installed in the environment, so `python -m unittest` was the reliable verification path.

Reusable knowledge:
- On this MLX path, the user prefers quality-preserving behavior over spatial ROI tiling when the seam becomes visible.
- `max_restore_roi_area` can be treated as a time-window guard or a logging aid, but using it to cut one connected ROI into spatial tiles introduces obvious seam artifacts.

References:
- `experiments/mlx_dcnv2/roi_restore.py:318-336` — connected ROI restore now takes the bbox as a single tile when no disconnected components are found.
- `experiments/mlx_dcnv2/restore_fixture.py:492-525` — window restore path mirrors the no-spatial-tiling behavior.
- `tests/test_mlx_dcnv2_roi_restore.py:65-87` — regression test for not splitting a large connected ROI.
- `tests/test_mlx_dcnv2_restore_fixture.py:342-373` — regression test for mask-window restore behavior.
- `experiments/mlx_dcnv2/README.md:213-217` — docs updated to match the no-seam policy.
- Verification: `python -m unittest tests.test_mlx_dcnv2_roi_restore tests.test_mlx_dcnv2_restore_fixture tests.test_mlx_dcnv2_run_restore_fixture tests.test_process_video_parallel_mlx` -> `Ran 55 tests ... OK`.

## Task 2: Expose ROI stats in MLX progress output

Outcome: success

Preference signals:
- The user pasted logs showing wildly different `window ... fps` values and asked `セグメント数を変えたらこんなに数字が違うのはなぜ？` -> they want the logs to explain behavior, not just report raw speed.
- After hearing that the `window timing` line already carried ROI stats internally, the user said `やって` -> they wanted the missing info surfaced in the parent `[MLX]` output.

Key steps:
- Traced `process_video_parallel.py` progress parsing and found `format_mlx_progress_line(...)` only consumed `start`, `frames`, `restore`, and `fps`.
- Added parsing for `roi=...`, `area=max/sum`, and memory fields from the child `window timing` line.
- Extended the formatted parent progress line to append `| roi N area max/sum | mem rss ... max ...` when those fields are present.
- Added a regression test for the new formatted string and verified with `python -m unittest tests.test_process_video_parallel_mlx`.

Failures and how to do differently:
- The first test update failed because the parser still ignored ROI fields; that failure was useful and led directly to the parent-log patch.
- Because the progress line now carries more signal, future debugging can separate “fast because ROI was skipped/small” from “slow because a big ROI actually ran.”

Reusable knowledge:
- The MLX `window timing` line is already the right place to expose ROI statistics; the parent `process_video_parallel.py` log just needed to preserve them.
- The formatted parent progress line now preserves both memory and ROI details, so a single line can distinguish empty/small/big ROI windows.

References:
- `process_video_parallel.py:171-195` — parses `roi=` and `area=` from child timing text.
- `process_video_parallel.py:245-263` — adds ROI stats to the rendered `[MLX] ...` progress line.
- `tests/test_process_video_parallel_mlx.py:278-305` — regression test for ROI and memory fields in the formatted progress line.
- Verification: `python -m unittest tests.test_process_video_parallel_mlx` -> `Ran 15 tests ... OK`.

## Task 3: Low-memory follow-up and default-parameter clarification

Outcome: partial

Preference signals:
- The user said `メモリがきついな` -> they are actively balancing memory against the seam fix.
- When the assistant proposed lowering MLX defaults, the user corrected with `いや、分割やめるのはそのまま。MLXのデフォルトを window=15 / overlap 3 / temporal ROI area 65536がだめ。` -> they rejected the low-memory default proposal but kept the seam fix.

Key steps:
- Inspected the current MLX defaults and related tests.
- Noted that the checked-in state already had `window=20`, auto overlap `4`, and `max-roi-area=131072`; the user’s correction indicated the proposed `window=15 / overlap 3 / ROI area 65536` change should not be applied.
- Confirmed the current defaults and related tests remained passing after the clarification.

Failures and how to do differently:
- I briefly tried to move toward a lower-memory default, but the user rejected that specific direction. Future memory follow-ups should be treated as separate from the seam fix, and default changes should be explicitly confirmed before editing.
- The user’s correction shows they want fine-grained control over what is reverted vs. what stays.

Reusable knowledge:
- Current MLX CLI defaults in this rollout remained `window=20`, auto overlap `4` when unspecified, and `max-roi-area=131072`.
- The user can distinguish between “remove a bad behavior” and “re-tune defaults for memory”; those are not automatically the same request.

References:
- `process_video_parallel.py:2028-2038` — current MLX defaults shown in the parser.
- `tests/test_process_video_parallel_mlx.py:149-220` — tests around window/overlap default derivation and ROI area default.
- Verification after clarification: `python -m unittest tests.test_process_video_parallel_mlx tests.test_mlx_dcnv2_roi_restore tests.test_mlx_dcnv2_restore_fixture tests.test_mlx_dcnv2_run_restore_fixture` -> `Ran 55 tests ... OK`.

## Task 4: Conceptual explanation of frame extraction vs video compression

Outcome: success

Preference signals:
- The user asked `どうしても1枚1枚画像をとりださないとマスクと復元は当てられないのかね？` and then `動画ファイルのフレームはキーフレーム以外は動きを補完して容量やパワーを抑えてるのよね？同じことはできないの？` -> they want conceptual, practical guidance about whether the pipeline can behave more like video compression.

Key steps:
- Explained that per-image file extraction is not required, but per-frame image data is still needed conceptually for masks and restoration.
- Explained that the current streaming MLX path already avoids writing PNG/JPEG masks unless debug output is requested.
- Clarified that compression-style motion/delta reuse is possible in spirit, but the restoration model still consumes RGB frames, so the most practical savings come from windowing, mask interpolation, reduced ROI size, and avoiding disk writes.

Reusable knowledge:
- The current MLX pipeline is already a streaming frame-based approach, not a “save every frame as an image file” workflow.
- Compression-like reuse is more plausible for masks, ROI tracking, or reused restoration outputs than for feeding compressed video directly into the restoration model.
