# mioh Buffered Real-Time Player Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a native buffered player that previews restored video with synchronized source audio, seeking, original/restored switching, and bounded compressed look-ahead.

**Architecture:** A bundled persistent Python worker reuses loaded models across seeks and writes two-second VideoToolbox H.264 MP4 segments while emitting JSON events. A SwiftUI `RealtimePlayerController` queues those segments in `AVQueuePlayer`, uses the original `AVPlayer` as the audio/master clock, and pauses both players during underruns.

**Tech Stack:** Python 3.12, PyAV, PyTorch/MPS, Core ML/CoreAI, SwiftUI, AVFoundation, AVKit, unittest/pytest.

## Global Constraints

- The main application continues to target arm64 macOS 26.0.
- CoreAI remains available only on macOS 27 and newer.
- Preview models load once and are reused across seek operations.
- Restored preview segments are two seconds each with a six-second startup target and eight-second production ceiling.
- The original source player is the audio and playback clock authority.
- Preview must not add a parallel-worker limit or change export behavior.

---

### Task 1: Preview protocol and segment encoder

**Files:**
- Create: `packaging/macOS/standalone/mioh_preview_worker.py`
- Create: `tests/test_mioh_preview_worker.py`

**Interfaces:**
- Produces: `emit_event(kind: str, **payload)`, `PreviewCommand`, `SegmentEncoder`, and `select_encoder(preferred: str) -> str`.
- `SegmentEncoder.add_frame(frame: numpy.ndarray, pts_ns: int) -> list[dict]` returns completed segment events.
- `SegmentEncoder.finish() -> list[dict]` closes and reports the final partial segment.

- [ ] **Step 1: Write failing protocol and encoder tests**

Test that every emitted event has `kind` and `generation`, command JSON parses `seek`, independent two-second segments are created in sequence, the final partial segment is emitted, and a failed `h264_videotoolbox` open retries with `libx264` plus `preset=ultrafast`.

- [ ] **Step 2: Run the focused test and observe the missing module failure**

Run: `python -m pytest tests/test_mioh_preview_worker.py -q`

Expected: collection fails because `mioh_preview_worker.py` does not exist.

- [ ] **Step 3: Implement the protocol and encoder**

Use stdout exclusively for compact JSON events, stderr for diagnostics, `av.VideoFrame.from_ndarray(..., format="bgr24")`, `h264_videotoolbox` as the preferred codec, and independent MP4 containers named `preview-g{generation}-{sequence:06d}.mp4`.

- [ ] **Step 4: Run focused tests**

Run: `python -m pytest tests/test_mioh_preview_worker.py -q`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit**

Run: `git add packaging/macOS/standalone/mioh_preview_worker.py tests/test_mioh_preview_worker.py && git commit -m "feat: add realtime preview segment worker"`

### Task 2: Persistent model session, seek, and backpressure

**Files:**
- Modify: `packaging/macOS/standalone/mioh_preview_worker.py`
- Modify: `tests/test_mioh_preview_worker.py`

**Interfaces:**
- Produces: `PreviewSession`, `PreviewSession.run(start_ns: int)`, `PreviewSession.seek(position_ns: int)`, and `PreviewSession.stop()`.
- Consumes model names and effect settings matching the standalone GUI fields.

- [ ] **Step 1: Write failing session tests**

Use fake loaders and frame restorers to prove models load once, seeking increments generation and creates a new frame restorer with the same model objects, stale segments are deleted, production waits when four two-second files exist, and stop releases the restorer and session directory.

- [ ] **Step 2: Run focused tests and verify behavioral failures**

Run: `python -m pytest tests/test_mioh_preview_worker.py -q`

Expected: the new session tests fail because `PreviewSession` is absent.

- [ ] **Step 3: Implement the persistent session**

Resolve model paths with `ModelFiles`, call `load_models` exactly once, create `FrameRestorer` instances with the captured composition/ROI settings, read `(frame, pts)` from its output queue, process stdin commands on a control thread, and poll commands while waiting for restored frames or buffer capacity.

- [ ] **Step 4: Verify focused tests**

Run: `python -m pytest tests/test_mioh_preview_worker.py -q`

Expected: protocol, encoder, reuse, seek, buffer, and cleanup tests all pass.

- [ ] **Step 5: Commit**

Run: `git add packaging/macOS/standalone/mioh_preview_worker.py tests/test_mioh_preview_worker.py && git commit -m "feat: keep preview models alive across seeks"`

### Task 3: Native Swift player controller and UI

**Files:**
- Create: `packaging/macOS/standalone/RealtimePlayer.swift`
- Modify: `packaging/macOS/standalone/MiohApp.swift`
- Modify: `tests/test_standalone_app_options.py`

**Interfaces:**
- Produces: `RealtimePlayerController`, `RealtimePlayerState`, `PreviewWorkerEvent`, and `RealtimePlayerView`.
- Consumes: `RestorationRunner.previewArguments(resources:)`, `RestorationRunner.environment(resources:python:)`, and the runner's selected input/settings.

- [ ] **Step 1: Write failing Swift source contract tests**

Assert the AVFoundation/AVKit imports, `AVQueuePlayer` restored player, `AVPlayer` source clock, JSON generation filtering, three-segment startup, two-segment rebuffer threshold, 80ms drift threshold, seek command, original/restored toggle, controls, and `再生` tab.

- [ ] **Step 2: Run the source contract and observe failures**

Run: `python -m unittest tests.test_standalone_app_options -v`

Expected: new player assertions fail because the Swift player source and tab are absent.

- [ ] **Step 3: Implement controller and view**

Read worker stdout incrementally, queue only contiguous current-generation segments, observe `AVPlayerItemDidPlayToEndTime`, delete consumed files, use the source player's periodic time observer for position, pause both players on underrun, seek both players together, and expose restart-with-settings behavior.

- [ ] **Step 4: Expose preview configuration from the runner**

Add preview arguments for restoration/detection/effects/memory settings, reuse the platform-aware CoreAI validation, and keep `--parallel-workers` behavior untouched for export.

- [ ] **Step 5: Compile and run focused tests**

Run:

`python -m unittest tests.test_standalone_app_options -v`

`xcrun swiftc -O -parse-as-library -target arm64-apple-macosx26.0 -framework AppKit -framework SwiftUI -framework AVFoundation -framework AVKit packaging/macOS/standalone/MiohApp.swift packaging/macOS/standalone/RealtimePlayer.swift -o /tmp/mioh-player-check`

Expected: all focused tests pass and Swift compilation exits zero.

- [ ] **Step 6: Commit**

Run: `git add packaging/macOS/standalone/MiohApp.swift packaging/macOS/standalone/RealtimePlayer.swift tests/test_standalone_app_options.py && git commit -m "feat: add native restored preview player"`

### Task 4: Bundle worker and integration test

**Files:**
- Modify: `packaging/macOS/standalone/build_app.sh`
- Modify: `tests/test_standalone_app_options.py`
- Modify: `tests/test_mioh_preview_worker.py`

**Interfaces:**
- Produces: `mioh_preview_worker.py` at `Contents/Resources/runtime/lib/python3.12/site-packages/` and a standalone app linked with AVFoundation/AVKit.

- [ ] **Step 1: Write failing packaging tests**

Assert both Swift sources are compiled, both AV frameworks are linked, and the preview worker is copied into the runtime. Add a synthetic 12fps video test that invokes the encoder path and verifies ordered playable segments with ffprobe.

- [ ] **Step 2: Run focused tests and observe packaging failures**

Run: `python -m pytest tests/test_mioh_preview_worker.py tests/test_standalone_app_options.py -q`

Expected: build-script assertions fail until packaging is updated.

- [ ] **Step 3: Update standalone packaging**

Compile `RealtimePlayer.swift` with the main app, link AVFoundation and AVKit, copy the worker after installing the lada package, and preserve the macOS 27-only CoreAI helper target.

- [ ] **Step 4: Run focused tests and standalone build**

Run:

`python -m pytest tests/test_mioh_preview_worker.py tests/test_standalone_app_options.py -q`

`packaging/macOS/standalone/build_app.sh`

Expected: tests pass, worker smoke test passes, app and DMG are produced.

- [ ] **Step 5: Commit**

Run: `git add packaging/macOS/standalone/build_app.sh tests/test_standalone_app_options.py tests/test_mioh_preview_worker.py && git commit -m "build: bundle mioh realtime preview worker"`

### Task 5: Complete verification

**Files:**
- Verify only.

**Interfaces:**
- Consumes the completed worker, controller, and app bundle.
- Produces fresh test, compilation, signature, architecture, and packaging evidence.

- [ ] **Step 1: Run the complete test suite**

Run: `python -m pytest -q`

Expected: zero failures.

- [ ] **Step 2: Verify the built bundle**

Confirm the main executable has minimum macOS 26.0, does not link CoreAI, links AVFoundation/AVKit, the helper has minimum macOS 27.0 and links CoreAI, the preview worker exists, the app signature verifies, and the DMG exists.

- [ ] **Step 3: Review repository state**

Run: `git diff --check && git status --short`

Expected: no whitespace errors and a clean worktree.
