# Coordinated Progress Output Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render one replaceable progress row per active Lada worker in the CLI and macOS app while preserving ordinary logs and processing behavior.

**Architecture:** Worker subprocess output is converted into progress events and sent to the parent parallel runner. A parent-owned renderer assigns stable lanes, redraws interactive terminals, and emits JSON events for the macOS app; Swift keeps those events outside durable log history and rebuilds the visible log from history plus active lanes.

**Tech Stack:** Python 3.12, `multiprocessing`/`concurrent.futures`, ANSI terminal control, JSON Lines, Swift 6, SwiftUI, Python `unittest`.

## Global Constraints

- Video processing commands, model selection, encoder settings, and worker counts are unchanged.
- Warnings, errors, phase changes, summaries, and completion results remain durable log lines.
- Malformed progress events are preserved as ordinary text.
- Renderer failure must not terminate restoration.
- Existing log size caps remain in force.

---

### Task 1: Parent-owned progress events and terminal lanes

**Files:**
- Modify: `process_video_parallel.py`
- Create: `tests/test_process_video_parallel_progress.py`

**Interfaces:**
- Produces: `parse_progress_line(line: str) -> float | None`
- Produces: `ParallelProgressRenderer(stream, app_protocol: bool, min_interval: float = 0.25)` with `progress(event)`, `complete(lane, message)`, and `close()`
- Produces: `process_segment_worker(segment_info, config, progress_queue=None)`

- [ ] **Step 1: Write failing parser and renderer tests**

Add tests that assert `Processing video:  13%` parses as `13.0`, two lane events remain separately addressable, app protocol output starts with `@@LADA_PROGRESS@@` and contains JSON, repeated events inside `min_interval` are suppressed, and completion removes its lane while preserving a completion message.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m unittest tests.test_process_video_parallel_progress -v`

Expected: FAIL because `parse_progress_line` and `ParallelProgressRenderer` do not exist.

- [ ] **Step 3: Implement event reporting and rendering**

Add a percent regex, an event dictionary containing `kind`, `lane`, `segment`, `text`, and `percent`, and a renderer that is the sole owner of progress presentation. Interactive output redraws the complete active-lane block with ANSI clear-line/cursor-up sequences. App protocol emits `@@LADA_PROGRESS@@` followed by compact JSON. Non-interactive non-app output emits throttled labeled progress lines.

Change workers so each child `Processing video:` line is queued rather than printed. Queue warning/error messages as durable log events. Use a spawn-context manager queue for process execution and `queue.Queue` for thread execution. Replace the blocking `as_completed` loop with a short-timeout loop that drains events and checks completed futures, keeping all terminal writes in the parent.

- [ ] **Step 4: Run focused and adjacent tests**

Run: `python -m unittest tests.test_process_video_parallel_progress tests.test_process_video_parallel_executor_selection tests.test_process_video_parallel_shutdown -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add process_video_parallel.py tests/test_process_video_parallel_progress.py
git commit -m "Coordinate parallel progress output"
```

### Task 2: Replaceable progress rows in the macOS app

**Files:**
- Modify: `packaging/macOS/standalone/LadaApp.swift`
- Modify: `tests/test_standalone_app_options.py`

**Interfaces:**
- Consumes: JSON lines prefixed by `@@LADA_PROGRESS@@`
- Produces: Swift `AppProgressEvent: Decodable`
- Produces: `consumeLine(_:)` that routes protocol events or durable log text

- [ ] **Step 1: Write a failing app source test**

Assert that the app environment enables `LADA_APP_PROGRESS`, the Swift source declares `AppProgressEvent`, recognizes `@@LADA_PROGRESS@@`, stores active rows separately from history, and rebuilds visible output rather than converting every carriage return into a durable newline.

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m unittest tests.test_standalone_app_options.StandaloneAppOptionTests.test_app_replaces_structured_progress_rows -v`

Expected: FAIL because structured progress handling is absent.

- [ ] **Step 3: Implement Swift line buffering and progress state**

Add a partial-line buffer, durable history string, stable lane order, and lane-to-event dictionary. Parse prefixed JSON with `JSONDecoder`; progress replaces the lane entry, completion removes it and appends its message, and malformed events go to history. Rebuild `log` as capped history plus the ordered active rows. Continue deriving the existing status and overall progress from event percentages. Set `LADA_APP_PROGRESS=1` in the child environment.

- [ ] **Step 4: Run the app test and build**

Run: `python -m unittest tests.test_standalone_app_options -v`

Expected: PASS.

Run: `packaging/macOS/standalone/build_app.sh`

Expected: Swift compilation and app assembly complete with exit code 0.

- [ ] **Step 5: Commit**

```bash
git add packaging/macOS/standalone/LadaApp.swift tests/test_standalone_app_options.py
git commit -m "Render app progress in replaceable rows"
```

### Task 3: Integrated verification

**Files:**
- Verify: `process_video_parallel.py`
- Verify: `packaging/macOS/standalone/LadaApp.swift`

**Interfaces:**
- Consumes all interfaces from Tasks 1 and 2.
- Produces a verified standalone application and regression-test evidence.

- [ ] **Step 1: Run the focused regression suite**

Run: `python -m unittest tests.test_process_video_parallel_progress tests.test_process_video_parallel_executor_selection tests.test_process_video_parallel_shutdown tests.test_standalone_app_options -v`

Expected: PASS with no failures or errors.

- [ ] **Step 2: Check repository hygiene**

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only intentional build artifacts, if the build script creates ignored output.

- [ ] **Step 3: Inspect the built app protocol path**

Run a short fixture or mocked worker stream through the Python renderer with `LADA_APP_PROGRESS=1` and confirm successive events for the same lane share the same lane key while different workers have different keys.

Expected: one JSON line per throttled update, stable same-worker lane keys, distinct cross-worker lane keys, and a completion event that removes the lane in Swift tests.
