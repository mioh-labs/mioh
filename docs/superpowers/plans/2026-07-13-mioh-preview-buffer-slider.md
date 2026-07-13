# mioh Preview Buffer Slider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 1-to-60-second realtime preview buffer slider that defaults to 8 seconds and updates an active worker immediately.

**Architecture:** `RestorationRunner` remains the single source of truth for the selected buffer limit and supplies it to new preview workers. `RealtimePlayerController` exposes one live-update method that forwards the existing `set_buffer_limit` command without seeking or restarting. `RealtimePlayerView` binds the slider to both paths.

**Tech Stack:** SwiftUI, Foundation `Process` pipes, JSON line commands, Python `pytest`, `swiftc`.

## Global Constraints

- Slider range is exactly 1 through 60 seconds.
- Slider step is exactly 1 second.
- Default is exactly 8 seconds.
- An active preview updates immediately without pause, seek, restart, or queue clearing.
- An idle preview uses the selected value on its next start.
- Continue working directly on `main` as requested by the user.

---

### Task 1: Buffer Limit Setting and Live Update

**Files:**
- Modify: `tests/test_standalone_app_options.py`
- Modify: `packaging/macOS/standalone/MiohApp.swift:115-124,264-296`
- Modify: `packaging/macOS/standalone/RealtimePlayer.swift:248-260,509-552`

**Interfaces:**
- Consumes: existing `RealtimePlayerController.sendCommand(_ payload: [String: Any])` and worker command `{"command":"set_buffer_limit","seconds":Double}`.
- Produces: `RestorationRunner.previewBufferLimit: Double` and `RealtimePlayerController.setBufferLimit(_ seconds: Double)`.

- [ ] **Step 1: Write the failing source-contract test**

Add this test to `StandaloneAppOptionTests`:

```python
def test_preview_buffer_slider_supports_one_minute_and_live_updates(self):
    app = APP_SOURCE.read_text()
    player = PLAYER_SOURCE.read_text()

    for contract in [
        "@Published var previewBufferLimit = 8.0",
        'add(&args, "--buffer-limit", previewBufferLimit)',
    ]:
        self.assertIn(contract, app)
    for contract in [
        "func setBufferLimit(_ seconds: Double)",
        '["command": "set_buffer_limit", "seconds": seconds]',
        'Text("バッファ上限")',
        "in: 1...60",
        "step: 1",
        "controller.setBufferLimit(value)",
        'Text("\\(Int(runner.previewBufferLimit))秒")',
    ]:
        self.assertIn(contract, player)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest -q tests/test_standalone_app_options.py::StandaloneAppOptionTests::test_preview_buffer_slider_supports_one_minute_and_live_updates
```

Expected: FAIL because `previewBufferLimit`, `setBufferLimit`, and the slider contracts do not exist.

- [ ] **Step 3: Add the runner setting and use it for worker startup**

Add the property beside the other preview settings in `RestorationRunner`:

```swift
@Published var previewBufferLimit = 8.0
```

Replace the fixed preview argument:

```swift
add(&args, "--buffer-limit", previewBufferLimit)
```

- [ ] **Step 4: Add the controller live-update method**

Add this public method beside `setVolume` and `setMuted`:

```swift
func setBufferLimit(_ seconds: Double) {
  guard worker != nil else { return }
  sendCommand(["command": "set_buffer_limit", "seconds": seconds])
}
```

This method deliberately does not change playback state, generation, position, or queued segments.

- [ ] **Step 5: Add the playback-tab slider**

Insert this row between the seek row and playback controls:

```swift
HStack(spacing: 12) {
  Text("バッファ上限")
  Slider(
    value: Binding(
      get: { runner.previewBufferLimit },
      set: { value in
        runner.previewBufferLimit = value
        controller.setBufferLimit(value)
      }
    ),
    in: 1...60,
    step: 1
  )
  .frame(maxWidth: 320)
  Text("\(Int(runner.previewBufferLimit))秒")
    .font(.caption.monospacedDigit())
    .frame(width: 48, alignment: .trailing)
  Spacer()
}
```

- [ ] **Step 6: Run focused tests and compile the main Swift target**

Run:

```bash
python -m pytest -q tests/test_standalone_app_options.py
xcrun swiftc -O -parse-as-library -target arm64-apple-macosx26.0 \
  -framework AppKit -framework SwiftUI -framework AVFoundation -framework AVKit \
  packaging/macOS/standalone/MiohApp.swift \
  packaging/macOS/standalone/RealtimePlayer.swift \
  -o /tmp/mioh-buffer-slider-verify
```

Expected: all focused tests PASS and `swiftc` exits 0.

- [ ] **Step 7: Commit the feature**

```bash
git add tests/test_standalone_app_options.py \
  packaging/macOS/standalone/MiohApp.swift \
  packaging/macOS/standalone/RealtimePlayer.swift
git commit -m "feat: add realtime preview buffer slider"
```

### Task 2: Full Build and GUI Verification

**Files:**
- Verify: `build/macos-standalone/mioh.app`
- Verify: `build/macos-standalone/mioh-0.11.0-unsigned.dmg`

**Interfaces:**
- Consumes: the slider, runner property, and controller method from Task 1.
- Produces: freshly built and signed app/DMG artifacts verified on `main`.

- [ ] **Step 1: Run the complete automated test suite**

Run:

```bash
python -m pytest -q
```

Expected: all tests pass, with only the repository's existing skips and warnings.

- [ ] **Step 2: Rebuild the standalone app and DMG**

Run:

```bash
packaging/macOS/standalone/build_app.sh
```

Expected: the MPS deform-conv smoke test passes and both artifact paths are printed.

- [ ] **Step 3: Verify the application signature**

Run:

```bash
codesign --verify --deep --strict --verbose=2 build/macos-standalone/mioh.app
```

Expected: `valid on disk` and `satisfies its Designated Requirement`.

- [ ] **Step 4: Exercise the slider in the built app**

Using Computer Use, open the playback tab and verify:

1. The displayed initial value is `8秒`.
2. The slider can be moved to `60秒`.
3. Starting a preview launches the worker with `--buffer-limit 60.0`.
4. Moving the slider while playing leaves the player active and sends `set_buffer_limit`.
5. Stopping the preview returns the controller to `待機中`.

- [ ] **Step 5: Verify repository cleanliness and recent commits**

Run:

```bash
git diff --check
git status --short --branch
git log -5 --oneline
```

Expected: no uncommitted source changes; branch remains `main`.
