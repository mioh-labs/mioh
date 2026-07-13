# mioh Smooth Player and Independent Input Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent repeated generation-start seeks during restored playback and give the playback tab its own movie input independent from normal export.

**Architecture:** `RealtimePlayerController` tracks whether the current worker generation has started and separates one-time initial seeking from pause and underrun resume. The controller also owns `previewInputURL`; `RestorationRunner.previewArguments` receives that URL explicitly while all processing settings remain shared.

**Tech Stack:** Swift 6, SwiftUI, AVFoundation, AVKit, Python unittest/pytest source-contract tests, Computer Use live verification.

## Global Constraints

- The main application continues to target arm64 macOS 26.0.
- Preview remains T36 on macOS 27 and standard BasicVSR++ v1.2 on macOS 26.
- The basic-tab input and normal export behavior remain unchanged.
- Restoration, detection, composition, ROI enhancer, device, and memory settings remain shared.
- Initial playback waits for three two-second segments; underrun recovery waits for two.
- The existing 80 ms drift correction is not changed in this work.
- The user explicitly authorized direct work on `main`.

---

### Task 1: Make generation start a one-time transition

**Files:**
- Modify: `packaging/macOS/standalone/RealtimePlayer.swift:50-360`
- Test: `tests/test_standalone_app_options.py`

**Interfaces:**
- Produces: `generationHasStarted: Bool`, `generationStartPending: Bool`, `startPlayersFromCurrentPosition()`, and generation-aware `resumeIfBuffered(endOfFile:)`.
- Consumes: `generation`, `requestedStartSeconds`, `startupSegmentCount`, `rebufferSegmentCount`, `sourcePlayer`, and `restoredPlayer`.

- [ ] **Step 1: Write a failing playback-transition contract test**

Add this test to `StandaloneAppOptionTests`:

```python
def test_player_starts_each_generation_once_and_resumes_without_seeking(self):
    player = PLAYER_SOURCE.read_text()

    for contract in [
        "private var generationHasStarted = false",
        "private var generationStartPending = false",
        "guard state != .playing, !generationStartPending else { return }",
        "generationHasStarted ? rebufferSegmentCount : startupSegmentCount",
        "private func startPlayersFromCurrentPosition()",
        "let startingGeneration = generation",
        "guard self.generation == startingGeneration else { return }",
    ]:
        self.assertIn(contract, player)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m unittest \
  tests.test_standalone_app_options.StandaloneAppOptionTests.test_player_starts_each_generation_once_and_resumes_without_seeking \
  -v
```

Expected: FAIL because the generation transition flags do not exist.

- [ ] **Step 3: Implement the one-time generation transition**

Add controller state:

```swift
private var generationHasStarted = false
private var generationStartPending = false
```

Reset both flags in `start`, `seek`, and `stop`. Replace the current start logic in `resumeIfBuffered` with:

```swift
private func resumeIfBuffered(endOfFile: Bool = false) {
  guard shouldPlay else { return }
  guard state != .playing, !generationStartPending else { return }
  if state == .paused {
    startPlayersFromCurrentPosition()
    return
  }
  let required = generationHasStarted ? rebufferSegmentCount : startupSegmentCount
  guard queuedSegments.count >= required || (endOfFile && !queuedSegments.isEmpty) else {
    if state != .loading && state != .seeking { state = .buffering }
    return
  }
  if generationHasStarted {
    startPlayersFromCurrentPosition()
    return
  }

  let startingGeneration = generation
  generationStartPending = true
  sourcePlayer.seek(
    to: CMTime(seconds: requestedStartSeconds, preferredTimescale: 600),
    toleranceBefore: .zero,
    toleranceAfter: .zero
  ) { [weak self] _ in
    Task { @MainActor in
      guard let self else { return }
      guard self.generation == startingGeneration else { return }
      self.generationStartPending = false
      guard self.shouldPlay else { return }
      self.generationHasStarted = true
      self.startPlayersFromCurrentPosition()
    }
  }
}

private func startPlayersFromCurrentPosition() {
  sourcePlayer.play()
  restoredPlayer.play()
  state = .playing
}
```

The segment handler may continue to call `resumeIfBuffered`; its new guards make segment delivery a no-op while already playing. Pause and underrun resume call `startPlayersFromCurrentPosition` and therefore do not seek.

- [ ] **Step 4: Run focused tests and compile**

Run:

```bash
python -m unittest tests.test_standalone_app_options -v
xcrun swiftc -O -parse-as-library \
  -target arm64-apple-macosx26.0 \
  -framework AppKit -framework SwiftUI \
  -framework AVFoundation -framework AVKit \
  packaging/macOS/standalone/MiohApp.swift \
  packaging/macOS/standalone/RealtimePlayer.swift \
  -o /tmp/mioh-player-state-check
```

Expected: all standalone option tests pass and Swift compilation exits zero.

- [ ] **Step 5: Commit the playback-state fix**

```bash
git add packaging/macOS/standalone/RealtimePlayer.swift \
  tests/test_standalone_app_options.py
git commit -m "fix: prevent repeated preview generation seeks"
```

### Task 2: Add a playback-only movie input

**Files:**
- Modify: `packaging/macOS/standalone/RealtimePlayer.swift:38-145,430-520`
- Modify: `packaging/macOS/standalone/MiohApp.swift:263-300`
- Test: `tests/test_standalone_app_options.py`

**Interfaces:**
- Produces: `RealtimePlayerController.previewInputURL: URL?`, `choosePreviewInput()`, and `RestorationRunner.previewArguments(resources:outputDirectory:input:)`.
- Consumes: `PathRow`, `RealtimePlayerController.stop()`, and the runner's processing settings.

- [ ] **Step 1: Write a failing independent-input contract test**

Add this test to `StandaloneAppOptionTests`:

```python
def test_playback_input_is_independent_from_export_input(self):
    player = PLAYER_SOURCE.read_text()
    app = APP_SOURCE.read_text()

    for contract in [
        "@Published var previewInputURL: URL?",
        "func choosePreviewInput()",
        "guard let input = previewInputURL",
        'PathRow(title: "再生動画"',
        "controller.previewInputURL == nil",
    ]:
        self.assertIn(contract, player)
    self.assertNotIn("guard let input = runner.inputURL", player)
    self.assertIn(
        "func previewArguments(resources: URL, outputDirectory: URL, input: URL)",
        app,
    )
    self.assertIn(
        'var args = ["--input", input.path, "--output-dir", outputDirectory.path]',
        app,
    )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m unittest \
  tests.test_standalone_app_options.StandaloneAppOptionTests.test_playback_input_is_independent_from_export_input \
  -v
```

Expected: FAIL because `previewInputURL` and the explicit input argument do not exist.

- [ ] **Step 3: Implement controller-owned preview input**

Import AppKit in `RealtimePlayer.swift` and add:

```swift
@Published var previewInputURL: URL?

func choosePreviewInput() {
  let panel = NSOpenPanel()
  panel.title = "再生動画を選択"
  panel.canChooseFiles = true
  panel.canChooseDirectories = false
  panel.allowsMultipleSelection = false
  guard panel.runModal() == .OK, let url = panel.url else { return }
  stop()
  previewInputURL = url
  position = 0
  duration = 0
  errorMessage = ""
  sourcePlayer.replaceCurrentItem(with: AVPlayerItem(url: url))
}
```

Change `start` to use only the controller URL:

```swift
guard let input = previewInputURL else {
  fail("再生タブで入力動画を選択してください")
  return
}
```

Pass that URL to the runner:

```swift
try runner.previewArguments(
  resources: resources,
  outputDirectory: session,
  input: input
)
```

Update the runner signature and arguments:

```swift
func previewArguments(resources: URL, outputDirectory: URL, input: URL) throws -> [String] {
  var args = ["--input", input.path, "--output-dir", outputDirectory.path]
```

- [ ] **Step 4: Add the playback-tab file row**

Place this above the video surface in `RealtimePlayerView`:

```swift
PathRow(
  title: "再生動画",
  icon: "film",
  url: controller.previewInputURL,
  action: controller.choosePreviewInput
)
```

Change the initial play button to disable against the preview input:

```swift
.disabled(controller.previewInputURL == nil)
```

- [ ] **Step 5: Run focused tests and compile**

Run:

```bash
python -m unittest tests.test_standalone_app_options -v
xcrun swiftc -O -parse-as-library \
  -target arm64-apple-macosx26.0 \
  -framework AppKit -framework SwiftUI \
  -framework AVFoundation -framework AVKit \
  packaging/macOS/standalone/MiohApp.swift \
  packaging/macOS/standalone/RealtimePlayer.swift \
  -o /tmp/mioh-player-input-check
```

Expected: all standalone option tests pass and Swift compilation exits zero.

- [ ] **Step 6: Commit the independent input**

```bash
git add packaging/macOS/standalone/MiohApp.swift \
  packaging/macOS/standalone/RealtimePlayer.swift \
  tests/test_standalone_app_options.py
git commit -m "feat: add independent preview video input"
```

### Task 3: Build and verify live playback

**Files:**
- Verify only.

**Interfaces:**
- Consumes: the completed standalone app, the preview worker, and a local test video.
- Produces: fresh test, build, signature, input-isolation, and monotonic-playback evidence.

- [ ] **Step 1: Run the complete automated test suite**

Run:

```bash
python -m pytest -q
```

Expected: zero failures.

- [ ] **Step 2: Rebuild and verify the standalone artifacts**

Run:

```bash
packaging/macOS/standalone/build_app.sh
codesign --verify --deep --strict --verbose=2 \
  build/macos-standalone/mioh.app
test -f build/macos-standalone/mioh-0.11.0-unsigned.dmg
```

Expected: the build exits zero, the app satisfies its designated requirement, and the DMG exists.

- [ ] **Step 3: Verify playback input isolation through the UI**

Launch the rebuilt app with Computer Use, open the playback tab, choose a movie
using the new `再生動画` row, and confirm the play button becomes enabled. Open
the basic tab and confirm its input path did not change.

- [ ] **Step 4: Verify monotonic restored playback through the UI**

Start restored playback, wait until the state becomes `再生中`, then sample the
displayed position and look-ahead at least once per second for 12 seconds.
Expected: position never returns to the generation start, state does not
restart on new segment delivery, and stop returns the controller to `待機中`.

- [ ] **Step 5: Review repository state**

Run:

```bash
git diff --check
git status --short --branch
git log -5 --oneline
```

Expected: no whitespace errors, a clean worktree, and both implementation commits on `main`.
