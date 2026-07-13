# mioh Preview Buffer Start Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make realtime preview wait for the selected buffered duration before starting, with an optional four-second fast rebuffer mode.

**Architecture:** A small pure Swift `PreviewBufferPolicy` calculates the required duration and readiness from playback state, making the edge cases executable without AVPlayer. `RestorationRunner` owns the selected limit and fast-rebuffer preference; `RealtimePlayerController` applies the policy to actual queued duration while the Python worker continues enforcing the generation cap.

**Tech Stack:** Swift 6, SwiftUI, AVFoundation, Python `unittest`/`pytest`, `swiftc`, bundled Python preview worker.

## Global Constraints

- The buffer slider remains exactly 1 through 60 seconds in one-second steps, with an 8-second default.
- Initial playback always waits for the selected buffered duration.
- Normal rebuffer recovery waits for the selected buffered duration.
- `再バッファを短縮` defaults to off and only changes recovery after playback has already started.
- Fast recovery waits for `min(selected buffer limit, 4.0 seconds)`.
- End of file may start with less than the target only when at least one restored segment is queued.
- Changing policy never seeks, restarts, pauses active playback, changes generation, or clears queued segments.
- Continue working directly on `main` as requested by the user.

---

### Task 1: Executable Buffer Readiness Policy

**Files:**
- Create: `packaging/macOS/standalone/PreviewBufferPolicy.swift`
- Create: `tests/test_preview_buffer_policy.py`
- Modify: `packaging/macOS/standalone/build_app.sh:23-33`
- Modify: `tests/test_standalone_app_options.py:238-249`

**Interfaces:**
- Produces: `PreviewBufferPolicy.requiredSeconds(selectedBufferLimit:generationHasStarted:shortenRebuffer:) -> Double`.
- Produces: `PreviewBufferPolicy.canStart(bufferedSeconds:selectedBufferLimit:generationHasStarted:shortenRebuffer:endOfFile:hasQueuedSegments:) -> Bool`.
- Consumed by: `RealtimePlayerController.resumeIfBuffered(endOfFile:)` in Task 2.

- [ ] **Step 1: Write the failing executable Swift policy test**

Create `tests/test_preview_buffer_policy.py`:

```python
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "packaging" / "macOS" / "standalone" / "PreviewBufferPolicy.swift"


@unittest.skipUnless(sys.platform == "darwin" and shutil.which("xcrun"), "Swift toolchain required")
class PreviewBufferPolicyTests(unittest.TestCase):
    def test_selected_duration_and_fast_rebuffer_policy(self):
        harness = r'''
        @main
        struct PolicyTest {
          static func main() {
            precondition(PreviewBufferPolicy.requiredSeconds(
              selectedBufferLimit: 60, generationHasStarted: false, shortenRebuffer: true
            ) == 60)
            precondition(PreviewBufferPolicy.requiredSeconds(
              selectedBufferLimit: 60, generationHasStarted: true, shortenRebuffer: false
            ) == 60)
            precondition(PreviewBufferPolicy.requiredSeconds(
              selectedBufferLimit: 60, generationHasStarted: true, shortenRebuffer: true
            ) == 4)
            precondition(PreviewBufferPolicy.requiredSeconds(
              selectedBufferLimit: 2, generationHasStarted: true, shortenRebuffer: true
            ) == 2)
            precondition(!PreviewBufferPolicy.canStart(
              bufferedSeconds: 59.9, selectedBufferLimit: 60,
              generationHasStarted: false, shortenRebuffer: false,
              endOfFile: false, hasQueuedSegments: true
            ))
            precondition(PreviewBufferPolicy.canStart(
              bufferedSeconds: 60, selectedBufferLimit: 60,
              generationHasStarted: false, shortenRebuffer: false,
              endOfFile: false, hasQueuedSegments: true
            ))
            precondition(PreviewBufferPolicy.canStart(
              bufferedSeconds: 1.5, selectedBufferLimit: 60,
              generationHasStarted: false, shortenRebuffer: false,
              endOfFile: true, hasQueuedSegments: true
            ))
            precondition(!PreviewBufferPolicy.canStart(
              bufferedSeconds: 0, selectedBufferLimit: 60,
              generationHasStarted: false, shortenRebuffer: false,
              endOfFile: true, hasQueuedSegments: false
            ))
            print("ok")
          }
        }
        '''
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            main = directory / "main.swift"
            binary = directory / "policy-test"
            main.write_text(harness)
            subprocess.run(
                ["xcrun", "swiftc", str(POLICY), str(main), "-o", str(binary)],
                check=True,
                capture_output=True,
                text=True,
            )
            result = subprocess.run(
                [str(binary)], check=True, capture_output=True, text=True
            )
        self.assertEqual(result.stdout.strip(), "ok")
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python -m pytest -q tests/test_preview_buffer_policy.py
```

Expected: FAIL because `PreviewBufferPolicy.swift` does not exist.

- [ ] **Step 3: Add the minimal pure Swift policy**

Create `packaging/macOS/standalone/PreviewBufferPolicy.swift`:

```swift
struct PreviewBufferPolicy {
  static let shortenedRebufferSeconds = 4.0

  static func requiredSeconds(
    selectedBufferLimit: Double,
    generationHasStarted: Bool,
    shortenRebuffer: Bool
  ) -> Double {
    let selected = max(0, selectedBufferLimit)
    guard generationHasStarted, shortenRebuffer else { return selected }
    return min(selected, shortenedRebufferSeconds)
  }

  static func canStart(
    bufferedSeconds: Double,
    selectedBufferLimit: Double,
    generationHasStarted: Bool,
    shortenRebuffer: Bool,
    endOfFile: Bool,
    hasQueuedSegments: Bool
  ) -> Bool {
    guard hasQueuedSegments else { return false }
    if endOfFile { return true }
    let required = requiredSeconds(
      selectedBufferLimit: selectedBufferLimit,
      generationHasStarted: generationHasStarted,
      shortenRebuffer: shortenRebuffer
    )
    return bufferedSeconds + 0.001 >= required
  }
}
```

- [ ] **Step 4: Include the policy in the app build and bundle contract**

Add the policy source before `RealtimePlayer.swift` in `build_app.sh`:

```zsh
  "$PACKAGE_DIR/MiohApp.swift" \
  "$PACKAGE_DIR/PreviewBufferPolicy.swift" \
  "$PACKAGE_DIR/RealtimePlayer.swift" \
```

Add this assertion to `test_app_bundles_realtime_player_and_preview_worker`:

```python
self.assertIn('"$PACKAGE_DIR/PreviewBufferPolicy.swift"', script)
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
python -m pytest -q tests/test_preview_buffer_policy.py \
  tests/test_standalone_app_options.py::StandaloneAppOptionTests::test_app_bundles_realtime_player_and_preview_worker
```

Expected: `2 passed`.

- [ ] **Step 6: Commit the policy**

```bash
git add packaging/macOS/standalone/PreviewBufferPolicy.swift \
  packaging/macOS/standalone/build_app.sh \
  tests/test_preview_buffer_policy.py \
  tests/test_standalone_app_options.py
git commit -m "feat: add preview buffer readiness policy"
```

---

### Task 2: Apply Selected Duration to Playback Start and Rebuffering

**Files:**
- Modify: `packaging/macOS/standalone/MiohApp.swift:115-123`
- Modify: `packaging/macOS/standalone/RealtimePlayer.swift:65-72,256-263,374-389,531-555`
- Modify: `tests/test_standalone_app_options.py:14-54,102-124`

**Interfaces:**
- Consumes: `PreviewBufferPolicy.canStart(...)` from Task 1.
- Produces: `RestorationRunner.previewShortenedRebuffer: Bool` with default `false`.
- Produces: `RealtimePlayerController.bufferPolicyDidChange()` for immediate reevaluation while buffering.

- [ ] **Step 1: Write the failing source-integration test**

Replace the fixed segment-count assertions with this test in
`StandaloneAppOptionTests`:

```python
def test_player_waits_for_selected_buffer_duration_with_optional_fast_rebuffer(self):
    app = APP_SOURCE.read_text()
    player = PLAYER_SOURCE.read_text()

    self.assertIn("@Published var previewShortenedRebuffer = false", app)
    for contract in [
        "PreviewBufferPolicy.canStart(",
        "bufferedSeconds: bufferedSeconds",
        "selectedBufferLimit: runner.previewBufferLimit",
        "generationHasStarted: generationHasStarted",
        "shortenRebuffer: runner.previewShortenedRebuffer",
        "endOfFile: endOfFile",
        "hasQueuedSegments: !queuedSegments.isEmpty",
        "func bufferPolicyDidChange()",
        '"再バッファを短縮",',
        ".toggleStyle(.checkbox)",
        "controller.bufferPolicyDidChange()",
    ]:
        self.assertIn(contract, player)
    self.assertNotIn("startupSegmentCount = 3", player)
    self.assertNotIn("rebufferSegmentCount = 2", player)
```

Remove the obsolete string expectations for `startupSegmentCount`,
`rebufferSegmentCount`, and their ternary selection from the earlier player
contract tests.

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m pytest -q tests/test_standalone_app_options.py::StandaloneAppOptionTests::test_player_waits_for_selected_buffer_duration_with_optional_fast_rebuffer
```

Expected: FAIL because the preference, policy call, checkbox, and reevaluation
method do not exist.

- [ ] **Step 3: Add the preference and replace fixed segment counts**

In `RestorationRunner`, add:

```swift
@Published var previewBufferLimit = 8.0
@Published var previewShortenedRebuffer = false
```

Remove these controller constants:

```swift
let startupSegmentCount = 3
let rebufferSegmentCount = 2
```

Replace the fixed count guard in `resumeIfBuffered` with:

```swift
guard let runner else { return }
let ready = PreviewBufferPolicy.canStart(
  bufferedSeconds: bufferedSeconds,
  selectedBufferLimit: runner.previewBufferLimit,
  generationHasStarted: generationHasStarted,
  shortenRebuffer: runner.previewShortenedRebuffer,
  endOfFile: endOfFile,
  hasQueuedSegments: !queuedSegments.isEmpty
)
guard ready else {
  if state != .loading && state != .seeking { state = .buffering }
  return
}
```

- [ ] **Step 4: Reevaluate immediately when a buffering preference changes**

Add beside `setBufferLimit`:

```swift
func setBufferLimit(_ seconds: Double) {
  guard worker != nil else { return }
  sendCommand(["command": "set_buffer_limit", "seconds": seconds])
  bufferPolicyDidChange()
}

func bufferPolicyDidChange() {
  guard worker != nil else { return }
  resumeIfBuffered()
}
```

This remains a no-op during active playback because `resumeIfBuffered` returns
when `state == .playing`.

- [ ] **Step 5: Add the checkbox to the buffer row**

After the selected-second label, add:

```swift
Toggle(
  "再バッファを短縮",
  isOn: Binding(
    get: { runner.previewShortenedRebuffer },
    set: { value in
      runner.previewShortenedRebuffer = value
      controller.bufferPolicyDidChange()
    }
  )
)
.toggleStyle(.checkbox)
.help("再生途中のバッファ切れだけ最大4秒で復帰します")
```

- [ ] **Step 6: Run focused tests and compile the complete Swift app target**

Run:

```bash
python -m pytest -q tests/test_preview_buffer_policy.py tests/test_standalone_app_options.py
xcrun swiftc -O -parse-as-library -target arm64-apple-macosx26.0 \
  -framework AppKit -framework SwiftUI -framework AVFoundation -framework AVKit \
  packaging/macOS/standalone/MiohApp.swift \
  packaging/macOS/standalone/PreviewBufferPolicy.swift \
  packaging/macOS/standalone/RealtimePlayer.swift \
  -o /tmp/mioh-buffer-start-policy-verify
```

Expected: all focused tests PASS and `swiftc` exits 0.

- [ ] **Step 7: Commit the player integration**

```bash
git add packaging/macOS/standalone/MiohApp.swift \
  packaging/macOS/standalone/RealtimePlayer.swift \
  tests/test_standalone_app_options.py
git commit -m "feat: wait for selected preview buffer"
```

---

### Task 3: Full Build and Runtime Verification

**Files:**
- Verify: `build/macos-standalone/mioh.app`
- Verify: `build/macos-standalone/mioh-0.11.0-unsigned.dmg`

**Interfaces:**
- Consumes: the policy and player integration from Tasks 1 and 2.
- Produces: freshly built and signed artifacts verified on `main`.

- [ ] **Step 1: Run the complete automated suite**

```bash
python -m pytest -q
```

Expected: all tests pass with only the repository's existing skips and
Torch/MMEngine deprecation warnings.

- [ ] **Step 2: Rebuild the standalone app and DMG**

```bash
packaging/macOS/standalone/build_app.sh
```

Expected: the MPS deform-conv smoke test passes and both artifact paths print.

- [ ] **Step 3: Verify the app signature**

```bash
codesign --verify --deep --strict --verbose=2 build/macos-standalone/mioh.app
```

Expected: `valid on disk` and `satisfies its Designated Requirement`.

- [ ] **Step 4: Verify initial buffering in the built app**

Using Computer Use on the exact built app:

1. Open the playback tab and select an independent video.
2. Confirm `再バッファを短縮` is off by default.
3. Set the buffer limit to 60 seconds and start playback.
4. Confirm the state remains `バッファ中` after more than the former three
   segments are available.
5. Confirm playback changes to `再生中` only when the displayed ahead buffer
   reaches at least 60 seconds, or when the worker reports end of file with
   queued restored media.
6. While actively playing, change the buffer limit and checkbox and confirm the
   worker PID and generation remain unchanged.
7. Stop and confirm `待機中` and worker exit.

- [ ] **Step 5: Verify final repository state**

```bash
git diff --check
git status --short --branch
git log -7 --oneline
```

Expected: no uncommitted source changes and branch remains `main`.
