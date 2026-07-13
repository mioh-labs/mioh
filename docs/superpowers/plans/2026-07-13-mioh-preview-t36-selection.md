# mioh Preview T36 Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select T36 automatically for real-time preview on macOS 27 while preserving the existing T90 export default and standard v1.2 fallback on macOS 26.

**Architecture:** `PlatformCapabilities` owns the platform-dependent preview model policy. `RestorationRunner.previewArguments` consumes that preview-only model and derives its automatic clip length from it, while `processingArguments` continues to resolve the export picker unchanged.

**Tech Stack:** Swift 6, SwiftUI, Python unittest/pytest source-contract tests.

## Global Constraints

- The main application continues to target arm64 macOS 26.0.
- CoreAI remains available only on macOS 27 and newer.
- Preview uses `basicvsrpp-v1.2-coreai-t36` on macOS 27 or newer.
- Preview uses `basicvsrpp-v1.2` on macOS 26.
- Normal export model selection and its macOS 27 T90 default remain unchanged.
- An explicitly enabled maximum clip length continues to override the automatic preview clip length.

---

### Task 1: Add preview-only model policy

**Files:**
- Modify: `packaging/macOS/standalone/MiohApp.swift:15-38,260-282`
- Test: `tests/test_standalone_app_options.py`

**Interfaces:**
- Produces: `PlatformCapabilities.previewRestorationModel: String`.
- Consumes: `RestorationRunner.capabilities`, `RestorationRunner.useMaxClipLength`, and `RestorationRunner.maxClipLength`.

- [ ] **Step 1: Write the failing source-contract test**

Add this test to `StandaloneAppOptionTests`:

```python
def test_preview_uses_t36_without_changing_t90_export_default(self):
    source = APP_SOURCE.read_text()

    self.assertIn("var previewRestorationModel: String", source)
    self.assertIn(
        'supportsCoreAI ? "basicvsrpp-v1.2-coreai-t36" : "basicvsrpp-v1.2"',
        source,
    )
    self.assertIn("let previewModel = capabilities.previewRestorationModel", source)
    self.assertIn('add(&args, "--restoration-model", previewModel)', source)
    self.assertIn("switch previewModel", source)
    self.assertIn(
        'supportsCoreAI ? "basicvsrpp-v1.2-coreai-t90" : "basicvsrpp-v1.2"',
        source,
    )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
python -m unittest \
  tests.test_standalone_app_options.StandaloneAppOptionTests.test_preview_uses_t36_without_changing_t90_export_default \
  -v
```

Expected: FAIL because `previewRestorationModel` and `previewModel` do not exist.

- [ ] **Step 3: Implement the preview-only model selection**

Add this property to `PlatformCapabilities`, leaving `defaultRestorationModel` unchanged:

```swift
var previewRestorationModel: String {
  supportsCoreAI ? "basicvsrpp-v1.2-coreai-t36" : "basicvsrpp-v1.2"
}
```

In `previewArguments`, replace export-model resolution with the preview policy and switch clip-length selection to the same value:

```swift
let previewModel = capabilities.previewRestorationModel
try rejectUnsupportedCoreAIModel(previewModel)
let detection = try resolvedDetectionModel(in: resources)
// ...
add(&args, "--restoration-model", previewModel)
// ...
switch previewModel {
case "basicvsrpp-v1.2-coreai-t36": automaticClipLength = 104
default: automaticClipLength = 180
}
```

Keep `processingArguments` and `defaultRestorationModel` unchanged so export continues to use the picker and defaults to T90 on macOS 27.

- [ ] **Step 4: Run focused tests and compile the app sources**

Run:

```bash
python -m unittest tests.test_standalone_app_options -v
xcrun swiftc -O -parse-as-library \
  -target arm64-apple-macosx26.0 \
  -framework AppKit -framework SwiftUI \
  -framework AVFoundation -framework AVKit \
  packaging/macOS/standalone/MiohApp.swift \
  packaging/macOS/standalone/RealtimePlayer.swift \
  -o /tmp/mioh-preview-t36-check
```

Expected: all standalone option tests pass and Swift compilation exits zero.

- [ ] **Step 5: Run the complete test suite**

Run:

```bash
python -m pytest -q
```

Expected: zero failures.

- [ ] **Step 6: Commit**

```bash
git add packaging/macOS/standalone/MiohApp.swift \
  tests/test_standalone_app_options.py
git commit -m "feat: use T36 for realtime preview"
```
