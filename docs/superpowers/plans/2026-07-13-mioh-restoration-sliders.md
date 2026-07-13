# mioh Restoration Slider Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace seven restoration numeric-only controls with clamped slider-plus-number rows while retaining both multiplier steppers.

**Architecture:** Add reusable `Double` and `Int` slider row helpers inside `ContentView`, keeping existing `RestorationRunner` properties and CLI argument generation unchanged. Replace only the five composition effect fields and ROI enhancer strength/tile rows, preserving their current disabled state and all defaults.

**Tech Stack:** SwiftUI, Swift 6, Python 3 unittest/pytest source-contract tests

## Global Constraints

- Composition ranges: sharpen 0–1 step 0.05; detail 0–1 step 0.05; feather 0–3 step 0.05; texture 0–1 step 0.01; smoothing 0–1 step 0.05.
- ROI ranges: strength 0–1 step 0.05; tile 0–1024 step 32.
- Keep composition effect scale as the existing 1–4 `Stepper`.
- Keep ROI enhancer scale as the existing 1–8 `Stepper`.
- Keep ROI strength and tile disabled when enhancer method is `none`.
- Keep all `@Published` properties, defaults, and CLI option generation unchanged.

---

### Task 1: Specify slider rows in the standalone GUI source contract

**Files:**
- Modify: `tests/test_standalone_app_options.py`
- Test: `tests/test_standalone_app_options.py`

**Interfaces:**
- Consumes: `packaging/macOS/standalone/MiohApp.swift` as text.
- Produces: `test_restoration_effects_use_slider_number_rows`, which fixes the exact helper calls, ranges, steps, multiplier steppers, and ROI disabled behavior as the UI contract.

- [ ] **Step 1: Add the failing source-contract test**

```python
def test_restoration_effects_use_slider_number_rows(self):
    source = APP_SOURCE.read_text()
    expected_slider_rows = [
        'doubleSliderField("シャープ", value: $runner.sharpenStrength, range: 0...1, step: 0.05)',
        'doubleSliderField("ディテール", value: $runner.detailBoost, range: 0...1, step: 0.05)',
        'doubleSliderField("境界フェザー", value: $runner.blendFeather, range: 0...3, step: 0.05)',
        'doubleSliderField("テクスチャ", value: $runner.textureMix, range: 0...1, step: 0.01)',
        'doubleSliderField("スムージング", value: $runner.smoothStrength, range: 0...1, step: 0.05)',
        'doubleSliderField("強度", value: $runner.roiEnhancerStrength, range: 0...1, step: 0.05)',
        'integerSliderField("タイル", value: $runner.roiEnhancerTile, range: 0...1024, step: 32)',
    ]

    self.assertIn("private func doubleSliderField", source)
    self.assertIn("private func integerSliderField", source)
    for row in expected_slider_rows:
        self.assertIn(row, source)
    self.assertIn(
        'LabeledContent("エフェクト倍率") { Stepper(value: $runner.effectUpscale, in: 1...4)',
        source,
    )
    self.assertIn(
        'LabeledContent("倍率") { Stepper(value: $runner.roiEnhancerScale, in: 1...8)',
        source,
    )
    self.assertIn(
        'doubleSliderField("強度", value: $runner.roiEnhancerStrength, range: 0...1, step: 0.05).disabled(runner.roiEnhancer == "none")',
        source,
    )
    self.assertIn(
        'integerSliderField("タイル", value: $runner.roiEnhancerTile, range: 0...1024, step: 32).disabled(runner.roiEnhancer == "none")',
        source,
    )
```

- [ ] **Step 2: Run the new test and verify RED**

Run: `pytest -q tests/test_standalone_app_options.py::StandaloneAppOptionTests::test_restoration_effects_use_slider_number_rows`

Expected: FAIL because `doubleSliderField` and `integerSliderField` do not exist.

- [ ] **Step 3: Commit the failing contract test**

```bash
git add tests/test_standalone_app_options.py
git commit -m "test: specify mioh restoration slider controls"
```

---

### Task 2: Implement clamped slider-plus-number helpers and replace rows

**Files:**
- Modify: `packaging/macOS/standalone/MiohApp.swift:591-610`
- Modify: `packaging/macOS/standalone/MiohApp.swift:706-712`
- Test: `tests/test_standalone_app_options.py`

**Interfaces:**
- Consumes: `Binding<Double>` or `Binding<Int>`, a closed range, and a step.
- Produces: `doubleSliderField(_:value:range:step:) -> some View` and `integerSliderField(_:value:range:step:) -> some View`.

- [ ] **Step 1: Add the reusable helpers**

```swift
private func doubleSliderField(
  _ title: String,
  value: Binding<Double>,
  range: ClosedRange<Double>,
  step: Double
) -> some View {
  let clampedValue = Binding<Double>(
    get: { min(max(value.wrappedValue, range.lowerBound), range.upperBound) },
    set: { value.wrappedValue = min(max($0, range.lowerBound), range.upperBound) }
  )
  return LabeledContent(title) {
    HStack(spacing: 12) {
      Slider(value: clampedValue, in: range, step: step)
        .frame(minWidth: 220)
      TextField("", value: clampedValue, format: .number.precision(.fractionLength(0...3)))
        .multilineTextAlignment(.trailing)
        .frame(width: 72)
    }
  }
}

private func integerSliderField(
  _ title: String,
  value: Binding<Int>,
  range: ClosedRange<Int>,
  step: Int
) -> some View {
  let clampedValue = Binding<Int>(
    get: { min(max(value.wrappedValue, range.lowerBound), range.upperBound) },
    set: { value.wrappedValue = min(max($0, range.lowerBound), range.upperBound) }
  )
  let sliderValue = Binding<Double>(
    get: { Double(clampedValue.wrappedValue) },
    set: { clampedValue.wrappedValue = Int($0.rounded()) }
  )
  return LabeledContent(title) {
    HStack(spacing: 12) {
      Slider(
        value: sliderValue,
        in: Double(range.lowerBound)...Double(range.upperBound),
        step: Double(step)
      )
      .frame(minWidth: 220)
      TextField("", value: clampedValue, format: .number)
        .multilineTextAlignment(.trailing)
        .frame(width: 72)
    }
  }
}
```

- [ ] **Step 2: Replace the seven approved rows only**

```swift
Section("合成") {
  doubleSliderField("シャープ", value: $runner.sharpenStrength, range: 0...1, step: 0.05)
  doubleSliderField("ディテール", value: $runner.detailBoost, range: 0...1, step: 0.05)
  doubleSliderField("境界フェザー", value: $runner.blendFeather, range: 0...3, step: 0.05)
  doubleSliderField("テクスチャ", value: $runner.textureMix, range: 0...1, step: 0.01)
  doubleSliderField("スムージング", value: $runner.smoothStrength, range: 0...1, step: 0.05)
  LabeledContent("エフェクト倍率") { Stepper(value: $runner.effectUpscale, in: 1...4) { Text("\(runner.effectUpscale)x") } }
}
```

```swift
LabeledContent("倍率") { Stepper(value: $runner.roiEnhancerScale, in: 1...8) { Text("\(runner.roiEnhancerScale)x") } }
  .disabled(runner.roiEnhancer == "none")
doubleSliderField("強度", value: $runner.roiEnhancerStrength, range: 0...1, step: 0.05).disabled(runner.roiEnhancer == "none")
integerSliderField("タイル", value: $runner.roiEnhancerTile, range: 0...1024, step: 32).disabled(runner.roiEnhancer == "none")
```

- [ ] **Step 3: Run the focused test and verify GREEN**

Run: `pytest -q tests/test_standalone_app_options.py::StandaloneAppOptionTests::test_restoration_effects_use_slider_number_rows`

Expected: PASS.

- [ ] **Step 4: Run all standalone option tests**

Run: `pytest -q tests/test_standalone_app_options.py`

Expected: all tests pass.

- [ ] **Step 5: Compile the Swift app source without packaging**

Run:

```bash
xcrun swiftc -O -parse-as-library -target arm64-apple-macosx27.0 \
  -framework AppKit -framework SwiftUI -framework CoreAI \
  packaging/macOS/standalone/MiohApp.swift -o /tmp/mioh-slider-build
```

Expected: exit code 0 and `/tmp/mioh-slider-build` exists.

- [ ] **Step 6: Commit the implementation**

```bash
git add packaging/macOS/standalone/MiohApp.swift
git commit -m "feat: add restoration slider controls to mioh"
```

---

### Task 3: Verify the complete repository

**Files:**
- Verify: `packaging/macOS/standalone/MiohApp.swift`
- Verify: `tests/test_standalone_app_options.py`

**Interfaces:**
- Consumes: the slider helper implementation and static source contract.
- Produces: a clean full-suite result and a reviewable final diff.

- [ ] **Step 1: Run the complete test suite**

Run: `pytest -q`

Expected: all tests pass with no failures; platform skips and existing deprecation warnings may remain.

- [ ] **Step 2: Validate formatting and scope**

Run:

```bash
git diff --check
git status --short
git log -4 --oneline
```

Expected: no whitespace errors; only intended committed files are present; latest commits are the slider contract and implementation.
