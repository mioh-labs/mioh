# mioh Additional FFmpeg Options Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an always-visible multiline field that safely augments final-output encoder options in mioh and the parallel-processing CLI.

**Architecture:** `process_video_parallel.py` owns one `shlex`-based parser and merge function, then resolves automatic, preset, and custom modes to one explicit encoder plus one final option string before invoking `lada-cli`. `MiohApp.swift` only collects the value and forwards it through the existing `--encoder-options` interface, avoiding duplicated parsing logic in Swift.

**Tech Stack:** Python 3.12, `argparse`, `shlex`, `unittest`, SwiftUI, macOS standalone build script.

## Global Constraints

- Apply additional options only to the restored output video's encoder.
- Do not inject options into segment splitting or final concatenation commands.
- Additional values override matching base keys; unmatched base keys remain.
- Treat negative numeric tokens such as `-1` as values, not option keys.
- Preserve current output when the additional field is empty.
- Keep the underlying `lada-cli` interface unchanged.
- Keep using the standalone app's bundled FFmpeg and ffprobe binaries.

---

### Task 1: Shared encoder-option parser and merge

**Files:**
- Modify: `process_video_parallel.py:35-60,844-1035,2239-2380`
- Test: `tests/test_process_video_parallel_encoding.py`

**Interfaces:**
- Produces: `parse_encoder_options(options: str | None) -> list[tuple[str, str | None]]`
- Produces: `merge_encoder_options(base: str | None, additional: str | None) -> str`
- Produces: `encoder_options_argument(value: str) -> str`

- [ ] **Step 1: Write the failing parser and merge tests**

Add these methods to `ProcessVideoParallelEncodingTests`:

```python
def test_additional_options_override_base_and_keep_order(self):
    merged = pvp.merge_encoder_options(
        "-q:v 55 -pix_fmt yuv420p -realtime 0",
        "-pix_fmt yuv420p10le -profile:v main10",
    )
    self.assertEqual(pvp.parse_encoder_options(merged), [
        ("-q:v", "55"),
        ("-pix_fmt", "yuv420p10le"),
        ("-realtime", "0"),
        ("-profile:v", "main10"),
    ])

def test_parser_supports_quotes_flags_and_negative_values(self):
    self.assertEqual(
        pvp.parse_encoder_options('-metadata "title=My Video" -qmin -1 -fast'),
        [("-metadata", "title=My Video"), ("-qmin", "-1"), ("-fast", None)],
    )

def test_empty_additional_options_preserve_base(self):
    base = "-crf 18 -preset slow"
    self.assertEqual(
        pvp.parse_encoder_options(pvp.merge_encoder_options(base, "")),
        pvp.parse_encoder_options(base),
    )

def test_parser_rejects_malformed_input(self):
    with self.assertRaisesRegex(ValueError, "引用符"):
        pvp.parse_encoder_options('-metadata "unterminated')
    with self.assertRaisesRegex(ValueError, "オプション名"):
        pvp.parse_encoder_options("orphan -crf 18")
```

- [ ] **Step 2: Run the tests and verify the red state**

Run: `python -m unittest tests.test_process_video_parallel_encoding -v`

Expected: errors state that `merge_encoder_options` and `parse_encoder_options` do not exist.

- [ ] **Step 3: Implement parsing, merging, and argparse validation**

Add `import shlex`, then add these helpers near the encoding helpers:

```python
def _is_encoder_option_key(token: str) -> bool:
    return len(token) > 1 and token.startswith("-") and not (
        token[1].isdigit() or token[1] == "."
    )


def parse_encoder_options(options: str | None) -> list[tuple[str, str | None]]:
    if not options or not options.strip():
        return []
    try:
        tokens = shlex.split(options)
    except ValueError as exc:
        raise ValueError(f"追加FFmpegオプションの引用符が不正です: {exc}") from exc
    parsed = []
    index = 0
    while index < len(tokens):
        key = tokens[index]
        if not _is_encoder_option_key(key):
            raise ValueError(f"追加FFmpegオプションにはオプション名が必要です: {key}")
        value = None
        if index + 1 < len(tokens) and not _is_encoder_option_key(tokens[index + 1]):
            value = tokens[index + 1]
            index += 1
        parsed.append((key, value))
        index += 1
    return parsed


def merge_encoder_options(base: str | None, additional: str | None) -> str:
    merged: dict[str, str | None] = {}
    for key, value in [*parse_encoder_options(base), *parse_encoder_options(additional)]:
        merged[key] = value
    tokens = []
    for key, value in merged.items():
        tokens.append(key)
        if value is not None:
            tokens.append(value)
    return shlex.join(tokens)


def encoder_options_argument(value: str) -> str:
    try:
        parse_encoder_options(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value
```

Replace the hand-written `.split()` merge inside `get_optimal_encoder_options` with:

```python
options = merge_encoder_options(options, user_options)
if user_options:
    print(f"📝 ユーザーオプション適用: {user_options}")
```

Set `type=encoder_options_argument` on the existing `--encoder-options` parser argument.

- [ ] **Step 4: Run the focused tests and verify green**

Run: `python -m unittest tests.test_process_video_parallel_encoding -v`

Expected: all encoding tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add process_video_parallel.py tests/test_process_video_parallel_encoding.py
git commit -m "Add safe encoder option merging"
```

---

### Task 2: Resolve presets and custom settings through one path

**Files:**
- Modify: `process_video_parallel.py:55,342-425,1435-1455,1770-1800`
- Test: `tests/test_process_video_parallel_encoding.py`
- Test: `tests/test_process_video_parallel_coreai.py`

**Interfaces:**
- Consumes: `merge_encoder_options(base, additional) -> str` from Task 1.
- Produces: `resolve_worker_encoding(config: WorkerRuntimeConfig) -> tuple[str | None, str | None]`.
- Updates: `build_lada_cli_command(...)` to pass one resolved encoder and option string.

- [ ] **Step 1: Write failing resolution and command tests**

Add `from dataclasses import replace` to the test file and use it with the existing `make_config` fixture:

```python
def test_preset_options_are_overridden_by_additional_values(self):
    config = replace(
        make_config("basicvsrpp-v1.2"),
        encoder_options="-q:v 70 -profile:v main10",
        optimal_encoder_options="-b:v 99M",
    )
    encoder, options = pvp.resolve_worker_encoding(config)
    parsed = pvp.parse_encoder_options(options)
    self.assertEqual(encoder, "hevc_videotoolbox")
    self.assertIn(("-q:v", "70"), parsed)
    self.assertIn(("-profile:v", "main10"), parsed)
    self.assertNotIn(("-b:v", "99M"), parsed)

def test_preset_command_uses_resolved_encoder_and_options(self):
    config = replace(make_config("basicvsrpp-v1.2"), encoder_options="-q:v 70")
    cmd = pvp.build_lada_cli_command(config, pvp.Path("in.mp4"), pvp.Path("out.mp4"))
    self.assertNotIn("--encoding-preset", cmd)
    self.assertEqual(cmd[cmd.index("--encoder") + 1], "hevc_videotoolbox")
    final_options = cmd[cmd.index("--encoder-options") + 1]
    self.assertIn(("-q:v", "70"), pvp.parse_encoder_options(final_options))

def test_automatic_and_custom_modes_use_resolved_final_options(self):
    automatic = replace(
        make_config("basicvsrpp-v1.2"),
        encoding_preset=None,
        optimal_encoder_options="-q:v 72",
    )
    custom = replace(
        automatic,
        encoder="libx264",
        optimal_encoder_options="-crf 19",
    )
    self.assertEqual(
        pvp.resolve_worker_encoding(automatic),
        ("hevc_videotoolbox", "-q:v 72"),
    )
    self.assertEqual(pvp.resolve_worker_encoding(custom), ("libx264", "-crf 19"))
```

- [ ] **Step 2: Run the tests and verify the red state**

Run:

```bash
python -m unittest \
  tests.test_process_video_parallel_encoding \
  tests.test_process_video_parallel_coreai -v
```

Expected: failure because `resolve_worker_encoding` is missing and preset commands still use `--encoding-preset`.

- [ ] **Step 3: Implement shared resolution**

Import `get_encoding_presets` beside `get_default_preset_name`, then add:

```python
def resolve_worker_encoding(
    config: WorkerRuntimeConfig,
) -> tuple[str | None, str | None]:
    if config.encoding_preset:
        preset = next(
            (item for item in get_encoding_presets() if item.name == config.encoding_preset),
            None,
        )
        if preset is None:
            raise ValueError(f"不明なエンコーディングプリセット: {config.encoding_preset}")
        return preset.encoder_name, merge_encoder_options(
            preset.encoder_options,
            config.encoder_options,
        )
    if config.encoder:
        return config.encoder, config.optimal_encoder_options or config.encoder_options or ""
    if config.optimal_encoder_options and config.device == "mps":
        return "hevc_videotoolbox", config.optimal_encoder_options
    if config.encoder_options:
        default_encoder = "hevc_videotoolbox" if config.device == "mps" else "libx264"
        return default_encoder, config.encoder_options
    return None, None
```

Replace the encoding branches in `build_lada_cli_command` with:

```python
encoder, encoder_options = resolve_worker_encoding(config)
if encoder:
    cmd.extend(["--encoder", encoder])
    cmd.extend(["--encoder-options", encoder_options or ""])
```

In `VideoProcessor.__init__`, initialize a separate intermediate setting:

```python
self.intermediate_encoder_options = None
```

During input analysis, call `get_optimal_encoder_options` with `user_options=None`, retain that result for pre-FPS conversion, and merge the free-form value only into the final-output setting:

```python
self.intermediate_encoder_options = get_optimal_encoder_options(
    input_video,
    None,
    self.args.auto_optimize,
    self.args.fps,
    self.args.bitrate_multiplier,
    self.args.qmin,
    self.args.qmax,
    getattr(self.args, "quality", None),
    use_pre_fps_conversion,
)
self.optimal_encoder_options = merge_encoder_options(
    self.intermediate_encoder_options,
    self.args.encoder_options,
)
```

Pass only `self.intermediate_encoder_options` into `split_video(..., encoder_options=...)`. Add a regression assertion to `tests/test_process_video_parallel_encoding.py` that reads the source and requires `encoder_options=self.intermediate_encoder_options`, while rejecting `encoder_options=self.optimal_encoder_options` in the `split_video` call.

After the final-output setting is computed and before splitting, log the resolved settings once:

```python
resolved_encoder, resolved_options = resolve_worker_encoding(
    self._build_worker_runtime_config()
)
if resolved_encoder:
    print(f"🎬 最終エンコーダー設定: {resolved_encoder} {resolved_options or ''}".rstrip())
```

Route the legacy duplicate worker command construction through `build_lada_cli_command` so both executor paths share the precedence rules.

- [ ] **Step 4: Run command, split, and resume regressions**

Run:

```bash
python -m unittest \
  tests.test_process_video_parallel_encoding \
  tests.test_process_video_parallel_coreai \
  tests.test_process_video_parallel_segment_count \
  tests.test_process_video_parallel_resume -v
```

Expected: all tests pass and split/merge behavior is unchanged.

- [ ] **Step 5: Commit Task 2**

```bash
git add process_video_parallel.py tests/test_process_video_parallel_encoding.py tests/test_process_video_parallel_coreai.py
git commit -m "Apply extra options to encoding presets"
```

---

### Task 3: Always-visible multiline field in mioh

**Files:**
- Modify: `packaging/macOS/standalone/MiohApp.swift:44-50,246-255,631-660`
- Test: `tests/test_standalone_app_options.py`

**Interfaces:**
- Consumes: existing `RestorationRunner.encoderOptions: String`.
- Produces: `--encoder-options <text>` in automatic, preset, and custom modes when non-empty.

- [ ] **Step 1: Write a failing Swift source contract test**

```python
def test_gui_has_always_visible_multiline_ffmpeg_options(self):
    source = APP_SOURCE.read_text()
    self.assertIn('Section("FFmpeg詳細設定")', source)
    self.assertIn('Text("追加FFmpegオプション")', source)
    self.assertIn('TextEditor(text: $runner.encoderOptions)', source)
    self.assertEqual(
        source.count('addOptional(&args, "--encoder-options", encoderOptions)'),
        1,
    )
```

- [ ] **Step 2: Run the source contract and verify the red state**

Run:

```bash
python -m unittest \
  tests.test_standalone_app_options.StandaloneAppOptionTests.test_gui_has_always_visible_multiline_ffmpeg_options -v
```

Expected: failure because no `TextEditor` or always-visible section exists.

- [ ] **Step 3: Forward the value independently of mode and add the editor**

Change argument construction to:

```swift
if encodingMode == "preset" {
  add(&args, "--encoding-preset", encodingPreset)
} else if encodingMode == "custom" {
  add(&args, "--encoder", encoder)
}
addOptional(&args, "--encoder-options", encoderOptions)
```

Replace the custom-only options `TextField` with an always-visible section:

```swift
Section("FFmpeg詳細設定") {
  Text("追加FFmpegオプション").font(.headline)
  TextEditor(text: $runner.encoderOptions)
    .font(.system(.body, design: .monospaced))
    .frame(minHeight: 72)
    .overlay(
      RoundedRectangle(cornerRadius: 6)
        .stroke(Color.secondary.opacity(0.25))
    )
  Text("例: -pix_fmt yuv420p10le -profile:v main10 -b:v 20M")
    .font(.caption)
    .foregroundStyle(.secondary)
}
```

- [ ] **Step 4: Run standalone source tests**

Run: `python -m unittest tests.test_standalone_app_options -v`

Expected: all source-contract tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add packaging/macOS/standalone/MiohApp.swift tests/test_standalone_app_options.py
git commit -m "Add detailed FFmpeg options field to mioh"
```

---

### Task 4: End-to-end verification and standalone artifacts

**Files:**
- Verify: `build/macos-standalone/mioh.app`
- Verify: `build/macos-standalone/mioh-0.11.0-unsigned.dmg`

**Interfaces:**
- Consumes all interfaces from Tasks 1-3.
- Produces signed standalone app and DMG artifacts containing the feature.

- [ ] **Step 1: Run the focused regression suite**

```bash
python -m unittest \
  tests.test_process_video_parallel_encoding \
  tests.test_process_video_parallel_coreai \
  tests.test_process_video_parallel_segment_count \
  tests.test_process_video_parallel_resume \
  tests.test_standalone_app_options -v
```

Expected: all selected tests pass.

- [ ] **Step 2: Verify malformed CLI input exits before processing**

Run: `python process_video_parallel.py --encoder-options '-metadata "unterminated' 2>&1`

Expected: nonzero exit with a concise Japanese quote error and no restoration startup log.

- [ ] **Step 3: Build the standalone app and DMG**

Run: `packaging/macOS/standalone/build_app.sh`

Expected: build completes, the MPS deform-conv smoke test passes, and both artifacts are created.

- [ ] **Step 4: Verify signatures and bundled sources**

```bash
codesign --verify --deep --strict --verbose=2 build/macos-standalone/mioh.app
rg -n "追加FFmpegオプション|--encoder-options" \
  packaging/macOS/standalone/MiohApp.swift \
  build/macos-standalone/mioh.app/Contents/Resources/process_video_parallel.py
```

Expected: signature verification succeeds and both the UI and bundled processor contain the new path.

- [ ] **Step 5: Verify diff hygiene**

```bash
git diff --check
git status --short --branch
```

Expected: no unstaged implementation changes remain after the task commits. If verification exposes a defect, fix only the responsible Task 1-3 file, rerun its focused test, and commit as `Fix standalone FFmpeg option integration`.
