# mioh macOS 26 M1 Compatibility Design

## Goal

Ship one mioh application that launches and processes video on an M1 Mac running macOS 26, while retaining CoreAI acceleration on macOS 27 and newer.

## Requirements

- Set the application deployment target and `LSMinimumSystemVersion` to macOS 26.0.
- Do not impose a new parallel-worker limit. Preserve the user's selected worker count.
- On macOS 26, use the PyTorch/MPS BasicVSR++ restoration model and Core ML detection model by default.
- On macOS 26, omit CoreAI-only restoration and detection choices because their runtime APIs require macOS 27.
- On macOS 27 and newer, preserve the current CoreAI choices and CoreAI default restoration model.
- Keep the existing arm64 Python, PyTorch, MPS deformable-convolution extension, Core ML models, ffmpeg, and ffprobe runtime.

## Approaches Considered

### One adaptive application (selected)

Build the GUI for macOS 26 and determine the available model lists and defaults from the operating-system version. Keep the CoreAI helper as a macOS 27-only executable and launch it only when CoreAI models are available.

This keeps one download and preserves the faster macOS 27 path without making macOS 26 load CoreAI.

### Separate macOS 26 and macOS 27 applications

Build two application bundles with different model sets. This simplifies each individual bundle but doubles packaging, testing, and distribution work and risks configuration drift.

### Remove CoreAI from all builds

Use MPS and Core ML everywhere. This has the smallest compatibility surface but discards the existing macOS 27 acceleration and is therefore rejected.

## Architecture

`MiohApp.swift` must not import or link CoreAI. A small platform-capability value derived from `ProcessInfo.processInfo.operatingSystemVersion` controls the restoration-model list, detection-model list, and their initial selections.

The standalone build compiles the main executable for `arm64-apple-macosx26.0`. It continues compiling `CoreAIRunner.swift` for `arm64-apple-macosx27.0` because the CoreAI API is unavailable on macOS 26. The helper can remain in the bundle: macOS 26 never launches it because no CoreAI option is exposed or selected there.

The Python runtime already selects the implementation from the model suffix. The macOS 26 restoration default `basicvsrpp-v1.2` resolves to the bundled `.pth` model and runs through PyTorch/MPS plus `mps_deform_conv`. Detection defaults to `v2-coreml`, which avoids CoreAI and runs through Core ML.

## Runtime Behavior

On macOS 26:

- restoration default: `basicvsrpp-v1.2`
- restoration choices: `basicvsrpp-v1.2`, `カスタム`
- detection default: `v2-coreml`
- detection choices: Core ML, PyTorch, and custom choices; no `v4-fast-coreai`
- CoreAI helper environment variable is not exported

On macOS 27 and newer:

- restoration default remains `basicvsrpp-v1.2-coreai-t90`
- existing CoreAI restoration and detection choices remain visible
- CoreAI helper environment variable is exported as before

The selected parallel-worker count is unchanged on both systems.

## Error Handling

Model selection must be normalized immediately before starting a process. If preferences or restored view state contain a CoreAI selection on macOS 26, it is replaced with the safe platform default. This prevents a stale selection from starting the macOS 27-only helper.

## Testing

Static contract tests will verify the deployment targets, plist minimum version, absence of a CoreAI link from the GUI executable, retention of the macOS 27 helper, conditional model lists/defaults, safe selection normalization, and unchanged `--parallel-workers` forwarding.

The final verification will compile the GUI for `arm64-apple-macosx26.0`, inspect its Mach-O deployment version and linked frameworks, compile the helper for macOS 27, and run the full Python test suite.

