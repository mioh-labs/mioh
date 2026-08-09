# mioh

**Motion-Informed Optical Healing**

mioh is a macOS / iPad video restoration app focused on mosaic restoration, realtime preview, HLS playback workflows, and local-network worker experiments.

This repository is the public source tree for the current **mioh** application. Some internal package names, paths, and historical scripts still use the older `lada` namespace; those are implementation details and do not indicate a separate product.

## What mioh does

- Restores mosaic / pixelated regions in video using BasicVSR++-based models.
- Provides native macOS standalone builds with Core AI / Apple Silicon optimized inference.
- Provides **mioh Remote** for iPad, including iPad-side restoration and worker mode experiments.
- Includes browser-assisted HLS discovery and realtime HLS restoration playback.
- Supports local export workflows and realtime preview workflows.
- Includes training, evaluation, and model conversion utilities used for the current restoration models.

## Apps

### macOS

The macOS app is the main desktop interface.

- **Dedicated build**: includes the selected restoration model for normal use.
- **Universal build**: portable/model-flexible packaging path.
- **Realtime HLS playback**: browser tab, HLS candidate detection, segment prefetching, and restored playback.
- **Cluster mode**: experimental local-network worker distribution.

Build scripts live under:

```shell
packaging/macOS/standalone/
```

Common build commands:

```shell
# Dedicated app / DMG
packaging/macOS/standalone/build_app.sh

# Universal app / DMG
packaging/macOS/standalone/build_universal_app.sh
```

Recent build outputs are written under:

```shell
build/macos-standalone/
build/macos-standalone-universal/
```

### iPad / mioh Remote

The iPad app lives under:

```shell
apps/MiohRemote/
```

It includes:

- iPad standalone restoration.
- HLS/browser workflow.
- Realtime playback UI.
- Local-network worker mode for cluster experiments.

## Releases

Public builds are published from this repository’s GitHub releases:

[GitHub Releases](https://github.com/mioh-labs/mioh/releases)

Release assets may include unsigned development DMGs. Validate checksums when they are provided in the release notes.

## Models and assets

Large model weights and generated binary artifacts are not expected to live in git. The repository contains code, packaging scripts, tests, and metadata; local model files are resolved from the configured model directory during build or runtime.

Typical model-related paths:

```shell
model_weights/
packaging/macOS/standalone/model-tools/
```

## Development notes

Useful checks while editing:

```shell
pytest tests/test_mioh_mac_hls_browser.py
pytest tests/test_mioh_mac_hls_resilience.py
pytest tests/test_standalone_app_options.py
```

The codebase still contains historical names such as `lada`, `LADA_*`, and `lada/` because mioh evolved from the Lada restoration project. Please prefer **mioh** for user-facing product names, release names, documentation titles, and app UI.

## Repository layout

```text
apps/MiohRemote/                 iPad app
packaging/macOS/standalone/      macOS standalone build and app sources
lada/                            shared restoration/library code
configs/                         training and model configs
scripts/                         training, evaluation, packaging helpers
tests/                           regression and contract tests
docs/                            implementation notes and runbooks
```

## Safety and scope

mioh is intended for local, user-controlled media processing and trusted local-network workflows. HLS/browser features depend on the user’s own browsing session and should respect site terms, authorization, and applicable law.

## License and acknowledgements

See [LICENSE.md](LICENSE.md) for licensing details.

mioh builds on work from Lada, BasicVSR++ / MMagic, YOLO / Ultralytics, FFmpeg, PyTorch, Apple platform frameworks, and the broader open-source video restoration ecosystem.
