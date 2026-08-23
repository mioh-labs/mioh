# mioh

**Motion-Informed Optical Healing**

mioh is a macOS-focused fork of [Lada](https://github.com/ladaapp/lada) for video mosaic detection/restoration workflows on Apple Silicon.

This repository is the public source/distribution repository for the macOS build. The only app build intended for public distribution here is the **mioh universal macOS build**.

## Download

The universal build should be published as a GitHub Release asset:

- `mioh-universal-0.14.3-unsigned.dmg`

Large DMG files are not stored directly in git. See [releases/README.md](releases/README.md).

## Build from source

The source needed to build the public universal macOS app is included under
`packaging/macOS/standalone/`.

From the repository root on macOS:

```zsh
LADA_STANDALONE_PYTHON_ENV=/path/to/python-venv \
MIOH_SKIP_HARDWARE_SMOKE=1 \
packaging/macOS/standalone/build_universal_app.sh
```

The build creates:

```text
build/macos-standalone-universal/mioh-universal.app
build/macos-standalone-universal/mioh-universal-0.14.3-unsigned.dmg
```

See [packaging/macOS/README.md](packaging/macOS/README.md) for full build
notes, model download/export steps, and requirements.

### Build mioh upscaler

`mioh upscaler` is a separate Apple Silicon application for FlashVSR/AdcSR
upscaling and MiniMax H3 video generation. It is not embedded in the universal
app or its DMG. The repository includes the Swift sources, packaging scripts,
and the minimal vendored FlashVSR build-time sources required to compile it.
Model weights and converted model packages remain external.

Requirements:

- Apple Silicon Mac
- Xcode and Command Line Tools with the macOS 27 SDK
- Network access on the first build to download the arm64 FFmpeg tools

From the repository root:

```zsh
packaging/macOS/upscaler/build_app.sh
```

The build creates:

```text
build/mioh-upscaler/mioh upscaler.app
build/mioh-upscaler/mioh-upscaler-0.14.3-unsigned.dmg
```

The DMG contains an alias to `/Applications`. When a selected upscaler model
is missing, version 0.14.3 can download, convert, and configure it from the
first-launch model setup screen; the user chooses the external model folder.
MiniMax H3 model assets are selected separately from the **動画生成** tab.
The one-time external model preparation procedure is documented in
[docs/minimax_h3_coreai_build.md](docs/minimax_h3_coreai_build.md).

See [packaging/macOS/upscaler/README.md](packaging/macOS/upscaler/README.md)
for model locations and implementation details.

<div align="center">

## 💛 Support mioh

### Help keep Apple Silicon development, testing, and model packaging moving.

<a href="https://ko-fi.com/miohlabs">
  <img src="https://storage.ko-fi.com/cdn/kofi2.png?v=3" alt="Support mioh on Ko-fi" width="260">
</a>

<br>

**If mioh helps your workflow, a small donation makes a real difference.**

Thank you for supporting independent development.

</div>

## Models are not included

The public mioh distribution should not commit model weight binaries or Core ML/Core AI packages directly to git.

Model files are obtained or regenerated separately using the scripts and license information in this repository. See [model_weights/README.md](model_weights/README.md).

## Source and license

mioh is based on Lada and is distributed under the **GNU Affero General Public License v3.0**.

You may charge for builds, accept donations, or offer paid support, but recipients keep the AGPL rights to inspect, modify, and redistribute the software. If you distribute a binary build, provide the corresponding source code and build scripts under the same license.

See:

- [LICENSE.md](LICENSE.md)
- [NOTICE.md](NOTICE.md)
- [LICENSES/](LICENSES/)

## What belongs in this repository

Keep:

- Source code required to build mioh.
- macOS packaging/build scripts.
- License and notice files.
- Model download/export/validation scripts.
- Small model license metadata files.
- Release instructions for the universal macOS build.

Do not commit:

- `.app`, `.dmg`, `.zip`, or build output directories.
- Raw model weights such as `.pt`, `.pth`, `.safetensors`, `.onnx`.
- Core ML / Core AI generated packages such as `.mlpackage` or `.aimodel`.
- Local Codex imports, logs, caches, datasets, or experiment outputs.

## Upstream

mioh is a modified fork of Lada. Lada authors retain copyright in upstream portions.

Upstream project: <https://github.com/ladaapp/lada>
