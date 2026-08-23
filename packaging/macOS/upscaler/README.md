# mioh upscaler

`mioh upscaler.app` is the standalone macOS 27 application for video
upscaling and MiniMax H3 video generation. These features are intentionally
not linked into or bundled with `mioh.app`.

Supported backends:

- FlashVSR Tiny/Compact, with the shared 85-frame streaming decoder.
- AdcSR x4 FP32 Core AI, with 128px tiles, 16px overlap, Metal feather
  composition, and optional optical-flow high-frequency stabilization.
- MiniMax H3 native Swift generation, with video or up to ten
  identity-reference images. Its converted graph, tokenizer and manifest stay
  external and are selected from the **動画生成** tab.

Build the signed-local app and unsigned distribution image with:

```bash
packaging/macOS/upscaler/build_app.sh
```

The repository includes the minimal FlashVSR Swift runtime and conversion
sources under `packaging/macOS/upscaler/vendor/flashvsr`, so a separate
`FlashVSR_plus` checkout is not required. `FLASHVSR_SOURCE_DIR` remains
available as an optional override for development. The first build requires
Xcode with the macOS 27 SDK and network access to fetch the arm64 FFmpeg tools.

Outputs:

- `build/mioh-upscaler/mioh upscaler.app`
- `build/mioh-upscaler/mioh-upscaler-0.14.3-unsigned.dmg`

Model weights are never bundled. The default external locations are:

- `model_weights/FlashVSR-v1.1-coreai-grid16`
- `model_weights/adcsr_x4_float32.aimodel`
- `model_weights/minimax-h3-native/manifest.json`

The first two can also be selected from the **アップスケール** tab and the
MiniMax H3 manifest from the **動画生成** tab.

See [../../../docs/minimax_h3_coreai_build.md](../../../docs/minimax_h3_coreai_build.md)
for the one-time external model conversion and validation procedure.

When the selected upscaler model is missing, version 0.14.3 opens the model
setup sheet on first launch. The user chooses an external destination;
FlashVSR official weights are downloaded and converted to Core AI on that Mac,
while the maintainer-provided AdcSR Core AI asset is downloaded and verified.
