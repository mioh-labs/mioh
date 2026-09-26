# mioh upscaler

`mioh upscaler.app` is the standalone macOS 27 application for video
upscaling and MiniMax H3 video generation. These features are intentionally
not linked into or bundled with `mioh.app`.

Supported backends:

- FlashVSR Tiny/Compact, with the shared 85-frame streaming decoder.
- AdcSR x4 FP32 Core AI, with 128px tiles, 16px overlap, Metal feather
  composition, and optional optical-flow high-frequency stabilization.
- PiperSR x2 Core ML, with an enumerated-shape full-frame path and overlapping
  256px tiles for other sizes, including 1920×1080 → 3840×2160. Frames are
  enhanced independently; it is not a temporal video-restoration model. The
  full-frame path uses a derived FP16-I/O model, two reusable frame sessions,
  Metal output conversion and overlapped frame scheduling.
- SwiftVR x2/x4 Core ML, using the same temporal 256px model pack as mioh
  Universal. The full frame is divided into overlapping 256px tiles, each
  processed as a video scene; tiles and 8-frame scene boundaries are blended.
  This is substantially slower than PiperSR and the model pack stays external.
- MiniMax H3 / 10Eros-Max H3 native Swift generation. Ref2VA accepts a video
  or up to eight identity-reference images; the separate FL2VA profile generates
  video and audio from a prompt alone. Converted graphs, tokenizer and
  manifests stay external and are selected from the **動画生成** tab.

Build the signed-local app and unsigned distribution image with:

```bash
packaging/macOS/upscaler/build_app.sh
```

Outputs:

- `build/mioh-upscaler/mioh upscaler.app`
- `build/mioh-upscaler/mioh-upscaler-0.14.3-unsigned.dmg`

FlashVSR, AdcSR, SwiftVR and H3 weights remain external. The small CC BY 4.0 PiperSR
models are bundled with attribution so the 2x option works without setup.
The default external locations are:

- `model_weights/FlashVSR-v1.1-coreai-grid16`
- `model_weights/adcsr_x4_float32.aimodel`
- `/Volumes/Project_HD/model_weights/minimax-h3-native/manifest.json`
- `/Volumes/Project_HD/model_weights/minimax-h3-native/manifest-fl2va.json`
- `/Volumes/Project_HD/swiftvr-eval` (or the folder produced by mioh
  Universal's `install-swiftvr-models.zsh`)

The first two and SwiftVR can also be selected from the **アップスケール** tab and the
MiniMax H3 manifest from the **動画生成** tab. The MiniMax H3 weights live on
the external `Project_HD` volume, so that volume must be mounted before 動画生成
can run.

## Long-form H3 continuation

The default **Hybrid AV** continuation mode combines two complementary,
model-free techniques:

- A Continuum-style masked target prefix protects the exact final 22 video
  frames and the matching 37 audio-latent ticks from the preceding Part.
- The two video tokens immediately before that prefix, plus their matching
  13 audio-latent ticks, are supplied as older H3-Extend attention context.

The protected prefix is not repeated in the attention context. This keeps the
Part boundary exact while retaining incoming motion history, and it uses the
same Ref2VA checkpoint as Part 1. The previous video-only `latent-prefix` mode
remains selectable for compatibility. Continuation state files use a versioned
property-list schema; existing version-1 video-only files still load.

The design is adapted from the MIT-licensed
[ComfyUI-MiniMax-H3-Extend](https://github.com/kat3ri/ComfyUI-MiniMax-H3-Extend)
and
[ComfyUI-H3-Continuum](https://github.com/ukr8b3g-cmyk/ComfyUI-H3-Continuum)
projects. No source checkpoint or additional model weight from either project
is bundled.

When the selected upscaler model is missing, version 0.14.3 opens the model
setup sheet on first launch. The user chooses an external destination;
FlashVSR official weights are downloaded and converted to Core AI on that Mac,
while the maintainer-provided AdcSR Core AI asset is downloaded and verified.

## Codex MCP

The app bundles a native Swift stdio MCP server at:

```text
mioh upscaler.app/Contents/Resources/bin/mioh-upscaler-mcp
```

Register an installed copy with Codex:

```bash
codex mcp add mioh-upscaler -- \
  "/Applications/mioh upscaler.app/Contents/Resources/bin/mioh-upscaler-mcp"
```

The server exposes capability inspection, MiniMax H3 video generation,
FlashVSR/AdcSR/PiperSR/SwiftVR upscaling, job status/list/stop, and app opening tools. The
`prompt` passed to `mioh_start_video_generation` is forwarded byte-for-byte to
the native H3 runner without summarizing or rewriting it. Model weights remain
external and are not exposed through MCP responses.
