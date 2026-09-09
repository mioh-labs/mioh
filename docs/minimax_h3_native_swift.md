# 10Eros-Max H3: Swift / Core AI runtime

## Status

The runtime and mioh upscaler integration are native Swift. Python and ComfyUI
are not launched by this path. The active reference model is 10Eros-Max H3
TURBO Hybrid Beta3, INT8 ConvRot with BF16 skip-edge blocks.

The original `.safetensors` files cannot be loaded by Core AI or Core ML. A
one-time model conversion is still required. That conversion is a build-time
operation; it is not part of the shipped runtime.

mioh upscaler ships the native Swift runner and UI only. The converted model
graphs, tokenizer and manifest are intentionally not embedded in the
application or DMG. Select an external `manifest.json` in the Video Generation
tab; mioh upscaler remembers that path for the next launch. mioh itself does
not contain this UI or runner.

## Reproduced H3 path

1. AVFoundation decodes the reference at 24fps and resizes it to the selected
   canvas.
2. The reference frame count is trimmed down to `17n + 5`.
3. The video VAE and 32kHz stereo audio VAE produce reference latents.
4. Swift performs Qwen byte-level BPE. The presentation is raw MiniMax H3
   text, not a chat template: `<Video 1>:`, timestamps, vision blocks, prompt.
5. Qwen sees frames at 2fps. Swift packs two-frame 16x16 visual patches,
   `image_grid_thw`, MRoPE position IDs and MiniMax modality tags.
6. Swift creates the video `[1,24,T,H/16,W/16]` and audio
   `[1,32,2,T40]` noise streams.
7. Swift packs `[text | reference audio | reference video | target audio |
   target video]`, constructs H3 RoPE and runs 50 exact-shape Core AI DiT
   blocks in fused groups of up to four layers.
8. Swift implements the model author's current TURBO recipe: ER-SDE with the
   ComfyUI Simple scheduler at six steps. For video shift 12 this is
   `[1.0, 0.9836839, 0.9600576, 0.9230769, 0.8575097, 0.7063800, 0.0]`.
9. The video/audio VAEs decode the final latents and AVFoundation writes
   HEVC/AAC MP4.

Every expensive boundary is cached using the input video digest, prompt,
geometry and model asset fingerprint. Changing only the prompt reuses the VAE
latents. A failed denoiser does not require rerunning video VAE or Qwen.

## Model graph contract

The manifest binds stable semantic names to backend-specific feature names.
Both `.aimodelc` and `.mlmodelc` profiles may live in one manifest.

| Stage | Required semantic output |
|---|---|
| `videoEncoder` | `referenceVideoLatent` |
| `audioEncoder` | `referenceAudioLatent` |
| `qwenComposite` | `context`, `tokenTags` |
| `denoiserComposite` | refined text, projected AV rows, 50 DiT blocks and AV heads |
| `videoDecoder` | `video` in NCTHW RGB `[0,1]` |
| `audioDecoder` | `audio` as stereo float waveform |

Swift owns MiniMax H3's packed layout, reference re-injection, visual/audio
condition augmentation and audio shift conversion. Core AI owns all learned
10Eros-Max projections, transformer blocks and final heads.

The DiT uses BF16 boundaries to avoid FP16 overflow and therefore runs through
Core AI on the GPU. macOS 27 beta 6 exposes only a GPU preference for Core AI,
not a GPU-only/ANE-exclusion policy. Exact-shape source assets avoid the
dynamic specialization path that repeatedly attempted unsupported ANE
compilation. The 13 fused assets are loaded sequentially in one Swift process
so their roughly 19.5 GiB of weights are not all retained in unified memory.
Every weight-distinct repeated layer also has a structurally unique salt output
to prevent the beta runtime from aliasing programs with different weights. The
Swift runtime never calls PyTorch/MPS `aten::_int_mm`.

## mioh upscaler

The **動画生成** tab provides:

- editable prompt, initially `モザイクを除去して最高品質の動画を生成する。`;
- the current fixed-shape Qwen Core AI profile accepts up to 16 prompt tokens;
  longer input is truncated after tokenization and reported in the progress log;
- Core AI execution;
- the checked 864x480 / 10-second profile;
- seed, manifest and output selection;
- persistent selection of an external model manifest (model weights are not
  bundled with mioh upscaler);
- native progress, stop and log handling.

The bundled runner is `mioh-minimax-h3-native`. It also supports standalone
validation/planning/running, for example:

```sh
'/path/to/mioh upscaler.app/Contents/Resources/bin/mioh-minimax-h3-native' plan \
  --manifest /path/to/manifest.json \
  --input /path/to/input.mp4 \
  --output /path/to/output.mp4 \
  --prompt 'モザイクを除去して最高品質の動画を生成する。' \
  --cache /path/to/cache \
  --backend coreai \
  --width 864 --height 480 --duration 10 --seed 261662374822964
```
