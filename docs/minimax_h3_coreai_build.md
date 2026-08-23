# MiniMax H3 Core AI model preparation

`mioh upscaler` runs MiniMax H3 through native Swift and Core AI. Python is
used only while converting the original checkpoints. Model weights and
generated `.aimodel` / `.aimodelc` assets stay outside this repository and are
never embedded in the app or DMG.

## Requirements

- Apple Silicon Mac running macOS 27 or later
- Xcode 27 and its Command Line Tools
- Python 3.12
- a local ComfyUI checkout containing
  `comfy/ldm/minimax/audio_vae.py` and `comfy/ldm/minimax/vae.py`
- MiniMax H3 checkpoints obtained from the
  [official model repository](https://huggingface.co/MiniMaxAI/MiniMax-H3)
  under its upstream license

Confirm that Core AI is available:

```zsh
xcrun --find coreai-build
xcrun coreai-build --version
```

Create an isolated conversion environment from the repository root:

```zsh
python3.12 -m venv .venv-minimax-h3
.venv-minimax-h3/bin/python -m pip install --upgrade pip
.venv-minimax-h3/bin/pip install -r scripts/apple/requirements-minimax-h3-coreai.txt
```

## External model directory

Choose an external destination. It may be anywhere writable and must not be
inside the application bundle:

```zsh
export MINIMAX_H3_MODELS='/path/to/model_weights/minimax-h3-native'
mkdir -p "$MINIMAX_H3_MODELS/coreai-source" "$MINIMAX_H3_MODELS/coreai"
```

The final directory selected in the app must contain a top-level
`manifest.json`, a tokenizer directory, and all assets referenced by that
manifest. Paths inside the manifest are relative to the directory containing
the manifest, so the directory can be moved as one unit.

## Export the audio and video VAEs

Set the checkpoint and ComfyUI paths first:

```zsh
export COMFY_ROOT='/path/to/ComfyUI'
export MINIMAX_H3_AUDIO_VAE='/path/to/minimax_h3_audio_vae_fp32.safetensors'
export MINIMAX_H3_VIDEO_VAE='/path/to/minimax_h3_video_vae_fp16.safetensors'
```

Export the four portable Core AI programs:

```zsh
.venv-minimax-h3/bin/python scripts/apple/export_minimax_h3_native.py \
  --stage audio-encoder --backend coreai \
  --checkpoint "$MINIMAX_H3_AUDIO_VAE" \
  --comfy-root "$COMFY_ROOT" \
  --output "$MINIMAX_H3_MODELS/coreai-source/audio-encoder.aimodel" \
  --skip-reference

.venv-minimax-h3/bin/python scripts/apple/export_minimax_h3_native.py \
  --stage audio-decoder --backend coreai \
  --checkpoint "$MINIMAX_H3_AUDIO_VAE" \
  --comfy-root "$COMFY_ROOT" \
  --output "$MINIMAX_H3_MODELS/coreai-source/audio-decoder.aimodel" \
  --skip-reference

.venv-minimax-h3/bin/python scripts/apple/export_minimax_h3_native.py \
  --stage video-encoder-tile --backend coreai \
  --checkpoint "$MINIMAX_H3_VIDEO_VAE" \
  --comfy-root "$COMFY_ROOT" \
  --output "$MINIMAX_H3_MODELS/coreai-source/video-encoder-tile256.aimodel" \
  --skip-reference

.venv-minimax-h3/bin/python scripts/apple/export_minimax_h3_native.py \
  --stage video-decoder-tile --backend coreai \
  --checkpoint "$MINIMAX_H3_VIDEO_VAE" \
  --comfy-root "$COMFY_ROOT" \
  --output "$MINIMAX_H3_MODELS/coreai-source/video-decoder-raw-tile7x16.aimodel" \
  --skip-reference
```

Compile each portable program for the current Apple Silicon generation:

```zsh
for model in "$MINIMAX_H3_MODELS"/coreai-source/*.aimodel; do
  xcrun coreai-build compile "$model" \
    --output "$MINIMAX_H3_MODELS/coreai" \
    --platform macOS \
    --min-deployment-version 27.0 \
    --preferred-compute gpu \
    --architecture h17s
done
```

`coreai-build` may add the architecture to a compiled directory name. The
manifest must use the exact generated relative path.

## Export the Qwen condition encoder

The exporter creates the fixed MiniMax H3 Qwen profile and its component
manifest. The checkpoint argument is the single Qwen safetensors file used by
the model profile:

```zsh
export MINIMAX_H3_QWEN='/path/to/qwen3vl_32b_minimax_h3_checkpoint.safetensors'

.venv-minimax-h3/bin/python scripts/apple/export_minimax_h3_qwen_coreai.py \
  --checkpoint "$MINIMAX_H3_QWEN" \
  --source-directory "$MINIMAX_H3_MODELS/coreai-source/qwen" \
  --compiled-directory "$MINIMAX_H3_MODELS/coreai" \
  --architecture h17s \
  --preferred-compute gpu
```

Copy the matching upstream tokenizer files into an external tokenizer
directory, preserving `vocab.json`, `merges.txt`, and
`tokenizer_config.json`. Do not substitute a tokenizer from a different Qwen
release.

## Validate before using the app

The repository includes Swift probes under `tests/swift/` for the VAE, Qwen,
ER-SDE schedule, and movie writer. At minimum, verify that every asset listed
by `manifest.json` exists before selecting it in **動画生成**:

```zsh
.venv-minimax-h3/bin/python - "$MINIMAX_H3_MODELS/manifest.json" <<'PY'
import json, pathlib, sys
manifest = pathlib.Path(sys.argv[1]).resolve()
payload = json.loads(manifest.read_text())
missing = []
def walk(value):
    if isinstance(value, dict):
        asset = value.get("asset")
        if isinstance(asset, str) and not (manifest.parent / asset).exists():
            missing.append(asset)
        for child in value.values(): walk(child)
    elif isinstance(value, list):
        for child in value: walk(child)
walk(payload)
if missing:
    raise SystemExit("missing assets:\n" + "\n".join(sorted(set(missing))))
print("MiniMax H3 manifest assets: OK")
PY
```

Then open `mioh upscaler`, choose **動画生成**, and select the external
`manifest.json`. The app stores only that path; it does not copy the model.
