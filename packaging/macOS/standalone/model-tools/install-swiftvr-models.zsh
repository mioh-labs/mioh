#!/bin/zsh
set -euo pipefail

# Build the external SwiftVR Core ML pack used by mioh Universal's ROI enhancer.
# The application itself stays model-free; select the printed model directory
# in Restoration > ROI Enhancer > SwiftVR after this command finishes.

APP=/Applications/mioh-universal.app
WORKSPACE="$HOME/Library/Application Support/mioh/SwiftVR"
MODEL_ROOT=""
SOURCE=""
SOURCE_USER_SET=0
CHECKPOINTS=""
PY=""
SCALE=both
DOWNLOAD=1
VERIFY_ONLY=0
UPSTREAM_COMMIT=dbec2f993ab5b0abf9a2ae2eae67117d5cadb8a7
HF_REVISION=5e89ae51d17b564b40cb14029002b48a83c2ff50

usage() {
  cat <<'EOF'
usage: install-swiftvr-models.zsh [options]

  --app PATH           mioh-universal.app (default: /Applications/mioh-universal.app)
  --workspace PATH     source, checkpoint and Python workspace
  --model-root PATH    converted model directory; select this folder in mioh
  --source PATH        existing official SwiftVR source checkout
  --checkpoints PATH   existing official SwiftVR checkpoint directory
  --python PATH        existing Python with Torch, Core ML Tools and SwiftVR deps
  --scale 2|4|both     export scale (default: both; 4x is always required)
  --no-download        use only existing source and checkpoints
  --verify-only        validate an already converted pack without modifying it
  -h, --help           show this help

Requires Apple Silicon, macOS 27+, Xcode Command Line Tools and free disk space.
Without --python, uv creates a separate conversion environment in --workspace.
The app is not modified. Downloaded weights are ~19 GiB; converted 2x+4x
packs are ~19 GiB, plus temporary conversion space and Core ML runtime cache.
EOF
}

die() { print -u2 -- "SwiftVR setup: $*"; exit 1; }
log() { print -- "SwiftVR setup: $*"; }

while (( $# )); do
  case "$1" in
    --app|--workspace|--model-root|--source|--checkpoints|--python|--scale)
      (( $# >= 2 )) || die "missing value for $1"
      case "$1" in
        --app) APP="$2" ;;
        --workspace) WORKSPACE="$2" ;;
        --model-root) MODEL_ROOT="$2" ;;
        --source) SOURCE="$2"; SOURCE_USER_SET=1 ;;
        --checkpoints) CHECKPOINTS="$2" ;;
        --python) PY="$2" ;;
        --scale) SCALE="$2" ;;
      esac
      shift 2 ;;
    --no-download) DOWNLOAD=0; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

[[ "$SCALE" == 2 || "$SCALE" == 4 || "$SCALE" == both ]] || die "--scale must be 2, 4 or both"
[[ -d "$APP/Contents/Resources/model-tools" ]] || die "mioh Universal not found at $APP"
[[ -x "$APP/Contents/Resources/bin/mioh-native-coreai-preview" ]] || die "this app has no native export helper; install a SwiftVR-capable Universal build"
TOOLS="$APP/Contents/Resources/model-tools"

WORKSPACE="${WORKSPACE:A}"
MODEL_ROOT="${${MODEL_ROOT:-$WORKSPACE/models}:A}"
SOURCE="${${SOURCE:-$WORKSPACE/source}:A}"
CHECKPOINTS="${${CHECKPOINTS:-$WORKSPACE/checkpoints}:A}"

valid_package() {
  local package="$1"
  [[ -s "$package/Manifest.json" && -s "$package/Data/com.apple.CoreML/model.mlmodel" && -s "$package/Data/com.apple.CoreML/weights/weight.bin" ]]
}

valid_components() {
  local directory="$1" name
  for name in context.f32 modulation.f32 rope-cosine.f32 rope-sine.f32 components.json; do
    [[ -s "$directory/$name" ]] || return 1
  done
  valid_package "$directory/patch.mlpackage" && valid_package "$directory/head.mlpackage"
}

verify_scale() {
  local scale="$1" size=$(( 256 * scale )) variant part prefix name first last
  for variant in t6 t7; do
    prefix="$MODEL_ROOT/native-${scale}x-${variant}-fp16/components"
    valid_components "$prefix" || die "incomplete components: $prefix"
  done
  for part in 'encoder-4f' 'encoder-24f' 'encoder-28f' 'decoder-1latent' 'decoder-6latent' 'decoder-7latent'; do
    valid_package "$MODEL_ROOT/reae-stateful-${part}-${size}-fp32.mlpackage" || die "incomplete ReAE: $part ($size)"
  done
  for first in 0 6 12 18 24; do
    last=$(( first + 5 ))
    name="dit-group-$(printf '%02d' $first)-$(printf '%02d' $last)-${scale}x-float16.mlpackage"
    valid_package "$MODEL_ROOT/native-${scale}x-fp16-grouped/$name" || die "incomplete DiT group: $name"
  done
  log "${scale}x pack verified"
}

typeset -a scales
scales=(4)
[[ "$SCALE" == both || "$SCALE" == 2 ]] && scales+=(2)
if (( VERIFY_ONLY )); then
  for scale in "${scales[@]}"; do verify_scale "$scale"; done
  log "select this model folder in mioh Universal: $MODEL_ROOT"
  exit 0
fi

[[ "$(uname -m)" == arm64 ]] || die "mioh Universal SwiftVR needs Apple Silicon"
(( ${$(sw_vers -productVersion)%%.*} >= 27 )) || die "SwiftVR export needs macOS 27 or later"
[[ -f "$TOOLS/scripts/apple/export_swiftvr_dit_group_coreml.py" ]] || die "this Universal build lacks the SwiftVR converters; rebuild/update mioh Universal first"
command -v xcrun >/dev/null || die "Xcode Command Line Tools are required"
mkdir -p "$WORKSPACE" "$MODEL_ROOT"
available_kib="$(df -Pk "$WORKSPACE" | awk 'END {print $4}')"
if [[ ! -s "$CHECKPOINTS/transformer/diffusion_pytorch_model.safetensors" ]]; then
  (( available_kib >= 25 * 1024 * 1024 )) || die "at least 25 GiB free is needed for the checkpoint in $WORKSPACE"
fi
if [[ -z "$PY" ]]; then
  command -v uv >/dev/null || die "install uv, or pass --python with the required packages installed"
  PY="$WORKSPACE/.venv/bin/python"
  if [[ ! -x "$PY" ]]; then
    uv venv "$WORKSPACE/.venv" --python 3.12
  fi
  # Never alter the signed application's bundled Python runtime.
  if ! "$PY" -c 'import coremltools, torch, torchvision, safetensors, huggingface_hub, diffusers, einops' >/dev/null 2>&1; then
    uv pip install --python "$PY" \
      'torch==2.12.1' 'torchvision==0.27.1' 'coremltools==9.0' 'numpy<2' \
      'safetensors==0.7.0' 'huggingface_hub>=1.0,<2' \
      'diffusers==0.36.0' 'transformers==5.2.0' \
      'accelerate==1.12.0' 'einops==0.8.2' 'pillow==11.3.0'
  fi
fi
[[ -x "$PY" ]] || die "Python is not executable: $PY"
"$PY" -c 'import coremltools, torch, torchvision, safetensors, huggingface_hub, diffusers, einops' >/dev/null || die "Python lacks SwiftVR conversion dependencies"

if [[ ! -f "$SOURCE/swiftvr/models/transformer.py" ]]; then
  (( DOWNLOAD )) || die "SwiftVR source missing: $SOURCE"
  [[ ! -e "$SOURCE" ]] || die "source path exists but is not a SwiftVR checkout: $SOURCE"
  git clone https://github.com/H-oliday/SwiftVR.git "$SOURCE"
  git -C "$SOURCE" checkout --detach "$UPSTREAM_COMMIT"
fi
if [[ "$(git -C "$SOURCE" rev-parse HEAD 2>/dev/null)" != "$UPSTREAM_COMMIT" && "$SOURCE_USER_SET" == 0 ]]; then
  [[ -z "$(git -C "$SOURCE" status --porcelain)" ]] || die "source checkout has changes and is not at the pinned revision: $SOURCE"
  git -C "$SOURCE" checkout --detach "$UPSTREAM_COMMIT"
fi
[[ "$(git -C "$SOURCE" rev-parse HEAD 2>/dev/null)" == "$UPSTREAM_COMMIT" ]] || die "SwiftVR source must be at $UPSTREAM_COMMIT: $SOURCE"
"$PY" - "$SOURCE" "$TOOLS/scripts/apple" <<'PY'
import sys
sys.path.insert(0, sys.argv[2])
from swiftvr_imports import load_transformer
load_transformer(sys.argv[1])
PY

TRANSFORMER="$CHECKPOINTS/transformer/diffusion_pytorch_model.safetensors"
REAE="$CHECKPOINTS/reae.safetensors"
PROMPT="$CHECKPOINTS/prompt_embedding.safetensors"
if [[ ! -s "$TRANSFORMER" || ! -s "$REAE" || ! -s "$PROMPT" ]]; then
  (( DOWNLOAD )) || die "official SwiftVR checkpoints missing in $CHECKPOINTS"
  log "downloading the pinned official checkpoint (~19 GiB)"
  "$PY" - "$CHECKPOINTS" "$HF_REVISION" <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import sys
snapshot_download(
    repo_id="H-oliday/SwiftVR", revision=sys.argv[2], local_dir=Path(sys.argv[1]),
    allow_patterns=["transformer/diffusion_pytorch_model.safetensors", "transformer/config.json", "reae.safetensors", "prompt_embedding.safetensors"],
)
PY
fi
[[ -s "$TRANSFORMER" && -s "$REAE" && -s "$PROMPT" ]] || die "checkpoint download is incomplete"

convert_package() {
  local output="$1" script="$2"
  shift 2
  if valid_package "$output"; then log "reuse: $output"; return; fi
  [[ ! -e "$output" ]] || die "incomplete existing package; inspect before retry: $output"
  mkdir -p "${output:h}"
  local staging
  staging="$(mktemp -d "${output:h}/.swiftvr-export.XXXXXX")"
  log "convert: $output"
  "$PY" "$TOOLS/scripts/apple/$script" "$@" --output "$staging/model.mlpackage"
  valid_package "$staging/model.mlpackage" || die "conversion produced an incomplete package: $staging"
  mv "$staging/model.mlpackage" "$output"
  rmdir "$staging"
}

for scale in "${scales[@]}"; do
  size=$(( 256 * scale ))
  for frames in 4 24 28; do
    convert_package "$MODEL_ROOT/reae-stateful-encoder-${frames}f-${size}-fp32.mlpackage" \
      probe_swiftvr_reae_state_coreml.py \
      --source "$SOURCE" --checkpoint "$REAE" --size "$size" --frames "$frames"
  done
  for latent in 1 6 7; do
    convert_package "$MODEL_ROOT/reae-stateful-decoder-${latent}latent-${size}-fp32.mlpackage" \
      probe_swiftvr_reae_decoder_state_coreml.py \
      --source "$SOURCE" --checkpoint "$REAE" --size "$size" --latent-frames "$latent"
  done
  for latent in 6 7; do
    components="$MODEL_ROOT/native-${scale}x-t${latent}-fp16/components"
    if valid_components "$components"; then
      log "reuse: $components"
    else
      [[ ! -e "$components" ]] || die "incomplete existing components; inspect before retry: $components"
      mkdir -p "${components:h}"
      staging="$(mktemp -d "${components:h}/.swiftvr-components.XXXXXX")"
      "$PY" "$TOOLS/scripts/apple/export_swiftvr_components_coreml.py" \
        --source "$SOURCE" --checkpoint "$TRANSFORMER" --prompt-embedding "$PROMPT" \
        --output-directory "$staging/components" --scale "$scale" \
        --latent-frames "$latent" --precision float16
      valid_components "$staging/components" || die "conversion produced incomplete components: $staging"
      mv "$staging/components" "$components"
      rmdir "$staging"
    fi
  done
  for first in 0 6 12 18 24; do
    last=$(( first + 5 ))
    name="dit-group-$(printf '%02d' $first)-$(printf '%02d' $last)-${scale}x-float16.mlpackage"
    convert_package "$MODEL_ROOT/native-${scale}x-fp16-grouped/$name" \
      export_swiftvr_dit_group_coreml.py \
      --source "$SOURCE" --checkpoint "$TRANSFORMER" \
      --first-layer "$first" --end-layer "$((last + 1))" \
      --latent-frames 7 6 --scale "$scale" --precision float16
  done
  verify_scale "$scale"
done

log "ready. In mioh Universal choose ROI enhancer SwiftVR, then select: $MODEL_ROOT"
log "The model is external to the app and survives application updates."
