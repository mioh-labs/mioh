#!/bin/zsh
set -euo pipefail

# Install the external ROI Refine pack (Sulphur 2 / LTX-2.3 with the mioh ROI IC-LoRA,
# Core AI DiT + MLX VAE) used by mioh Universal's ROI enhancer. The application itself
# stays model-free; select the printed pack folder in Restoration > ROI Enhancer > ROI Refine.
#
#   --mode download  fetch the converted pack from Hugging Face (default, ~18 GiB)
#   --mode build     convert it locally from the official Sulphur 2 checkpoint (~43 GiB
#                    download) and the published LoRA; needs the app's Core AI converter

APP=/Applications/mioh-universal.app
WORKSPACE="$HOME/Library/Application Support/mioh/ROIRefine"
PACK=""
MODE=download
CHECKPOINT=""
LTX_SOURCE=""
DOWNLOAD=1
VERIFY_ONLY=0
PACK_REPO=okatti/mioh-roi-refine
PACK_REVISION=ef593b113ce47cb82b115f1999fc828e9c42f265
CHECKPOINT_REPO=SulphurAI/Sulphur-2-base
CHECKPOINT_REVISION=65587bbb38c700622f9434a16e732c8b280c115e
CHECKPOINT_FILE=sulphur_distil_bf16.safetensors
LTX_COMMIT=e98f6d927d83c92ef2be6aec76c8440b19306e39
PACK_NAME=pack-sulphur-iclora1750-v5
FPS_RATES=(23.976:23.976023976023978 24.000:24.0 25.000:25.0 29.970:29.97002997002997 30.000:30.0
  50.000:50.0 59.940:59.94005994005994 60.000:60.0)
SQUARE_SIDES=(384 768 1024)   # 512 lives in constants/ itself; no 256 window
RECTANGLES=(256x512 512x256)

usage() {
  cat <<'EOF'
usage: install-roi-refine.zsh [options]

  --app PATH           mioh-universal.app (default: /Applications/mioh-universal.app)
  --mode download|build
                       download the converted pack (default) or convert it locally
  --workspace PATH     download and conversion workspace
  --pack PATH          pack folder to create; select this folder in mioh
  --checkpoint PATH    existing sulphur_distil_bf16.safetensors (build mode)
  --ltx-source PATH    existing Lightricks/LTX-2 checkout (build mode)
  --no-download        use only files that already exist
  --verify-only        check an existing pack without modifying it
  -h, --help           show this help

Requires Apple Silicon and macOS 27+. The Hugging Face repository is gated: accept the
LTX-2 Community License on https://huggingface.co/okatti/mioh-roi-refine first, then log
in with `hf auth login` (or set HF_TOKEN). The pack is ~18 GiB; build mode also downloads
the ~43 GiB checkpoint and needs ~70 GiB free. The pack is outside the app and survives
application updates. Outputs are machine generated: the LTX-2 license requires you to
say so wherever you share them.
EOF
}

die() { print -u2 -- "ROI Refine setup: $*"; exit 1; }
log() { print -- "ROI Refine setup: $*"; }

while (( $# )); do
  case "$1" in
    --app|--mode|--workspace|--pack|--checkpoint|--ltx-source)
      (( $# >= 2 )) || die "missing value for $1"
      case "$1" in
        --app) APP="$2" ;;
        --mode) MODE="$2" ;;
        --workspace) WORKSPACE="$2" ;;
        --pack) PACK="$2" ;;
        --checkpoint) CHECKPOINT="$2" ;;
        --ltx-source) LTX_SOURCE="$2" ;;
      esac
      shift 2 ;;
    --no-download) DOWNLOAD=0; shift ;;
    --verify-only) VERIFY_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown option: $1 (see --help)" ;;
  esac
done

[[ "$MODE" == download || "$MODE" == build ]] || die "--mode must be download or build"
[[ -d "$APP/Contents/Resources/model-tools" ]] || die "mioh Universal not found at $APP"
TOOLS="$APP/Contents/Resources/model-tools"
PY="$APP/Contents/Resources/runtime/bin/python3.12"
[[ -x "$PY" ]] || die "missing packaged Python: $PY"
# The packaged runtime is relocatable: point it at its own home and site-packages.
export PYTHONHOME="$APP/Contents/Resources/runtime" PYTHONUNBUFFERED=1
export PYTHONPATH="$PYTHONHOME/lib/python3.12/site-packages:$TOOLS/scripts/apple:$TOOLS/scripts"

WORKSPACE="${WORKSPACE:A}"
PACK="${${PACK:-$WORKSPACE/$PACK_NAME}:A}"
CHECKPOINT="${${CHECKPOINT:-$WORKSPACE/checkpoint/$CHECKPOINT_FILE}:A}"
LTX_SOURCE="${${LTX_SOURCE:-$WORKSPACE/LTX-2}:A}"

verify_pack() {
  local directory="$1" group rate side
  for name in bin/ltx-video bin/mlx-swift_Cmlx.bundle vae.safetensors roi-refine-pack.json; do
    [[ -e "$directory/$name" ]] || die "incomplete pack, missing $name: $directory"
  done
  [[ -x "$directory/bin/ltx-video" ]] || die "server is not executable: $directory/bin/ltx-video"
  for group in 00 04 08 12 16 20 24 28 32 36 40 44; do
    [[ -d "$directory/blocks/group$group.aimodel" && -s "$directory/blocks/group$group.aimodel.json" ]] \
      || die "incomplete DiT group $group: $directory/blocks"
  done
  for rate in "${FPS_RATES[@]}"; do
    rate="${rate%%:*}"
    [[ -s "$directory/constants/fps-$rate.safetensors" ]] || die "missing constants/fps-$rate.safetensors"
    for side in "${SQUARE_SIDES[@]}" "${RECTANGLES[@]}"; do
      [[ -s "$directory/constants/size-$side/fps-$rate.safetensors" ]] \
        || die "missing constants/size-$side/fps-$rate.safetensors"
    done
  done
  log "pack verified: $directory"
}

if (( VERIFY_ONLY )); then
  verify_pack "$PACK"
  log "select this folder in mioh Universal (ROI Refine): $PACK"
  exit 0
fi

[[ "$(uname -m)" == arm64 ]] || die "ROI Refine needs Apple Silicon"
(( ${$(sw_vers -productVersion)%%.*} >= 27 )) || die "ROI Refine needs macOS 27 or later (Core AI)"
[[ -n "$PACK_REVISION" ]] || die "this script has no pinned pack revision; update mioh Universal"
mkdir -p "$WORKSPACE"

# snapshot REPO REVISION DESTINATION PATTERN... : gated downloads need a logged-in token.
snapshot() {
  (( DOWNLOAD )) || die "missing files in $3 and --no-download was given"
  "$PY" - "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, revision, destination, *patterns = sys.argv[1:]
snapshot_download(repo_id=repo, revision=revision, local_dir=destination, allow_patterns=patterns)
PY
}

if [[ "$MODE" == download ]]; then
  available_kib="$(df -Pk "$WORKSPACE" | awk 'END {print $4}')"
  (( available_kib >= 20 * 1024 * 1024 )) || die "at least 20 GiB free is needed in $WORKSPACE"
  staging="$WORKSPACE/.download"
  log "downloading the converted pack (~18 GiB) from $PACK_REPO"
  snapshot "$PACK_REPO" "$PACK_REVISION" "$staging" "pack/*" "LICENSE" "NOTICE" "README.md"
  [[ ! -e "$PACK" ]] || die "pack folder already exists; inspect or remove it first: $PACK"
  mkdir -p "${PACK:h}"
  mv "$staging/pack" "$PACK"
  cp "$staging/LICENSE" "$staging/NOTICE" "$PACK/"
  chmod +x "$PACK/bin/ltx-video"
  xattr -dr com.apple.quarantine "$PACK/bin" 2>/dev/null || true
  verify_pack "$PACK"
  log "ready. In mioh Universal choose ROI enhancer ROI Refine, then select: $PACK"
  exit 0
fi

# --- build mode -------------------------------------------------------------------------
for script in export_ltx23_dit_block_coreai.py export_ltx23_roi_refine_constants.py run_ltx23_roi_refine_coreai.py; do
  [[ -f "$TOOLS/scripts/apple/$script" ]] || die "this Universal build lacks the ROI Refine converters; update mioh Universal"
done
"$PY" -c 'import coreai_torch, torch, safetensors' >/dev/null 2>&1 || die "the packaged Python lacks the Core AI converter"
command -v git >/dev/null || die "git (Xcode Command Line Tools) is required"
available_kib="$(df -Pk "$WORKSPACE" | awk 'END {print $4}')"
if [[ ! -s "$CHECKPOINT" ]]; then
  (( available_kib >= 70 * 1024 * 1024 )) || die "at least 70 GiB free is needed in $WORKSPACE"
  log "downloading the official Sulphur 2 distilled checkpoint (~43 GiB)"
  snapshot "$CHECKPOINT_REPO" "$CHECKPOINT_REVISION" "$WORKSPACE/checkpoint" "$CHECKPOINT_FILE" "LICENSE.txt"
fi
[[ -s "$CHECKPOINT" ]] || die "checkpoint missing: $CHECKPOINT"

SUPPORT="$WORKSPACE/support"
if [[ ! -s "$SUPPORT/lora/lora_weights_step_01750.safetensors" || ! -s "$SUPPORT/prompt-embedding.safetensors" || ! -x "$SUPPORT/pack/bin/ltx-video" ]]; then
  log "downloading the ROI IC-LoRA, prompt embedding and server from $PACK_REPO"
  snapshot "$PACK_REPO" "$PACK_REVISION" "$SUPPORT" "lora/*" "prompt-embedding.safetensors" \
    "pack/bin/*" "pack/roi-refine-pack.json" "LICENSE" "NOTICE"
fi
LORA="$SUPPORT/lora/lora_weights_step_01750.safetensors"
PROMPT="$SUPPORT/prompt-embedding.safetensors"

if [[ ! -d "$LTX_SOURCE/packages/ltx-core/src/ltx_core" ]]; then
  (( DOWNLOAD )) || die "LTX-2 source missing: $LTX_SOURCE"
  [[ ! -e "$LTX_SOURCE" ]] || die "path exists but is not an LTX-2 checkout: $LTX_SOURCE"
  git clone https://github.com/Lightricks/LTX-2.git "$LTX_SOURCE"
fi
[[ "$(git -C "$LTX_SOURCE" rev-parse HEAD)" == "$LTX_COMMIT" ]] || git -C "$LTX_SOURCE" checkout --detach "$LTX_COMMIT"
export PYTHONPATH="$LTX_SOURCE/packages/ltx-core/src:$PYTHONPATH"

mkdir -p "$PACK/blocks" "$PACK/constants"
if [[ ! -x "$PACK/bin/ltx-video" ]]; then
  ditto "$SUPPORT/pack/bin" "$PACK/bin"
  chmod +x "$PACK/bin/ltx-video"
  xattr -dr com.apple.quarantine "$PACK/bin" 2>/dev/null || true
fi
cp "$SUPPORT/pack/roi-refine-pack.json" "$SUPPORT/LICENSE" "$SUPPORT/NOTICE" "$PACK/"

for layer in 0 4 8 12 16 20 24 28 32 36 40 44; do
  group="$PACK/blocks/group$(printf '%02d' $layer).aimodel"
  if [[ -d "$group" && -s "$group.json" ]]; then log "reuse: ${group:t}"; continue; fi
  log "convert DiT layers $layer-$((layer + 3)) (int8, up to 10240 tokens)"
  "$PY" "$TOOLS/scripts/apple/export_ltx23_dit_block_coreai.py" \
    --checkpoint "$CHECKPOINT" --lora "$LORA" --layer "$layer" --layer-count 4 \
    --int8-group 64 --max-tokens 10240 --benchmark 0 --skip-reference-check --overwrite \
    --output "$group"
done

if [[ ! -s "$PACK/vae.safetensors" ]]; then
  log "extract the video VAE"
  "$PY" - "$CHECKPOINT" "$PACK/vae.safetensors.partial" <<'PY'
import sys
from safetensors import safe_open
from safetensors.torch import save_file
with safe_open(sys.argv[1], framework="pt") as handle:
    tensors = {key: handle.get_tensor(key) for key in handle.keys() if key.startswith("vae.")}
    metadata = handle.metadata()
save_file(tensors, sys.argv[2], metadata=metadata)
PY
  mv "$PACK/vae.safetensors.partial" "$PACK/vae.safetensors"
fi

constants() {   # constants OUTPUT FPS [exporter options...]
  local output="$1" fps="$2"
  shift 2
  [[ -s "$output" ]] && return
  mkdir -p "${output:h}"
  "$PY" "$TOOLS/scripts/apple/export_ltx23_roi_refine_constants.py" \
    --checkpoint "$CHECKPOINT" --prompt-embedding "$PROMPT" --output "$output.partial" \
    --fps "$fps" "$@" >/dev/null
  mv "$output.partial" "$output"
}
log "export window constants (8 frame rates x 7 window shapes)"
for rate in "${FPS_RATES[@]}"; do
  name="${rate%%:*}" fps="${rate#*:}"
  constants "$PACK/constants/fps-$name.safetensors" "$fps" --size 512
  for side in "${SQUARE_SIDES[@]}"; do
    constants "$PACK/constants/size-$side/fps-$name.safetensors" "$fps" --size "$side"
  done
  for rectangle in "${RECTANGLES[@]}"; do
    constants "$PACK/constants/size-$rectangle/fps-$name.safetensors" "$fps" \
      --base-constants "$PACK/constants/fps-$name.safetensors" \
      --width "${rectangle%x*}" --height "${rectangle#*x}"
  done
done

verify_pack "$PACK"
log "ready. In mioh Universal choose ROI enhancer ROI Refine, then select: $PACK"
log "The pack is external to the app and survives application updates."
