#!/bin/zsh
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

set -euo pipefail

DESTINATION=""
INSTALL_FLASHVSR=0
INSTALL_ADCSR=0
DRY_RUN=0
PROGRESS_BASE=0
PROGRESS_SPAN=1

trap 'trap - INT TERM; pkill -TERM -P $$ 2>/dev/null || true; exit 130' INT TERM

usage() {
  cat <<'EOF'
usage: setup-upscaler-models.zsh --destination DIR [--flashvsr] [--adcsr] [--dry-run]

Downloads verified upstream weights, converts FlashVSR to Core AI on this Mac,
and installs the resulting models below DIR. AdcSR is distributed by its
maintainer as an already-converted Core AI asset and is verified in place.
EOF
}

while (( $# > 0 )); do
  case "$1" in
    --destination)
      DESTINATION="$2"
      shift 2
      ;;
    --flashvsr)
      INSTALL_FLASHVSR=1
      shift
      ;;
    --adcsr)
      INSTALL_ADCSR=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      print -u2 "unknown option: $1"
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$DESTINATION" || "$DESTINATION" == "/" || "$DESTINATION" == "$HOME" ]]; then
  print -u2 "unsafe or missing model destination: $DESTINATION"
  exit 2
fi
if (( ! INSTALL_FLASHVSR && ! INSTALL_ADCSR )); then
  print -u2 "select at least one model"
  exit 2
fi

DESTINATION="${DESTINATION:A}"
WORK="$DESTINATION/.mioh-upscaler-setup"
if [[ "$WORK" != "$DESTINATION/"* ]]; then
  print -u2 "unsafe setup workspace: $WORK"
  exit 2
fi

progress() {
  local fraction="$1"
  shift
  local overall
  overall=$(awk -v base="$PROGRESS_BASE" -v span="$PROGRESS_SPAN" \
    -v value="$fraction" 'BEGIN { printf "%.4f", base + span * value }')
  print "MIOH_SETUP|$overall|$*"
}

sha256() {
  shasum -a 256 "$1" | awk '{print $1}'
}

download() {
  local url="$1"
  local output="$2"
  local expected="$3"
  local actual=""
  mkdir -p "${output:h}"
  if [[ -f "$output" ]]; then
    actual="$(sha256 "$output")"
    if [[ "$actual" == "$expected" ]]; then
      print "verified: $output"
      return 0
    fi
    mv "$output" "$output.invalid-$(date +%Y%m%d-%H%M%S)"
  fi
  if [[ -f "$output.part" ]]; then
    print "resuming: $output.part"
  else
    print "downloading: $output"
  fi
  curl -fsSL --retry 5 --retry-all-errors --retry-delay 2 \
    --connect-timeout 30 --continue-at - -o "$output.part" "$url"
  actual="$(sha256 "$output.part")"
  if [[ "$actual" != "$expected" ]]; then
    print -u2 "checksum mismatch for ${output:t}: expected $expected, got $actual"
    exit 1
  fi
  mv "$output.part" "$output"
}

install_adcsr() {
  local final="$DESTINATION/adcsr_x4_float32.aimodel"
  if [[ -f "$final/main.mlirb" && -f "$final/metadata.json" ]] \
      && [[ "$(sha256 "$final/main.mlirb")" == \
        "33d2a727e24044912ca1f352ed3b946863f6770990f6fc553c1c134ac3d5423c" ]]; then
    progress 1.0 "AdcSRは配置済みです"
    return
  fi
  local staging="$WORK/adcsr_x4_float32.aimodel.installing"
  mkdir -p "$staging"
  local base="https://huggingface.co/mlboydaisuke/AdcSR-CoreAI/resolve/main/adcsr_x4_float32.aimodel"
  progress 0.05 "AdcSR Core AIモデルをダウンロード中（約1.7 GB）"
  download "$base/main.mlirb" "$staging/main.mlirb" \
    "33d2a727e24044912ca1f352ed3b946863f6770990f6fc553c1c134ac3d5423c"
  curl -fsSL --retry 5 -o "$staging/main.hash" "$base/main.hash"
  curl -fsSL --retry 5 -o "$staging/metadata.json" "$base/metadata.json"
  curl -fsSL --retry 5 -o "$DESTINATION/AdcSR-CoreAI-LICENSE.txt" \
    "https://huggingface.co/mlboydaisuke/AdcSR-CoreAI/resolve/main/LICENSE"
  if [[ -e "$final" ]]; then
    mv "$final" "$final.invalid-$(date +%Y%m%d-%H%M%S)"
  fi
  mv "$staging" "$final"
  progress 1.0 "AdcSRの検証と設定が完了しました"
}

install_flashvsr() {
  local final="$DESTINATION/FlashVSR-v1.1-coreai-grid16"
  local block_count=0
  if [[ -d "$final" ]]; then
    block_count=$(find "$final" -maxdepth 1 -type d -name 'dit_block_*.aimodel' | wc -l | tr -d ' ')
  fi
  if [[ "$block_count" == "30" \
        && -d "$final/patch_head.aimodel" \
        && -d "$final/lq_projection.aimodel" \
        && -d "$final/tcdecoder.aimodel" ]]; then
    progress 1.0 "FlashVSRは配置済みです"
    return
  fi

  local tool_source="${0:A:h}/flashvsr"
  if [[ ! -f "$tool_source/deployment/coreai/export_native.py" ]]; then
    print -u2 "bundled FlashVSR converter is missing: $tool_source"
    exit 1
  fi
  local raw="$WORK/FlashVSR-v1.1"
  local compact="$WORK/FlashVSR-v1.1-upscale"
  local converted="$WORK/coreai-native/grid16"
  mkdir -p "$raw" "$WORK/coreai-native"

  local available_kib
  available_kib=$(df -Pk "$DESTINATION" | awk 'END { print $4 }')
  local required_kib=$((18 * 1024 * 1024))
  if (( available_kib < required_kib )); then
    print -u2 "FlashVSR setup needs at least 18 GiB free in $DESTINATION"
    exit 1
  fi

  progress 0.03 "FlashVSR公式重みをダウンロード中（約6.5 GB）"
  local hf="https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1/resolve/main"
  download "$hf/diffusion_pytorch_model_streaming_dmd.safetensors" \
    "$raw/diffusion_pytorch_model_streaming_dmd.safetensors" \
    "bd28180edcf3446c028e32fc6b731a80bf7e4da2ab4caac3186b9499964d37be"
  progress 0.25 "FlashVSR LQ投影をダウンロード中"
  download "$hf/LQ_proj_in.ckpt" "$raw/LQ_proj_in.ckpt" \
    "d6d011cdaaba6a52645086caa08fa04124e746f6ca568140a24007591142bfd2"
  progress 0.28 "FlashVSR時間デコーダーをダウンロード中"
  download "$hf/TCDecoder.ckpt" "$raw/TCDecoder.ckpt" \
    "e224bdcf2f52745cbf4d393ff5374c2ba09e90285d5d19062d2bf63b915b6161"
  download \
    "https://raw.githubusercontent.com/sh202603/FlashVSR_plus/f489dd4eb8e5da6687351f8332e6a7cd88c01f63/models/posi_prompt.pth" \
    "$raw/posi_prompt.pth" \
    "4601107a11e4e11a936a6b79df579e54dbc99872132bf542151f0ffd65b4b1ef"

  progress 0.31 "Mac変換環境を準備中（初回のみ）"
  local uv_archive="$WORK/uv-aarch64-apple-darwin.tar.gz"
  download \
    "https://github.com/astral-sh/uv/releases/download/0.12.5/uv-aarch64-apple-darwin.tar.gz" \
    "$uv_archive" \
    "5bb0e5fe008a773c3dbcb97ff79cd89e1241464fe9d2f986d52ad8f1b037bd62"
  local uv_root="$WORK/uv"
  if [[ ! -x "$uv_root/uv" ]]; then
    mkdir -p "$uv_root"
    tar -xzf "$uv_archive" -C "$uv_root" --strip-components=1
  fi
  local venv="$WORK/coreai-venv"
  "$uv_root/uv" venv --clear --python 3.12 "$venv"
  "$uv_root/uv" pip install --python "$venv/bin/python" \
    'torch==2.11.0' 'coreai-torch==0.4.2' 'coreai-opt==0.2.1' \
    'coremltools==9.0' 'safetensors==0.8.0' 'einops==0.8.2' \
    'tqdm==4.68.4' 'numpy==2.2.6'

  progress 0.42 "FlashVSRをアップスケール専用重みに整理中"
  "$venv/bin/python" "$tool_source/deployment/build_upscale_bundle.py" \
    --checkpoint "$raw/diffusion_pytorch_model_streaming_dmd.safetensors" \
    --prompt "$raw/posi_prompt.pth" \
    --lq-projection "$raw/LQ_proj_in.ckpt" \
    --tcdecoder "$raw/TCDecoder.ckpt" \
    --output-dir "$compact" --device auto --force

  progress 0.52 "FlashVSRをMac Core AI形式へ変換中（30ブロック）"
  (
    cd "$tool_source"
    PYTHONPATH="$tool_source" "$venv/bin/python" \
      -m deployment.coreai.export_native \
      --component all --output-dir "$converted" \
      --checkpoint "$compact/diffusion_pytorch_model_streaming_dmd.compact-bf16.safetensors" \
      --dtype float16 --force --validate
  )

  block_count=$(find "$converted" -maxdepth 1 -type d -name 'dit_block_*.aimodel' | wc -l | tr -d ' ')
  if [[ "$block_count" != "30" \
        || ! -d "$converted/patch_head.aimodel" \
        || ! -d "$converted/lq_projection.aimodel" \
        || ! -d "$converted/tcdecoder.aimodel" ]]; then
    print -u2 "FlashVSR Core AI conversion is incomplete"
    exit 1
  fi
  if [[ -e "$final" ]]; then
    mv "$final" "$final.invalid-$(date +%Y%m%d-%H%M%S)"
  fi
  mv "$converted" "$final"
  progress 1.0 "FlashVSRのMac変換と設定が完了しました"
}

if (( DRY_RUN )); then
  print "destination=$DESTINATION"
  print "flashvsr=$INSTALL_FLASHVSR"
  print "adcsr=$INSTALL_ADCSR"
  print "converter=${0:A:h}/flashvsr"
  exit 0
fi

mkdir -p "$DESTINATION" "$WORK"
if (( INSTALL_ADCSR )); then
  PROGRESS_BASE=0
  if (( INSTALL_FLASHVSR )); then
    PROGRESS_SPAN=0.20
  else
    PROGRESS_SPAN=1
  fi
  install_adcsr
fi
if (( INSTALL_FLASHVSR )); then
  if (( INSTALL_ADCSR )); then
    PROGRESS_BASE=0.20
    PROGRESS_SPAN=0.80
  else
    PROGRESS_BASE=0
    PROGRESS_SPAN=1
  fi
  install_flashvsr
fi

# All installed models have already been moved out of this private workspace.
# Keep cleanup strictly confined to the selected destination.
if [[ "$WORK" == "$DESTINATION/.mioh-upscaler-setup" ]]; then
  rm -rf "$WORK"
fi
PROGRESS_BASE=0
PROGRESS_SPAN=1
progress 1.0 "モデル自動設定が完了しました"
