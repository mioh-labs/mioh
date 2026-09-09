#!/bin/zsh
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

set -euo pipefail

DESTINATION=""
INSTALL_FLASHVSR=0
INSTALL_ADCSR=0
INSTALL_MINIMAX_H3=0
DRY_RUN=0
PROGRESS_BASE=0
PROGRESS_SPAN=1

trap 'trap - INT TERM; pkill -TERM -P $$ 2>/dev/null || true; exit 130' INT TERM

usage() {
  cat <<'EOF'
usage: setup-upscaler-models.zsh --destination DIR [--flashvsr] [--adcsr] [--minimax-h3] [--dry-run]

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
    --minimax-h3)
      INSTALL_MINIMAX_H3=1
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
if (( ! INSTALL_FLASHVSR && ! INSTALL_ADCSR && ! INSTALL_MINIMAX_H3 )); then
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
    if [[ -z "$expected" ]]; then
      print "available: $output"
      return 0
    fi
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
  if [[ -n "$expected" ]]; then
    actual="$(sha256 "$output.part")"
    if [[ "$actual" != "$expected" ]]; then
      print -u2 "checksum mismatch for ${output:t}: expected $expected, got $actual"
      exit 1
    fi
  else
    print "downloaded without bundled checksum: $output"
  fi
  mv "$output.part" "$output"
}

compile_coreai_asset() {
  local source="$1"
  local output_dir="$2"
  [[ -d "$source" ]] || { print -u2 "missing Core AI source asset: $source"; exit 1; }
  mkdir -p "$output_dir"
  local stem="${source:t:r}"
  local expected="$output_dir/$stem.aimodelc"
  if [[ -d "$expected" ]]; then
    return
  fi
  xcrun coreai-build compile "$source" \
    --output "$output_dir" \
    --platform macOS \
    --min-deployment-version 27.0 \
    --preferred-compute gpu
  if [[ ! -d "$expected" ]]; then
    print -u2 "Core AI compile did not produce expected asset: $expected"
    exit 1
  fi
}

ensure_uv_python() {
  local uv_archive="$WORK/uv-aarch64-apple-darwin.tar.gz"
  local uv_root="$WORK/uv"
  local venv="$WORK/coreai-venv"
  download \
    "https://github.com/astral-sh/uv/releases/download/0.12.5/uv-aarch64-apple-darwin.tar.gz" \
    "$uv_archive" \
    "5bb0e5fe008a773c3dbcb97ff79cd89e1241464fe9d2f986d52ad8f1b037bd62" >&2
  if [[ ! -x "$uv_root/uv" ]]; then
    mkdir -p "$uv_root"
    tar -xzf "$uv_archive" -C "$uv_root" --strip-components=1 >&2
  fi
  if [[ ! -x "$venv/bin/python" ]]; then
    "$uv_root/uv" venv --clear --python 3.12 "$venv" >&2
  fi
  local requirements=(
    'torch==2.11.0'
    'coreai-torch==0.4.2'
    'coreai-opt==0.2.1'
    'coremltools==9.0'
    'safetensors==0.7.0'
    'einops==0.8.2'
    'tqdm==4.68.4'
    'numpy==2.2.6'
    'pillow==12.0.0'
  )
  local stamp="$venv/.mioh-upscaler-coreai-requirements"
  local signature="${(j: :)requirements}"
  if [[ ! -f "$stamp" ]] || [[ "$(<"$stamp")" != "$signature" ]]; then
    UV_LINK_MODE=copy "$uv_root/uv" pip install --python "$venv/bin/python" \
      "${requirements[@]}" >&2
    print -r -- "$signature" > "$stamp"
  fi
  print "$venv/bin/python"
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
    'coremltools==9.0' 'safetensors==0.7.0' 'einops==0.8.2' \
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

is_h3_manifest() {
  local manifest="$1"
  [[ -f "$manifest" ]] || return 1
  /usr/bin/python3 - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

try:
    data = json.loads(Path(sys.argv[1]).read_text())
except Exception:
    sys.exit(1)
identifier = str(data.get("modelIdentifier", "")).lower()
required = ["schemaVersion", "stages", "sigmas"]
if all(key in data for key in required) and "h3" in identifier:
    sys.exit(0)
sys.exit(1)
PY
}

is_beta5_h3_manifest() {
  local manifest="$1"
  [[ -f "$manifest" ]] || return 1
  /usr/bin/python3 - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

try:
    data = json.loads(Path(sys.argv[1]).read_text())
except Exception:
    sys.exit(1)
identifier = str(data.get("modelIdentifier", "")).lower()
required = ["schemaVersion", "stages", "sigmas"]
if all(key in data for key in required) and "h3" in identifier and "beta5" in identifier:
    sys.exit(0)
sys.exit(1)
PY
}

install_minimax_h3() {
  local candidates=(
    "$DESTINATION/minimax-h3-native/manifest-beta5-s16384.json"
    "$DESTINATION/minimax-h3-native/manifest-beta5.json"
    "$DESTINATION/minimax-h3-native/manifest.json"
    "$DESTINATION/manifest-beta5-s16384.json"
    "$DESTINATION/manifest-beta5.json"
    "$DESTINATION/manifest.json"
    "/Volumes/Project_HD/model_weights/minimax-h3-native/manifest-beta5-s16384.json"
    "/Volumes/Project_HD/model_weights/minimax-h3-native/manifest-beta5.json"
    "/Volumes/Project_HD/model_weights/minimax-h3-native/manifest.json"
  )
  local manifest=""
  for candidate in "${candidates[@]}"; do
    if is_beta5_h3_manifest "$candidate"; then
      manifest="$candidate"
      break
    fi
  done
  if [[ -n "$manifest" ]]; then
    progress 1.0 "MiniMax H3は配置済みです: $manifest"
    return
  fi

  local tool_source="${0:A:h}/h3/scripts/apple"
  if [[ ! -f "$tool_source/export_minimax_h3_native.py" \
        || ! -f "$tool_source/export_minimax_h3_qwen_coreai.py" \
        || ! -f "$tool_source/export_10eros_max_h3_dit_coreai.py" \
        || ! -f "$tool_source/build_10eros_max_h3_manifest.py" ]]; then
    print -u2 "bundled MiniMax H3 converter is missing: ${0:A:h}/h3"
    exit 1
  fi

  local available_kib
  available_kib=$(df -Pk "$DESTINATION" | awk 'END { print $4 }')
  local required_kib=$((140 * 1024 * 1024))
  if (( available_kib < required_kib )); then
    print -u2 "MiniMax H3 setup needs at least 140 GiB free in $DESTINATION"
    exit 1
  fi

  local final="$DESTINATION/minimax-h3-native"
  local staging="$WORK/minimax-h3-native.installing"
  local raw="$staging/checkpoints"
  local coreai="$staging/coreai"
  local source_qwen="$staging/source-qwen-int8"
  local source_dit="$staging/10eros-max-h3-beta5-dit-configuration"
  local comfy_root="$WORK/ComfyUI"
  mkdir -p "$raw/text_encoders" "$raw/vae" "$coreai" "$source_qwen" "$source_dit"

  progress 0.03 "MiniMax H3変換環境を準備中"
  local python
  python="$(ensure_uv_python)"

  if [[ ! -d "$comfy_root/.git" ]]; then
    progress 0.08 "ComfyUIのMiniMax H3 VAE定義を取得中"
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$comfy_root"
  fi

  local comfy_hf="https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
  local teneros_hf="https://huggingface.co/TenStrip/10Eros-Max/resolve/main"
  progress 0.12 "10Eros-Max H3 beta5 INT8 ConvRotをダウンロード中（約21 GB）"
  download \
    "$teneros_hf/10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors" \
    "$raw/10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors" \
    "4dd965496e5b1b83cd13c65cbe7a535b8a4d94ae768a7646b4e336d52c4781cf"
  progress 0.24 "MiniMax H3 Qwen 16384用text encoderをダウンロード中"
  download \
    "$comfy_hf/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" \
    "$raw/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" \
    ""
  progress 0.34 "MiniMax H3 VAEをダウンロード中"
  download \
    "$comfy_hf/vae/minimax_h3_video_vae_fp16.safetensors" \
    "$raw/vae/minimax_h3_video_vae_fp16.safetensors" \
    ""
  download \
    "$comfy_hf/vae/minimax_h3_audio_vae_fp32.safetensors" \
    "$raw/vae/minimax_h3_audio_vae_fp32.safetensors" \
    ""

  progress 0.38 "Qwen tokenizerをダウンロード中"
  mkdir -p "$staging/tokenizer/qwen25"
  local qwen_tokenizer_hf="https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct/resolve/main"
  download "$qwen_tokenizer_hf/tokenizer_config.json" \
    "$staging/tokenizer/qwen25/tokenizer_config.json" ""
  download "$qwen_tokenizer_hf/vocab.json" \
    "$staging/tokenizer/qwen25/vocab.json" ""
  download "$qwen_tokenizer_hf/merges.txt" \
    "$staging/tokenizer/qwen25/merges.txt" ""

  progress 0.42 "MiniMax H3 VAEをMac Core AI形式へ変換中"
  "$python" "$tool_source/export_minimax_h3_native.py" \
    --stage video-encoder-tile --backend coreai \
    --checkpoint "$raw/vae/minimax_h3_video_vae_fp16.safetensors" \
    --comfy-root "$comfy_root" \
    --output "$coreai/video-encoder-tile256.aimodel" \
    --video-tile-size 256 --skip-reference --overwrite
  compile_coreai_asset "$coreai/video-encoder-tile256.aimodel" "$coreai"
  "$python" "$tool_source/export_minimax_h3_native.py" \
    --stage video-decoder-tile --backend coreai \
    --checkpoint "$raw/vae/minimax_h3_video_vae_fp16.safetensors" \
    --comfy-root "$comfy_root" \
    --output "$coreai/video-decoder-raw-tile7x16.aimodel" \
    --video-latent-frames 7 --skip-reference --overwrite
  compile_coreai_asset "$coreai/video-decoder-raw-tile7x16.aimodel" "$coreai"
  "$python" "$tool_source/export_minimax_h3_native.py" \
    --stage audio-encoder --backend coreai \
    --checkpoint "$raw/vae/minimax_h3_audio_vae_fp32.safetensors" \
    --comfy-root "$comfy_root" \
    --output "$coreai/audio-encoder.aimodel" \
    --audio-samples 320000 --skip-reference --overwrite
  compile_coreai_asset "$coreai/audio-encoder.aimodel" "$coreai"
  "$python" "$tool_source/export_minimax_h3_native.py" \
    --stage audio-decoder --backend coreai \
    --checkpoint "$raw/vae/minimax_h3_audio_vae_fp32.safetensors" \
    --comfy-root "$comfy_root" \
    --output "$coreai/audio-decoder.aimodel" \
    --audio-latent-frames 405 --skip-reference --overwrite
  compile_coreai_asset "$coreai/audio-decoder.aimodel" "$coreai"

  progress 0.50 "Qwen 16384 text/vision encoderをMac Core AI形式へ変換中"
  "$python" "$tool_source/export_minimax_h3_qwen_coreai.py" \
    --checkpoint "$raw/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors" \
    --source-directory "$source_qwen" \
    --compiled-directory "$coreai" \
    --sequence-length 16384 \
    --preferred-compute gpu \
    --overwrite

  progress 0.72 "10Eros-Max H3 DiTをMac Core AI形式へ変換中"
  "$python" "$tool_source/export_10eros_max_h3_dit_coreai.py" \
    --checkpoint "$raw/10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors" \
    --source-directory "$source_dit" \
    --compiled-directory "$coreai" \
    --asset-prefix 10eros-max-h3-beta5 \
    --configuration-name 10eros-max-h3-beta5-dit-configuration \
    --model-name "10Eros-Max H3 TURBO Hybrid Beta5 INT8 ConvRot" \
    --preferred-compute gpu \
    --dynamic-max-tokens 131072 \
    --dynamic-sample-tokens 16384 \
    --block-group-size 4 \
    --block-scalar-type bfloat16 \
    --fragment-output "$staging/10eros-max-h3-beta5-denoiser-dynamic-composite-manifest.json" \
    --overwrite

  progress 0.96 "MiniMax H3 manifestを作成中"
  "$python" "$tool_source/build_10eros_max_h3_manifest.py" \
    --model-directory "$staging" \
    --denoiser-manifest "$staging/10eros-max-h3-beta5-denoiser-dynamic-composite-manifest.json" \
    --model-identifier "10eros-max-h3-turbo-hybrid-beta5-native-v1-s16384" \
    --conditioning-mode ref2va \
    --output "$staging/manifest-beta5-s16384.json"
  "$python" "$tool_source/build_10eros_max_h3_manifest.py" \
    --model-directory "$staging" \
    --denoiser-manifest "$staging/10eros-max-h3-beta5-denoiser-dynamic-composite-manifest.json" \
    --model-identifier "10eros-max-h3-turbo-hybrid-beta5-native-v1-s16384" \
    --conditioning-mode fl2va \
    --output "$staging/manifest-fl2va-beta5-s16384.json"
  cp "$staging/manifest-beta5-s16384.json" "$staging/manifest.json"
  cp "$staging/manifest-fl2va-beta5-s16384.json" "$staging/manifest-fl2va.json"
  if ! is_h3_manifest "$staging/manifest-beta5-s16384.json"; then
    print -u2 "MiniMax H3 manifest generation failed"
    exit 1
  fi

  if [[ -e "$final" ]]; then
    mv "$final" "$final.invalid-$(date +%Y%m%d-%H%M%S)"
  fi
  mv "$staging" "$final"
  progress 1.0 "MiniMax H3のダウンロード、Mac変換、manifest設定が完了しました: $final/manifest-beta5-s16384.json"
}

if (( DRY_RUN )); then
  print "destination=$DESTINATION"
  print "flashvsr=$INSTALL_FLASHVSR"
  print "adcsr=$INSTALL_ADCSR"
  print "minimax_h3=$INSTALL_MINIMAX_H3"
  print "converter=${0:A:h}/flashvsr"
  print "h3_converter=${0:A:h}/h3"
  exit 0
fi

mkdir -p "$DESTINATION" "$WORK"
selected_count=$((INSTALL_ADCSR + INSTALL_FLASHVSR + INSTALL_MINIMAX_H3))
if (( INSTALL_ADCSR )); then
  PROGRESS_BASE=0
  if (( selected_count > 1 )); then
    PROGRESS_SPAN=$(awk -v n="$selected_count" 'BEGIN { printf "%.4f", 1 / n }')
  else
    PROGRESS_SPAN=1
  fi
  install_adcsr
fi
if (( INSTALL_FLASHVSR )); then
  if (( selected_count > 1 )); then
    PROGRESS_BASE=$(awk -v done="$INSTALL_ADCSR" -v n="$selected_count" 'BEGIN { printf "%.4f", done / n }')
    PROGRESS_SPAN=$(awk -v n="$selected_count" 'BEGIN { printf "%.4f", 1 / n }')
  else
    PROGRESS_BASE=0
    PROGRESS_SPAN=1
  fi
  install_flashvsr
fi
if (( INSTALL_MINIMAX_H3 )); then
  if (( selected_count > 1 )); then
    PROGRESS_BASE=$(awk -v done="$((INSTALL_ADCSR + INSTALL_FLASHVSR))" -v n="$selected_count" 'BEGIN { printf "%.4f", done / n }')
    PROGRESS_SPAN=$(awk -v n="$selected_count" 'BEGIN { printf "%.4f", 1 / n }')
  else
    PROGRESS_BASE=0
    PROGRESS_SPAN=1
  fi
  install_minimax_h3
fi

# All installed models have already been moved out of this private workspace.
# Keep cleanup strictly confined to the selected destination.
if [[ "$WORK" == "$DESTINATION/.mioh-upscaler-setup" ]]; then
  rm -rf "$WORK"
fi
PROGRESS_BASE=0
PROGRESS_SPAN=1
progress 1.0 "モデル自動設定が完了しました"
