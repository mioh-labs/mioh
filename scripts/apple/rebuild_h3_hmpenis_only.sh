#!/bin/zsh
# Rebuild every installed MiniMax H3 / 10Eros variant without Realism People.
# HMPenis and the appropriate Turbo adapter are retained.

set -euo pipefail

SCRIPT_DIR=${0:A:h}
PROJECT_ROOT=${SCRIPT_DIR:h:h}
MODEL_ROOT=${MIOH_H3_MODEL_ROOT:-/Volumes/Project_HD/model_weights}
SOURCE_ROOT=${MIOH_H3_SOURCE_ROOT:-${MODEL_ROOT}/minimax-h3-source}
LORA_ROOT=${MIOH_H3_LORA_ROOT:-${MODEL_ROOT}/minimax-h3-loras}
TEN_EROS_CHECKPOINT=${MIOH_10EROS_CHECKPOINT:-/Volumes/Firewire_HD/10Eros_Max_h3_TURBO-hybrid_beta5_int8.safetensors}
RUNNER=${MIOH_H3_RUNNER:-/Applications/mioh\ upscaler.app/Contents/Resources/bin/mioh-minimax-h3-native}
PYTHON=${MIOH_H3_PYTHON:-python3}

HMPENIS=${LORA_ROOT}/hmpenis/HMPenis_v2_e35.safetensors
COMBINED_ROOT=${LORA_ROOT}/hmpenis-only

require_file() {
  [[ -f "$1" ]] || { print -u2 "missing file: $1"; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { print -u2 "missing directory: $1"; exit 1; }
}

clone_directory() {
  local source=$1
  local destination=$2
  rm -rf -- "$destination"
  # APFS clone-copy keeps temporary disk use low while preserving the source.
  cp -cR -- "$source" "$destination"
  if [[ ! -f "$destination/qwen-composite-manifest.json" ]]; then
    "$PYTHON" - "$destination/manifest.json" "$destination/qwen-composite-manifest.json" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
output = Path(sys.argv[2])
manifest = json.loads(source.read_text(encoding="utf-8"))
output.write_text(
    json.dumps(manifest["qwenComposite"], indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
PY
  fi
}

combine_lora() {
  local turbo=$1
  local output=$2
  require_file "$turbo"
  if [[ ! -f "$output" ]]; then
    "$PYTHON" "$SCRIPT_DIR/combine_minimax_h3_loras.py" \
      --component "$HMPENIS" 1.0 \
      --component "$turbo" 1.0 \
      --output "$output"
  fi
}

validate_and_install() {
  local destination=$1
  local staging=$2
  local backup=${destination}.pre-hmpenis-only

  "$RUNNER" validate --manifest "$staging/manifest.json"
  if [[ ! -e "$destination" ]]; then
    mv -- "$staging" "$destination"
    return
  fi
  rm -rf -- "$backup"
  mv -- "$destination" "$backup"
  if mv -- "$staging" "$destination"; then
    rm -rf -- "$backup"
  else
    mv -- "$backup" "$destination"
    return 1
  fi
}

rebuild_minimax() {
  local mode=$1
  local steps=$2
  local checkpoint turbo turbo_mode source destination staging combined prefix fragment label

  checkpoint=${SOURCE_ROOT}/diffusion_models/minimax_h3_${mode}_pruned_fp8_scaled.safetensors
  if [[ "$mode" == ref2va ]]; then
    turbo_mode=ref2v
  else
    turbo_mode=fl2v
  fi
  turbo=${LORA_ROOT}/minimax_h3_${turbo_mode}_turbo_${steps}step_v1.0_768p_comfyui_bf16.safetensors
  if [[ "$mode" == ref2va && "$steps" == 4 ]]; then
    turbo=${LORA_ROOT}/minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors
  fi
  combined=${COMBINED_ROOT}/HMPenis-s100_${mode:u}Turbo${steps}-s100.safetensors
  source=${MODEL_ROOT}/minimax-h3-native-${mode}-turbo${steps}-nofp8
  if [[ "$mode" == ref2va ]]; then
    label=MiniMax_Ref2VA_${steps}_step
  else
    label=MiniMax_FL2VA_${steps}_step
  fi
  destination=${MODEL_ROOT}/${label}
  staging=${destination}.rebuild-hmpenis-only
  prefix=minimax-h3-${mode}-turbo${steps}-nofp8
  fragment=${staging}/${prefix}-denoiser-composite-manifest.json

  print "\n==> Rebuilding ${prefix} (HMPenis + Turbo, no Realism People)"
  if [[ -f "$destination/manifest.json" ]] && \
    "$RUNNER" validate --manifest "$destination/manifest.json" >/dev/null; then
    print "Already complete: $destination"
    [[ "$source" == "$destination" ]] || rm -rf -- "$source"
    return
  fi
  require_file "$checkpoint"
  require_dir "$source"
  combine_lora "$turbo" "$combined"
  clone_directory "$source" "$staging"

  "$PYTHON" "$SCRIPT_DIR/export_10eros_max_h3_dit_coreai.py" \
    --checkpoint "$checkpoint" \
    --lora "$combined" \
    --lora-strength 1.0 \
    --source-directory "$staging/coreai" \
    --compiled-directory "$staging/coreai" \
    --asset-prefix "$prefix" \
    --configuration-name "${prefix}-dit-configuration" \
    --model-name "$prefix (HMPenis 1.0, no Realism People)" \
    --block-group-size 4 \
    --expand-fp8-scaled \
    --fragment-output "$fragment" \
    --overwrite

  "$PYTHON" "$SCRIPT_DIR/build_10eros_max_h3_manifest.py" \
    --model-directory "$staging" \
    --denoiser-manifest "$fragment" \
    --output "$staging/manifest.json" \
    --model-identifier "$prefix" \
    --conditioning-mode "$mode" \
    --sampler euler \
    --steps "$steps" \
    --video-shift 6

  validate_and_install "$destination" "$staging"
  if [[ "$source" != "$destination" ]]; then
    rm -rf -- "$source"
  fi
}

rebuild_10eros() {
  local base destination staging combined prefix fragment
  base=${MODEL_ROOT}/minimax-h3-native-beta5-turbo-dit4
  destination=${MODEL_ROOT}/10Eros_Beta5_Turbo
  staging=${destination}.rebuild-hmpenis-only
  combined=${COMBINED_ROOT}/HMPenis-s100.safetensors
  prefix=10eros-max-h3-turbo-hybrid-beta5-int8-dense-dit4-hmpenis
  fragment=${staging}/${prefix}-denoiser-composite-manifest.json

  print "\n==> Rebuilding 10Eros Beta5 Turbo (HMPenis only)"
  require_file "$TEN_EROS_CHECKPOINT"
  require_dir "$base"
  if [[ ! -f "$combined" ]]; then
    "$PYTHON" "$SCRIPT_DIR/combine_minimax_h3_loras.py" \
      --component "$HMPENIS" 1.0 \
      --output "$combined"
  fi
  clone_directory "$base" "$staging"

  "$PYTHON" "$SCRIPT_DIR/export_10eros_max_h3_dit_coreai.py" \
    --checkpoint "$TEN_EROS_CHECKPOINT" \
    --lora "$combined" \
    --lora-strength 1.0 \
    --source-directory "$staging/coreai" \
    --compiled-directory "$staging/coreai" \
    --asset-prefix "$prefix" \
    --configuration-name "${prefix}-dit-configuration" \
    --model-name "10Eros-Max H3 TURBO Beta5 INT8 + HMPenis 1.0 / DiT 4-layer AOT" \
    --block-group-size 4 \
    --expand-int8-convrot \
    --fragment-output "$fragment"

  "$PYTHON" "$SCRIPT_DIR/build_10eros_max_h3_manifest.py" \
    --model-directory "$staging" \
    --denoiser-manifest "$fragment" \
    --output "$staging/manifest.json" \
    --model-identifier "10eros-max-h3-turbo-beta5-hmpenis-dit4" \
    --conditioning-mode ref2va \
    --sampler res_multistep \
    --steps 6 \
    --video-shift 12

  validate_and_install "$destination" "$staging"
  if [[ "$base" != "$destination" ]]; then
    rm -rf -- "$base"
  fi
}

require_file "$HMPENIS"
require_file "$RUNNER"
mkdir -p -- "$COMBINED_ROOT"

rebuild_minimax ref2va 4
rebuild_minimax ref2va 8
rebuild_minimax fl2va 4
rebuild_minimax fl2va 8
rebuild_10eros

defaults write com.okatti.mioh.upscaler \
  com.okatti.mioh.upscaler.10erosMaxH3ManifestPath \
  "${MODEL_ROOT}/MiniMax_Ref2VA_8_step/manifest.json"

print "\nAll H3 variants were rebuilt without Realism People."
