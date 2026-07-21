#!/bin/zsh
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

set -euo pipefail

repo_root="/Users/okatti/Documents/lada"
python_bin="/Users/okatti/.pyenv/versions/lada/bin/python"
dataset_root="/Volumes/Project_HD/lada_finetune_aozora_hikari/dataset_representative"
work_dir="${WORK_DIR:-/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v31/runs/hierarchical-9-3-1-pilot}"
source_checkpoint="${V3_CHECKPOINT:-/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v3/runs/pilot-500/mioh-restorer-v3-step-0003000.pth}"
teacher_checkpoint="${TEACHER_CHECKPOINT:-$repo_root/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth}"
steps="${STEPS:-2000}"
validation_batches="${VALIDATION_BATCHES:-16}"

for required in "$python_bin" "$source_checkpoint" "$teacher_checkpoint"; do
  if [[ ! -e "$required" ]]; then
    echo "Required file not found: $required" >&2
    exit 1
  fi
done

count_metadata() {
  find "$1" -maxdepth 1 \( -type f -o -type l \) -name '*.json' \
    ! -name '._*' -print | wc -l | tr -d ' '
}

train_count="$(count_metadata "$dataset_root/train/crop_unscaled_meta")"
validation_count="$(count_metadata "$dataset_root/validation/crop_unscaled_meta")"
test_count="$(count_metadata "$dataset_root/test/crop_unscaled_meta")"
if [[ "$train_count" != 1500 || "$validation_count" != 188 || "$test_count" != 187 ]]; then
  echo "Unexpected dataset counts: train=$train_count validation=$validation_count test=$test_count" >&2
  exit 1
fi

resume_args=()
initialization_args=(--initialize-from-v3 "$source_checkpoint")
if [[ -n "${RESUME:-}" ]]; then
  if [[ ! -f "$RESUME" ]]; then
    echo "Resume checkpoint not found: $RESUME" >&2
    exit 1
  fi
  resume_args=(--resume "$RESUME")
  initialization_args=()
elif [[ -e "$work_dir/mioh-restorer-v31-latest.pth" && "${CHECK_ONLY:-0}" != 1 ]]; then
  echo "A V3.1 checkpoint already exists in $work_dir" >&2
  echo "Resume it with RESUME=... $0" >&2
  exit 1
fi

log_file="${LOG_FILE:-$work_dir/training.log}"

echo "Starting MiohRestorerV3.1 hierarchical 9/3/1 pilot"
echo "Coverage: +/-13 feature pixels (+/-52 input pixels)"
echo "Dataset: train=$train_count validation=$validation_count test=$test_count"
echo "Initialization: $source_checkpoint"
echo "Work directory: $work_dir"

if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  echo "Preflight check complete; training was not started."
  exit 0
fi

mkdir -p "$work_dir"
cd "$repo_root"

caffeinate -dimsu "$python_bin" -u \
  scripts/training/train-mioh-restorer-v2.py \
  --model-version 3 \
  --train-metadata-root "$dataset_root/train/crop_unscaled_meta" \
  --val-metadata-root "$dataset_root/validation/crop_unscaled_meta" \
  --work-dir "$work_dir" \
  "${resume_args[@]}" \
  "${initialization_args[@]}" \
  --teacher-checkpoint "$teacher_checkpoint" \
  --teacher-weight 0.25 \
  --teacher-feature-weight 0.05 \
  --teacher-alignment-weight 0.02 \
  --teacher-distill-calls 2 \
  --teacher-shift-temperature 0.5 \
  --teacher-fp16 \
  --gradient-checkpointing \
  --steps "$steps" \
  --batch-size 1 \
  --workers 2 \
  --validation-workers 0 \
  --no-validate-at-start \
  --prefetch-factor 1 \
  --window-frames 24 \
  --chunk-frames 4 \
  --image-size 384 \
  --channels 96 \
  --blocks 7 \
  --encoder-blocks 5 \
  --reconstruction-blocks 5 \
  --alignment-radius 1 \
  --first-order-dilation 1 \
  --second-order-dilation 2 \
  --hierarchical-alignment-dilations 9 3 1 \
  --alignment-temperature 0.5 \
  --alignment-key-channels 64 \
  --alignment-groups 8 \
  --detail-scale 0.25 \
  --learning-rate 5e-5 \
  --minimum-learning-rate 2e-6 \
  --warmup-steps 250 \
  --gradient-weight 0.30 \
  --temporal-weight 0.20 \
  --high-frequency-weight 0.15 \
  --perceptual-weight 0.02 \
  --perceptual-frame-stride 4 \
  --perceptual-image-size 224 \
  --structural-weight 0.10 \
  --structural-frame-stride 2 \
  --directional-aux-weight 0.15 \
  --direction-consistency-weight 0.02 \
  --gan-weight 0 \
  --max-grad-norm 1.0 \
  --ema-decay 0.999 \
  --save-latest-every 100 \
  --save-every 500 \
  --validate-every 500 \
  --validation-batches "$validation_batches" \
  --log-every 20 \
  --memory-log-every 20 \
  --memory-warning-ratio 0.80 \
  --memory-critical-ratio 0.92 \
  --memory-warning-available-gib 8 \
  --memory-critical-available-gib 4 \
  --memory-emergency-stop \
  --seed 0 \
  --device mps \
  --degrade \
  --horizontal-flip \
  2>&1 | tee -a "$log_file"
