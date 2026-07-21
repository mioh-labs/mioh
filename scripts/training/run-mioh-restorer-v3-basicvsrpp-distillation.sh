#!/bin/zsh
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

set -euo pipefail

repo_root="/Users/okatti/Documents/lada"
python_bin="/Users/okatti/.pyenv/versions/lada/bin/python"
dataset_root="/Volumes/Project_HD/lada_finetune_aozora_hikari/dataset_representative"
work_dir="${WORK_DIR:-/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v3/runs/basicvsrpp-alignment-distilled-stage1}"
log_dir="/Volumes/Project_HD/lada_finetune_aozora_hikari/logs"
v2_checkpoint="${V2_CHECKPOINT:-/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v2/runs/stage1/mioh-restorer-v2-step-0008000.pth}"
teacher_checkpoint="${TEACHER_CHECKPOINT:-$repo_root/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth}"
steps="${STEPS:-40000}"
validation_batches="${VALIDATION_BATCHES:-16}"
validate_at_start="${VALIDATE_AT_START:-1}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python not found: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$teacher_checkpoint" ]]; then
  echo "BasicVSR++ teacher checkpoint not found: $teacher_checkpoint" >&2
  exit 1
fi

count_metadata() {
  find "$1" -maxdepth 1 \( -type f -o -type l \) -name '*.json' \
    ! -name '._*' -print | wc -l | tr -d ' '
}

train_count="$(count_metadata "$dataset_root/train/crop_unscaled_meta")"
validation_count="$(count_metadata "$dataset_root/validation/crop_unscaled_meta")"
test_count="$(count_metadata "$dataset_root/test/crop_unscaled_meta")"
if [[ "$train_count" != 1500 || "$validation_count" != 188 || "$test_count" != 187 ]]; then
  echo "Unexpected dataset counts: train=$train_count validation=$validation_count test=$test_count" >&2
  echo "Expected: train=1500 validation=188 test=187" >&2
  exit 1
fi

broken_link="$(find -L "$dataset_root" -type l -print -quit)"
if [[ -n "$broken_link" ]]; then
  echo "Broken dataset link: $broken_link" >&2
  exit 1
fi

resume_args=()
initialization_args=()
if [[ -n "${RESUME:-}" ]]; then
  if [[ ! -f "$RESUME" ]]; then
    echo "Resume checkpoint not found: $RESUME" >&2
    exit 1
  fi
  resume_args=(--resume "$RESUME")
else
  if [[ ! -f "$v2_checkpoint" ]]; then
    echo "V2 initialization checkpoint not found: $v2_checkpoint" >&2
    exit 1
  fi
  initialization_args=(--initialize-from-v2 "$v2_checkpoint")
  if [[ -e "$work_dir/mioh-restorer-v3-latest.pth" && "${CHECK_ONLY:-0}" != 1 ]]; then
    echo "A V3 checkpoint already exists in $work_dir" >&2
    echo "Resume it with RESUME=... $0" >&2
    exit 1
  fi
fi

mkdir -p "$work_dir" "$log_dir"
log_file="${LOG_FILE:-$work_dir/training.log}"
validation_start_args=(--validate-at-start)
if [[ "$validate_at_start" != 1 ]]; then
  validation_start_args=(--no-validate-at-start)
fi

echo "Starting Core AI aligned MiohRestorerV3 training"
echo "Dataset: train=$train_count validation=$validation_count test=$test_count"
echo "Teacher: $teacher_checkpoint"
echo "Work directory: $work_dir"
echo "Log: $log_file"

if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  echo "Preflight check complete; training was not started."
  exit 0
fi

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
  --teacher-shift-temperature 1.0 \
  --teacher-fp16 \
  --gradient-checkpointing \
  --steps "$steps" \
  --batch-size 1 \
  --workers 2 \
  --validation-workers 0 \
  "${validation_start_args[@]}" \
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
  --alignment-key-channels 64 \
  --alignment-groups 8 \
  --detail-scale 0.25 \
  --learning-rate 5e-5 \
  --minimum-learning-rate 2e-6 \
  --warmup-steps 500 \
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
