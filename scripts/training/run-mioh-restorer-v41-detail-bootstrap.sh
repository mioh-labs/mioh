#!/bin/zsh
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

set -euo pipefail

repo_root="/Users/okatti/Documents/lada"
python_bin="/Users/okatti/.pyenv/versions/lada/bin/python"
dataset_root="/Volumes/Project_HD/lada_finetune_aozora_hikari/dataset_representative"
default_parent="/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v4/runs/stage-3-detail-recovery/mioh-restorer-v4-step-0020000.pth"
parent="${INITIALIZE_FROM:-$default_parent}"
work_dir="${WORK_DIR:-/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v41/runs/stage-3a-detail-bootstrap}"
steps="${STEPS:-3000}"
fixed_lr="${FIXED_LR:-1e-5}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python not found: $python_bin" >&2
  exit 1
fi
if [[ ! -f "$parent" ]]; then
  echo "Completed V4 parent checkpoint not found: $parent" >&2
  exit 1
fi
if [[ -e "$work_dir/mioh-restorer-v4-latest.pth" ]]; then
  echo "A V4.1 checkpoint already exists in $work_dir" >&2
  echo "Use the explicit resume command printed in docs/mioh-restorer-v4.md" >&2
  exit 1
fi

train_root="$dataset_root/train/crop_unscaled_meta"
validation_root="$dataset_root/validation/crop_unscaled_meta"
mkdir -p "$work_dir"
cd "$repo_root"

echo "Starting MiohRestorer V4.1 high-resolution detail bootstrap"
echo "Parent: $parent (EMA; whitelist migration; optimizer and EMA rebuilt)"
echo "Trainable: V4.1 detail projections, 1/2+full fine alignment, detail fusion/output"
echo "Frozen: all V4 encoder/coarse alignment/base/texture/confidence parameters"
echo "Loss/degradation: unchanged Stage 3 objective and 20/50/30 clean/mild/full mix"
echo "Steps: $steps, fixed learning rate: $fixed_lr"
echo "Work directory: $work_dir"

if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  echo "Preflight check complete; training was not started."
  exit 0
fi

caffeinate -dimsu "$python_bin" -u \
  scripts/training/train-mioh-restorer-v4.py \
  --train-metadata-root "$train_root" \
  --val-metadata-root "$validation_root" \
  --work-dir "$work_dir" \
  --initialize-from "$parent" \
  --initialize-weights ema \
  --initialize-v4-upgrade \
  --stage 3 \
  --steps "$steps" \
  --device mps \
  --image-size 384 \
  --batch-size 1 \
  --workers 2 \
  --validation-workers 0 \
  --prefetch-factor 1 \
  --alignment-variant hier27 \
  --execution-mode batch \
  --high-resolution-detail \
  --train-high-resolution-detail-only \
  --gradient-checkpointing \
  --stage-transition-steps 0 \
  --fixed-learning-rate "$fixed_lr" \
  --learning-rate "$fixed_lr" \
  --minimum-learning-rate 2e-6 \
  --warmup-steps 0 \
  --weight-decay 1e-4 \
  --max-grad-norm 1.0 \
  --ema-decay 0.999 \
  --confidence-scale 0.05 \
  --perceptual-image-size 224 \
  --save-latest-every 100 \
  --save-every 500 \
  --validate-every 500 \
  --validation-batches 32 \
  --log-every 20 \
  --memory-warning-ratio 0.80 \
  --memory-critical-ratio 0.92 \
  --memory-warning-available-gib 8 \
  --memory-critical-available-gib 4 \
  --degrade \
  --horizontal-flip \
  2>&1 | tee -a "$work_dir/training.log"
