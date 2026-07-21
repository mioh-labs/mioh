#!/bin/zsh
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

set -euo pipefail

repo_root="/Users/okatti/Documents/lada"
python_bin="/Users/okatti/.pyenv/versions/lada/bin/python"
dataset_root="/Volumes/Project_HD/lada_finetune_aozora_hikari/dataset_representative"
teacher_checkpoint="${TEACHER_CHECKPOINT:-$repo_root/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth}"
stage="${STAGE:-1}"
case "$stage" in
  1) stage_name="foundation"; default_steps=10000 ;;
  2) stage_name="faithful-reconstruction"; default_steps=15000 ;;
  3) stage_name="detail-recovery"; default_steps=20000 ;;
  4) stage_name="temporal-consistency"; default_steps=15000 ;;
  5) stage_name="fidelity-polish"; default_steps=10000 ;;
  *) echo "STAGE must be between 1 and 5" >&2; exit 1 ;;
esac
work_dir="${WORK_DIR:-/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v4/runs/stage-${stage}-${stage_name}}"
steps="${STEPS:-$default_steps}"
warmup_steps="${WARMUP_STEPS:-500}"

if [[ ! -x "$python_bin" ]]; then
  echo "Python not found: $python_bin" >&2
  exit 1
fi
teacher_args=()
if [[ "$stage" == 1 || "$stage" == 2 ]]; then
  if [[ ! -f "$teacher_checkpoint" ]]; then
    echo "BasicVSR++ teacher checkpoint not found: $teacher_checkpoint" >&2
    exit 1
  fi
  teacher_args=(--teacher-checkpoint "$teacher_checkpoint")
fi

count_metadata() {
  find "$1" -maxdepth 1 \( -type f -o -type l \) -name '*.json' \
    ! -name '._*' -print | wc -l | tr -d ' '
}

train_root="$dataset_root/train/crop_unscaled_meta"
validation_root="$dataset_root/validation/crop_unscaled_meta"
test_root="$dataset_root/test/crop_unscaled_meta"
train_count="$(count_metadata "$train_root")"
validation_count="$(count_metadata "$validation_root")"
test_count="$(count_metadata "$test_root")"
if [[ "$train_count" != 1500 || "$validation_count" != 188 || "$test_count" != 187 ]]; then
  echo "Unexpected dataset counts: train=$train_count validation=$validation_count test=$test_count" >&2
  exit 1
fi

transfer_args=()
if [[ -n "${RESUME:-}" && -n "${INITIALIZE_FROM:-}" ]]; then
  echo "RESUME and INITIALIZE_FROM cannot be used together" >&2
  exit 1
elif [[ -n "${RESUME:-}" ]]; then
  if [[ ! -f "$RESUME" ]]; then
    echo "Resume checkpoint not found: $RESUME" >&2
    exit 1
  fi
  transfer_args=(--resume "$RESUME")
elif [[ -n "${INITIALIZE_FROM:-}" ]]; then
  if [[ ! -f "$INITIALIZE_FROM" ]]; then
    echo "Initialization checkpoint not found: $INITIALIZE_FROM" >&2
    exit 1
  fi
  transfer_args=(
    --initialize-from "$INITIALIZE_FROM"
    --initialize-weights "${INITIALIZE_WEIGHTS:-ema}"
  )
elif [[ "$stage" != 1 && "${CHECK_ONLY:-0}" != 1 ]]; then
  echo "Stage $stage requires INITIALIZE_FROM=<completed previous-stage checkpoint>" >&2
  exit 1
elif [[ -e "$work_dir/mioh-restorer-v4-latest.pth" && "${CHECK_ONLY:-0}" != 1 ]]; then
  echo "A V4 checkpoint already exists in $work_dir" >&2
  echo "Resume it with RESUME=... $0" >&2
  exit 1
fi

echo "Starting independent MiohRestorerV4-Q training stage"
echo "Dataset: train=$train_count validation=$validation_count test=$test_count"
echo "Geometry: 9 input -> 5 output, stride 4, hier27 reach +/-40px"
echo "Stage: $stage ($stage_name), local steps: $steps"
echo "Targets: direct clean GT is always primary; no BasicVSR++ pixel teacher, no GAN hallucination"
if [[ "$stage" == 1 ]]; then
  echo "Auxiliary supervision: SPyNet flow KL + exact synthetic known motion"
  echo "Teacher: $teacher_checkpoint (SPyNet only; no DCN offset projection)"
elif [[ "$stage" == 2 ]]; then
  echo "Auxiliary supervision: low-weight 1x1 feature distillation on the same 9-frame window"
  echo "Teacher: $teacher_checkpoint (quarter-resolution reconstruction features only)"
else
  echo "Auxiliary supervision: none (GT-only quality training)"
fi
echo "Work directory: $work_dir"
if [[ -n "${INITIALIZE_FROM:-}" ]]; then
  echo "Initialize from: $INITIALIZE_FROM (${INITIALIZE_WEIGHTS:-ema} weights; new optimizer/EMA)"
elif [[ -n "${RESUME:-}" ]]; then
  echo "Resume same stage: $RESUME (optimizer/EMA/RNG retained)"
else
  if [[ "$stage" == 1 ]]; then
    echo "Initialization: new random weights"
  else
    echo "Initialization: not supplied (required for an actual Stage $stage run)"
  fi
fi

if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  echo "Preflight check complete; training was not started."
  exit 0
fi

mkdir -p "$work_dir"
cd "$repo_root"

caffeinate -dimsu "$python_bin" -u \
  scripts/training/train-mioh-restorer-v4.py \
  --train-metadata-root "$train_root" \
  --val-metadata-root "$validation_root" \
  --work-dir "$work_dir" \
  "${transfer_args[@]}" \
  "${teacher_args[@]}" \
  --stage "$stage" \
  --steps "$steps" \
  --device mps \
  --image-size 384 \
  --batch-size 1 \
  --workers 2 \
  --validation-workers 0 \
  --prefetch-factor 1 \
  --alignment-variant hier27 \
  --execution-mode batch \
  --gradient-checkpointing \
  --stage-transition-steps 500 \
  --learning-rate 5e-5 \
  --minimum-learning-rate 2e-6 \
  --warmup-steps "$warmup_steps" \
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
