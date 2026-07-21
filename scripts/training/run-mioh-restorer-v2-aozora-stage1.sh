#!/bin/zsh

set -euo pipefail

repo_root="${0:A:h:h:h}"
dataset_root="/Volumes/Project_HD/lada_finetune_aozora_hikari/dataset_representative"
work_dir="/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v2/runs/stage1"
log_dir="/Volumes/Project_HD/lada_finetune_aozora_hikari/logs"
python_bin="${LADA_TRAIN_PYTHON:-/Users/okatti/.pyenv/versions/lada/bin/python}"
latest_checkpoint="$work_dir/mioh-restorer-v2-latest.pth"
resume_args=()

count_metadata() {
  find "$1" -maxdepth 1 \( -type f -o -type l \) -name '*.json' \
    ! -name '._*' -print | wc -l | tr -d ' '
}

if [[ ! -x "$python_bin" ]]; then
  echo "Training Python not found: $python_bin" >&2
  exit 1
fi

for split in train validation test; do
  metadata_root="$dataset_root/$split/crop_unscaled_meta"
  if [[ ! -d "$metadata_root" ]]; then
    echo "Dataset split not found: $metadata_root" >&2
    exit 1
  fi
done

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

if [[ -n "${RESUME:-}" ]]; then
  if [[ ! -f "$RESUME" ]]; then
    echo "Resume checkpoint not found: $RESUME" >&2
    exit 1
  fi
  resume_args=(--resume "$RESUME")
elif [[ -e "$latest_checkpoint" ]]; then
  echo "A checkpoint already exists: $latest_checkpoint" >&2
  echo "This launcher will not overwrite an existing run." >&2
  echo "Resume with: RESUME='$latest_checkpoint' $0" >&2
  exit 1
fi

mkdir -p "$work_dir" "$log_dir"
log_file="$log_dir/mioh-restorer-v2-aozora-stage1.log"

echo "Starting a new MiohRestorerV2 training run"
echo "Dataset: train=$train_count validation=$validation_count test=$test_count"
echo "Work directory: $work_dir"
echo "Log: $log_file"

if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  echo "Preflight check complete; training was not started."
  exit 0
fi

cd "$repo_root"

caffeinate -dimsu "$python_bin" -u \
  scripts/training/train-mioh-restorer-v2.py \
  --train-metadata-root "$dataset_root/train/crop_unscaled_meta" \
  --val-metadata-root "$dataset_root/validation/crop_unscaled_meta" \
  --work-dir "$work_dir" \
  "${resume_args[@]}" \
  --steps 40000 \
  --batch-size 1 \
  --workers 2 \
  --validation-workers 0 \
  --prefetch-factor 1 \
  --window-frames 24 \
  --chunk-frames 4 \
  --image-size 384 \
  --channels 96 \
  --blocks 12 \
  --fusion-full-channels 32 \
  --fusion-half-channels 64 \
  --fusion-quarter-channels 96 \
  --detail-scale 0.25 \
  --learning-rate 1e-4 \
  --minimum-learning-rate 2e-6 \
  --warmup-steps 1000 \
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
  --gan-weight 0.002 \
  --gan-start-step 10000 \
  --gan-learning-rate 5e-5 \
  --gan-warmup-steps 500 \
  --gan-frame-stride 4 \
  --gan-image-size 192 \
  --discriminator-channels 32 \
  --max-grad-norm 1.0 \
  --ema-decay 0.999 \
  --save-latest-every 100 \
  --save-every 500 \
  --validate-every 500 \
  --validation-batches 16 \
  --validate-at-start \
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
