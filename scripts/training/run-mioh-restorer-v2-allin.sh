#!/bin/zsh

set -euo pipefail

repo_root="${0:A:h:h:h}"
work_dir="/Volumes/Project_HD/lada_finetune/mioh_restorer_v2_allin/runs/stage1"
latest_checkpoint="$work_dir/mioh-restorer-v2-latest.pth"
resume_args=()

if [[ -n "${RESUME:-}" ]]; then
  resume_args=(--resume "$RESUME")
elif [[ -e "$latest_checkpoint" ]]; then
  echo "Existing V2 checkpoint found: $latest_checkpoint" >&2
  echo "Resume with: RESUME='$latest_checkpoint' $0" >&2
  exit 1
fi

cd "$repo_root"

exec caffeinate -dimsu .venv-coreai/bin/python \
  scripts/training/train-mioh-restorer-v2.py \
  --train-metadata-root /Volumes/Project_HD/lada_finetune/mioh_restorer_v1/dataset/train/crop_unscaled_meta \
  --val-metadata-root /Volumes/Project_HD/lada_finetune/mioh_restorer_v1/dataset/val/crop_unscaled_meta \
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
  --no-validate-at-start \
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
  --horizontal-flip
