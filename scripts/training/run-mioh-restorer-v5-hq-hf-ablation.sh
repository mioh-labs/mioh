#!/bin/zsh
set -euo pipefail

# Bounded diagnostic run. It preserves the deployed BasicVSR++ backbone and
# asks whether the V5-HQ refiner can recover faithful high-frequency energy.

repo_root="${0:A:h:h:h}"
dataset_root="${DATASET_ROOT:-/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v5_balanced_v3}"
work_root="${WORK_ROOT:-/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v5_hq_ablation/hf-v3-zero-500}"
deployed_checkpoint="${BASICVSRPP_CHECKPOINT:-$repo_root/model_weights/basicvsrpp-v1.2-clean4000-unified-hf3000-forward-consistency-w005-500-raw.pth}"
python_bin="${PYTHON_BIN:-/Users/okatti/.pyenv/versions/lada/bin/python}"

train_manifest="${TRAIN_MANIFEST:-$dataset_root/manifests/train-native-balanced.jsonl}"
validation_manifest="${VALIDATION_MANIFEST:-$dataset_root/manifests/validation-native-balanced.jsonl}"

for required in "$train_manifest" "$validation_manifest" "$deployed_checkpoint"; do
  if [[ ! -f "$required" ]]; then
    echo "Required input is missing: $required" >&2
    exit 2
  fi
done

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export LADA_DEFORM_CONV_BACKEND=mps_deform_conv

# RESTART=1 discards an existing stage checkpoint and starts the pilot over.
# The default now continues from ``*-latest.pth`` so a run stopped by the
# safety gate can be carried to its full step budget for a real verdict.
extra_flags=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  extra_flags+=(--resume "$RESUME_CHECKPOINT")
fi
if [[ "${RESUME_MODEL_ONLY:-0}" == "1" ]]; then
  extra_flags+=(--resume-model-only)
fi
if [[ "${RESTART:-0}" == "1" ]]; then
  extra_flags+=(--restart-stage)
fi
if [[ -n "${FIXED_LEARNING_RATE:-}" ]]; then
  extra_flags+=(--hq-fixed-learning-rate "$FIXED_LEARNING_RATE")
fi
# EARLY_STOP=1 opts into the legacy amplitude/correlation gate.  It is
# advisory by default because local-NCC runs use projection metrics instead.
if [[ "${EARLY_STOP:-0}" == "1" ]]; then
  extra_flags+=(--hq-hf-early-stop)
fi
if [[ "${FREEZE_BASE_HEAD:-0}" == "1" ]]; then
  extra_flags+=(--hq-freeze-base-head)
fi
if [[ "${FREEZE_CONFIDENCE_HEAD:-0}" == "1" ]]; then
  extra_flags+=(--hq-freeze-confidence-head)
fi
if [[ "${HF_LOCAL_CORRELATION_ON_RESIDUAL:-0}" == "1" ]]; then
  extra_flags+=(--hq-hf-local-correlation-on-residual)
fi
if [[ "${RAW_TEMPORAL_CANDIDATES:-0}" == "1" ]]; then
  extra_flags+=(--hq-raw-temporal-candidates)
fi
if [[ "${RAW_TEMPORAL_ENCODER_CHANNELS:-0}" != "0" ]]; then
  extra_flags+=(--hq-raw-temporal-encoder-channels "${RAW_TEMPORAL_ENCODER_CHANNELS}")
fi
if [[ "${RAW_TEMPORAL_NEAREST_WARP:-0}" == "1" ]]; then
  extra_flags+=(--hq-raw-temporal-nearest-warp)
fi
if [[ "${RAW_TEMPORAL_ZERO_INPUT:-0}" == "1" ]]; then
  extra_flags+=(--hq-raw-temporal-zero-input)
fi
if [[ "${TEXTURE_EFFECTIVE_MASK:-0}" == "1" ]]; then
  extra_flags+=(--hq-texture-effective-mask)
  extra_flags+=(
    --hq-texture-effective-mask-feather-radius
    "${TEXTURE_EFFECTIVE_MASK_FEATHER_RADIUS:-0}"
  )
fi
if [[ "${GAN_TEMPORAL:-0}" == "1" ]]; then
  extra_flags+=(--hq-gan-temporal)
fi
if [[ "${GAN_NORMALIZE_SECONDARY_RMS:-0}" == "1" ]]; then
  extra_flags+=(--hq-gan-normalize-secondary-rms)
fi
if [[ "${GAN_CANDIDATE_PRIMARY:-0}" == "1" ]]; then
  extra_flags+=(--hq-gan-candidate-primary)
fi
if [[ "${GAN_FREEZE_DISCRIMINATOR:-0}" == "1" ]]; then
  extra_flags+=(--hq-gan-freeze-discriminator)
fi
if [[ "${GAN_DISCRIMINATOR_PRETRAIN_UNTIL_STEP:-0}" != "0" ]]; then
  extra_flags+=(
    --hq-gan-discriminator-pretrain-until-step
    "${GAN_DISCRIMINATOR_PRETRAIN_UNTIL_STEP}"
  )
fi
if [[ -n "${GAN_INITIALIZE_DISCRIMINATOR_FROM:-}" ]]; then
  extra_flags+=(
    --hq-gan-initialize-discriminator-from
    "$GAN_INITIALIZE_DISCRIMINATOR_FROM"
  )
fi

exec caffeinate -dimsu "$python_bin" -u scripts/training/train-mioh-restorer-v5.py \
  --train-manifest "$train_manifest" \
  --validation-manifest "$validation_manifest" \
  --work-root "$work_root" \
  --variant hq \
  --stage 3 \
  --steps "${STEPS:-500}" \
  --device mps \
  --amp off \
  --batch-size 1 \
  --accumulate 1 \
  --workers 0 \
  --basicvsrpp-checkpoint "$deployed_checkpoint" \
  --hq-hf-ablation \
  --hq-hf-amplitude-weight "${HF_AMPLITUDE_WEIGHT:-0.05}" \
  --hq-hf-correlation-weight "${HF_CORRELATION_WEIGHT:-0.02}" \
  --hq-hf-local-correlation-weight "${HF_LOCAL_CORRELATION_WEIGHT:-0.0}" \
  --hq-hf-local-correlation-patch-size "${HF_LOCAL_CORRELATION_PATCH_SIZE:-32}" \
  --hq-hf-residual-reconstruction-weight "${HF_RESIDUAL_RECONSTRUCTION_WEIGHT:-0.0}" \
  --hq-hf-correction-target-projection-weight "${HF_CORRECTION_TARGET_PROJECTION_WEIGHT:-0.0}" \
  --hq-hf-correction-target-orthogonal-energy-weight "${HF_CORRECTION_TARGET_ORTHOGONAL_ENERGY_WEIGHT:-0.0}" \
  --hq-guard-ring-identity-weight "${GUARD_RING_IDENTITY_WEIGHT:-0.0}" \
  --hq-confidence-weight "${CONFIDENCE_WEIGHT:-0.03}" \
  --hq-gan-weight "${GAN_WEIGHT:-0.0}" \
  --hq-gan-generator-hinge-weight "${GAN_GENERATOR_HINGE_WEIGHT:-1.0}" \
  --hq-gan-feature-matching-weight "${GAN_FEATURE_MATCHING_WEIGHT:-0.0}" \
  --hq-gan-start-step "${GAN_START_STEP:-0}" \
  --hq-gan-warmup-steps "${GAN_WARMUP_STEPS:-50}" \
  --hq-gan-learning-rate "${GAN_LEARNING_RATE:-0.0001}" \
  --hq-gan-discriminator-channels "${GAN_DISCRIMINATOR_CHANNELS:-16}" \
  --hq-gan-discriminator-architecture "${GAN_DISCRIMINATOR_ARCHITECTURE:-patch}" \
  --hq-gan-image-size "${GAN_IMAGE_SIZE:-192}" \
  --hq-gan-frame-stride "${GAN_FRAME_STRIDE:-1}" \
  --hq-gan-crop-padding "${GAN_CROP_PADDING:-16}" \
  --hq-gan-minimum-crop-size "${GAN_MINIMUM_CROP_SIZE:-96}" \
  --mosaic-block-minimum "${MOSAIC_BLOCK_MINIMUM:-6}" \
  --mosaic-block-maximum "${MOSAIC_BLOCK_MAXIMUM:-12}" \
  --perceptual-image-size "${PERCEPTUAL_IMAGE_SIZE:-224}" \
  --save-every "${SAVE_EVERY:-250}" \
  --validate-every "${VALIDATE_EVERY:-100}" \
  --validation-batches "${VALIDATION_BATCHES:-24}" \
  --log-every "${LOG_EVERY:-10}" \
  "${extra_flags[@]}"
