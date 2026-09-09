#!/bin/zsh
set -euo pipefail

# Signal-selection pilot following the amplitude ablation.  The deployed
# BasicVSR++ backbone remains frozen, amplitude/global-correlation rewards are
# removed, and the V5-HQ refiner learns native-pixel local NCC while the clean
# compositor guard ring is explicitly held to its identity target.

script_dir="${0:A:h}"
export WORK_ROOT="${WORK_ROOT:-/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v5_hq_ablation/hf-v8-raw-temporal-residual-reg-100}"
export STEPS="${STEPS:-100}"
export RESTART="${RESTART:-0}"
export EARLY_STOP=0
export HF_AMPLITUDE_WEIGHT=0
export HF_CORRELATION_WEIGHT=0
export HF_LOCAL_CORRELATION_WEIGHT="${HF_LOCAL_CORRELATION_WEIGHT:-0.005}"
export HF_LOCAL_CORRELATION_PATCH_SIZE="${HF_LOCAL_CORRELATION_PATCH_SIZE:-32}"
export HF_LOCAL_CORRELATION_ON_RESIDUAL=1
export HF_RESIDUAL_RECONSTRUCTION_WEIGHT="${HF_RESIDUAL_RECONSTRUCTION_WEIGHT:-0.05}"
export GUARD_RING_IDENTITY_WEIGHT="${GUARD_RING_IDENTITY_WEIGHT:-2.0}"
export FREEZE_BASE_HEAD=1
export FREEZE_CONFIDENCE_HEAD=1
export RAW_TEMPORAL_CANDIDATES=1
export CONFIDENCE_WEIGHT=0
export SAVE_EVERY="${SAVE_EVERY:-50}"
export VALIDATE_EVERY="${VALIDATE_EVERY:-50}"
export VALIDATION_BATCHES="${VALIDATION_BATCHES:-24}"

exec "$script_dir/run-mioh-restorer-v5-hq-hf-ablation.sh"
