#!/bin/zsh
set -euo pipefail

# Full from-scratch V5-HQ curriculum on the source-balanced native V3 dataset.
# Existing V5/V5-HQ runs use different work roots and are never overwritten.

repo_root="${0:A:h:h:h}"
dataset_root="${DATASET_ROOT:-/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v5_balanced_v3}"
export VARIANT=hq
export TRAIN_MANIFEST="${TRAIN_MANIFEST:-$dataset_root/manifests/train-native-balanced.jsonl}"
export VALIDATION_MANIFEST="${VALIDATION_MANIFEST:-$dataset_root/manifests/validation-native-balanced.jsonl}"
export WORK_ROOT="${WORK_ROOT:-/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v5_hq/runs/balanced-native-v3-hq-384}"
export BASICVSRPP_CHECKPOINT="${BASICVSRPP_CHECKPOINT:-$repo_root/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth}"
export START_STAGE="${START_STAGE:-1}"
export END_STAGE="${END_STAGE:-6}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export ACCUMULATE="${ACCUMULATE:-1}"
export WORKERS="${WORKERS:-0}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export LADA_DEFORM_CONV_BACKEND=mps_deform_conv

for required in "$TRAIN_MANIFEST" "$VALIDATION_MANIFEST" "$BASICVSRPP_CHECKPOINT"; do
  if [[ ! -f "$required" ]]; then
    echo "Required training input is missing: $required" >&2
    exit 2
  fi
done

exec zsh "$repo_root/scripts/training/run-mioh-restorer-v5-local.sh"
