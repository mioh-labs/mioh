#!/bin/zsh
set -euo pipefail

repo_root="${0:A:h:h:h}"
export VARIANT=hq
export WORK_ROOT="${WORK_ROOT:-/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v5_hq/runs/hybrid-native-256}"
export BASICVSRPP_CHECKPOINT="${BASICVSRPP_CHECKPOINT:-$repo_root/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export ACCUMULATE="${ACCUMULATE:-1}"
export WORKERS="${WORKERS:-0}"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export LADA_DEFORM_CONV_BACKEND=mps_deform_conv

exec zsh "$repo_root/scripts/training/run-mioh-restorer-v5-local.sh"
