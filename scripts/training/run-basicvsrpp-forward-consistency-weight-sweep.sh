#!/usr/bin/env bash
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/okatti/.pyenv/versions/lada/bin/python}"
CONFIG="${CONFIG:-configs/basicvsrpp/mosaic_restoration_generic_stage2.11_forward_consistency.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/Volumes/Project_HD/lada_finetune_aozora_hikari/forward_consistency_weight_sweep}"
WEIGHTS="${WEIGHTS:-0.05 0.10 0.20 0.40}"
SEED="${SEED:-20260808}"

for WEIGHT in ${WEIGHTS}; do
  LABEL="${WEIGHT/./}"
  LABEL="${LABEL#0}"
  WORK_DIR="${OUTPUT_ROOT}/run-w${LABEL}-seed${SEED}"
  echo "==> forward consistency weight ${WEIGHT}: ${WORK_DIR}"
  PYTORCH_ENABLE_MPS_FALLBACK=1 \
  LADA_DEFORM_CONV_BACKEND=mps_deform_conv \
  "${PYTHON_BIN}" scripts/training/train-mosaic-restoration-basicvsrpp.py \
    "${CONFIG}" \
    --work-dir "${WORK_DIR}" \
    --trust-checkpoint \
    --cfg-options \
      randomness.deterministic=False \
      randomness.seed="${SEED}" \
      train_dataloader.dataset.seed="${SEED}" \
      model.mosaic_forward_consistency_loss.loss_weight="${WEIGHT}"
done
