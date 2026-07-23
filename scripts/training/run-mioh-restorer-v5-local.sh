#!/bin/zsh
set -euo pipefail

repo_root="${0:A:h:h:h}"
python_bin="${PYTHON_BIN:-/Users/okatti/.pyenv/versions/lada/bin/python}"
train_manifest="${TRAIN_MANIFEST:-}"
validation_manifest="${VALIDATION_MANIFEST:-}"
work_root="${WORK_ROOT:-/Volumes/Project_HD/lada_finetune_aozora_hikari/mioh_restorer_v5/runs/v5-q-native}"
start_stage="${START_STAGE:-1}"
end_stage="${END_STAGE:-6}"
variant="${VARIANT:-q}"
basicvsrpp_checkpoint="${BASICVSRPP_CHECKPOINT:-$repo_root/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth}"
batch_size="${BATCH_SIZE:-1}"
accumulate="${ACCUMULATE:-4}"
# macOS/PyTorch can fail in torch_shm_manager when DataLoader workers try to
# create shared storage, especially when TMPDIR is on an external volume.
# Decoding in the training process is slower but reliable and avoids the same
# multiprocessing failure seen in the packaged GUI runtime.
workers="${WORKERS:-0}"

if [[ -z "$train_manifest" || -z "$validation_manifest" ]]; then
  echo "Set TRAIN_MANIFEST and VALIDATION_MANIFEST to V5 native JSONL files." >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  echo "Python not found: $python_bin" >&2
  exit 2
fi
if (( start_stage < 1 || end_stage > 6 || start_stage > end_stage )); then
  echo "START_STAGE/END_STAGE must describe a range within 1...6." >&2
  exit 2
fi

cd "$repo_root"
mkdir -p "$work_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
if [[ "$variant" == "hq" ]]; then
  export LADA_DEFORM_CONV_BACKEND="${LADA_DEFORM_CONV_BACKEND:-mps_deform_conv}"
fi

echo "MiohRestorer V5 native six-stage training"
echo "Device: mps"
echo "Variant: $variant"
echo "Stages: $start_stage...$end_stage"
echo "Work root: $work_root"
if [[ "$variant" == "hq" ]]; then
  echo "Quality-first hybrid: BasicVSR++ recurrent initialization + V5-HQ refiner"
  echo "BasicVSR++ initialization: $basicvsrpp_checkpoint"
  echo "MPS grid-sample backward uses the documented CPU fallback."
else
  echo "No V3/V4/BasicVSR++ checkpoint or external flow model is used."
fi

for stage in {$start_stage..$end_stage}; do
  command=(
    "$python_bin" -u scripts/training/train-mioh-restorer-v5.py
    --train-manifest "$train_manifest"
    --validation-manifest "$validation_manifest"
    --work-root "$work_root"
    --variant "$variant"
    --stage "$stage"
    --device mps
    --amp off
    --batch-size "$batch_size"
    --accumulate "$accumulate"
    --workers "$workers"
  )
  if [[ "$variant" == "hq" ]]; then
    command+=(--basicvsrpp-checkpoint "$basicvsrpp_checkpoint")
  fi
  if [[ "${RESTART_STAGE:-0}" == "1" ]]; then
    command+=(--restart-stage)
  fi
  if [[ -n "${STEPS:-}" ]]; then
    command+=(--steps "$STEPS")
  fi
  echo "Starting V5 Stage $stage"
  caffeinate -dimsu "${command[@]}"
done
