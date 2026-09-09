#!/bin/zsh
set -euo pipefail

repo_root="${0:A:h:h:h}"
python_bin="${PYTHON_BIN:-$repo_root/.venv_torch213/bin/python}"
config="$repo_root/configs/basicvsrpp/mosaic_restoration_generic_stage2.19_clean_detail_recovery.py"
default_work_dir="/Volumes/Project_HD/lada_finetune_aozora_hikari/basicvsrpp_clean_no_interlace_fc2_v3/stage-03-detail-recovery-projreward0035-orth0015-1500-seed20260817"
work_dir="${WORK_DIR:-$default_work_dir}"
checkpoint="$repo_root/model_weights/basicvsrpp-v1.2-clean4000-unified-hf3000-forward-consistency-w005-500-raw.pth"
max_iters="${MAX_ITERS:-1500}"
val_interval="${VAL_INTERVAL:-250}"
checkpoint_interval="${CHECKPOINT_INTERVAL:-250}"
max_keep_ckpts="${MAX_KEEP_CKPTS:-13}"

for required in "$python_bin" "$config" "$checkpoint"; do
  if [[ ! -e "$required" ]]; then
    print -u2 "Required input is missing: $required"
    exit 2
  fi
done

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export LADA_DEFORM_CONV_BACKEND=mps_deform_conv
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset PYTORCH_ENABLE_MPS_FALLBACK || true

"$python_bin" - <<'PY'
import importlib.metadata
import torch

if not torch.backends.mps.is_available():
    raise SystemExit("MPS is unavailable")
version = importlib.metadata.version("mps-deform-conv")
if version != "0.2.2":
    raise SystemExit(f"Expected mps-deform-conv 0.2.2, found {version}")
print(f"MPS detail-recovery backend: mps-deform-conv {version}")
PY

extra_flags=(
  --trust-checkpoint
  --cfg-options
  "train_cfg.max_iters=$max_iters"
  "train_cfg.val_interval=$val_interval"
  "default_hooks.checkpoint.interval=$checkpoint_interval"
  "default_hooks.checkpoint.max_keep_ckpts=$max_keep_ckpts"
)
if [[ "${RESUME:-0}" == "1" ]]; then
  resume_checkpoint="${RESUME_CHECKPOINT:-}"
  if [[ -z "$resume_checkpoint" && -f "$work_dir/last_checkpoint" ]]; then
    resume_checkpoint="$(<"$work_dir/last_checkpoint")"
  fi
  if [[ -z "$resume_checkpoint" || ! -f "$resume_checkpoint" ]]; then
    print -u2 "Resume checkpoint is missing: ${resume_checkpoint:-<unset>}"
    exit 2
  fi
  extra_flags=(
    --resume
    --load-from "$resume_checkpoint"
    "${extra_flags[@]}"
  )
fi

exec caffeinate -dimsu "$python_bin" \
  scripts/training/train-mosaic-restoration-basicvsrpp.py \
  "$config" --work-dir "$work_dir" "${extra_flags[@]}"
