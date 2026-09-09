#!/bin/zsh
set -euo pipefail

# Reproduce Stage A of the full BasicVSR++ reactivation experiment.  The
# torchvision MPS operator currently implements only the forward pass, so the
# training run explicitly selects mioh's native Metal forward/backward DCNv2
# extension instead of silently falling back to CPU during backpropagation.

repo_root="${0:A:h:h:h}"
python_bin="${PYTHON_BIN:-$repo_root/.venv_torch213/bin/python}"
config="${CONFIG:-$repo_root/configs/basicvsrpp/mosaic_restoration_generic_stage2.14_full_gan_reactivation.py}"
work_dir="${WORK_DIR:-/Volumes/Project_HD/lada_finetune_aozora_hikari/basicvsrpp_full_gan_reactivation/stage-a-v1-seed20260814}"

for required in "$python_bin" "$config"; do
  if [[ ! -e "$required" ]]; then
    echo "Required input is missing: $required" >&2
    exit 2
  fi
done

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export LADA_DEFORM_CONV_BACKEND=mps_deform_conv

# Default to a strict native-MPS run.  This makes a newly unsupported operator
# fail loudly instead of changing the experiment's performance characteristics.
if [[ "${ALLOW_MPS_FALLBACK:-0}" == "1" ]]; then
  export PYTORCH_ENABLE_MPS_FALLBACK=1
else
  unset PYTORCH_ENABLE_MPS_FALLBACK || true
fi

"$python_bin" - <<'PY'
import importlib.metadata
import torch

if not torch.backends.mps.is_available():
    raise SystemExit("MPS is unavailable")
version = importlib.metadata.version("mps-deform-conv")
if version != "0.2.2":
    raise SystemExit(f"Expected mps-deform-conv 0.2.2, found {version}")
print(f"MPS training backend: mps-deform-conv {version} (native forward/backward)")
PY

extra_flags=()
if [[ "${RESUME:-0}" == "1" ]]; then
  resume_checkpoint="${RESUME_CHECKPOINT:-}"
  if [[ -z "$resume_checkpoint" && -f "$work_dir/last_checkpoint" ]]; then
    resume_checkpoint="$(<"$work_dir/last_checkpoint")"
  fi
  if [[ -z "$resume_checkpoint" || ! -f "$resume_checkpoint" ]]; then
    echo "Resume checkpoint is missing: ${resume_checkpoint:-<unset>}" >&2
    exit 2
  fi
  extra_flags+=(
    --resume
    --load-from "$resume_checkpoint"
    --trust-checkpoint
  )
fi

cfg_options=()
if [[ -n "${MAX_ITERS:-}" ]]; then
  cfg_options+=("train_cfg.max_iters=${MAX_ITERS}")
fi
if [[ -n "${VAL_INTERVAL:-}" ]]; then
  cfg_options+=("train_cfg.val_interval=${VAL_INTERVAL}")
fi
if [[ -n "${CHECKPOINT_INTERVAL:-}" ]]; then
  cfg_options+=("default_hooks.checkpoint.interval=${CHECKPOINT_INTERVAL}")
fi
if [[ -n "${MAX_KEEP_CKPTS:-}" ]]; then
  cfg_options+=("default_hooks.checkpoint.max_keep_ckpts=${MAX_KEEP_CKPTS}")
fi
if (( ${#cfg_options[@]} > 0 )); then
  extra_flags+=(--cfg-options "${cfg_options[@]}")
fi

exec caffeinate -dimsu "$python_bin" \
  scripts/training/train-mosaic-restoration-basicvsrpp.py \
  "$config" \
  --work-dir "$work_dir" \
  "${extra_flags[@]}"
