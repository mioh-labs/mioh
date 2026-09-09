#!/bin/zsh
set -euo pipefail

# Clean, reproducible 10k-step full-generator run.  Unlike the historical
# full-GAN stage, this arm keeps the promoted model's ROI/HF/temporal fidelity
# constraints active from step zero.

repo_root="${0:A:h:h:h}"
python_bin="${PYTHON_BIN:-$repo_root/.venv_torch213/bin/python}"
config="${CONFIG:-$repo_root/configs/basicvsrpp/mosaic_restoration_generic_stage2.14_full_gan_temporal_fidelity.py}"
work_dir="${WORK_DIR:-/Volumes/Project_HD/lada_finetune_aozora_hikari/basicvsrpp_full_gan_reactivation/stage-a-temporal-fidelity-v1-seed20260814}"

for required in "$python_bin" "$config"; do
  if [[ ! -e "$required" ]]; then
    echo "Required input is missing: $required" >&2
    exit 2
  fi
done

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export LADA_DEFORM_CONV_BACKEND=mps_deform_conv

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
  extra_flags+=(--resume --load-from "$resume_checkpoint" --trust-checkpoint)
fi

exec caffeinate -dimsu "$python_bin" \
  scripts/training/train-mosaic-restoration-basicvsrpp.py \
  "$config" \
  --work-dir "$work_dir" \
  "${extra_flags[@]}"
