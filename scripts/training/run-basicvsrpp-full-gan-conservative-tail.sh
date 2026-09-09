#!/bin/zsh
set -euo pipefail

# Reproduce Stage B: a non-GAN, tail-only fidelity pass initialized from the
# Stage A EMA generator.  The native Metal deformable convolution is still
# selected for the frozen backbone forward path; no CPU fallback is allowed.

repo_root="${0:A:h:h:h}"
python_bin="${PYTHON_BIN:-$repo_root/.venv_torch213/bin/python}"
config="${CONFIG:-$repo_root/configs/basicvsrpp/mosaic_restoration_generic_stage2.15_full_gan_conservative_tail.py}"
work_dir="${WORK_DIR:-/Volumes/Project_HD/lada_finetune_aozora_hikari/basicvsrpp_full_gan_reactivation/stage-b-conservative-v1-seed20260814}"

for required in "$python_bin" "$config"; do
  if [[ ! -e "$required" ]]; then
    echo "Required input is missing: $required" >&2
    exit 2
  fi
done

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export LADA_DEFORM_CONV_BACKEND=mps_deform_conv
unset PYTORCH_ENABLE_MPS_FALLBACK || true

extra_flags=()
if [[ "${RESUME:-0}" == "1" ]]; then
  extra_flags+=(--resume)
fi
if [[ -n "${MAX_ITERS:-}" ]]; then
  extra_flags+=(--cfg-options "train_cfg.max_iters=${MAX_ITERS}")
fi

exec caffeinate -dimsu "$python_bin" \
  scripts/training/train-mosaic-restoration-basicvsrpp.py \
  "$config" \
  --work-dir "$work_dir" \
  "${extra_flags[@]}"
