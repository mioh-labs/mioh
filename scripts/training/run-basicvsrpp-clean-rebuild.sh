#!/bin/zsh
set -euo pipefail

repo_root="${0:A:h:h:h}"
python_bin="${PYTHON_BIN:-$repo_root/.venv_torch213/bin/python}"
stage="${STAGE:-foundation}"
experiment_root="/Volumes/Project_HD/lada_finetune_aozora_hikari/basicvsrpp_clean_no_interlace_fc2_v3"

case "$stage" in
  foundation)
    config="$repo_root/configs/basicvsrpp/mosaic_restoration_generic_stage2.16_clean_rebuild.py"
    work_dir="$experiment_root/stage-00-generic-v1.2-full-finetune-9000-seed229883930"
    ;;
  unified-hf)
    config="$repo_root/configs/basicvsrpp/mosaic_restoration_generic_stage2.17_clean_unified_hf.py"
    work_dir="$experiment_root/stage-01-unified-hf-3000-seed20260804"
    source_checkpoint="$experiment_root/stage-00-generic-v1.2-full-finetune-9000-seed229883930/iter_9000.pth"
    initialization_checkpoint="$experiment_root/initialization/generic-v1.2-full-clean-9000-ema-as-generator.pth"
    ;;
  forward-consistency)
    config="$repo_root/configs/basicvsrpp/mosaic_restoration_generic_stage2.18_clean_forward_consistency.py"
    work_dir="$experiment_root/stage-02-forward-consistency-w005-500"
    source_checkpoint="$experiment_root/stage-01-unified-hf-3000-seed20260804/iter_3000.pth"
    initialization_checkpoint="$experiment_root/initialization/unified-hf-3000-ema-as-generator.pth"
    ;;
  *)
    echo "Unknown STAGE: $stage" >&2
    exit 2
    ;;
esac

for required in "$python_bin" "$config"; do
  if [[ ! -e "$required" ]]; then
    echo "Required input is missing: $required" >&2
    exit 2
  fi
done

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export LADA_DEFORM_CONV_BACKEND=mps_deform_conv
# The selected generic v1.2 full foundation and MMEngine resume checkpoints
# are trusted project-produced archives that predate PyTorch's weights-only
# default.  MMEngine does not expose a weights_only argument at this callsite.
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
print(f"MPS training backend: mps-deform-conv {version}")
PY

if [[ "$stage" != "foundation" && ! -f "$initialization_checkpoint" ]]; then
  if [[ ! -f "$source_checkpoint" ]]; then
    echo "Previous-stage checkpoint is missing: $source_checkpoint" >&2
    exit 2
  fi
  "$python_bin" scripts/training/prepare-basicvsrpp-ema-finetune.py \
    "$source_checkpoint" "$initialization_checkpoint" \
    --trust-checkpoint
fi

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
  "$config" --work-dir "$work_dir" "${extra_flags[@]}"
