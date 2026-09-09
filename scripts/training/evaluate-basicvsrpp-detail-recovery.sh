#!/bin/zsh
set -euo pipefail

repo_root="${0:A:h:h:h}"
python_bin="${PYTHON_BIN:-$repo_root/.venv_torch213/bin/python}"
run_dir="${RUN_DIR:-/Volumes/Project_HD/lada_finetune_aozora_hikari/basicvsrpp_clean_no_interlace_fc2_v3/stage-03-detail-recovery-projreward0035-orth0015-1500-seed20260817}"
output_dir="${OUTPUT_DIR:-$repo_root/output/evaluations/basicvsrpp-detail-recovery-projreward0035-orth0015-1500-20260816}"
config="$repo_root/configs/basicvsrpp/mosaic_restoration_generic_stage2.19_clean_detail_recovery.py"
manifest="/Volumes/Project_HD/lada_finetune_aozora_hikari/basicvsrpp_clean_no_interlace_fc2_v3/manifests/validation-native-hf-clean-v1.jsonl"
adopted="$repo_root/model_weights/basicvsrpp-v1.2-clean4000-unified-hf3000-forward-consistency-w005-500-raw.pth"

checkpoints=("adopted@raw=$adopted")
for step in 250 500 750 1000 1250 1500; do
  checkpoint="$run_dir/iter_${step}.pth"
  if [[ ! -f "$checkpoint" ]]; then
    print -u2 "Detail-recovery checkpoint is missing: $checkpoint"
    exit 2
  fi
  checkpoints+=("step${step}@raw=$checkpoint")
done

args=()
for checkpoint in "${checkpoints[@]}"; do
  args+=(--checkpoint "$checkpoint")
done

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export LADA_DEFORM_CONV_BACKEND=mps_deform_conv
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset PYTORCH_ENABLE_MPS_FALLBACK || true

exec "$python_bin" scripts/training/evaluate-basicvsrpp-hf-checkpoints.py \
  --config "$config" \
  --manifest "$manifest" \
  "${args[@]}" \
  --output-dir "$output_dir" \
  --device mps \
  --skip-videos \
  --overwrite \
  --trust-checkpoint
