#!/bin/zsh
set -euo pipefail

repo_root="${0:A:h:h:h}"
python_bin="${PYTHON_BIN:-$repo_root/.venv_torch213/bin/python}"
phase_a_config="$repo_root/configs/basicvsrpp/mosaic_restoration_generic_stage2.21_large_roi_full_trunk.py"
phase_b_config="$repo_root/configs/basicvsrpp/mosaic_restoration_generic_stage2.22_large_roi_tail_consolidation.py"
experiment_root="/Volumes/Project_HD/lada_finetune_aozora_hikari/basicvsrpp_clean_no_interlace_fc2_v3"
phase_a_work_dir="${PHASE_A_WORK_DIR:-$experiment_root/stage-05-large-roi-full/phase-a-full-trunk}"
phase_b_work_dir="${PHASE_B_WORK_DIR:-$experiment_root/stage-05-large-roi-full/phase-b-tail-consolidation}"
action="${ACTION:-preflight}"

required_paths=(
  "$python_bin"
  "$phase_a_config"
  "$phase_b_config"
  "$repo_root/model_weights/basicvsrpp-v1.2-large-roi-native-tiles-27000-ema.pth"
  "$experiment_root/manifests/train-large-roi-native-tiles-v1.jsonl"
  "$experiment_root/manifests/validation-large-roi-native-tiles-v1.jsonl"
  "$experiment_root/manifests/validation-native-hf-clean-v1.jsonl"
)
for required in "${required_paths[@]}"; do
  if [[ ! -e "$required" ]]; then
    print -u2 "Required large-ROI input is missing: $required"
    exit 2
  fi
done

cd "$repo_root"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export LADA_DEFORM_CONV_BACKEND=mps_deform_conv
export LADA_MPS_GRID_SAMPLE_BACKWARD=1
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
unset PYTORCH_ENABLE_MPS_FALLBACK || true

run_preflight() {
  "$python_bin" - \
    "$phase_a_config" \
    "$phase_b_config" \
    "$repo_root/model_weights/basicvsrpp-v1.2-large-roi-native-tiles-27000-ema.pth" \
    "$experiment_root/manifests/train-large-roi-native-tiles-v1.jsonl" \
    "$experiment_root/manifests/validation-large-roi-native-tiles-v1.jsonl" \
    "$experiment_root/manifests/validation-native-hf-clean-v1.jsonl" <<'PY'
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import torch
from mmengine.config import Config

from lada.utils.mps_grid_sample_backward import (
    enable_native_mps_grid_sample_backward,
)

(
    phase_a_path,
    phase_b_path,
    checkpoint_path,
    train_path,
    large_val_path,
    generic_val_path,
) = map(Path, sys.argv[1:])


def read_rows(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if "start612" in text.lower() or "start-612" in text.lower():
        raise SystemExit(f"START-612 data leaked into the generic run: {path}")
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise SystemExit(f"manifest is empty: {path}")
    return rows


def source_ids(rows: list[dict]) -> set[str]:
    return {
        str(row.get("source_video_id") or row["target_video"])
        for row in rows
    }


train = read_rows(train_path)
large_val = read_rows(large_val_path)
generic_val = read_rows(generic_val_path)
train_sources = source_ids(train)
large_val_sources = source_ids(large_val)
generic_val_sources = source_ids(generic_val)
if train_sources & (large_val_sources | generic_val_sources):
    raise SystemExit("large-ROI train/validation source leakage detected")

missing_assets = sorted(
    {
        str(Path(row[key]))
        for rows in (train, large_val, generic_val)
        for row in rows
        for key in ("target_video", "mask_video")
        if not Path(row[key]).is_file()
    }
)
if missing_assets:
    raise SystemExit(
        "large-ROI manifest assets are missing; first item: " + missing_assets[0]
    )

digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
expected_digest = "bd7585c0643d86be931a256c210f67085ad8e7e88ef70b1d9c700b31f1b3a18b"
if digest != expected_digest:
    raise SystemExit(
        f"adopted large-ROI checkpoint SHA-256 mismatch: {digest} != {expected_digest}"
    )

phase_a = Config.fromfile(phase_a_path)
phase_b = Config.fromfile(phase_b_path)
for path, config in ((phase_a_path, phase_a), (phase_b_path, phase_b)):
    text = path.read_text(encoding="utf-8").lower()
    if "start612" in text or "start-612" in text:
        raise SystemExit(f"START-612 reference found in training config: {path}")
    if Path(config.train_manifest) != train_path:
        raise SystemExit(f"unexpected training manifest in {path}")

if "trainable_modules" in phase_a.model.generator:
    raise SystemExit("phase A must train the full restoration trunk")
expected_tail = {"reconstruction", "upsample1", "upsample2", "conv_hr", "conv_last"}
if set(phase_b.model.generator.trainable_modules) != expected_tail:
    raise SystemExit("phase B tail module contract changed")
if int(phase_a.train_cfg.max_iters) != 20_000:
    raise SystemExit("phase A must retain the reviewed 20,000-step budget")
if int(phase_b.train_cfg.max_iters) != 10_000:
    raise SystemExit("phase B must retain the reviewed 10,000-step budget")

if not torch.backends.mps.is_available():
    raise SystemExit("MPS is unavailable")
deform_version = importlib.metadata.version("mps-deform-conv")
if deform_version != "0.2.2":
    raise SystemExit(
        f"Expected mps-deform-conv 0.2.2, found {deform_version}"
    )
if not enable_native_mps_grid_sample_backward(raise_on_error=True):
    raise SystemExit("native MPS grid_sample backward is unavailable")

forced_tiles = sum("forced_final_crop_offset" in row for row in train)
anchor_entries = len(train) - forced_tiles
summary = {
    "checkpoint_sha256": digest,
    "train_entries": len(train),
    "train_source_videos": len(train_sources),
    "native_large_roi_tiles": forced_tiles,
    "ordinary_roi_anchors": anchor_entries,
    "large_roi_validation_entries": len(large_val),
    "generic_validation_entries": len(generic_val),
    "phase_a_steps": int(phase_a.train_cfg.max_iters),
    "phase_b_steps": int(phase_b.train_cfg.max_iters),
    "mps_deform_conv": deform_version,
    "native_grid_sample_backward": True,
    "start612_references": 0,
}
print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
PY
}

train_phase() {
  local config="$1"
  local work_dir="$2"
  local max_iters="$3"
  local val_interval="$4"
  local checkpoint_interval="$5"
  local load_checkpoint="${6:-}"

  if [[ "${CONFIRM_TRAINING:-0}" != "1" ]]; then
    print -u2 "Training is armed but not started. Set CONFIRM_TRAINING=1."
    exit 2
  fi

  local flags=(
    --trust-checkpoint
    --work-dir "$work_dir"
    --cfg-options
    "train_cfg.max_iters=$max_iters"
    "train_cfg.val_interval=$val_interval"
    "default_hooks.checkpoint.interval=$checkpoint_interval"
  )
  if [[ "${RESUME:-0}" == "1" ]]; then
    local resume_checkpoint="${RESUME_CHECKPOINT:-}"
    if [[ -z "$resume_checkpoint" && -f "$work_dir/last_checkpoint" ]]; then
      resume_checkpoint="$(<"$work_dir/last_checkpoint")"
    fi
    if [[ -z "$resume_checkpoint" || ! -f "$resume_checkpoint" ]]; then
      print -u2 "Resume checkpoint is missing: ${resume_checkpoint:-<unset>}"
      exit 2
    fi
    flags=(--resume --load-from "$resume_checkpoint" "${flags[@]}")
  elif [[ -n "$load_checkpoint" ]]; then
    if [[ ! -f "$load_checkpoint" ]]; then
      print -u2 "Initialization checkpoint is missing: $load_checkpoint"
      exit 2
    fi
    flags=(--load-from "$load_checkpoint" "${flags[@]}")
  fi

  exec caffeinate -dimsu "$python_bin" \
    scripts/training/train-mosaic-restoration-basicvsrpp.py \
    "$config" "${flags[@]}"
}

case "$action" in
  preflight)
    run_preflight
    print "Preflight passed. No training was started."
    print "Phase A: ACTION=phase-a CONFIRM_TRAINING=1 $0"
    print "Phase B: ACTION=phase-b CONFIRM_TRAINING=1 PHASE_A_CHECKPOINT=/absolute/path/to/checkpoint.pth $0"
    ;;
  phase-a)
    run_preflight
    train_phase \
      "$phase_a_config" "$phase_a_work_dir" \
      "${MAX_ITERS:-20000}" "${VAL_INTERVAL:-1000}" \
      "${CHECKPOINT_INTERVAL:-500}"
    ;;
  phase-b)
    run_preflight
    phase_a_checkpoint="${PHASE_A_CHECKPOINT:-}"
    if [[ -z "$phase_a_checkpoint" ]]; then
      print -u2 "PHASE_A_CHECKPOINT must name the accepted phase-A raw or EMA checkpoint."
      exit 2
    fi
    train_phase \
      "$phase_b_config" "$phase_b_work_dir" \
      "${MAX_ITERS:-10000}" "${VAL_INTERVAL:-500}" \
      "${CHECKPOINT_INTERVAL:-250}" "$phase_a_checkpoint"
    ;;
  *)
    print -u2 "Unsupported ACTION: $action (use preflight, phase-a, or phase-b)"
    exit 2
    ;;
esac
