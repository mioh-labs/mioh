#!/bin/zsh
set -euo pipefail

APP="/Applications/mioh-universal.app"
COREML=1
COREAI=1
COMPILE_COREAI=0
ARCHITECTURE="${LADA_COREAI_ARCHITECTURE:-}"

usage() {
  cat <<'EOF'
usage: convert-mioh-models.zsh [--app /Applications/mioh-universal.app]
                               [--coreml-only|--coreai-only]
                               [--compile-coreai --architecture h17s]

Exports downloaded source weights to Core ML .mlpackage and Core AI .aimodel
assets in the app's Contents/Resources/models folder.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app)
      APP="$2"
      shift 2
      ;;
    --coreml-only)
      COREML=1
      COREAI=0
      shift
      ;;
    --coreai-only)
      COREML=0
      COREAI=1
      shift
      ;;
    --compile-coreai)
      COMPILE_COREAI=1
      shift
      ;;
    --architecture)
      ARCHITECTURE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      print -u2 "unknown option: $1"
      usage >&2
      exit 2
      ;;
  esac
done

RESOURCES="$APP/Contents/Resources"
MODELS="$RESOURCES/models"
TOOLS="$RESOURCES/model-tools"
PY="$RESOURCES/runtime/bin/python3.12"
if [[ ! -x "$PY" ]]; then
  print -u2 "missing packaged Python: $PY"
  exit 1
fi
mkdir -p "$MODELS"

run_py() {
  PYTHONHOME="$RESOURCES/runtime" \
  PYTHONPATH="$RESOURCES/runtime/lib/python3.12/site-packages:$TOOLS/scripts/apple:$TOOLS/scripts" \
    "$PY" "$@"
}

if [[ "$COREML" == 1 ]]; then
  for detector in \
    lada_mosaic_detection_model_v4_fast.pt \
    lada_mosaic_detection_model_v4_accurate.pt \
    lada_mosaic_detection_model_vr_v2_accurate.pt; do
    if [[ -f "$MODELS/$detector" ]]; then
      run_py "$TOOLS/scripts/apple/export_v4_fast_coreml.py" \
        --model "$MODELS/$detector" \
        --output-dir "$MODELS" \
        --allow-overwrite
    fi
  done

  if [[ -f "$MODELS/RealESRGAN_x4plus.pth" ]]; then
    run_py "$TOOLS/scripts/apple/export_realesrgan_coreml.py" \
      --model "$MODELS/RealESRGAN_x4plus.pth" \
      --scale 4 \
      --allow-overwrite
  fi
fi

if [[ "$COREAI" == 1 ]]; then
  if [[ -f "$MODELS/lada_mosaic_restoration_model_generic_v1.2.pth" ]]; then
    for frames in 18 36 90; do
      run_py "$TOOLS/scripts/apple/export_basicvsrpp_coreai.py" \
        --model "$MODELS/lada_mosaic_restoration_model_generic_v1.2.pth" \
        --frames "$frames" \
        --output "$MODELS/basicvsrpp-v1.2-t${frames}-fp16.aimodel" \
        --skip-reference-inference \
        --allow-overwrite
    done

    run_py "$TOOLS/scripts/apple/export_basicvsrpp_variable_chunk6.py" \
      --checkpoint "$MODELS/lada_mosaic_restoration_model_generic_v1.2.pth" \
      --output-dir "$MODELS/basicvsrpp-v1.2-variable-coreai.aimodel" \
      --overwrite
  fi

  if [[ -f "$MODELS/lada_mosaic_detection_model_v4_fast.pt" ]]; then
    run_py "$TOOLS/scripts/apple/export_v4_fast_coreai.py" \
      --model "$MODELS/lada_mosaic_detection_model_v4_fast.pt" \
      --output "$MODELS/lada_mosaic_detection_model_v4_fast-fp16.aimodel" \
      --allow-overwrite
  fi

  if [[ -f "$MODELS/RealESRGAN_x2plus.pth" ]]; then
    run_py "$TOOLS/scripts/apple/export_realesrgan_coreai.py" \
      --model "$MODELS/RealESRGAN_x2plus.pth" \
      --scale 2 \
      --output "$MODELS/RealESRGAN_x2plus-256-fp16.aimodel" \
      --allow-overwrite
  fi
  if [[ -f "$MODELS/RealESRGAN_x4plus.pth" ]]; then
    run_py "$TOOLS/scripts/apple/export_realesrgan_coreai.py" \
      --model "$MODELS/RealESRGAN_x4plus.pth" \
      --scale 4 \
      --output "$MODELS/RealESRGAN_x4plus-256-fp16.aimodel" \
      --allow-overwrite
  fi
  if [[ -f "$MODELS/realesr-general-x4v3.pth" ]]; then
    run_py "$TOOLS/scripts/apple/export_srvgg_coreai.py" \
      --model "$MODELS/realesr-general-x4v3.pth" \
      --output "$MODELS/realesr-general-x4v3-256-fp16.aimodel" \
      --allow-overwrite
  fi
  if [[ -f "$MODELS/4xNomosWebPhoto_RealPLKSR.safetensors" ]]; then
    run_py "$TOOLS/scripts/apple/export_spandrel_coreai.py" \
      --model "$MODELS/4xNomosWebPhoto_RealPLKSR.safetensors" \
      --output "$MODELS/4xNomosWebPhoto_RealPLKSR-256-fp16.aimodel" \
      --allow-overwrite
  fi
fi

if [[ "$COMPILE_COREAI" == 1 ]]; then
  if [[ -z "$ARCHITECTURE" ]]; then
    print -u2 "pass --architecture h17s, or set LADA_COREAI_ARCHITECTURE"
    exit 2
  fi
  COMPILED="$MODELS/compiled-$ARCHITECTURE"
  mkdir -p "$COMPILED"
  for asset in "$MODELS"/*.aimodel(N); do
    xcrun coreai-build compile \
      "$asset" \
      --output "$COMPILED" \
      --platform macOS \
      --min-deployment-version 27.0 \
      --preferred-compute gpu \
      --architecture "$ARCHITECTURE"
  done
fi

print "Converted models in: $MODELS"
