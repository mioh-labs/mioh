#!/bin/zsh
set -euo pipefail

APP="/Applications/mioh-universal.app"
COREML=1
COREAI=1
COMPILE_COREAI=0
ARCHITECTURE="${LADA_COREAI_ARCHITECTURE:-}"
COREAI_EXPLICIT=0

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
      COREAI_EXPLICIT=1
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

OS_MAJOR="${MIOH_OS_MAJOR:-$(sw_vers -productVersion | cut -d. -f1)}"
if (( OS_MAJOR < 27 )) && [[ "$COREAI" == 1 ]]; then
  if [[ "$COREAI_EXPLICIT" == 1 || "$COMPILE_COREAI" == 1 ]]; then
    print -u2 "Core AI model conversion requires macOS 27 or later (current: macOS $OS_MAJOR)"
    exit 2
  fi
  print "macOS $OS_MAJOR detected: Core AI conversion is unavailable; converting Core ML models only"
  COREAI=0
fi

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
  local script="$1"
  shift
  print ""
  print "======================================================================"
  print "Converting: ${script:t}"
  print "======================================================================"
  PYTHONHOME="$RESOURCES/runtime" \
  PYTHONPATH="$RESOURCES/runtime/lib/python3.12/site-packages:$TOOLS/scripts/apple:$TOOLS/scripts" \
  PYTHONUNBUFFERED=1 \
    "$PY" "$script" "$@"
}

if [[ "$COREML" == 1 ]]; then
  for detector in \
    lada_mosaic_detection_model_v2.pt \
    lada_mosaic_detection_model_v3.1_fast.pt \
    lada_mosaic_detection_model_v3.1_accurate.pt \
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
      --output-dir "$MODELS" \
      --scale 4 \
      --allow-overwrite
  fi
  if [[ -f "$MODELS/RealESRGAN_x2plus.pth" ]]; then
    run_py "$TOOLS/scripts/apple/export_realesrgan_coreml.py" \
      --model "$MODELS/RealESRGAN_x2plus.pth" \
      --output-dir "$MODELS" \
      --scale 2 \
      --allow-overwrite
  fi
  if [[ -f "$MODELS/realesr-general-x4v3.pth" ]]; then
    run_py "$TOOLS/scripts/apple/export_srvgg_coreml.py" \
      --model "$MODELS/realesr-general-x4v3.pth" \
      --output-dir "$MODELS" \
      --allow-overwrite
  fi
  if [[ -f "$MODELS/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth" \
        && -f "$MODELS/3rd_party/SwinIR/models/network_swinir.py" ]]; then
    run_py "$TOOLS/scripts/apple/export_swinir_coreml.py" \
      --model "$MODELS/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth" \
      --swinir-repo-dir "$MODELS/3rd_party/SwinIR" \
      --output-dir "$MODELS" \
      --allow-overwrite
  fi
  if [[ -f "$MODELS/4xNomosWebPhoto_RealPLKSR.safetensors" ]]; then
    run_py "$TOOLS/scripts/apple/export_spandrel_coreml.py" \
      --model "$MODELS/4xNomosWebPhoto_RealPLKSR.safetensors" \
      --output-dir "$MODELS" \
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

  for detector in \
    lada_mosaic_detection_model_v2.pt \
    lada_mosaic_detection_model_v3.1_fast.pt \
    lada_mosaic_detection_model_v3.1_accurate.pt \
    lada_mosaic_detection_model_v4_fast.pt \
    lada_mosaic_detection_model_v4_accurate.pt \
    lada_mosaic_detection_model_vr_v2_accurate.pt; do
    if [[ -f "$MODELS/$detector" ]]; then
      run_py "$TOOLS/scripts/apple/export_v4_fast_coreai.py" \
        --model "$MODELS/$detector" \
        --output "$MODELS/${detector:r}-fp16.aimodel" \
        --allow-overwrite
    fi
  done

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
  COMPILE_STAGE="$MODELS/.coreai-compile-$ARCHITECTURE"
  rm -rf "$COMPILE_STAGE"
  mkdir -p "$COMPILE_STAGE"
  for asset in "$MODELS"/*.aimodel(N); do
    if [[ "${asset:t}" == "basicvsrpp-v1.2-variable-coreai.aimodel" ]]; then
      continue
    fi
    xcrun coreai-build compile \
      "$asset" \
      --output "$MODELS" \
      --platform macOS \
      --min-deployment-version 27.0 \
      --preferred-compute gpu \
      --architecture "$ARCHITECTURE"
  done

  VARIABLE_SOURCE="$MODELS/basicvsrpp-v1.2-variable-coreai.aimodel"
  if [[ -d "$VARIABLE_SOURCE" ]]; then
    VARIABLE_COMPILED="$MODELS/basicvsrpp-v1.2-variable-coreai.$ARCHITECTURE.aimodelc"
    rm -rf "$VARIABLE_COMPILED"
    mkdir -p "$VARIABLE_COMPILED"
    for asset in "$VARIABLE_SOURCE"/*.aimodel(N); do
      rm -rf "$COMPILE_STAGE"/*(N)
      xcrun coreai-build compile \
        "$asset" \
        --output "$COMPILE_STAGE" \
        --platform macOS \
        --min-deployment-version 27.0 \
        --preferred-compute gpu \
        --architecture "$ARCHITECTURE"
      compiled=("$COMPILE_STAGE"/${asset:r:t}.*.aimodelc(N))
      if (( ${#compiled[@]} != 1 )); then
        print -u2 "Expected one compiled asset for ${asset:t}; found ${#compiled[@]}"
        exit 1
      fi
      ditto "$compiled[1]" "$VARIABLE_COMPILED/${compiled[1]:t}"
    done
  fi
  rm -rf "$COMPILE_STAGE"
fi

if [[ "$COREAI" == 1 \
      && -f "$MODELS/lada_mosaic_restoration_model_generic_v1.2.pth" ]]; then
  VARIABLE_SOURCE="$MODELS/basicvsrpp-v1.2-variable-coreai.aimodel"
  if [[ ! -d "$VARIABLE_SOURCE" ]]; then
    print -u2 "Missing converted variable restoration collection: $VARIABLE_SOURCE"
    exit 1
  fi
  VARIABLE_ASSET_COUNT=$(find "$VARIABLE_SOURCE" -maxdepth 1 -type d -name '*.aimodel' | wc -l | tr -d ' ')
  if [[ "$VARIABLE_ASSET_COUNT" != 11 ]]; then
    print -u2 "Variable restoration collection is incomplete: expected 11 assets, found $VARIABLE_ASSET_COUNT"
    exit 1
  fi
fi

print "Converted models in: $MODELS"
