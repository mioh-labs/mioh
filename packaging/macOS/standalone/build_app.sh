#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h:h:h}"
PACKAGE_DIR="$ROOT/packaging/macos/standalone"
BUILD_DIR="${BUILD_DIR:-$ROOT/build/macos-standalone}"
COREAI_DISTRIBUTION="${COREAI_DISTRIBUTION:-dedicated}"
APP_BASENAME="${APP_BASENAME:-mioh}"
DMG_BASENAME="${DMG_BASENAME:-mioh-0.14.3-unsigned}"
case "$COREAI_DISTRIBUTION" in
  dedicated|portable) ;;
  *)
    print -u2 "Unsupported COREAI_DISTRIBUTION: $COREAI_DISTRIBUTION"
    exit 2
    ;;
esac
APP="$BUILD_DIR/$APP_BASENAME.app"
DMG="$BUILD_DIR/$DMG_BASENAME.dmg"
INCLUDE_USER_MANUAL="${INCLUDE_USER_MANUAL:-0}"
MIOH_MODELESS_DISTRIBUTION="${MIOH_MODELESS_DISTRIBUTION:-0}"
USER_MANUAL_PDF="${USER_MANUAL_PDF:-$ROOT/output/pdf/mioh-user-manual-ja.pdf}"
CONTENTS="$APP/Contents"
RESOURCES="$CONTENTS/Resources"
LADA_STANDALONE_PYTHON_ENV="${LADA_STANDALONE_PYTHON_ENV:-${LADA_STANDALONE_VENV:-$ROOT/.venv-coreai}}"
LADA_STANDALONE_PYTHON_ENV="${LADA_STANDALONE_PYTHON_ENV:A}"
# The dedicated build targets Core AI hardware and runs entirely inside the
# Swift pipeline, so it bundles no interpreter. The portable/universal build
# still ships the Python restoration and preview path for machines without a
# usable Core AI restorer, so it carries a full runtime in Resources/runtime.
if [[ "$COREAI_DISTRIBUTION" == "portable" ]]; then
  MIOH_BUNDLE_PYTHON_RUNTIME="${MIOH_BUNDLE_PYTHON_RUNTIME:-1}"
else
  MIOH_BUNDLE_PYTHON_RUNTIME="${MIOH_BUNDLE_PYTHON_RUNTIME:-0}"
fi
if [[ "$MIOH_BUNDLE_PYTHON_RUNTIME" == 1 || "$MIOH_MODELESS_DISTRIBUTION" != 1 ]]; then
  if [[ ! -x "$LADA_STANDALONE_PYTHON_ENV/bin/python" ]]; then
    print -u2 "Missing build-time Python: $LADA_STANDALONE_PYTHON_ENV/bin/python"
    print -u2 "Set LADA_STANDALONE_PYTHON_ENV to the environment to package."
    exit 1
  fi
fi
PYTHON_SOURCE="${PYTHON_SOURCE:-$HOME/.local/share/uv/python/cpython-3.12-macos-aarch64-none}"
PYTHON_SOURCE="${PYTHON_SOURCE:A}"
SITE_PACKAGES="${SITE_PACKAGES:-$LADA_STANDALONE_PYTHON_ENV/lib/python3.12/site-packages}"
if [[ "$MIOH_BUNDLE_PYTHON_RUNTIME" == 1 ]]; then
  if [[ ! -d "$PYTHON_SOURCE" ]]; then
    print -u2 "Missing interpreter to bundle: $PYTHON_SOURCE"
    print -u2 "Set PYTHON_SOURCE, or install it with: uv python install 3.12"
    exit 1
  fi
  if [[ ! -d "$SITE_PACKAGES" ]]; then
    print -u2 "Missing site-packages for standalone build: $SITE_PACKAGES"
    print -u2 "Set LADA_STANDALONE_PYTHON_ENV to the single Python environment to package."
    exit 1
  fi
fi
COMPILED_MODELS="${COMPILED_MODELS:-$BUILD_DIR/compiled-models}"
COREAI_ARCHITECTURE="${COREAI_ARCHITECTURE:-h17s}"
COMPILED_COREML_MODELS="${COMPILED_COREML_MODELS:-$BUILD_DIR/compiled-coreml-models}"
FFMPEG_CACHE="${FFMPEG_CACHE:-$BUILD_DIR/ffmpeg-static}"
VENDORED_MPS_DEFORM_CONV="$PACKAGE_DIR/vendor/mps-deform-conv-0.2.2"
MPS_DEFORM_BUILD_SOURCE="$BUILD_DIR/mps-deform-conv-source"
PREVIEW_ENCODER_TARGET="arm64-apple-macosx26.0"

rm -rf "$APP" "$BUILD_DIR/Lada.app"
rm -f "$DMG" "$BUILD_DIR/Lada-0.11.0-unsigned.dmg" "$BUILD_DIR/mioh-0.11.0-unsigned.dmg"
mkdir -p "$CONTENTS/MacOS" "$RESOURCES/bin" "$RESOURCES/models"

typeset -a APP_SWIFT_FLAGS
APP_SWIFT_FLAGS=()
if [[ "$COREAI_DISTRIBUTION" == "portable" ]]; then
  APP_SWIFT_FLAGS+=(-D MIOH_PORTABLE_COREAI)
else
  APP_SWIFT_FLAGS+=(-D MIOH_DEDICATED_VARIABLE_HQ)
fi
typeset -a SWIFT_SUBPROCESS_FLAGS
SWIFT_SUBPROCESS_FLAGS=()
if [[ "${MIOH_DISABLE_SWIFT_SANDBOX:-0}" == "1" ]]; then
  SWIFT_SUBPROCESS_FLAGS+=(-disable-sandbox)
fi

xcrun swiftc \
  "${SWIFT_SUBPROCESS_FLAGS[@]}" \
  -O \
  -parse-as-library \
  -target arm64-apple-macosx26.0 \
  -framework AppKit \
  -framework SwiftUI \
  -framework AVFoundation \
  -framework AVKit \
  -framework Metal \
  -framework Network \
  -framework SceneKit \
  "${APP_SWIFT_FLAGS[@]}" \
  "$PACKAGE_DIR/MiohApp.swift" \
  "$PACKAGE_DIR/RealtimePlayer.swift" \
  -o "$CONTENTS/MacOS/mioh"
xcrun swiftc \
  "${SWIFT_SUBPROCESS_FLAGS[@]}" \
  -O \
  -parse-as-library \
  -target "$PREVIEW_ENCODER_TARGET" \
  -framework Accelerate \
  -framework AVFoundation \
  -framework CoreVideo \
  -framework VideoToolbox \
  "$PACKAGE_DIR/PreviewVideoToolboxEncoder.swift" \
  -o "$RESOURCES/bin/mioh-preview-videotoolbox-encoder"
xcrun swiftc \
  "${SWIFT_SUBPROCESS_FLAGS[@]}" \
  -O \
  -parse-as-library \
  -D MIOH_NATIVE_PREVIEW_PIPELINE \
  "${APP_SWIFT_FLAGS[@]}" \
  -target arm64-apple-macosx27.0 \
  -framework Accelerate \
  -framework AVFoundation \
  -framework CoreAI \
  -framework CoreImage \
  -framework CoreML \
  -framework CoreVideo \
  -framework VideoToolbox \
  "$PACKAGE_DIR/PreviewVideoToolboxEncoder.swift" \
  "$PACKAGE_DIR/NativePreviewPipeline.swift" \
  -o "$RESOURCES/bin/mioh-native-coreai-preview"
xcrun swiftc \
  "${SWIFT_SUBPROCESS_FLAGS[@]}" \
  -O \
  -parse-as-library \
  -target arm64-apple-macosx27.0 \
  -framework CoreAI \
  "$PACKAGE_DIR/CoreAIRunner.swift" \
  -o "$RESOURCES/bin/lada-coreai-runner"
VARIABLE_RUNNER_SOURCE="$PACKAGE_DIR/VariableBasicVSRPPChunk6Runner.swift"
xcrun swiftc \
  "${SWIFT_SUBPROCESS_FLAGS[@]}" \
  -O \
  -parse-as-library \
  -target arm64-apple-macosx27.0 \
  -framework CoreAI \
  -framework Metal \
  "$VARIABLE_RUNNER_SOURCE" \
  -o "$RESOURCES/bin/lada-basicvsrpp-variable-runner"
if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
  xcrun swiftc \
    "${SWIFT_SUBPROCESS_FLAGS[@]}" \
    -O \
    -parse-as-library \
    -target arm64-apple-macosx27.0 \
    -framework CoreAI \
    -framework Metal \
    "$PACKAGE_DIR/VariableBasicVSRPPRunner.swift" \
    -o "$RESOURCES/bin/lada-basicvsrpp-variable-hq-runner"
fi

cp "$PACKAGE_DIR/Info.plist" "$CONTENTS/Info.plist"
if [[ -d "$PACKAGE_DIR/Localizations" ]]; then
  for localization in "$PACKAGE_DIR/Localizations"/*.lproj(N); do
    ditto "$localization" "$RESOURCES/${localization:t}"
  done
fi
if [[ "$MIOH_BUNDLE_PYTHON_RUNTIME" == 1 ]]; then
  ditto "$PYTHON_SOURCE" "$RESOURCES/runtime"
  mkdir -p "$RESOURCES/runtime/lib/python3.12/site-packages"
  rsync -a --exclude '.DS_Store' \
    "$SITE_PACKAGES/" "$RESOURCES/runtime/lib/python3.12/site-packages/"
  rm -f "$RESOURCES/runtime/lib/python3.12/site-packages"/__editable__.lada-*.pth(N)
  rm -f "$RESOURCES/runtime/lib/python3.12/site-packages"/__editable___lada_*_finder.py(N)
  rm -f "$RESOURCES/runtime/lib/python3.12/site-packages/_virtualenv.pth"
  rm -f "$RESOURCES/runtime/lib/python3.12/site-packages/_virtualenv.py"
  rm -rf "$RESOURCES/runtime/lib/python3.12/site-packages"/lada-*.dist-info(N)
  uv pip install \
    --python "$RESOURCES/runtime/bin/python3.12" \
    --break-system-packages \
    --no-deps \
    --no-build-isolation \
    --reinstall \
    "$ROOT"
  rm -rf "$MPS_DEFORM_BUILD_SOURCE"
  ditto "$VENDORED_MPS_DEFORM_CONV" "$MPS_DEFORM_BUILD_SOURCE"
  uv pip install \
    --python "$RESOURCES/runtime/bin/python3.12" \
    --break-system-packages \
    --no-deps \
    --no-build-isolation \
    --reinstall \
    "$MPS_DEFORM_BUILD_SOURCE"
  if [[ "${MIOH_SKIP_HARDWARE_SMOKE:-0}" != "1" ]]; then
    PYTHONHOME="$RESOURCES/runtime" \
    PYTHONPATH="$RESOURCES/runtime/lib/python3.12/site-packages" \
      "$RESOURCES/runtime/bin/python3.12" \
      "$PACKAGE_DIR/verify_mps_deform_conv.py"
  fi
  cp "$ROOT/process_video_parallel.py" \
    "$RESOURCES/runtime/lib/python3.12/site-packages/process_video_parallel.py"
  cp "$PACKAGE_DIR/mioh_preview_worker.py" \
    "$RESOURCES/runtime/lib/python3.12/site-packages/mioh_preview_worker.py"
  rm -f "$RESOURCES/runtime/lib/python3.12/site-packages"/lada-*.dist-info/direct_url.json(N)
fi
mkdir -p "$FFMPEG_CACHE"
if [[ ! -x "$FFMPEG_CACHE/ffmpeg" ]]; then
  curl -fL --retry 3 \
    -o "$FFMPEG_CACHE/ffmpeg.zip" \
    https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release/ffmpeg.zip
  ditto -x -k "$FFMPEG_CACHE/ffmpeg.zip" "$FFMPEG_CACHE/ffmpeg-unpacked"
  mv "$FFMPEG_CACHE/ffmpeg-unpacked/ffmpeg" "$FFMPEG_CACHE/ffmpeg"
fi
if [[ ! -x "$FFMPEG_CACHE/ffprobe" ]]; then
  curl -fL --retry 3 \
    -o "$FFMPEG_CACHE/ffprobe.zip" \
    https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release/ffprobe.zip
  ditto -x -k "$FFMPEG_CACHE/ffprobe.zip" "$FFMPEG_CACHE/ffprobe-unpacked"
  mv "$FFMPEG_CACHE/ffprobe-unpacked/ffprobe" "$FFMPEG_CACHE/ffprobe"
fi
cp "$FFMPEG_CACHE/ffmpeg" "$RESOURCES/bin/ffmpeg"
cp "$FFMPEG_CACHE/ffprobe" "$RESOURCES/bin/ffprobe"

MODEL_TOOLS_SOURCE="$PACKAGE_DIR/model-tools"
if [[ -d "$MODEL_TOOLS_SOURCE" ]]; then
  mkdir -p "$RESOURCES/model-tools/scripts"
  ditto "$MODEL_TOOLS_SOURCE" "$RESOURCES/model-tools"
  ditto "$ROOT/scripts/apple" "$RESOURCES/model-tools/scripts/apple"
  # RF-DETR remains a local research prototype. Keep it out of the shipped
  # application and model-tools bundle until it is deliberately reintroduced.
  find "$RESOURCES/model-tools/scripts/apple" \
    -maxdepth 1 -type f -iname '*rfdetr*' -delete
  find "$RESOURCES/model-tools/scripts/apple" \
    -type d -name __pycache__ -prune -exec rm -rf {} +
  cp "$ROOT/scripts/download_nomos_roi_enhancers.py" \
    "$RESOURCES/model-tools/scripts/download_nomos_roi_enhancers.py"
chmod +x \
    "$RESOURCES/model-tools/download-mioh-models.zsh" \
    "$RESOURCES/model-tools/convert-mioh-models.zsh"
fi

if [[ "$MIOH_MODELESS_DISTRIBUTION" != 1 ]]; then

MODEL_ASSETS=(
  lada_mosaic_restoration_model_generic_v1.2.pth
  RealESRGAN_x2plus.pth
  RealESRGAN_x4plus.pth
  RealESRGAN_x2plus_256.mlpackage
  RealESRGAN_x4plus_256.mlpackage
  realesr-general-x4v3_256.mlpackage
  MewZoom-V1-4X-Unet_256.mlpackage
  swinir-real-x4_256.mlpackage
  4xNomosWebPhoto_RealPLKSR_256.mlpackage
)
for asset in "${MODEL_ASSETS[@]}"; do
  if [[ -e "$ROOT/model_weights/$asset" ]]; then
    ditto "$ROOT/model_weights/$asset" "$RESOURCES/models/$asset"
  fi
done
if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
  RFDETR_SOURCE_ASSETS=(
    rfdetr-v6-576-fp32.aimodel
    rfdetr-v6-large-768-fp32.aimodel
  )
  for asset in "${RFDETR_SOURCE_ASSETS[@]}"; do
    if [[ -d "$ROOT/model_weights/$asset" ]]; then
      ditto "$ROOT/model_weights/$asset" "$RESOURCES/models/$asset"
    fi
  done
fi
COREML_DETECTION_ASSETS=(
  lada_mosaic_detection_model_v2.mlpackage
  lada_mosaic_detection_model_v3.1_fast.mlpackage
  lada_mosaic_detection_model_v3.1_accurate.mlpackage
  lada_mosaic_detection_model_v4_fast.mlpackage
  lada_mosaic_detection_model_v4_accurate.mlpackage
  lada_mosaic_detection_model_vr_v2_accurate.mlpackage
)
mkdir -p "$COMPILED_COREML_MODELS"
for package in "${COREML_DETECTION_ASSETS[@]}"; do
  source_model="$ROOT/model_weights/$package"
  compiled_name="${package:r}.mlmodelc"
  compiled_model="$COMPILED_COREML_MODELS/$compiled_name"
  if [[ ! -d "$compiled_model" || "$source_model" -nt "$compiled_model" ]]; then
    rm -rf "$compiled_model"
    xcrun coremlcompiler compile "$source_model" "$COMPILED_COREML_MODELS"
  fi
  ditto "$compiled_model" "$RESOURCES/models/$compiled_name"
done

# ROI enhancers are image-to-image Core ML programs. Ship their compiled
# form for immediate native use while retaining the source package for the
# portable model-management workflow. Missing optional models are skipped.
COREML_ENHANCER_ASSETS=(
  RealESRGAN_x2plus_256.mlpackage
  RealESRGAN_x4plus_256.mlpackage
  realesr-general-x4v3_256.mlpackage
  MewZoom-V1-4X-Unet_256.mlpackage
  swinir-real-x4_256.mlpackage
  4xNomosWebPhoto_RealPLKSR_256.mlpackage
)
for package in "${COREML_ENHANCER_ASSETS[@]}"; do
  source_model="$ROOT/model_weights/$package"
  [[ -d "$source_model" ]] || continue
  compiled_name="${package:r}.mlmodelc"
  compiled_model="$COMPILED_COREML_MODELS/$compiled_name"
  if [[ ! -d "$compiled_model" || "$source_model" -nt "$compiled_model" ]]; then
    rm -rf "$compiled_model"
    xcrun coremlcompiler compile "$source_model" "$COMPILED_COREML_MODELS"
  fi
  ditto "$compiled_model" "$RESOURCES/models/$compiled_name"
done

COREAI_DETECTION_STEMS=(
  lada_mosaic_detection_model_v2
  lada_mosaic_detection_model_v3.1_fast
  lada_mosaic_detection_model_v3.1_accurate
  lada_mosaic_detection_model_v4_fast
  lada_mosaic_detection_model_v4_accurate
  lada_mosaic_detection_model_vr_v2_accurate
)
for stem in "${COREAI_DETECTION_STEMS[@]}"; do
  detection_checkpoint="$ROOT/model_weights/$stem.pt"
  detection_asset="$ROOT/model_weights/$stem-fp16.aimodel"
  if [[ ! -d "$detection_asset" || "$detection_checkpoint" -nt "$detection_asset" ]]; then
    PYTHONPATH="$ROOT" "$LADA_STANDALONE_PYTHON_ENV/bin/python" \
      "$ROOT/scripts/apple/export_v4_fast_coreai.py" \
      --model "$detection_checkpoint" \
      --output "$detection_asset" \
      --allow-overwrite
  fi
done

COREAI_MODEL_ASSETS=(
  basicvsrpp-v1.2-t18-fp16.aimodel
  basicvsrpp-v1.2-t36-fp16.aimodel
  basicvsrpp-v1.2-t90-fp16.aimodel
  lada_mosaic_detection_model_v2-fp16.aimodel
  lada_mosaic_detection_model_v3.1_fast-fp16.aimodel
  lada_mosaic_detection_model_v3.1_accurate-fp16.aimodel
  lada_mosaic_detection_model_v4_fast-fp16.aimodel
  lada_mosaic_detection_model_v4_accurate-fp16.aimodel
  lada_mosaic_detection_model_vr_v2_accurate-fp16.aimodel
  RealESRGAN_x2plus-256-fp16.aimodel
  RealESRGAN_x4plus-256-fp16.aimodel
  realesr-general-x4v3-256-fp16.aimodel
  4xNomosWebPhoto_RealPLKSR-256-fp16.aimodel
)
VARIABLE_COREAI_SOURCE_MODELS="${VARIABLE_COREAI_SOURCE_MODELS:-$BUILD_DIR/variable-basicvsrpp-source}"
VARIABLE_COREAI_CHECKPOINT="${VARIABLE_COREAI_CHECKPOINT:-$ROOT/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth}"
VARIABLE_COREAI_ASSETS=(
  spatial6 flow6
  backward_1_start6 backward_1_continue6
  forward_1_start6 forward_1_continue6
  backward_2_start6 backward_2_continue6
  forward_2_start6 forward_2_continue6
  reconstruction6
)
VARIABLE_COREAI_STEP1_SOURCE_MODELS="${VARIABLE_COREAI_STEP1_SOURCE_MODELS:-$BUILD_DIR/variable-basicvsrpp-step1-source}"
VARIABLE_COREAI_STEP1_ASSETS=(
  spatial flow
  backward_1_init backward_1_first backward_1_later
  forward_1_init forward_1_first forward_1_later
  backward_2_init backward_2_first backward_2_later
  forward_2_init forward_2_first forward_2_later
  reconstruction
)
needs_variable_export=0
for name in "${VARIABLE_COREAI_ASSETS[@]}"; do
  source_asset="$VARIABLE_COREAI_SOURCE_MODELS/basicvsrpp-variable-$name.aimodel"
  if [[ ! -d "$source_asset" \
        || "$VARIABLE_COREAI_CHECKPOINT" -nt "$source_asset" \
        || "$ROOT/scripts/apple/basicvsrpp_coreai_kernels.py" -nt "$source_asset" \
        || "$ROOT/scripts/apple/export_basicvsrpp_variable_chunk6.py" -nt "$source_asset" ]]; then
    needs_variable_export=1
    break
  fi
done
if (( needs_variable_export )); then
  mkdir -p "$VARIABLE_COREAI_SOURCE_MODELS"
  PYTHONPATH="$ROOT" "$LADA_STANDALONE_PYTHON_ENV/bin/python" \
    "$ROOT/scripts/apple/export_basicvsrpp_variable_chunk6.py" \
    --checkpoint "$VARIABLE_COREAI_CHECKPOINT" \
    --output-dir "$VARIABLE_COREAI_SOURCE_MODELS" \
    --overwrite
fi
if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
  needs_step1_export=0
  for name in "${VARIABLE_COREAI_STEP1_ASSETS[@]}"; do
    source_asset="$VARIABLE_COREAI_STEP1_SOURCE_MODELS/basicvsrpp-variable-$name.aimodel"
    if [[ ! -d "$source_asset" \
          || "$VARIABLE_COREAI_CHECKPOINT" -nt "$source_asset" \
          || "$ROOT/scripts/apple/basicvsrpp_coreai_kernels.py" -nt "$source_asset" \
          || "$ROOT/scripts/apple/benchmark_basicvsrpp_variable_coreai.py" -nt "$source_asset" ]]; then
      needs_step1_export=1
      break
    fi
  done
  if (( needs_step1_export )); then
    mkdir -p "$VARIABLE_COREAI_STEP1_SOURCE_MODELS"
    PYTHONPATH="$ROOT" "$LADA_STANDALONE_PYTHON_ENV/bin/python" \
      "$ROOT/scripts/apple/benchmark_basicvsrpp_variable_coreai.py" \
      --checkpoint "$VARIABLE_COREAI_CHECKPOINT" \
      --output-dir "$VARIABLE_COREAI_STEP1_SOURCE_MODELS" \
      --export \
      --export-only \
      --overwrite
  fi
fi
if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
mkdir -p "$COMPILED_MODELS"
find "$COMPILED_MODELS" -maxdepth 1 -type d -name '*.aimodelc' \
  ! -name "*.$COREAI_ARCHITECTURE.aimodelc" -exec rm -rf {} +

typeset -A expected_coreai_models
for asset in "${COREAI_MODEL_ASSETS[@]}"; do
  compiled_name="${asset:r}.$COREAI_ARCHITECTURE.aimodelc"
  expected_coreai_models[$compiled_name]=1
done
generic_variable_collection_name="basicvsrpp-v1.2-variable-coreai.$COREAI_ARCHITECTURE.aimodelc"
variable_collection_name="$generic_variable_collection_name"
expected_coreai_models[$generic_variable_collection_name]=1
if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
  hq_variable_collection_name="basicvsrpp-v1.2-variable-hq-coreai.$COREAI_ARCHITECTURE.aimodelc"
  expected_coreai_models[$hq_variable_collection_name]=1
fi
for model in "$COMPILED_MODELS"/*.$COREAI_ARCHITECTURE.aimodelc(N); do
  if [[ -z "${expected_coreai_models[${model:t}]-}" ]]; then
    rm -rf "$model"
  fi
done

for asset in "${COREAI_MODEL_ASSETS[@]}"; do
  source_model="$ROOT/model_weights/$asset"
  compiled_name="${asset:r}.$COREAI_ARCHITECTURE.aimodelc"
  compiled_model="$COMPILED_MODELS/$compiled_name"
  if [[ ! -d "$compiled_model" || "$source_model" -nt "$compiled_model" ]]; then
    rm -rf "$compiled_model"
    xcrun coreai-build compile \
      "$source_model" \
      --output "$COMPILED_MODELS" \
      --platform macOS \
      --min-deployment-version 27.0 \
      --preferred-compute gpu \
      --architecture "$COREAI_ARCHITECTURE"
  fi

  inspect_file="$BUILD_DIR/${compiled_name}.inspect.json"
  xcrun coreai-build inspect "$compiled_model" --json > "$inspect_file"
  "$LADA_STANDALONE_PYTHON_ENV/bin/python" - "$inspect_file" "$COREAI_ARCHITECTURE" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    details = json.load(handle)
architecture = sys.argv[2]
if architecture not in details.get("supportedArchitectures", []):
    raise SystemExit(f"compiled model does not support {architecture}: {sys.argv[1]}")
if architecture == "h17s" and "M5 Pro" not in details.get("supportedChips", []):
    raise SystemExit(f"h17s model is not specialized for M5 Pro: {sys.argv[1]}")
PY
  rm -f "$inspect_file"
  ditto "$compiled_model" "$RESOURCES/models/$compiled_name"
done

variable_collection="$COMPILED_MODELS/$variable_collection_name"
if [[ -f "$variable_collection/metadata.json" ]]; then
  # coreai-build treats an output directory ending in .aimodelc as the model
  # destination itself. Remove an interrupted/legacy single-model payload so
  # this path can remain the collection that contains all chunk assets.
  rm -rf "$variable_collection"
fi
mkdir -p "$variable_collection"
variable_compile_output="$BUILD_DIR/variable-basicvsrpp-compiled-stage"
mkdir -p "$variable_compile_output"
typeset -A expected_variable_assets
for name in "${VARIABLE_COREAI_ASSETS[@]}"; do
  source_asset="$VARIABLE_COREAI_SOURCE_MODELS/basicvsrpp-variable-$name.aimodel"
  compiled_name="basicvsrpp-variable-$name.$COREAI_ARCHITECTURE.aimodelc"
  compiled_asset="$variable_collection/$compiled_name"
  expected_variable_assets[$compiled_name]=1
  if [[ ! -d "$compiled_asset" || "$source_asset" -nt "$compiled_asset" ]]; then
    rm -rf "$compiled_asset"
    rm -rf "$variable_compile_output/$compiled_name"
    xcrun coreai-build compile \
      "$source_asset" \
      --output "$variable_compile_output" \
      --platform macOS \
      --min-deployment-version 27.0 \
      --preferred-compute gpu \
      --architecture "$COREAI_ARCHITECTURE"
    ditto "$variable_compile_output/$compiled_name" "$compiled_asset"
  fi
done
for compiled_asset in "$variable_collection"/*.$COREAI_ARCHITECTURE.aimodelc(N); do
  if [[ -z "${expected_variable_assets[${compiled_asset:t}]-}" ]]; then
    rm -rf "$compiled_asset"
  fi
done
ditto "$variable_collection" "$RESOURCES/models/$variable_collection_name"
if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
  step1_collection="$COMPILED_MODELS/$hq_variable_collection_name"
  if [[ -f "$step1_collection/metadata.json" ]]; then
    rm -rf "$step1_collection"
  fi
  mkdir -p "$step1_collection"
  step1_compile_output="$BUILD_DIR/variable-basicvsrpp-step1-compiled-stage"
  mkdir -p "$step1_compile_output"
  typeset -A expected_step1_assets
  for name in "${VARIABLE_COREAI_STEP1_ASSETS[@]}"; do
    source_asset="$VARIABLE_COREAI_STEP1_SOURCE_MODELS/basicvsrpp-variable-$name.aimodel"
    compiled_name="basicvsrpp-variable-$name.$COREAI_ARCHITECTURE.aimodelc"
    compiled_asset="$step1_collection/$compiled_name"
    expected_step1_assets[$compiled_name]=1
    if [[ ! -d "$compiled_asset" || "$source_asset" -nt "$compiled_asset" ]]; then
      rm -rf "$compiled_asset"
      rm -rf "$step1_compile_output/$compiled_name"
      xcrun coreai-build compile \
        "$source_asset" \
        --output "$step1_compile_output" \
        --platform macOS \
        --min-deployment-version 27.0 \
        --preferred-compute gpu \
        --architecture "$COREAI_ARCHITECTURE"
      ditto "$step1_compile_output/$compiled_name" "$compiled_asset"
    fi
  done
  for compiled_asset in "$step1_collection"/*.$COREAI_ARCHITECTURE.aimodelc(N); do
    if [[ -z "${expected_step1_assets[${compiled_asset:t}]-}" ]]; then
      rm -rf "$compiled_asset"
    fi
  done
  ditto "$step1_collection" "$RESOURCES/models/$hq_variable_collection_name"
fi
else
  for asset in "${COREAI_MODEL_ASSETS[@]}"; do
    source_model="$ROOT/model_weights/$asset"
    if [[ ! -d "$source_model" ]]; then
      print -u2 "Missing portable Core AI model: $source_model"
      exit 1
    fi
    ditto "$source_model" "$RESOURCES/models/$asset"
  done
  variable_source_collection="$RESOURCES/models/basicvsrpp-v1.2-variable-coreai.aimodel"
  mkdir -p "$variable_source_collection"
  for name in "${VARIABLE_COREAI_ASSETS[@]}"; do
    source_asset="$VARIABLE_COREAI_SOURCE_MODELS/basicvsrpp-variable-$name.aimodel"
    ditto "$source_asset" "$variable_source_collection/${source_asset:t}"
  done
fi

# Keep the exact variable-restoration checkpoint auditable after the model has
# been split into eleven (and, for dedicated builds, fifteen HQ) compiled
# assets. The absolute build-machine path is intentionally omitted.
"$LADA_STANDALONE_PYTHON_ENV/bin/python" - \
  "$VARIABLE_COREAI_CHECKPOINT" \
  "$RESOURCES/models/basicvsrpp-v1.2-variable-coreai.provenance.json" \
  "$COREAI_DISTRIBUTION" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
destination = Path(sys.argv[2])
distribution = sys.argv[3]
digest = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
payload = {
    "format_version": 1,
    "checkpoint_filename": checkpoint.name,
    "checkpoint_sha256": digest.hexdigest(),
    "checkpoint_size": checkpoint.stat().st_size,
    "distribution": distribution,
    "chunk_size": 6,
    "chunk_asset_count": 11,
    "hq_asset_count": 15 if distribution == "dedicated" else 0,
}
destination.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
else
  print "Modeless distribution: skipping bundled model assets and Core ML/Core AI exports"
fi
cp "$ROOT/LICENSE.md" "$RESOURCES/LICENSE.md"
ditto "$ROOT/LICENSES" "$RESOURCES/LICENSES"
if [[ "$MIOH_BUNDLE_PYTHON_RUNTIME" == 1 ]]; then
  cp "$VENDORED_MPS_DEFORM_CONV/LICENSE" "$RESOURCES/LICENSES/mps-deform-conv.txt"
fi

if [[ -n "${MIOH_PREBUILT_APP_ICON:-}" ]]; then
  ditto "$MIOH_PREBUILT_APP_ICON" "$RESOURCES/AppIcon.icns"
else
  ICONSET="$BUILD_DIR/AppIcon.iconset"
  rm -rf "$ICONSET"
  mkdir -p "$ICONSET"
  SOURCE_ICON="$ROOT/lada/gui/icons/mioh-icon.png"
  for spec in "16 icon_16x16" "32 icon_16x16@2x" "32 icon_32x32" \
              "64 icon_32x32@2x" "128 icon_128x128" "256 icon_128x128@2x" \
              "256 icon_256x256" "512 icon_256x256@2x" "512 icon_512x512" \
              "1024 icon_512x512@2x"; do
    size="${spec%% *}"
    name="${spec#* }"
    sips -z "$size" "$size" "$SOURCE_ICON" --out "$ICONSET/$name.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$RESOURCES/AppIcon.icns"
fi

chmod +x "$CONTENTS/MacOS/mioh" \
  "$RESOURCES/bin/ffmpeg" "$RESOURCES/bin/ffprobe" \
  "$RESOURCES/bin/mioh-preview-videotoolbox-encoder" \
  "$RESOURCES/bin/lada-coreai-runner" \
  "$RESOURCES/bin/lada-basicvsrpp-variable-runner"
chmod +x "$RESOURCES/bin/mioh-native-coreai-preview"
if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
  chmod +x "$RESOURCES/bin/lada-basicvsrpp-variable-hq-runner"
fi
if [[ "$MIOH_BUNDLE_PYTHON_RUNTIME" == 1 ]]; then
  chmod +x "$RESOURCES/runtime/bin/python3.12"
fi

if [[ "$MIOH_MODELESS_DISTRIBUTION" == 1 ]]; then
  print "Skipping model smoke tests for modeless distribution"
elif [[ "${MIOH_SKIP_HARDWARE_SMOKE:-0}" == "1" ]]; then
  print "Skipping MPS/Core AI hardware smoke tests by request"
elif [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
  PYTHONPATH="$ROOT" \
  LADA_MODEL_WEIGHTS_DIR="$RESOURCES/models" \
  LADA_COREAI_ARCHITECTURE="$COREAI_ARCHITECTURE" \
  LADA_COREAI_SWIFT_RUNNER="$RESOURCES/bin/lada-coreai-runner" \
  LADA_VARIABLE_COREAI_SWIFT_RUNNER="$RESOURCES/bin/lada-basicvsrpp-variable-runner" \
  LADA_VARIABLE_COREAI_HQ_SWIFT_RUNNER="$RESOURCES/bin/lada-basicvsrpp-variable-hq-runner" \
    "$LADA_STANDALONE_PYTHON_ENV/bin/python" \
    "$PACKAGE_DIR/verify_coreai_models.py" \
    --resources "$RESOURCES" \
    --distribution "$COREAI_DISTRIBUTION" \
    --architecture "$COREAI_ARCHITECTURE"
else
  env -u LADA_COREAI_ARCHITECTURE -u LADA_COREAI_SWIFT_RUNNER \
    PYTHONPATH="$ROOT" \
    LADA_MODEL_WEIGHTS_DIR="$RESOURCES/models" \
    "$LADA_STANDALONE_PYTHON_ENV/bin/python" \
    "$PACKAGE_DIR/verify_coreai_models.py" \
    --resources "$RESOURCES" \
    --distribution "$COREAI_DISTRIBUTION" \
    --architecture "$COREAI_ARCHITECTURE" \
    --smoke-model basicvsrpp-v1.2-coreai
fi

if [[ "$MIOH_BUNDLE_PYTHON_RUNTIME" == 1 ]]; then
  find "$RESOURCES/runtime" -type d -name '__pycache__' -prune -exec rm -rf {} +
  find "$RESOURCES/runtime" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
  # The build environment also serves local RF-DETR experiments. Do not inherit
  # that prototype or its CLI into the production mioh runtime.
  rm -rf \
    "$RESOURCES/runtime/lib/python3.12/site-packages/rfdetr" \
    "$RESOURCES/runtime/lib/python3.12/site-packages/lada/models/rfdetr" \
    "$RESOURCES/runtime/lib/python3.12/site-packages"/rfdetr-*.dist-info(N)
  rm -f "$RESOURCES/runtime/bin/rfdetr"
  rm -rf \
    "$RESOURCES/runtime/bin/pip" \
    "$RESOURCES/runtime/bin/pip3" \
    "$RESOURCES/runtime/bin/pip3.12" \
    "$RESOURCES/runtime/lib/python3.12/site-packages/pip" \
    "$RESOURCES/runtime/lib/python3.12/site-packages"/pip-*.dist-info(N) \
    "$RESOURCES/runtime/lib/python3.12/site-packages/setuptools" \
    "$RESOURCES/runtime/lib/python3.12/site-packages"/setuptools-*.dist-info(N) \
    "$RESOURCES/runtime/lib/python3.12/site-packages/wheel" \
    "$RESOURCES/runtime/lib/python3.12/site-packages"/wheel-*.dist-info(N) \
    "$RESOURCES/runtime/lib/python3.12/site-packages/tests" \
    "$RESOURCES/runtime/lib/python3.12/site-packages/test" \
    "$RESOURCES/runtime/lib/python3.12/site-packages/yapftests"
fi

codesign --force --deep --sign - "$APP"

DMG_ROOT="$BUILD_DIR/dmg-root"
rm -f "$DMG"
rm -rf "$DMG_ROOT"
mkdir -p "$DMG_ROOT"
ditto "$APP" "$DMG_ROOT/$APP_BASENAME.app"
ln -s /Applications "$DMG_ROOT/Applications"
if [[ "$INCLUDE_USER_MANUAL" == 1 ]]; then
  if [[ ! -f "$USER_MANUAL_PDF" ]]; then
    print -u2 "Missing mioh user manual: $USER_MANUAL_PDF"
    print -u2 "Generate it with scripts/docs/build_mioh_manual_pdf.py before building."
    exit 1
  fi
  cp "$USER_MANUAL_PDF" "$DMG_ROOT/mioh ユーザーマニュアル.pdf"
fi
if [[ -d "$RESOURCES/model-tools" ]]; then
  ditto "$RESOURCES/model-tools" "$DMG_ROOT/model-tools"
  ln -s "model-tools/download-mioh-models.zsh" "$DMG_ROOT/download-mioh-models.zsh"
  ln -s "model-tools/convert-mioh-models.zsh" "$DMG_ROOT/convert-mioh-models.zsh"
fi
diskutil image create from \
  --volumeName "$APP_BASENAME" \
  --format UDZO \
  "$DMG_ROOT" \
  "$DMG"

print "App: $APP"
print "DMG: $DMG"
