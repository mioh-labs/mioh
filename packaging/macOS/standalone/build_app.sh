#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h:h:h}"
PACKAGE_DIR="$ROOT/packaging/macos/standalone"
BUILD_DIR="${BUILD_DIR:-$ROOT/build/macos-standalone}"
COREAI_DISTRIBUTION="${COREAI_DISTRIBUTION:-dedicated}"
APP_BASENAME="${APP_BASENAME:-mioh}"
DMG_BASENAME="${DMG_BASENAME:-mioh-0.11.0-unsigned}"
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
USER_MANUAL_PDF="${USER_MANUAL_PDF:-$ROOT/output/pdf/mioh-user-manual-ja.pdf}"
CONTENTS="$APP/Contents"
RESOURCES="$CONTENTS/Resources"
PYTHON_SOURCE="${PYTHON_SOURCE:-$HOME/.local/share/uv/python/cpython-3.12-macos-aarch64-none}"
PYTHON_SOURCE="${PYTHON_SOURCE:A}"
LADA_STANDALONE_PYTHON_ENV="${LADA_STANDALONE_PYTHON_ENV:-${LADA_STANDALONE_VENV:-$ROOT/.venv-coreai}}"
LADA_STANDALONE_PYTHON_ENV="${LADA_STANDALONE_PYTHON_ENV:A}"
SITE_PACKAGES="${SITE_PACKAGES:-$LADA_STANDALONE_PYTHON_ENV/lib/python3.12/site-packages}"
if [[ ! -d "$SITE_PACKAGES" ]]; then
  print -u2 "Missing site-packages for standalone build: $SITE_PACKAGES"
  print -u2 "Set LADA_STANDALONE_PYTHON_ENV to the single Python environment to package."
  exit 1
fi
COMPILED_MODELS="${COMPILED_MODELS:-$BUILD_DIR/compiled-models}"
COREAI_ARCHITECTURE="${COREAI_ARCHITECTURE:-h17s}"
COMPILED_COREML_MODELS="${COMPILED_COREML_MODELS:-$BUILD_DIR/compiled-coreml-models}"
FFMPEG_CACHE="${FFMPEG_CACHE:-$BUILD_DIR/ffmpeg-static}"
VENDORED_MPS_DEFORM_CONV="$PACKAGE_DIR/vendor/mps-deform-conv-0.2.2"
MPS_DEFORM_BUILD_SOURCE="$BUILD_DIR/mps-deform-conv-source"

rm -rf "$APP" "$BUILD_DIR/Lada.app"
rm -f "$DMG" "$BUILD_DIR/Lada-0.11.0-unsigned.dmg"
mkdir -p "$CONTENTS/MacOS" "$RESOURCES/bin" "$RESOURCES/models"

typeset -a APP_SWIFT_FLAGS
APP_SWIFT_FLAGS=()
if [[ "$COREAI_DISTRIBUTION" == "portable" ]]; then
  APP_SWIFT_FLAGS+=(-D MIOH_PORTABLE_COREAI)
fi

xcrun swiftc \
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
  -O \
  -parse-as-library \
  -target arm64-apple-macosx27.0 \
  -framework CoreAI \
  "$PACKAGE_DIR/CoreAIRunner.swift" \
  -o "$RESOURCES/bin/lada-coreai-runner"

cp "$PACKAGE_DIR/Info.plist" "$CONTENTS/Info.plist"
ditto "$PYTHON_SOURCE" "$RESOURCES/runtime"
mkdir -p "$RESOURCES/runtime/lib/python3.12/site-packages"
ditto "$SITE_PACKAGES" "$RESOURCES/runtime/lib/python3.12/site-packages"
rm -f "$RESOURCES/runtime/lib/python3.12/site-packages/__editable__.lada-0.11.0.pth"
rm -f "$RESOURCES/runtime/lib/python3.12/site-packages/__editable___lada_0_11_0_finder.py"
rm -f "$RESOURCES/runtime/lib/python3.12/site-packages/_virtualenv.pth"
rm -f "$RESOURCES/runtime/lib/python3.12/site-packages/_virtualenv.py"
rm -rf "$RESOURCES/runtime/lib/python3.12/site-packages/lada-0.11.0.dist-info"
uv pip install \
  --python "$RESOURCES/runtime/bin/python3.12" \
  --break-system-packages \
  --no-deps \
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
PYTHONHOME="$RESOURCES/runtime" \
PYTHONPATH="$RESOURCES/runtime/lib/python3.12/site-packages" \
  "$RESOURCES/runtime/bin/python3.12" \
  "$PACKAGE_DIR/verify_mps_deform_conv.py"
cp "$ROOT/process_video_parallel.py" \
  "$RESOURCES/runtime/lib/python3.12/site-packages/process_video_parallel.py"
cp "$PACKAGE_DIR/mioh_preview_worker.py" \
  "$RESOURCES/runtime/lib/python3.12/site-packages/mioh_preview_worker.py"
rm -f "$RESOURCES/runtime/lib/python3.12/site-packages/lada-0.11.0.dist-info/direct_url.json"

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

MODEL_ASSETS=(
  lada_mosaic_restoration_model_generic_v1.2.pth
  RealESRGAN_x2plus.pth
  RealESRGAN_x4plus.pth
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

COREAI_MODEL_ASSETS=(
  basicvsrpp-v1.2-t18-fp16.aimodel
  basicvsrpp-v1.2-t36-fp16.aimodel
  basicvsrpp-v1.2-t90-fp16.aimodel
  lada_mosaic_detection_model_v4_fast-fp16.aimodel
  RealESRGAN_x4plus-256-fp16.aimodel
  realesr-general-x4v3-256-fp16.aimodel
  4xNomosWebPhoto_RealPLKSR-256-fp16.aimodel
)
if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
mkdir -p "$COMPILED_MODELS"
find "$COMPILED_MODELS" -maxdepth 1 -type d -name '*.aimodelc' \
  ! -name "*.$COREAI_ARCHITECTURE.aimodelc" -exec rm -rf {} +

typeset -A expected_coreai_models
for asset in "${COREAI_MODEL_ASSETS[@]}"; do
  compiled_name="${asset:r}.$COREAI_ARCHITECTURE.aimodelc"
  expected_coreai_models[$compiled_name]=1
done
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
  "$RESOURCES/runtime/bin/python3.12" - "$inspect_file" "$COREAI_ARCHITECTURE" <<'PY'
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
else
  for asset in "${COREAI_MODEL_ASSETS[@]}"; do
    source_model="$ROOT/model_weights/$asset"
    if [[ ! -d "$source_model" ]]; then
      print -u2 "Missing portable Core AI model: $source_model"
      exit 1
    fi
    ditto "$source_model" "$RESOURCES/models/$asset"
  done
fi
cp "$ROOT/LICENSE.md" "$RESOURCES/LICENSE.md"
ditto "$ROOT/LICENSES" "$RESOURCES/LICENSES"
cp "$VENDORED_MPS_DEFORM_CONV/LICENSE" "$RESOURCES/LICENSES/mps-deform-conv.txt"

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

chmod +x "$CONTENTS/MacOS/mioh" "$RESOURCES/runtime/bin/python3.12" \
  "$RESOURCES/bin/ffmpeg" "$RESOURCES/bin/ffprobe" \
  "$RESOURCES/bin/lada-coreai-runner"

if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
  PYTHONHOME="$RESOURCES/runtime" \
  PYTHONPATH="$RESOURCES/runtime/lib/python3.12/site-packages" \
  LADA_MODEL_WEIGHTS_DIR="$RESOURCES/models" \
  LADA_COREAI_ARCHITECTURE="$COREAI_ARCHITECTURE" \
  LADA_COREAI_SWIFT_RUNNER="$RESOURCES/bin/lada-coreai-runner" \
    "$RESOURCES/runtime/bin/python3.12" \
    "$PACKAGE_DIR/verify_coreai_models.py" \
    --resources "$RESOURCES" \
    --distribution "$COREAI_DISTRIBUTION" \
    --architecture "$COREAI_ARCHITECTURE"
else
  env -u LADA_COREAI_ARCHITECTURE -u LADA_COREAI_SWIFT_RUNNER \
    PYTHONHOME="$RESOURCES/runtime" \
    PYTHONPATH="$RESOURCES/runtime/lib/python3.12/site-packages" \
    LADA_MODEL_WEIGHTS_DIR="$RESOURCES/models" \
    "$RESOURCES/runtime/bin/python3.12" \
    "$PACKAGE_DIR/verify_coreai_models.py" \
    --resources "$RESOURCES" \
    --distribution "$COREAI_DISTRIBUTION" \
    --architecture "$COREAI_ARCHITECTURE" \
    --smoke-model basicvsrpp-v1.2-coreai
fi

find "$RESOURCES/runtime" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "$RESOURCES/runtime" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
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
diskutil image create from \
  --volumeName "$APP_BASENAME" \
  --format UDZO \
  "$DMG_ROOT" \
  "$DMG"

print "App: $APP"
print "DMG: $DMG"
