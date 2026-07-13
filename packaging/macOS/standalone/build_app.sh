#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h:h:h}"
PACKAGE_DIR="$ROOT/packaging/macos/standalone"
BUILD_DIR="${BUILD_DIR:-$ROOT/build/macos-standalone}"
APP="$BUILD_DIR/mioh.app"
CONTENTS="$APP/Contents"
RESOURCES="$CONTENTS/Resources"
PYTHON_SOURCE="${PYTHON_SOURCE:-$HOME/.local/share/uv/python/cpython-3.12-macos-aarch64-none}"
PYTHON_SOURCE="${PYTHON_SOURCE:A}"
SITE_PACKAGES="$ROOT/.venv-coreai/lib/python3.12/site-packages"
COMPILED_MODELS="${COMPILED_MODELS:-$BUILD_DIR/compiled-models}"
FFMPEG_CACHE="${FFMPEG_CACHE:-$BUILD_DIR/ffmpeg-static}"
VENDORED_MPS_DEFORM_CONV="$PACKAGE_DIR/vendor/mps-deform-conv-0.2.2"
MPS_DEFORM_BUILD_SOURCE="$BUILD_DIR/mps-deform-conv-source"

rm -rf "$APP" "$BUILD_DIR/Lada.app"
rm -f "$BUILD_DIR/Lada-0.11.0-unsigned.dmg"
mkdir -p "$CONTENTS/MacOS" "$RESOURCES/bin" "$RESOURCES/models"

xcrun swiftc \
  -O \
  -parse-as-library \
  -target arm64-apple-macosx26.0 \
  -framework AppKit \
  -framework SwiftUI \
  "$PACKAGE_DIR/MiohApp.swift" \
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
  basicvsrpp-v1.2-t18-fp16.aimodel
  basicvsrpp-v1.2-t36-fp16.aimodel
  basicvsrpp-v1.2-t90-fp16.aimodel
  lada_mosaic_detection_model_v2.pt
  lada_mosaic_detection_model_v2.mlpackage
  lada_mosaic_detection_model_v3.pt
  lada_mosaic_detection_model_v3.1_fast.pt
  lada_mosaic_detection_model_v3.1_fast.mlpackage
  lada_mosaic_detection_model_v3.1_accurate.pt
  lada_mosaic_detection_model_v3.1_accurate.mlpackage
  lada_mosaic_detection_model_v4_fast.pt
  lada_mosaic_detection_model_v4_fast.mlpackage
  lada_mosaic_detection_model_v4_fast-fp16.aimodel
  lada_mosaic_detection_model_v4_accurate.pt
  lada_mosaic_detection_model_v4_accurate.mlpackage
  RealESRGAN_x2plus.pth
  RealESRGAN_x4plus.pth
  RealESRGAN_x4plus_256.mlpackage
  RealESRGAN_x4plus-256-fp16.aimodel
  realesr-general-x4v3_256.mlpackage
  realesr-general-x4v3-256-fp16.aimodel
  MewZoom-V1-4X-Unet_256.mlpackage
  MewZoom-V1-4X-Unet_512.mlpackage
  swinir-real-x4_256.mlpackage
)
for asset in "${MODEL_ASSETS[@]}"; do
  if [[ -e "$ROOT/model_weights/$asset" ]]; then
    ditto "$ROOT/model_weights/$asset" "$RESOURCES/models/$asset"
  fi
done

if [[ ! -d "$COMPILED_MODELS" ]] || ! find "$COMPILED_MODELS" -name '*.aimodelc' -print -quit | grep -q .; then
  mkdir -p "$COMPILED_MODELS"
  xcrun coreai-build compile \
    "$ROOT/model_weights/basicvsrpp-v1.2-t90-fp16.aimodel" \
    --output "$COMPILED_MODELS" \
    --platform macOS \
    --min-deployment-version 27.0 \
    --preferred-compute gpu
fi

for model in "$COMPILED_MODELS"/*.aimodelc; do
  ditto "$model" "$RESOURCES/models/${model:t}"
done
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

find "$RESOURCES/runtime" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "$RESOURCES/runtime" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

codesign --force --deep --sign - "$APP"

DMG="$BUILD_DIR/mioh-0.11.0-unsigned.dmg"
DMG_ROOT="$BUILD_DIR/dmg-root"
rm -f "$DMG"
rm -rf "$DMG_ROOT"
mkdir -p "$DMG_ROOT"
ditto "$APP" "$DMG_ROOT/mioh.app"
ln -s /Applications "$DMG_ROOT/Applications"
diskutil image create from \
  --volumeName "mioh" \
  --format UDZO \
  "$DMG_ROOT" \
  "$DMG"

print "App: $APP"
print "DMG: $DMG"
