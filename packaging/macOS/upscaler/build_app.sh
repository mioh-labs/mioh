#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h:h:h}"
UPSCALER_DIR="$ROOT/packaging/macOS/upscaler"
BUILD_DIR="${BUILD_DIR:-$ROOT/build/mioh-upscaler}"
APP="$BUILD_DIR/mioh upscaler.app"
DMG="$BUILD_DIR/mioh-upscaler-0.14.3-unsigned.dmg"
CONTENTS="$APP/Contents"
RESOURCES="$CONTENTS/Resources"

VENDORED_FLASHVSR_SOURCE_DIR="$UPSCALER_DIR/vendor/flashvsr"
if [[ -d "$VENDORED_FLASHVSR_SOURCE_DIR" ]]; then
  DEFAULT_FLASHVSR_SOURCE_DIR="$VENDORED_FLASHVSR_SOURCE_DIR"
else
  DEFAULT_FLASHVSR_SOURCE_DIR="$ROOT/../FlashVSR_plus"
fi
FLASHVSR_SOURCE_DIR="${FLASHVSR_SOURCE_DIR:-$DEFAULT_FLASHVSR_SOURCE_DIR}"
FLASHVSR_SOURCE_DIR="${FLASHVSR_SOURCE_DIR:A}"
FLASHVSR_NATIVE_PIPELINE="$FLASHVSR_SOURCE_DIR/deployment/coreai/FlashVSRNativePipeline.swift"
FLASHVSR_NATIVE_RUNNER="$FLASHVSR_SOURCE_DIR/deployment/coreai/FlashVSRNativeVideoRunner.swift"
FFMPEG_CACHE="${FFMPEG_CACHE:-$ROOT/build/macos-standalone/ffmpeg-static}"

for required in "$FLASHVSR_NATIVE_PIPELINE" "$FLASHVSR_NATIVE_RUNNER"; do
  if [[ ! -f "$required" ]]; then
    print -u2 "Missing FlashVSR Swift source: $required"
    exit 2
  fi
done

rm -rf "$APP"
rm -f "$DMG"
mkdir -p "$CONTENTS/MacOS" "$RESOURCES/bin"
cp "$UPSCALER_DIR/Info.plist" "$CONTENTS/Info.plist"

ICONSET="$BUILD_DIR/AppIcon.iconset"
rm -rf "$ICONSET"
mkdir -p "$ICONSET"
SOURCE_ICON="$UPSCALER_DIR/AppIcon-1024.png"
for spec in "16 icon_16x16" "32 icon_16x16@2x" "32 icon_32x32" \
            "64 icon_32x32@2x" "128 icon_128x128" "256 icon_128x128@2x" \
            "256 icon_256x256" "512 icon_256x256@2x" "512 icon_512x512" \
            "1024 icon_512x512@2x"; do
  size="${spec%% *}"
  name="${spec#* }"
  sips -z "$size" "$size" "$SOURCE_ICON" \
    --out "$ICONSET/$name.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$RESOURCES/AppIcon.icns"

xcrun swiftc \
  -O -parse-as-library -target arm64-apple-macosx27.0 \
  -framework AppKit -framework AVFoundation -framework AVKit -framework CoreMedia \
  -framework SwiftUI -framework UniformTypeIdentifiers \
  "$UPSCALER_DIR/UpscalerMediaProbe.swift" \
  "$UPSCALER_DIR/VideoUpscaleController.swift" \
  "$UPSCALER_DIR/UpscalerVideoPreview.swift" \
  "$UPSCALER_DIR/UpscalerModelSetup.swift" \
  "$UPSCALER_DIR/MiniMaxH3VideoGenerationView.swift" \
  "$UPSCALER_DIR/UpscalerApp.swift" \
  -o "$CONTENTS/MacOS/mioh-upscaler"

xcrun swiftc \
  -O -parse-as-library -target arm64-apple-macosx27.0 \
  -framework AVFoundation -framework CoreAI -framework CoreImage \
  -framework CoreMedia -framework CoreVideo -framework Metal \
  -framework Vision -framework VideoToolbox \
  "$UPSCALER_DIR/AdcSRNativePipeline.swift" \
  "$UPSCALER_DIR/AdcSRNativeVideoRunner.swift" \
  -o "$RESOURCES/bin/adcsr-coreai-video"

xcrun swiftc \
  -O -parse-as-library -target arm64-apple-macosx27.0 \
  -framework AVFoundation -framework CoreAI -framework CoreImage \
  -framework CoreML -framework CoreMedia -framework CoreVideo \
  -framework Metal -framework VideoToolbox \
  "$FLASHVSR_NATIVE_PIPELINE" "$FLASHVSR_NATIVE_RUNNER" \
  -o "$RESOURCES/bin/flashvsr-coreai-video"

# MiniMax H3 / 10Eros-Max H3 belongs to mioh upscaler. Only the native Swift
# runner is bundled; model graphs, tokenizer and manifest remain external.
xcrun swiftc \
  -O -parse-as-library -target arm64-apple-macosx27.0 \
  -framework AVFoundation -framework CoreAI -framework CoreImage \
  -framework CoreMedia -framework CoreML -framework CoreVideo \
  "$UPSCALER_DIR/MiniMaxH3NativeCore.swift" \
  "$UPSCALER_DIR/MiniMaxH3NativeModels.swift" \
  "$UPSCALER_DIR/MiniMaxH3NativeVideoVAE.swift" \
  "$UPSCALER_DIR/MiniMaxH3NativeQwen.swift" \
  "$UPSCALER_DIR/MiniMaxH3NativeQwenComposite.swift" \
  "$UPSCALER_DIR/TenErosMaxH3DenoiserComposite.swift" \
  "$UPSCALER_DIR/MiniMaxH3NativeMedia.swift" \
  "$UPSCALER_DIR/MiniMaxH3NativeRunner.swift" \
  -o "$RESOURCES/bin/mioh-minimax-h3-native"

# The first-launch installer contains conversion code, but no model weights.
# Models remain external in the folder selected by the user.
MODEL_TOOLS="$RESOURCES/model-tools"
FLASHVSR_CONVERTER_SOURCE="$FLASHVSR_SOURCE_DIR"
mkdir -p "$MODEL_TOOLS/flashvsr/deployment/coreai" \
  "$MODEL_TOOLS/flashvsr/src/models"
cp "$UPSCALER_DIR/model-tools/setup-upscaler-models.zsh" "$MODEL_TOOLS/"
cp "$FLASHVSR_CONVERTER_SOURCE/deployment/__init__.py" \
  "$MODEL_TOOLS/flashvsr/deployment/__init__.py"
cp "$FLASHVSR_CONVERTER_SOURCE/deployment/build_upscale_bundle.py" \
  "$MODEL_TOOLS/flashvsr/deployment/build_upscale_bundle.py"
for source in __init__.py export_native.py full_model.py model.py; do
  cp "$FLASHVSR_CONVERTER_SOURCE/deployment/coreai/$source" \
    "$MODEL_TOOLS/flashvsr/deployment/coreai/$source"
done
cp "$FLASHVSR_CONVERTER_SOURCE/src/models/TCDecoder.py" \
  "$MODEL_TOOLS/flashvsr/src/models/TCDecoder.py"
chmod +x "$MODEL_TOOLS/setup-upscaler-models.zsh"

mkdir -p "$FFMPEG_CACHE"
if [[ ! -x "$FFMPEG_CACHE/ffmpeg" ]]; then
  curl -fL --retry 3 -o "$FFMPEG_CACHE/ffmpeg.zip" \
    https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release/ffmpeg.zip
  mkdir -p "$FFMPEG_CACHE/ffmpeg-unpacked"
  ditto -x -k "$FFMPEG_CACHE/ffmpeg.zip" "$FFMPEG_CACHE/ffmpeg-unpacked"
  mv "$FFMPEG_CACHE/ffmpeg-unpacked/ffmpeg" "$FFMPEG_CACHE/ffmpeg"
fi
if [[ ! -x "$FFMPEG_CACHE/ffprobe" ]]; then
  curl -fL --retry 3 -o "$FFMPEG_CACHE/ffprobe.zip" \
    https://ffmpeg.martin-riedl.de/redirect/latest/macos/arm64/release/ffprobe.zip
  mkdir -p "$FFMPEG_CACHE/ffprobe-unpacked"
  ditto -x -k "$FFMPEG_CACHE/ffprobe.zip" "$FFMPEG_CACHE/ffprobe-unpacked"
  mv "$FFMPEG_CACHE/ffprobe-unpacked/ffprobe" "$FFMPEG_CACHE/ffprobe"
fi
cp "$FFMPEG_CACHE/ffmpeg" "$FFMPEG_CACHE/ffprobe" "$RESOURCES/bin/"

chmod +x "$CONTENTS/MacOS/mioh-upscaler" \
  "$RESOURCES/bin/adcsr-coreai-video" \
  "$RESOURCES/bin/flashvsr-coreai-video" \
  "$RESOURCES/bin/mioh-minimax-h3-native" \
  "$RESOURCES/bin/ffmpeg" "$RESOURCES/bin/ffprobe"

codesign --force --deep --sign - "$APP"
DMG_ROOT="$BUILD_DIR/dmg-root"
rm -rf "$DMG_ROOT"
mkdir -p "$DMG_ROOT"
ditto "$APP" "$DMG_ROOT/${APP:t}"
ln -s /Applications "$DMG_ROOT/Applications"
hdiutil create -volname "mioh upscaler" -srcfolder "$DMG_ROOT" \
  -ov -format UDZO "$DMG"
print "App: $APP"
print "DMG: $DMG"
