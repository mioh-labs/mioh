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
FFMPEG_VERSION="8.1.2"
FFMPEG_RELEASE="tas-ffmpeg-$FFMPEG_VERSION-macos-arm64"
FFMPEG_ARCHIVE="$FFMPEG_RELEASE.tar.xz"
FFMPEG_URL="https://github.com/NevermindNilas/TAS-FFMPEG/releases/download/v$FFMPEG_VERSION/$FFMPEG_ARCHIVE"
FFMPEG_SHA256="c57c509ffc3c5456fb9a37101ec25468f4bfe20d2f68394b9f307066422642d0"
FFMPEG_CACHE="${FFMPEG_CACHE:-$ROOT/build/macos-standalone/$FFMPEG_RELEASE}"
FFMPEG_PACKAGE="$FFMPEG_CACHE/$FFMPEG_RELEASE"

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
  -framework CoreImage -framework CoreVideo -framework ImageIO -framework SwiftUI \
  -framework UniformTypeIdentifiers -framework Vision \
  "$UPSCALER_DIR/UpscalerMediaProbe.swift" \
  "$UPSCALER_DIR/VideoUpscaleController.swift" \
  "$UPSCALER_DIR/UpscalerVideoPreview.swift" \
  "$UPSCALER_DIR/UpscalerModelSetup.swift" \
  "$UPSCALER_DIR/MiniMaxH3FaceReferences.swift" \
  "$UPSCALER_DIR/MiniMaxH3ReferenceEditPrompt.swift" \
  "$UPSCALER_DIR/MiniMaxH3ReferenceVideoMask.swift" \
  "$UPSCALER_DIR/MiniMaxH3PromptAssistant.swift" \
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

# MiniMax H3 / 10Eros-Max H3 belongs to mioh upscaler. The native Swift runner
# is bundled here; the first-launch model setup also bundles the Python
# exporters needed to download safetensors and convert them into external Core
# AI assets in the folder selected by the user.
xcrun swiftc \
  -O -parse-as-library -target arm64-apple-macosx27.0 \
  -framework AVFoundation -framework CoreAI -framework CoreImage \
  -framework CoreMedia -framework CoreML -framework CoreVideo -framework Vision \
  "$UPSCALER_DIR/MiniMaxH3ReferenceEditPrompt.swift" \
  "$UPSCALER_DIR/MiniMaxH3NativeCore.swift" \
  "$UPSCALER_DIR/MiniMaxH3NativeModels.swift" \
  "$UPSCALER_DIR/MiniMaxH3SpatialTileBlender.swift" \
  "$UPSCALER_DIR/MiniMaxH3NativeVideoVAE.swift" \
  "$UPSCALER_DIR/MiniMaxH3NativeQwen.swift" \
  "$UPSCALER_DIR/MiniMaxH3NativeQwenComposite.swift" \
  "$UPSCALER_DIR/TenErosMaxH3DenoiserComposite.swift" \
  "$UPSCALER_DIR/MiniMaxH3NativeMedia.swift" \
  "$UPSCALER_DIR/MiniMaxH3ReferenceVideoMask.swift" \
  "$UPSCALER_DIR/MiniMaxH3MusicAnalysis.swift" \
  "$UPSCALER_DIR/MiniMaxH3NativeRunner.swift" \
  -o "$RESOURCES/bin/mioh-minimax-h3-native"

# Native stdio MCP server for Codex. It starts the same bundled Swift runners
# as the UI and forwards Codex-authored H3 prompts byte-for-byte.
xcrun swiftc \
  -O -parse-as-library -target arm64-apple-macosx27.0 \
  -framework AppKit -framework AVFoundation -framework CoreImage \
  -framework CoreMedia -framework ImageIO -framework UniformTypeIdentifiers \
  -framework Vision \
  "$UPSCALER_DIR/UpscalerMediaProbe.swift" \
  "$UPSCALER_DIR/VideoUpscaleController.swift" \
  "$UPSCALER_DIR/MiniMaxH3FaceReferences.swift" \
  "$UPSCALER_DIR/MiniMaxH3ReferenceEditPrompt.swift" \
  "$UPSCALER_DIR/MiohUpscalerMCPServer.swift" \
  -o "$RESOURCES/bin/mioh-upscaler-mcp"

# The first-launch installer contains conversion code, but no model weights.
# Models remain external in the folder selected by the user.
MODEL_TOOLS="$RESOURCES/model-tools"
FLASHVSR_CONVERTER_SOURCE="$FLASHVSR_SOURCE_DIR"
mkdir -p "$MODEL_TOOLS/flashvsr/deployment/coreai" \
  "$MODEL_TOOLS/flashvsr/src/models" \
  "$MODEL_TOOLS/h3/scripts/apple"
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
for source in \
  build_10eros_max_h3_manifest.py \
  export_10eros_max_h3_dit_block.py \
  export_10eros_max_h3_dit_components.py \
  export_10eros_max_h3_dit_coreai.py \
  export_minimax_h3_native.py \
  export_minimax_h3_qwen_coreai.py \
  export_minimax_h3_qwen_embedding.py \
  export_minimax_h3_qwen_language_layer.py \
  export_minimax_h3_qwen_vision.py \
  pilot_10eros_max_h3_int8_convrot.py \
  pilot_minimax_h3_fp8_scaled.py \
  pilot_minimax_h3_qwen_nvfp4.py \
  reference_10eros_max_h3_dit_sequence.py \
  ten_eros_h3_coreai_kernels.py; do
  cp "$ROOT/scripts/apple/$source" "$MODEL_TOOLS/h3/scripts/apple/$source"
done
chmod +x "$MODEL_TOOLS/setup-upscaler-models.zsh"

mkdir -p "$FFMPEG_CACHE"
if [[ ! -f "$FFMPEG_CACHE/$FFMPEG_ARCHIVE" ]]; then
  curl -fL --retry 3 -o "$FFMPEG_CACHE/$FFMPEG_ARCHIVE.download" "$FFMPEG_URL"
  mv "$FFMPEG_CACHE/$FFMPEG_ARCHIVE.download" "$FFMPEG_CACHE/$FFMPEG_ARCHIVE"
fi
ffmpeg_actual_sha256="$(shasum -a 256 "$FFMPEG_CACHE/$FFMPEG_ARCHIVE" | awk '{print $1}')"
if [[ "$ffmpeg_actual_sha256" != "$FFMPEG_SHA256" ]]; then
  print -u2 "FFmpeg archive SHA-256 mismatch: $ffmpeg_actual_sha256"
  exit 2
fi
if [[ ! -x "$FFMPEG_PACKAGE/bin/ffmpeg" || ! -x "$FFMPEG_PACKAGE/bin/ffprobe" ]]; then
  tar -xJf "$FFMPEG_CACHE/$FFMPEG_ARCHIVE" -C "$FFMPEG_CACHE"
fi
for required in \
  "$FFMPEG_PACKAGE/bin/ffmpeg" \
  "$FFMPEG_PACKAGE/bin/ffprobe" \
  "$FFMPEG_PACKAGE/lib" \
  "$FFMPEG_PACKAGE/licenses" \
  "$FFMPEG_PACKAGE/manifest.json"; do
  if [[ ! -e "$required" ]]; then
    print -u2 "Incomplete FFmpeg package: $required"
    exit 2
  fi
done
if ! "$FFMPEG_PACKAGE/bin/ffmpeg" -hide_banner -h full 2>&1 | grep -- '-vsync' >/dev/null; then
  print -u2 "Bundled FFmpeg does not provide -vsync"
  exit 2
fi
if ! "$FFMPEG_PACKAGE/bin/ffmpeg" -hide_banner -encoders 2>/dev/null | grep ' png ' >/dev/null; then
  print -u2 "Bundled FFmpeg does not provide the PNG encoder"
  exit 2
fi
cp "$FFMPEG_PACKAGE/bin/ffmpeg" "$FFMPEG_PACKAGE/bin/ffprobe" "$RESOURCES/bin/"
ditto "$FFMPEG_PACKAGE/lib" "$RESOURCES/lib"
ditto "$FFMPEG_PACKAGE/licenses" "$RESOURCES/licenses/tas-ffmpeg"
cp "$FFMPEG_PACKAGE/manifest.json" \
  "$RESOURCES/licenses/tas-ffmpeg/build-manifest.json"

chmod +x "$CONTENTS/MacOS/mioh-upscaler" \
  "$RESOURCES/bin/adcsr-coreai-video" \
  "$RESOURCES/bin/flashvsr-coreai-video" \
  "$RESOURCES/bin/mioh-minimax-h3-native" \
  "$RESOURCES/bin/mioh-upscaler-mcp" \
  "$RESOURCES/bin/ffmpeg" "$RESOURCES/bin/ffprobe"

codesign --force --deep --sign - "$APP"
DMG_ROOT="$BUILD_DIR/dmg-root"
rm -rf "$DMG_ROOT"
mkdir -p "$DMG_ROOT"
ditto "$APP" "$DMG_ROOT/${APP:t}"
ln -s /Applications "$DMG_ROOT/Applications"
diskutil image create from --volumeName "mioh upscaler" --format UDZO \
  "$DMG_ROOT" "$DMG"
print "App: $APP"
print "DMG: $DMG"
