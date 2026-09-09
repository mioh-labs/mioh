#!/bin/zsh
set -euo pipefail

ROOT="${0:A:h:h:h:h}"
PACKAGE_DIR="$ROOT/packaging/macOS/standalone"
REMOTE_APP_SOURCE_DIR="$ROOT/apps/MiohRemote/MiohRemote"
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
COREAI_ARCHITECTURE="${COREAI_ARCHITECTURE:-h17s}"
DEDICATED_PREBUILT_MODELS="${DEDICATED_PREBUILT_MODELS:-$ROOT/model_weights/mioh-dedicated-$COREAI_ARCHITECTURE}"
LADA_STANDALONE_PYTHON_ENV="${LADA_STANDALONE_PYTHON_ENV:-${LADA_STANDALONE_VENV:-$ROOT/.venv-coreai}}"
LADA_STANDALONE_PYTHON_ENV="${LADA_STANDALONE_PYTHON_ENV:A}"
# Dedicated builds consume checked native assets and have no Python dependency.
# Python remains available only to the separate portable model-export path.
if [[ "$COREAI_DISTRIBUTION" == "portable" \
      && "$MIOH_MODELESS_DISTRIBUTION" != 1 ]]; then
  if [[ ! -x "$LADA_STANDALONE_PYTHON_ENV/bin/python" ]]; then
    print -u2 "Missing build-time Python: $LADA_STANDALONE_PYTHON_ENV/bin/python"
    print -u2 "Set LADA_STANDALONE_PYTHON_ENV to the model build environment."
    exit 1
  fi
fi
COMPILED_MODELS="${COMPILED_MODELS:-$BUILD_DIR/compiled-models}"
COMPILED_COREML_MODELS="${COMPILED_COREML_MODELS:-$BUILD_DIR/compiled-coreml-models}"
FFMPEG_CACHE="${FFMPEG_CACHE:-$BUILD_DIR/ffmpeg-static}"
PREVIEW_ENCODER_TARGET="arm64-apple-macosx26.0"

rm -rf "$APP" "$BUILD_DIR/Lada.app"
rm -f "$DMG" "$BUILD_DIR/Lada-0.11.0-unsigned.dmg" "$BUILD_DIR/mioh-0.11.0-unsigned.dmg"
mkdir -p "$CONTENTS/MacOS" "$RESOURCES/bin" "$RESOURCES/models"

typeset -a APP_SWIFT_FLAGS
APP_SWIFT_FLAGS=()
if [[ "$COREAI_DISTRIBUTION" == "portable" ]]; then
  APP_SWIFT_FLAGS+=(-D MIOH_PORTABLE_COREAI)
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
  -framework CoreMedia \
  -framework CoreVideo \
  -framework Metal \
  -framework Network \
  -framework QuickLookThumbnailing \
  -framework Security \
  -framework SceneKit \
  -framework UniformTypeIdentifiers \
  -framework WebKit \
  "${APP_SWIFT_FLAGS[@]}" \
  "$REMOTE_APP_SOURCE_DIR/IPadBrowserLibraryStore.swift" \
  "$REMOTE_APP_SOURCE_DIR/IPadWebMediaDiscovery.swift" \
  "$REMOTE_APP_SOURCE_DIR/IPadMediaURLResolver.swift" \
  "$REMOTE_APP_SOURCE_DIR/IPadAuthenticatedMediaProxy.swift" \
  "$REMOTE_APP_SOURCE_DIR/IPadMPEGTSRemuxer.swift" \
  "$REMOTE_APP_SOURCE_DIR/IPadInteractiveMediaBrowser.swift" \
  "$PACKAGE_DIR/MacChildProcessPipe.swift" \
  "$PACKAGE_DIR/MacNativeExportBatch.swift" \
  "$PACKAGE_DIR/InputPanelThumbnailCache.swift" \
  "$PACKAGE_DIR/MiohApp.swift" \
  "$PACKAGE_DIR/MacMediaBrowser.swift" \
  "$PACKAGE_DIR/MacHLSAVFoundationCapture.swift" \
  "$PACKAGE_DIR/MacHLSUnifiedPlayback.swift" \
  "$PACKAGE_DIR/MacHLSRealtimePipeline.swift" \
  "$PACKAGE_DIR/RealtimePlayer.swift" \
  "$PACKAGE_DIR/RemoteControlServer.swift" \
  "$PACKAGE_DIR/RemoteStreamingCoordinator.swift" \
  "$PACKAGE_DIR/RemoteClusterService.swift" \
  "$PACKAGE_DIR/RemoteClusterHTTPTransfer.swift" \
  "$PACKAGE_DIR/MiohClusterController.swift" \
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
  "$ROOT/packages/MiohRemoteKit/Sources/MiohRemoteKit/MiohHTTPRangeAsset.swift" \
  "$PACKAGE_DIR/MacChildProcessPipe.swift" \
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
    -framework CoreML \
    "$PACKAGE_DIR/DedicatedModelVerifier.swift" \
    -o "$RESOURCES/bin/mioh-dedicated-model-verifier"
fi

cp "$PACKAGE_DIR/Info.plist" "$CONTENTS/Info.plist"
if [[ -d "$PACKAGE_DIR/Localizations" ]]; then
  for localization in "$PACKAGE_DIR/Localizations"/*.lproj(N); do
    ditto "$localization" "$RESOURCES/${localization:t}"
  done
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
if [[ "$COREAI_DISTRIBUTION" == "portable" && -d "$MODEL_TOOLS_SOURCE" ]]; then
  mkdir -p "$RESOURCES/model-tools/scripts"
  ditto "$MODEL_TOOLS_SOURCE" "$RESOURCES/model-tools"
  ditto "$ROOT/scripts/apple" "$RESOURCES/model-tools/scripts/apple"
  # RF-DETR remains a local research prototype. Keep it out of the shipped
  # application and model-tools bundle until it is deliberately reintroduced.
  find "$RESOURCES/model-tools/scripts/apple" \
    -maxdepth 1 -type f -iname '*rfdetr*' -delete
  # Upscaler models and their download tooling ship only with Mioh Upscaler.
  find "$RESOURCES/model-tools/scripts/apple" \
    -maxdepth 1 -type f -name 'download_adcsr_coreai.sh' -delete
  find "$RESOURCES/model-tools/scripts/apple" \
    -type d -name __pycache__ -prune -exec rm -rf {} +
  cp "$ROOT/scripts/download_nomos_roi_enhancers.py" \
    "$RESOURCES/model-tools/scripts/download_nomos_roi_enhancers.py"
chmod +x \
    "$RESOURCES/model-tools/download-mioh-models.zsh" \
    "$RESOURCES/model-tools/convert-mioh-models.zsh"
fi

if [[ "$MIOH_MODELESS_DISTRIBUTION" != 1 ]]; then

if [[ "$COREAI_DISTRIBUTION" == "portable" ]]; then
  MODEL_ASSETS=(
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
fi
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
if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
  COREML_DETECTION_ASSETS+=(
    rfdetr-v6-576-fp32.mlpackage
    rfdetr-v6-large-768-fp32.mlpackage
  )
fi
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
if [[ "$COREAI_DISTRIBUTION" == "portable" ]]; then
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
fi

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
VARIABLE_COREAI_ASSETS=(
  spatial6 flow6
  backward_1_start6 backward_1_continue6
  forward_1_start6 forward_1_continue6
  backward_2_start6 backward_2_continue6
  forward_2_start6 forward_2_continue6
  reconstruction6
)
variable_continuations_use_native_state() {
  local source_root="$1"
  local name source_asset inspection state_name
  for name in \
    backward_1_continue6 forward_1_continue6 \
    backward_2_continue6 forward_2_continue6; do
    source_asset="$source_root/basicvsrpp-variable-$name.aimodel"
    [[ -d "$source_asset" ]] || return 1
    inspection="$(xcrun coreai-build inspect "$source_asset" --json 2>/dev/null)" \
      || return 1
    for state_name in state_n1 state_n2 flow_previous; do
      [[ "$inspection" == *"$state_name"* ]] || return 1
    done
  done
}
if [[ "$COREAI_DISTRIBUTION" == "portable" ]]; then
  VARIABLE_COREAI_SOURCE_MODELS="${VARIABLE_COREAI_SOURCE_MODELS:-$BUILD_DIR/variable-basicvsrpp-source}"
  VARIABLE_COREAI_CHECKPOINT="${VARIABLE_COREAI_CHECKPOINT:-$ROOT/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth}"
  if [[ ! -f "$VARIABLE_COREAI_CHECKPOINT" ]]; then
    print -u2 "Missing portable variable restoration checkpoint: $VARIABLE_COREAI_CHECKPOINT"
    exit 1
  fi
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
  if (( ! needs_variable_export )) \
    && ! variable_continuations_use_native_state "$VARIABLE_COREAI_SOURCE_MODELS"; then
    needs_variable_export=1
  fi
  if (( needs_variable_export )); then
    mkdir -p "$VARIABLE_COREAI_SOURCE_MODELS"
    PYTHONPATH="$ROOT" "$LADA_STANDALONE_PYTHON_ENV/bin/python" \
      "$ROOT/scripts/apple/export_basicvsrpp_variable_chunk6.py" \
      --checkpoint "$VARIABLE_COREAI_CHECKPOINT" \
      --output-dir "$VARIABLE_COREAI_SOURCE_MODELS" \
      --native-state-continuations \
      --overwrite
  fi
fi
if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
  DEDICATED_COREAI_ASSETS=(
    basicvsrpp-v1.2-t18-fp16.$COREAI_ARCHITECTURE.aimodelc
    basicvsrpp-v1.2-t36-fp16.$COREAI_ARCHITECTURE.aimodelc
    basicvsrpp-v1.2-t90-fp16.$COREAI_ARCHITECTURE.aimodelc
    lada_mosaic_detection_model_v2-fp16.$COREAI_ARCHITECTURE.aimodelc
    lada_mosaic_detection_model_v3.1_fast-fp16.$COREAI_ARCHITECTURE.aimodelc
    lada_mosaic_detection_model_v3.1_accurate-fp16.$COREAI_ARCHITECTURE.aimodelc
    lada_mosaic_detection_model_v4_fast-fp16.$COREAI_ARCHITECTURE.aimodelc
    lada_mosaic_detection_model_v4_accurate-fp16.$COREAI_ARCHITECTURE.aimodelc
    lada_mosaic_detection_model_vr_v2_accurate-fp16.$COREAI_ARCHITECTURE.aimodelc
    RealESRGAN_x2plus-256-fp16.$COREAI_ARCHITECTURE.aimodelc
    RealESRGAN_x4plus-256-fp16.$COREAI_ARCHITECTURE.aimodelc
    realesr-general-x4v3-256-fp16.$COREAI_ARCHITECTURE.aimodelc
    4xNomosWebPhoto_RealPLKSR-256-fp16.$COREAI_ARCHITECTURE.aimodelc
  )
  if [[ ! -d "$DEDICATED_PREBUILT_MODELS" ]]; then
    print -u2 "Missing native Dedicated model set: $DEDICATED_PREBUILT_MODELS"
    exit 1
  fi
  for asset in "${DEDICATED_COREAI_ASSETS[@]}"; do
    source_model="$DEDICATED_PREBUILT_MODELS/$asset"
    if [[ ! -d "$source_model" ]]; then
      print -u2 "Missing native Dedicated model: $source_model"
      exit 1
    fi
    ditto "$source_model" "$RESOURCES/models/$asset"
  done
  # The public variable identifier is the established pre-large-ROI model.
  # Large-ROI weights and their separate runtime path are not packaged.
  standard_variable_asset="basicvsrpp-v1.2-standard-variable-coreai.$COREAI_ARCHITECTURE.aimodelc"
  active_variable_asset="basicvsrpp-v1.2-variable-coreai.$COREAI_ARCHITECTURE.aimodelc"
  source_standard_variable="$DEDICATED_PREBUILT_MODELS/$standard_variable_asset"
  if [[ ! -d "$source_standard_variable" ]]; then
    print -u2 "Missing standard variable restoration model: $source_standard_variable"
    exit 1
  fi
  ditto "$source_standard_variable" "$RESOURCES/models/$active_variable_asset"
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

if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
  source_metadata="$DEDICATED_PREBUILT_MODELS/basicvsrpp-v1.2-standard-variable-coreai.provenance.json"
  if [[ ! -f "$source_metadata" ]]; then
    print -u2 "Missing standard variable restoration metadata: $source_metadata"
    exit 1
  fi
  cp "$source_metadata" \
    "$RESOURCES/models/basicvsrpp-v1.2-variable-coreai.provenance.json"
else
  # Portable exports retain an auditable checkpoint identity. The Dedicated
  # build copies immutable provenance alongside its prebuilt native assets.
  write_variable_coreai_provenance() {
  "$LADA_STANDALONE_PYTHON_ENV/bin/python" - \
    "$1" "$2" "$3" "$4" "$5" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1])
destination = Path(sys.argv[2])
distribution = sys.argv[3]
hq_asset_count = int(sys.argv[4])
expected_sha256 = sys.argv[5]
digest = hashlib.sha256()
with checkpoint.open("rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
checkpoint_sha256 = digest.hexdigest()
if expected_sha256 and checkpoint_sha256 != expected_sha256:
    raise SystemExit(
        f"checkpoint SHA-256 mismatch: {checkpoint_sha256}; "
        f"expected {expected_sha256}"
    )
payload = {
    "format_version": 1,
    "checkpoint_filename": checkpoint.name,
    "checkpoint_sha256": checkpoint_sha256,
    "checkpoint_size": checkpoint.stat().st_size,
    "distribution": distribution,
    "chunk_size": 6,
    "chunk_asset_count": 11,
    "hq_asset_count": hq_asset_count,
}
destination.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
  }
  write_variable_coreai_provenance \
    "$VARIABLE_COREAI_CHECKPOINT" \
    "$RESOURCES/models/basicvsrpp-v1.2-variable-coreai.provenance.json" \
    "$COREAI_DISTRIBUTION" \
    0 \
    ""
fi
else
  print "Modeless distribution: skipping bundled model assets and Core ML/Core AI exports"
fi

# Cluster identity is derived from portable source assets, never from the
# machine-specific compiled .aimodelc/.mlmodelc layout. Dedicated Macs and
# portable iPad/Mac Workers can therefore compare the same model identity.
# The variable restorer is one logical model made from exactly eleven source
# assets; its digest is the tree digest of that virtual collection.
CANONICAL_MODEL_MANIFEST="$RESOURCES/models/mioh-cluster-model-identities-v1.json"
if [[ "$MIOH_MODELESS_DISTRIBUTION" == 1 ]]; then
  print "Modeless distribution: skipping cluster model identity manifest"
elif [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
  source_manifest="$DEDICATED_PREBUILT_MODELS/mioh-cluster-model-identities-v1.json"
  if [[ ! -f "$source_manifest" ]]; then
    print -u2 "Missing native Dedicated identity manifest: $source_manifest"
    exit 1
  fi
  cp "$source_manifest" "$CANONICAL_MODEL_MANIFEST"
  # The runtime model was renamed to the public variable identifier above.
  # Keep the cluster identity tied to those standard weights and remove the
  # retired implementation-only alias.
  plutil -convert xml1 "$CANONICAL_MODEL_MANIFEST"
  /usr/libexec/PlistBuddy \
    -c 'Delete :models:basicvsrpp-v1.2-coreai-variable' \
    "$CANONICAL_MODEL_MANIFEST"
  /usr/libexec/PlistBuddy \
    -c 'Copy :models:basicvsrpp-v1.2-coreai-variable-standard :models:basicvsrpp-v1.2-coreai-variable' \
    "$CANONICAL_MODEL_MANIFEST"
  /usr/libexec/PlistBuddy \
    -c 'Delete :models:basicvsrpp-v1.2-coreai-variable-standard' \
    "$CANONICAL_MODEL_MANIFEST"
  plutil -convert json "$CANONICAL_MODEL_MANIFEST"
else
VARIABLE_COREAI_SOURCE_MODELS="${VARIABLE_COREAI_SOURCE_MODELS:-$BUILD_DIR/variable-basicvsrpp-source}"
canonical_standard_variable_root=""
VARIABLE_COREAI_CHECKPOINT="${VARIABLE_COREAI_CHECKPOINT:-$ROOT/model_weights/lada_mosaic_restoration_model_generic_v1.2.pth}"
CANONICAL_VARIABLE_ASSETS=(
  spatial6 flow6
  backward_1_start6 backward_1_continue6
  forward_1_start6 forward_1_continue6
  backward_2_start6 backward_2_continue6
  forward_2_start6 forward_2_continue6
  reconstruction6
)
needs_canonical_variable_export=0
for name in "${CANONICAL_VARIABLE_ASSETS[@]}"; do
  canonical_source_asset="$VARIABLE_COREAI_SOURCE_MODELS/basicvsrpp-variable-$name.aimodel"
  if [[ ! -d "$canonical_source_asset" \
        || "$VARIABLE_COREAI_CHECKPOINT" -nt "$canonical_source_asset" \
        || "$ROOT/scripts/apple/basicvsrpp_coreai_kernels.py" -nt "$canonical_source_asset" \
        || "$ROOT/scripts/apple/export_basicvsrpp_variable_chunk6.py" -nt "$canonical_source_asset" ]]; then
    needs_canonical_variable_export=1
    break
  fi
done
if (( ! needs_canonical_variable_export )) \
  && ! variable_continuations_use_native_state "$VARIABLE_COREAI_SOURCE_MODELS"; then
  needs_canonical_variable_export=1
fi
if (( needs_canonical_variable_export )); then
  if [[ ! -f "$VARIABLE_COREAI_CHECKPOINT" ]]; then
    print -u2 "Missing checkpoint required for cluster identity manifest: $VARIABLE_COREAI_CHECKPOINT"
    exit 1
  fi
  mkdir -p "$VARIABLE_COREAI_SOURCE_MODELS"
  PYTHONPATH="$ROOT" "$LADA_STANDALONE_PYTHON_ENV/bin/python" \
    "$ROOT/scripts/apple/export_basicvsrpp_variable_chunk6.py" \
    --checkpoint "$VARIABLE_COREAI_CHECKPOINT" \
    --output-dir "$VARIABLE_COREAI_SOURCE_MODELS" \
    --native-state-continuations \
    --overwrite
fi

"$LADA_STANDALONE_PYTHON_ENV/bin/python" - \
  "$ROOT/model_weights" \
  "$VARIABLE_COREAI_SOURCE_MODELS" \
  "$canonical_standard_variable_root" \
  "$CANONICAL_MODEL_MANIFEST" <<'PY'
import hashlib
import json
import sys
import unicodedata
from pathlib import Path

weights = Path(sys.argv[1])
variable_root = Path(sys.argv[2])
standard_variable_root = Path(sys.argv[3]) if sys.argv[3] else None
destination = Path(sys.argv[4])


def update_file(digest, path):
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)


def tree_digest(root):
    if root.is_symlink():
        raise SystemExit(f"canonical model asset contains symlink: {root}")
    digest = hashlib.sha256()
    if root.is_file():
        update_file(digest, root)
        return digest.hexdigest()
    if not root.is_dir():
        raise SystemExit(f"canonical model asset is missing: {root}")
    entries = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise SystemExit(f"canonical model asset contains symlink: {candidate}")
        if candidate.is_file():
            relative = unicodedata.normalize("NFC", candidate.relative_to(root).as_posix())
            entries.append((relative, candidate))
    entries.sort(key=lambda item: item[0].encode("utf-8"))
    if len({relative for relative, _ in entries}) != len(entries):
        raise SystemExit(f"canonical model asset has duplicate normalized paths: {root}")
    for relative, candidate in entries:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        update_file(digest, candidate)
        digest.update(b"\0")
    return digest.hexdigest()


def collection_digest(assets):
    digest = hashlib.sha256()
    entries = []
    for asset in assets:
        if asset.is_symlink() or not asset.is_dir():
            raise SystemExit(f"canonical variable asset is invalid: {asset}")
        for candidate in asset.rglob("*"):
            if candidate.is_symlink():
                raise SystemExit(f"canonical variable asset contains symlink: {candidate}")
            if candidate.is_file():
                relative = unicodedata.normalize(
                    "NFC", f"{asset.name}/{candidate.relative_to(asset).as_posix()}"
                )
                entries.append((relative, candidate))
    entries.sort(key=lambda item: item[0].encode("utf-8"))
    if len({relative for relative, _ in entries}) != len(entries):
        raise SystemExit("canonical variable collection has duplicate normalized paths")
    for relative, candidate in entries:
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        update_file(digest, candidate)
        digest.update(b"\0")
    return digest.hexdigest()


models = {}


def add(model_id, source, *, required=True, asset_type="source-tree"):
    path = weights / source
    if not path.exists():
        if required:
            raise SystemExit(f"canonical source asset is missing for {model_id}: {path}")
        return
    models[model_id] = {
        "sha256": tree_digest(path),
        "asset_type": asset_type,
        "source_assets": [source],
    }


add("basicvsrpp-v1.2-coreai", "basicvsrpp-v1.2-t18-fp16.aimodel")
add("basicvsrpp-v1.2-coreai-t36", "basicvsrpp-v1.2-t36-fp16.aimodel")
add("basicvsrpp-v1.2-coreai-t90", "basicvsrpp-v1.2-t90-fp16.aimodel")

variable_names = [
    "spatial6", "flow6",
    "backward_1_start6", "backward_1_continue6",
    "forward_1_start6", "forward_1_continue6",
    "backward_2_start6", "backward_2_continue6",
    "forward_2_start6", "forward_2_continue6",
    "reconstruction6",
]
variable_assets = [variable_root / f"basicvsrpp-variable-{name}.aimodel" for name in variable_names]
models["basicvsrpp-v1.2-coreai-variable"] = {
    "sha256": collection_digest(variable_assets),
    "asset_type": "source-collection",
    "source_assets": [asset.name for asset in variable_assets],
}
if standard_variable_root is not None:
    standard_variable_assets = [
        standard_variable_root / f"basicvsrpp-variable-{name}.aimodel"
        for name in variable_names
    ]
    models["basicvsrpp-v1.2-coreai-variable-standard"] = {
        "sha256": collection_digest(standard_variable_assets),
        "asset_type": "source-collection",
        "source_assets": [asset.name for asset in standard_variable_assets],
    }

detection_stems = {
    "v2": "lada_mosaic_detection_model_v2",
    "v3.1-fast": "lada_mosaic_detection_model_v3.1_fast",
    "v3.1-accurate": "lada_mosaic_detection_model_v3.1_accurate",
    "v4-fast": "lada_mosaic_detection_model_v4_fast",
    "v4-accurate": "lada_mosaic_detection_model_v4_accurate",
    "vr-v2-accurate": "lada_mosaic_detection_model_vr_v2_accurate",
}
for model_id, stem in detection_stems.items():
    add(f"{model_id}-coreai", f"{stem}-fp16.aimodel")
    add(f"{model_id}-coreml", f"{stem}.mlpackage")

optional_assets = {
    "realesrgan-x2-coreai": "RealESRGAN_x2plus-256-fp16.aimodel",
    "realesrgan-x2": "RealESRGAN_x2plus_256.mlpackage",
    "realesrgan-x4-coreai": "RealESRGAN_x4plus-256-fp16.aimodel",
    "realesrgan-x4": "RealESRGAN_x4plus_256.mlpackage",
    "realesrgan-x4-coreml": "RealESRGAN_x4plus_256.mlpackage",
    "realesr-general-x4v3-coreai": "realesr-general-x4v3-256-fp16.aimodel",
    "realesr-general-x4v3-coreml": "realesr-general-x4v3_256.mlpackage",
    "mewzoom-x4-coreml": "MewZoom-V1-4X-Unet_256.mlpackage",
    "mewzoom-x4-coreml-512": "MewZoom-V1-4X-Unet_512.mlpackage",
    "swinir-x4-coreml": "swinir-real-x4_256.mlpackage",
    "swinir-real-x4-coreml": "swinir-real-x4_256.mlpackage",
    "nomos-webphoto-realplksr-x4-coreai": "4xNomosWebPhoto_RealPLKSR-256-fp16.aimodel",
    "nomos-webphoto-realplksr-x4": "4xNomosWebPhoto_RealPLKSR_256.mlpackage",
    "nomos-webphoto-realplksr-x4-coreml": "4xNomosWebPhoto_RealPLKSR_256.mlpackage",
    "jasna-v6-coreai": "rfdetr-v6-576-fp32.aimodel",
    "jasna-v6-coreml": "rfdetr-v6-576-fp32.mlpackage",
    "jasna-v6-large-coreai": "rfdetr-v6-large-768-fp32.aimodel",
    "jasna-v6-large-coreml": "rfdetr-v6-large-768-fp32.mlpackage",
}
for model_id, source in optional_assets.items():
    add(model_id, source, required=False)

payload = {
    "format_version": 1,
    "digest_algorithm": "sha256-tree-v1",
    "models": dict(sorted(models.items())),
}
destination.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
fi

cp "$ROOT/LICENSE.md" "$RESOURCES/LICENSE.md"
ditto "$ROOT/LICENSES" "$RESOURCES/LICENSES"

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
  chmod +x \
    "$RESOURCES/bin/mioh-dedicated-model-verifier"
fi

if [[ "$MIOH_MODELESS_DISTRIBUTION" == 1 ]]; then
  print "Skipping model smoke tests for modeless distribution"
elif [[ "${MIOH_SKIP_HARDWARE_SMOKE:-0}" == "1" ]]; then
  print "Skipping MPS/Core AI hardware smoke tests by request"
elif [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
  "$RESOURCES/bin/mioh-dedicated-model-verifier" \
    "$RESOURCES/models" \
    "$COREAI_ARCHITECTURE"
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

if [[ "$COREAI_DISTRIBUTION" == "dedicated" ]]; then
  dedicated_python_files=("$APP"/**/*.(py|pyc|pyo)(N))
  if (( ${#dedicated_python_files[@]} )); then
    print -u2 "Dedicated app unexpectedly contains Python files:"
    print -u2 -- "${(F)dedicated_python_files}"
    exit 1
  fi
  dedicated_python_weights=("$RESOURCES/models"/**/*.pth(N))
  if (( ${#dedicated_python_weights[@]} )); then
    print -u2 "Dedicated app unexpectedly contains Python checkpoints:"
    print -u2 -- "${(F)dedicated_python_weights}"
    exit 1
  fi
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
