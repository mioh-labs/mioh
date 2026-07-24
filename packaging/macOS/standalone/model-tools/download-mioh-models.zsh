#!/bin/zsh
set -euo pipefail

APP="/Applications/mioh-universal.app"
FORCE=0
MINIMAL=0

usage() {
  cat <<'EOF'
usage: download-mioh-models.zsh [--app /Applications/mioh-universal.app] [--minimal] [--force]

Downloads source model weights into the app's Contents/Resources/models folder.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --app)
      APP="$2"
      shift 2
      ;;
    --minimal)
      MINIMAL=1
      shift
      ;;
    --force)
      FORCE=1
      shift
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
if [[ ! -d "$RESOURCES" ]]; then
  print -u2 "missing app resources: $RESOURCES"
  print -u2 "install mioh-universal.app first, or pass --app /path/to/mioh-universal.app"
  exit 1
fi
mkdir -p "$MODELS" "$MODELS/3rd_party"

download() {
  local url="$1"
  local output="$2"
  if [[ -e "$output" && "$FORCE" != 1 ]]; then
    print "exists: $output"
    return
  fi
  print "download: $output"
  curl -fL --retry 3 --continue-at - -o "$output.part" "$url"
  mv "$output.part" "$output"
}

hf_lada() {
  local file="$1"
  print "https://huggingface.co/ladaapp/lada/resolve/main/$file?download=true"
}

download "$(hf_lada lada_mosaic_restoration_model_generic_v1.2.pth)" "$MODELS/lada_mosaic_restoration_model_generic_v1.2.pth"
download "$(hf_lada lada_mosaic_detection_model_v2.pt)" "$MODELS/lada_mosaic_detection_model_v2.pt"
download "$(hf_lada lada_mosaic_detection_model_v4_fast.pt)" "$MODELS/lada_mosaic_detection_model_v4_fast.pt"
download "$(hf_lada lada_mosaic_detection_model_v4_accurate.pt)" "$MODELS/lada_mosaic_detection_model_v4_accurate.pt"
download "$(hf_lada lada_mosaic_detection_model_vr_v2_accurate.pt)" "$MODELS/lada_mosaic_detection_model_vr_v2_accurate.pt"

download \
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
  "$MODELS/RealESRGAN_x2plus.pth"
download \
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
  "$MODELS/RealESRGAN_x4plus.pth"

if [[ "$MINIMAL" != 1 ]]; then
  download \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5/realesr-general-x4v3.pth" \
    "$MODELS/realesr-general-x4v3.pth"
  download \
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth" \
    "$MODELS/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"

  typeset -a NOMOS_FLAGS
  NOMOS_FLAGS=(--output-dir "$MODELS")
  if [[ "$FORCE" == 1 ]]; then
    NOMOS_FLAGS+=(--force)
  fi

  PYTHONHOME="$RESOURCES/runtime" \
  PYTHONPATH="$RESOURCES/runtime/lib/python3.12/site-packages" \
    "$RESOURCES/runtime/bin/python3.12" \
    "$RESOURCES/model-tools/scripts/download_nomos_roi_enhancers.py" \
    "${NOMOS_FLAGS[@]}"
fi

print "Downloaded model sources to: $MODELS"
