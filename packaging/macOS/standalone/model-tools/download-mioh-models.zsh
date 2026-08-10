#!/bin/zsh
set -euo pipefail

APP="/Applications/mioh-universal.app"
FORCE=0
MINIMAL=0
MIOH_RELEASE_TAG="${MIOH_RELEASE_TAG:-v0.14.3-007}"
typeset -a FAILURES
FAILURES=()

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
  local expected_sha="${3:-}"
  local actual_sha
  if [[ -e "$output" && "$FORCE" != 1 ]]; then
    if [[ -z "$expected_sha" ]]; then
      print "exists: $output"
      return 0
    fi
    actual_sha="$(shasum -a 256 "$output" | awk '{print $1}')"
    if [[ "$actual_sha" == "$expected_sha" ]]; then
      print "verified: $output"
      return 0
    fi
    print -u2 "checksum mismatch; downloading again: $output"
    rm -f "$output"
  fi
  print "download: $output"
  mkdir -p "${output:h}"
  # Never resume an error response or a partial file from an older URL.
  # A model becomes visible to mioh only after the complete payload and its
  # checksum have both been verified.
  rm -f "$output.part"
  if ! curl -fL \
    --retry 3 \
    --retry-all-errors \
    --connect-timeout 30 \
    -o "$output.part" \
    "$url"; then
    print -u2 "download failed: $url"
    rm -f "$output.part"
    return 1
  fi
  if [[ ! -s "$output.part" ]]; then
    print -u2 "download produced an empty file: $url"
    rm -f "$output.part"
    return 1
  fi
  if [[ -n "$expected_sha" ]]; then
    actual_sha="$(shasum -a 256 "$output.part" | awk '{print $1}')"
    if [[ "$actual_sha" != "$expected_sha" ]]; then
      print -u2 "checksum mismatch: ${output:t}"
      print -u2 "  expected: $expected_sha"
      print -u2 "  actual:   $actual_sha"
      rm -f "$output.part"
      return 1
    fi
  fi
  mv "$output.part" "$output"
  return 0
}

fetch() {
  local url="$1"
  local output="$2"
  local expected_sha="${3:-}"
  if ! download "$url" "$output" "$expected_sha"; then
    FAILURES+=("${output:t}")
  fi
}

hf_lada() {
  local file="$1"
  print "https://huggingface.co/ladaapp/lada/resolve/main/$file?download=true"
}

fetch \
  "$(hf_lada lada_mosaic_restoration_model_generic_v1.2.pth)" \
  "$MODELS/lada_mosaic_restoration_model_generic_v1.2.pth" \
  "d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee"
fetch \
  "$(hf_lada lada_mosaic_detection_model_v2.pt)" \
  "$MODELS/lada_mosaic_detection_model_v2.pt" \
  "056756fcab250bcdf0833e75aac33e2197b8809b0ab8c16e14722dcec94269b5"
fetch \
  "$(hf_lada lada_mosaic_detection_model_v3.1_fast.pt)" \
  "$MODELS/lada_mosaic_detection_model_v3.1_fast.pt" \
  "25d62894c16bba00468f3bcc160360bb84726b2f92751b5e235578bf2f9b0820"
fetch \
  "$(hf_lada lada_mosaic_detection_model_v3.1_accurate.pt)" \
  "$MODELS/lada_mosaic_detection_model_v3.1_accurate.pt" \
  "2b6e5d6cd5a795a4dcc1205b817a7323a4bd3725cef1a7de3a172cb5689f0368"
fetch \
  "$(hf_lada lada_mosaic_detection_model_v4_fast.pt)" \
  "$MODELS/lada_mosaic_detection_model_v4_fast.pt" \
  "9a6b660d1d3e3797d39515e08b0e72fcc59815f38279faa7a4ab374ab2c1e3b4"
fetch \
  "$(hf_lada lada_mosaic_detection_model_v4_accurate.pt)" \
  "$MODELS/lada_mosaic_detection_model_v4_accurate.pt" \
  "c244d7e49d8f88e264b8dc15f91fb21f5908ad8fb6f300b7bc88462d0801bc1f"
# The VR checkpoint is not present in ladaapp/lada on Hugging Face. It is a
# mioh release asset and must stay pinned to the release that publishes it.
fetch \
  "https://github.com/mioh-labs/mioh/releases/download/$MIOH_RELEASE_TAG/lada_mosaic_detection_model_vr_v2_accurate.pt" \
  "$MODELS/lada_mosaic_detection_model_vr_v2_accurate.pt" \
  "91fe7a48b0e9edf51361918c8a30f752c64511005e643343a7382d951f3fe0f8"

fetch \
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" \
  "$MODELS/RealESRGAN_x2plus.pth" \
  "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"
fetch \
  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
  "$MODELS/RealESRGAN_x4plus.pth" \
  "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1"

if [[ "$MINIMAL" != 1 ]]; then
  fetch \
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth" \
    "$MODELS/realesr-general-x4v3.pth" \
    "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292"
  fetch \
    "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth" \
    "$MODELS/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth" \
    "b9afb61e65e04eb7f8aba5095d070bbe9af28df76acd0c9405aeb33b814bcfc6"
  fetch \
    "https://raw.githubusercontent.com/JingyunLiang/SwinIR/6545850fbf8df298df73d81f3e8cba638787c8bd/models/network_swinir.py" \
    "$MODELS/3rd_party/SwinIR/models/network_swinir.py"

  typeset -a NOMOS_FLAGS
  NOMOS_FLAGS=(--output-dir "$MODELS")
  if [[ "$FORCE" == 1 ]]; then
    NOMOS_FLAGS+=(--force)
  fi

  if ! PYTHONHOME="$RESOURCES/runtime" \
    PYTHONPATH="$RESOURCES/runtime/lib/python3.12/site-packages" \
      "$RESOURCES/runtime/bin/python3.12" \
      "$RESOURCES/model-tools/scripts/download_nomos_roi_enhancers.py" \
      "${NOMOS_FLAGS[@]}"; then
    FAILURES+=("Nomos ROI enhancer models")
  fi
fi

if (( ${#FAILURES[@]} > 0 )); then
  print -u2 ""
  print -u2 "The following downloads failed:"
  for failure in "${FAILURES[@]}"; do
    print -u2 "  - $failure"
  done
  print -u2 "Fix the reported URL/network issue and run this script again."
  exit 1
fi

print "Downloaded model sources to: $MODELS"
