#!/bin/zsh
set -euo pipefail

APP="/Applications/mioh-universal.app"
FORCE=0
MINIMAL=0
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
