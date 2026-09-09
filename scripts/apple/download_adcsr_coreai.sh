#!/bin/zsh
# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h:h}"
DESTINATION="${1:-$REPO_ROOT/model_weights/adcsr_x4_float32.aimodel}"
EXPECTED_SHA256="33d2a727e24044912ca1f352ed3b946863f6770990f6fc553c1c134ac3d5423c"
BASE_URL="https://huggingface.co/mlboydaisuke/AdcSR-CoreAI/resolve/main/adcsr_x4_float32.aimodel"
LICENSE_URL="https://huggingface.co/mlboydaisuke/AdcSR-CoreAI/resolve/main/LICENSE"
LICENSE_FILE="${DESTINATION:h}/AdcSR-CoreAI-LICENSE.txt"

if [[ -z "$DESTINATION" || "$DESTINATION" == "/" ]]; then
  print -u2 "Refusing an unsafe AdcSR destination: $DESTINATION"
  exit 2
fi

mkdir -p "$DESTINATION"
MODEL_FILE="$DESTINATION/main.mlirb"
PART_FILE="$DESTINATION/main.mlirb.part"

if [[ -f "$MODEL_FILE" ]] \
    && [[ "$(shasum -a 256 "$MODEL_FILE" | awk '{print $1}')" == "$EXPECTED_SHA256" ]]; then
  print "AdcSR Core AI weights already verified: $MODEL_FILE"
else
  print "Downloading AdcSR FP32 Core AI weights (about 1.7 GB)…"
  curl -fL --retry 5 --retry-delay 2 --continue-at - \
    -o "$PART_FILE" "$BASE_URL/main.mlirb"
  ACTUAL_SHA256="$(shasum -a 256 "$PART_FILE" | awk '{print $1}')"
  if [[ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]]; then
    print -u2 "AdcSR checksum mismatch: expected $EXPECTED_SHA256, got $ACTUAL_SHA256"
    exit 1
  fi
  mv "$PART_FILE" "$MODEL_FILE"
fi

curl -fL --retry 5 -o "$DESTINATION/main.hash" "$BASE_URL/main.hash"
curl -fL --retry 5 -o "$DESTINATION/metadata.json" "$BASE_URL/metadata.json"
curl -fL --retry 5 -o "$LICENSE_FILE" "$LICENSE_URL"
print "AdcSR Core AI model ready: $DESTINATION"
