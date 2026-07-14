#!/bin/zsh
set -euo pipefail

PACKAGE_DIR="${0:A:h}"
ROOT="${PACKAGE_DIR:h:h:h}"

export COREAI_DISTRIBUTION="portable"
export BUILD_DIR="${BUILD_DIR:-$ROOT/build/macos-standalone-universal}"
export APP_BASENAME="mioh-universal"
export DMG_BASENAME="mioh-universal-0.11.0-unsigned"

exec "$PACKAGE_DIR/build_app.sh"
