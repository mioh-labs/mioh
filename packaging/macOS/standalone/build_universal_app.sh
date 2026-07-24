#!/bin/zsh
set -euo pipefail

PACKAGE_DIR="${0:A:h}"
ROOT="${PACKAGE_DIR:h:h:h}"

export COREAI_DISTRIBUTION="portable"
export BUILD_DIR="${BUILD_DIR:-$ROOT/build/macos-standalone-universal}"
export APP_BASENAME="mioh-universal"
export DMG_BASENAME="mioh-universal-0.11.0-unsigned"
export INCLUDE_USER_MANUAL=1
export MIOH_MODELESS_DISTRIBUTION="${MIOH_MODELESS_DISTRIBUTION:-1}"

exec "$PACKAGE_DIR/build_app.sh"
