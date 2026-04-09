#!/usr/bin/env bash
set -euo pipefail

# Isolated enterprise-mode setup wrapper.
# Uses the existing hlf/setup.sh implementation with opt-in overrides.
# This keeps the default working path unchanged.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_HLF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/hlf"
CONSENSUS_MODE="${BATFL_ENTERPRISE_CONSENSUS:-bft}"

if [ ! -x "$BASE_HLF_DIR/setup.sh" ]; then
  echo "[ERROR] Base setup script not found: $BASE_HLF_DIR/setup.sh" >&2
  exit 1
fi

export BATFL_USE_CA=true
export BATFL_STATE_DB=couchdb
export BATFL_FABRIC_ENV_OUT="$SCRIPT_DIR/fabric_connection.env"
export BATFL_CONSENSUS="$CONSENSUS_MODE"

echo "[Enterprise] Starting isolated Fabric setup (CA + CouchDB + ${BATFL_CONSENSUS^^})"
echo "[Enterprise] Env file: $BATFL_FABRIC_ENV_OUT"

cd "$BASE_HLF_DIR"
./setup.sh

echo "[Enterprise] Setup complete."
