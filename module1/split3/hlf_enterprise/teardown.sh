#!/usr/bin/env bash
set -euo pipefail

# Enterprise-mode teardown wrapper.
# Uses the same underlying network cleanup script as default mode.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_HLF_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/hlf"

if [ ! -x "$BASE_HLF_DIR/teardown.sh" ]; then
  echo "[ERROR] Base teardown script not found: $BASE_HLF_DIR/teardown.sh" >&2
  exit 1
fi

cd "$BASE_HLF_DIR"
./teardown.sh

echo "[Enterprise] Teardown complete."
