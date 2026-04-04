#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
FABRIC_SAMPLES_DIR="$PROJECT_DIR/fabric-samples"
echo "Stopping Fabric network..."
[ -d "$FABRIC_SAMPLES_DIR/test-network" ] && \
    cd "$FABRIC_SAMPLES_DIR/test-network" && ./network.sh down 2>/dev/null || true
rm -rf "$PROJECT_DIR/organizations" "$PROJECT_DIR/channel-artifacts"
rm -f "$SCRIPT_DIR/fabric_connection.env" /tmp/trust-registry.tar.gz
echo "Done. Run ./setup.sh to restart."
