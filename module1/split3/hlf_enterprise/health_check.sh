#!/usr/bin/env bash
set -euo pipefail

# Enterprise preflight for isolated Fabric profile.
# Verifies container topology + channel + chaincode availability.

CONSENSUS_MODE="${1:-${BATFL_ENTERPRISE_CONSENSUS:-bft}}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TEST_NETWORK_DIR="$REPO_ROOT/fabric-samples/test-network"
ENV_FILE="$SCRIPT_DIR/fabric_connection.env"

if [ ! -d "$TEST_NETWORK_DIR" ]; then
  echo "[ERROR] test-network not found at $TEST_NETWORK_DIR" >&2
  exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
  echo "[ERROR] enterprise env file missing: $ENV_FILE" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$ENV_FILE"

export PATH="$REPO_ROOT/fabric-samples/bin:$PATH"
export FABRIC_CFG_PATH="$REPO_ROOT/fabric-samples/config"

if ! command -v peer >/dev/null 2>&1; then
  echo "[ERROR] peer binary not available on PATH" >&2
  exit 1
fi

check_running() {
  local name="$1"
  docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null | grep -q true
}

wait_for_containers() {
  local timeout_secs="${1:-120}"
  local interval_secs="${2:-4}"
  shift 2 || true
  local containers=("$@")
  local elapsed=0

  while [ "$elapsed" -lt "$timeout_secs" ]; do
    local missing=()
    for c in "${containers[@]}"; do
      if ! check_running "$c"; then
        missing+=("$c")
      fi
    done

    if [ "${#missing[@]}" -eq 0 ]; then
      return 0
    fi

    echo "[Enterprise] Waiting for containers: ${missing[*]}"
    sleep "$interval_secs"
    elapsed=$((elapsed + interval_secs))
  done

  return 1
}

required_common=(
  peer0.org1.example.com
  peer0.org2.example.com
)

required_orderers=(orderer.example.com)
if [ "$CONSENSUS_MODE" = "bft" ]; then
  required_orderers+=(orderer2.example.com orderer3.example.com orderer4.example.com)
fi

for c in "${required_common[@]}"; do
  :
done

for c in "${required_orderers[@]}"; do
  :
done

all_required=("${required_common[@]}" "${required_orderers[@]}")
if ! wait_for_containers 120 4 "${all_required[@]}"; then
  echo "[ERROR] Required containers did not become healthy in time." >&2
  echo "[ERROR] Expected (${CONSENSUS_MODE}): ${all_required[*]}" >&2
  echo "[ERROR] Currently running containers:" >&2
  docker ps --format '{{.Names}}' >&2 || true
  exit 1
fi

ORG1_CERT="$FABRIC_ORG1_CERT"
ORG1_MSP_DIR="${ORG1_CERT%/signcerts/*}"
ORDERER_CA="$FABRIC_ORDERER_TLS_CERT"
PEER1_TLS="$FABRIC_ORG1_TLS_CERT"
PEER2_TLS="$FABRIC_ORG2_TLS_CERT"

if [ ! -f "$ORDERER_CA" ] || [ ! -f "$PEER1_TLS" ] || [ ! -f "$PEER2_TLS" ]; then
  echo "[ERROR] TLS cert paths in enterprise env file are invalid" >&2
  exit 1
fi

export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID="Org1MSP"
export CORE_PEER_MSPCONFIGPATH="$ORG1_MSP_DIR"
export CORE_PEER_ADDRESS="${FABRIC_PEER_ORG1_ENDPOINT}"
export CORE_PEER_TLS_ROOTCERT_FILE="$PEER1_TLS"

cd "$TEST_NETWORK_DIR"

if ! peer channel getinfo -c "$FABRIC_CHANNEL" >/dev/null 2>&1; then
  echo "[ERROR] Channel check failed for $FABRIC_CHANNEL" >&2
  exit 1
fi

if ! peer lifecycle chaincode querycommitted -C "$FABRIC_CHANNEL" -n "$FABRIC_CHAINCODE" >/dev/null 2>&1; then
  echo "[ERROR] Chaincode $FABRIC_CHAINCODE is not committed on channel $FABRIC_CHANNEL" >&2
  exit 1
fi

if ! peer chaincode query -C "$FABRIC_CHANNEL" -n "$FABRIC_CHAINCODE" \
  -c '{"function":"ExportAuditTrail","Args":[]}' >/dev/null 2>&1; then
  echo "[ERROR] Chaincode query failed for $FABRIC_CHAINCODE on channel $FABRIC_CHANNEL" >&2
  echo "[ERROR] Chaincode may be unavailable on peer CLI path even if lifecycle metadata exists." >&2
  exit 1
fi

if ! peer chaincode invoke \
  -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
  --tls --cafile "$ORDERER_CA" -C "$FABRIC_CHANNEL" -n "$FABRIC_CHAINCODE" \
  --peerAddresses "${FABRIC_PEER_ORG1_ENDPOINT}" --tlsRootCertFiles "$PEER1_TLS" \
  --peerAddresses "${FABRIC_PEER_ORG2_ENDPOINT}" --tlsRootCertFiles "$PEER2_TLS" \
  --waitForEvent --waitForEventTimeout 30s \
  -c '{"function":"AppendEvent","Args":["HEALTHCHECK","0","enterprise-health","{}"]}' >/dev/null 2>&1; then
  echo "[ERROR] Chaincode invoke failed for $FABRIC_CHAINCODE on channel $FABRIC_CHANNEL" >&2
  echo "[ERROR] Re-run enterprise setup: module1/split3/hlf_enterprise/setup.sh" >&2
  exit 1
fi

echo "[Enterprise] Health check PASSED"
echo "[Enterprise] Consensus: $CONSENSUS_MODE"
echo "[Enterprise] Channel: $FABRIC_CHANNEL"
echo "[Enterprise] Chaincode: $FABRIC_CHAINCODE"
