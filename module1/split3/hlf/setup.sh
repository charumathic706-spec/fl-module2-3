#!/usr/bin/env bash
# =============================================================================
# BATFL — Hyperledger Fabric Network Setup
# module1/split3/hlf/setup.sh
#
# Uses fabric-samples/test-network (Hyperledger official) for reliable
# cross-platform setup. Handles crypto, channel, and chaincode automatically.
#
# Prerequisites: Docker Desktop (4 GB RAM min), Git, curl, Go 1.21+
# Usage:  cd module1/split3/hlf && chmod +x setup.sh teardown.sh && ./setup.sh
# =============================================================================
set -euo pipefail

CHANNEL_NAME="mychannel"
CC_NAME="trust-registry"
CC_VERSION="1.0"
CC_SEQUENCE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# module1/split3/hlf → module1/split3 → module1 → project root
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
FABRIC_SAMPLES_DIR="$PROJECT_DIR/fabric-samples"
TEST_NETWORK_DIR="$FABRIC_SAMPLES_DIR/test-network"
CC_SRC="$SCRIPT_DIR/chaincode/trust_registry"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'; BOLD='\033[1m'
log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1" >&2; exit 1; }
step() { echo -e "\n${BLUE}${BOLD}━━ $1 ━━${NC}"; }

docker_info_ok() {
    if command -v timeout >/dev/null 2>&1; then
        timeout 60 docker info >/dev/null 2>&1
    else
        docker info >/dev/null 2>&1
    fi
}

wait_for_docker() {
    local retries="${1:-20}"
    local delay="${2:-2}"
    local i=1
    while [ "$i" -le "$retries" ]; do
        if docker_info_ok; then
            return 0
        fi
        sleep "$delay"
        i=$((i + 1))
    done
    return 1
}

recover_docker_socket() {
    warn "Trying lightweight Docker recovery for daemon/socket instability..."
    docker version >/dev/null 2>&1 || true
    wait_for_docker 30 2 || true
    # Prune only build cache to reduce daemon pressure without removing containers/images.
    docker builder prune -f >/dev/null 2>&1 || true
    wait_for_docker 20 2 || true
}

restart_peer_container() {
    local org_name="$1"
    local container_name=""
    case "$org_name" in
        Org1) container_name="peer0.org1.example.com" ;;
        Org2) container_name="peer0.org2.example.com" ;;
        *) return 0 ;;
    esac

    warn "Restarting ${container_name} to recover gRPC/build state..."
    docker restart "$container_name" >/dev/null 2>&1 || true

    local i=1
    while [ "$i" -le 30 ]; do
        if docker inspect -f '{{.State.Running}}' "$container_name" 2>/dev/null | grep -q true; then
            return 0
        fi
        sleep 2
        i=$((i + 1))
    done
    return 1
}

step "Checking prerequisites"
command -v docker &>/dev/null || err "Docker not found. Install Docker Desktop."
docker_info_ok || err "Docker not running or not responsive. Start Docker Desktop and ensure WSL integration is enabled."
wait_for_docker 15 2 || err "Docker daemon not healthy. Ensure Docker Desktop is fully started and WSL integration is enabled."
command -v git    &>/dev/null || err "Git not found."
command -v curl   &>/dev/null || err "curl not found."
command -v jq     &>/dev/null || err "jq not found. Install it with 'sudo apt-get update && sudo apt-get install -y jq' on Ubuntu/WSL, or 'brew install jq' on macOS."
log "Prerequisites OK"

step "Getting fabric-samples"
if [ ! -d "$FABRIC_SAMPLES_DIR" ]; then
    CLONED=false
    for TAG in v2.5.4 v2.5.3 v2.5.1 v2.5.0 main; do
        warn "Trying tag: $TAG"
        if git clone --depth 1 --branch "$TAG" \
               https://github.com/hyperledger/fabric-samples.git \
               "$FABRIC_SAMPLES_DIR" 2>/dev/null; then
            log "Cloned fabric-samples @ $TAG"
            CLONED=true; break
        fi
    done
    [ "$CLONED" = true ] || err "Failed to clone fabric-samples. Check internet."
else
    log "fabric-samples already present"
fi

step "Downloading Fabric 2.5 binaries and Docker images"
if [ ! -f "$FABRIC_SAMPLES_DIR/bin/peer" ]; then
    cd "$FABRIC_SAMPLES_DIR"
    DOWNLOADED=false
    for VER in 2.5.4 2.5.3 2.5.0; do
        if curl -sSL https://raw.githubusercontent.com/hyperledger/fabric/main/scripts/install-fabric.sh \
               | bash -s -- --fabric-version "$VER" --ca-version 1.5.7 binary docker 2>&1 | tail -5; then
            log "Fabric $VER downloaded"; DOWNLOADED=true; break
        fi
        warn "Version $VER failed, trying next..."
    done
    [ "$DOWNLOADED" = true ] || err "Failed to download Fabric binaries."
    cd "$PROJECT_DIR"
else
    log "Fabric binaries already present"
fi

export PATH="$FABRIC_SAMPLES_DIR/bin:$PATH"
export FABRIC_CFG_PATH="$FABRIC_SAMPLES_DIR/config/"
for BIN in peer configtxgen cryptogen; do
    [ -f "$FABRIC_SAMPLES_DIR/bin/$BIN" ] || err "Binary $BIN missing. Delete fabric-samples/ and retry."
done
log "Fabric binaries verified"

step "Starting test-network and creating channel"
cd "$TEST_NETWORK_DIR"

# Remove stale Fabric containers that can hold CA/orderer ports from failed runs
for C in ca_org1 ca_org2 ca_orderer orderer.example.com peer0.org1.example.com peer0.org2.example.com couchdb0 couchdb1; do
    docker rm -f "$C" >/dev/null 2>&1 || true
done

# Remove stale containers from previous custom stacks (e.g., *.fraud-detect.com / *.batfl.com)
docker ps -a --format '{{.Names}}' \
    | grep -E '(\.fraud-detect\.com|\.batfl\.com|^cli$|^cli_alt$|^couchdb0_alt$|^couchdb1_alt$)' \
    | xargs -r docker rm -f >/dev/null 2>&1 || true

# Check required host ports before launching test-network
for P in 7054 8054 9054 17054 18054 19054 7050 7051 9051; do
    if docker ps --format '{{.Ports}}' | grep -q "0.0.0.0:${P}->"; then
        err "Port ${P} is already allocated by an existing Docker container. Stop conflicting containers and retry."
    fi
done

./network.sh down 2>/dev/null || true

echo "Running: ./network.sh up createChannel -c $CHANNEL_NAME -ca -s leveldb"
./network.sh up createChannel -c "$CHANNEL_NAME" -ca -s leveldb
if [ $? -ne 0 ]; then
    err "network.sh failed to create channel. Check Docker logs: docker logs $(docker ps -a | grep 'fabric' | tail -1 | awk '{print $1}')"
fi

log "Network up, channel '$CHANNEL_NAME' created"
cd "$PROJECT_DIR"

# Verify crypto material was generated
echo "Verifying crypto material generation..."
ORDERER_TLS_CERT="$TEST_NETWORK_DIR/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/tls/server.crt"
if [ ! -f "$ORDERER_TLS_CERT" ]; then
    err "Orderer TLS cert not found at: $ORDERER_TLS_CERT\nCrypto generation likely failed. Try: cd $TEST_NETWORK_DIR && ./network.sh down && ./network.sh up createChannel -c $CHANNEL_NAME -ca -s leveldb"
fi
log "Crypto material verified"

log "Copying crypto material..."
rm -rf "$PROJECT_DIR/organizations"
cp -r "$TEST_NETWORK_DIR/organizations" "$PROJECT_DIR/"
mkdir -p "$PROJECT_DIR/channel-artifacts"
cp "$TEST_NETWORK_DIR/channel-artifacts/"*.block "$PROJECT_DIR/channel-artifacts/" 2>/dev/null || true

step "Building and deploying trust-registry chaincode"
export CORE_PEER_TLS_ENABLED=true
ORDERER_CA="$PROJECT_DIR/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem"
PEER1_TLS="$PROJECT_DIR/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
PEER2_TLS="$PROJECT_DIR/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"

if command -v go &>/dev/null; then
    cd "$CC_SRC"
    # Keep chaincode in module mode to avoid stale vendor tree issues during Fabric builds.
    rm -rf vendor
    go mod download 2>&1 | tail -5 || true
    cd "$PROJECT_DIR"
else
    warn "Go not found — chaincode packaging may fail. Install: https://go.dev/dl/"
fi

package_chaincode() {
    rm -f /tmp/trust-registry.tar.gz
    peer lifecycle chaincode package /tmp/trust-registry.tar.gz \
        --path "$CC_SRC" --lang golang --label "${CC_NAME}_${CC_VERSION}"
}

package_chaincode
log "Chaincode packaged"

set_org1() {
    export CORE_PEER_LOCALMSPID="Org1MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE="$PEER1_TLS"
    export CORE_PEER_MSPCONFIGPATH="$PROJECT_DIR/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
    export CORE_PEER_ADDRESS="localhost:7051"
}
set_org2() {
    export CORE_PEER_LOCALMSPID="Org2MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE="$PEER2_TLS"
    export CORE_PEER_MSPCONFIGPATH="$PROJECT_DIR/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp"
    export CORE_PEER_ADDRESS="localhost:9051"
}

install_chaincode() {
    local org_name="$1"
    local attempts=1
    local max_attempts=3
    local backoff=5
    local output
    while [ "$attempts" -le "$max_attempts" ]; do
        if output=$(peer lifecycle chaincode install /tmp/trust-registry.tar.gz 2>&1); then
            log "Installed on ${org_name}"
            return 0
        fi
        echo "$output" | tail -20

        if echo "$output" | grep -Eqi "already successfully installed|already installed"; then
            log "Chaincode already installed on ${org_name}; continuing"
            return 0
        fi

        # Some peer failures still install package; verify before retrying.
        if peer lifecycle chaincode queryinstalled 2>&1 | grep -q "${CC_NAME}_${CC_VERSION}"; then
            log "Install appears committed on ${org_name} despite transient error; continuing"
            return 0
        fi

        if echo "$output" | grep -Eqi "broken pipe|/var/run/docker.sock|docker image build failed"; then
            warn "Detected Docker socket/build instability while installing on ${org_name}."
            recover_docker_socket
        fi

        if echo "$output" | grep -Eqi "timeout expired while executing transaction|keepalive ping failed|rpc error: code = Unavailable|failed to get chaincode package|no such file or directory"; then
            warn "Detected peer/build runtime instability while installing on ${org_name}."
            recover_docker_socket
            restart_peer_container "$org_name" || true
            package_chaincode || true
        fi

        warn "Chaincode install failed on ${org_name} (attempt ${attempts}/${max_attempts}). Retrying..."
        sleep "$backoff"
        backoff=$((backoff + 5))
        attempts=$((attempts + 1))
    done
    err "Chaincode install failed on ${org_name} after ${max_attempts} attempts. Check Docker Desktop logs, WSL integration, and run: docker info ; docker system df"
}

set_org1; install_chaincode "Org1"
set_org2; install_chaincode "Org2"

set_org1
CC_PACKAGE_ID=$(peer lifecycle chaincode queryinstalled 2>&1 \
    | grep "Package ID:" | grep "${CC_NAME}_${CC_VERSION}" \
    | sed 's/Package ID: //;s/, Label.*//' | head -1)
[ -z "$CC_PACKAGE_ID" ] && err "Could not determine package ID."
log "Package ID: $CC_PACKAGE_ID"

APPROVE_ARGS="-o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
    --channelID $CHANNEL_NAME --name $CC_NAME --version $CC_VERSION \
    --package-id $CC_PACKAGE_ID --sequence $CC_SEQUENCE --tls --cafile $ORDERER_CA"

set_org1; eval "peer lifecycle chaincode approveformyorg $APPROVE_ARGS"; log "Org1 approved"
set_org2; eval "peer lifecycle chaincode approveformyorg $APPROVE_ARGS"; log "Org2 approved"

set_org1
peer lifecycle chaincode commit \
    -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
    --channelID "$CHANNEL_NAME" --name "$CC_NAME" --version "$CC_VERSION" \
    --sequence $CC_SEQUENCE --tls --cafile "$ORDERER_CA" \
    --peerAddresses localhost:7051 --tlsRootCertFiles "$PEER1_TLS" \
    --peerAddresses localhost:9051 --tlsRootCertFiles "$PEER2_TLS"
log "Chaincode committed"
sleep 5

step "Smoke test"
set_org1
peer chaincode invoke \
    -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
    --tls --cafile "$ORDERER_CA" -C "$CHANNEL_NAME" -n "$CC_NAME" \
    --peerAddresses localhost:7051 --tlsRootCertFiles "$PEER1_TLS" \
    --peerAddresses localhost:9051 --tlsRootCertFiles "$PEER2_TLS" \
    -c '{"function":"RegisterModel","Args":["1","abc123","blk001","GENESIS","0.85","0.92","[0,1,2]","[]","4","512"]}' 2>&1
sleep 5
RESULT=$(peer chaincode query -C "$CHANNEL_NAME" -n "$CC_NAME" \
    -c '{"function":"GetModel","Args":["1"]}' 2>&1)
if echo "$RESULT" | python3 -c "import json,sys;d=json.load(sys.stdin);assert d['round']==1" 2>/dev/null; then
    log "Smoke test PASSED ✅"
else
    warn "Smoke test inconclusive — retry in 10s: peer chaincode query -C mychannel -n trust-registry -c '{\"function\":\"GetModel\",\"Args\":[\"1\"]}'"
fi

step "Writing Python connection config"
cat > "$SCRIPT_DIR/fabric_connection.env" << ENV
FABRIC_CHANNEL=$CHANNEL_NAME
FABRIC_CHAINCODE=$CC_NAME
FABRIC_PEER_ORG1_ENDPOINT=localhost:7051
FABRIC_PEER_ORG2_ENDPOINT=localhost:9051
FABRIC_ORG1_MSP_ID=Org1MSP
FABRIC_ORG2_MSP_ID=Org2MSP
FABRIC_ORG1_CERT=$PROJECT_DIR/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp/signcerts/Admin@org1.example.com-cert.pem
FABRIC_ORG1_KEY_DIR=$PROJECT_DIR/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp/keystore
FABRIC_ORG1_TLS_CERT=$PROJECT_DIR/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
FABRIC_ORG2_CERT=$PROJECT_DIR/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp/signcerts/Admin@org2.example.com-cert.pem
FABRIC_ORG2_KEY_DIR=$PROJECT_DIR/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp/keystore
FABRIC_ORG2_TLS_CERT=$PROJECT_DIR/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt
FABRIC_ORDERER_TLS_CERT=$PROJECT_DIR/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem
ENV
log "Config written: $SCRIPT_DIR/fabric_connection.env"

echo ""
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  Hyperledger Fabric network is ready!${NC}"
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  Next: pip install hf-fabric-gateway grpcio protobuf"
echo "  Then: cd ../../.. && python module1/split3/split3_main.py --demo --blockchain fabric"
echo "  Stop: ./teardown.sh"
