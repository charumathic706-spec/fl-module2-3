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
CC_VERSION="1.2"
# Channel is recreated by setup each run, so definition sequence must start at 1.
CC_SEQUENCE=1
STATE_DB="${BATFL_STATE_DB:-leveldb}"
USE_CA="${BATFL_USE_CA:-false}"
CONSENSUS_MODE="${BATFL_CONSENSUS:-raft}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FABRIC_ENV_OUT_FILE="${BATFL_FABRIC_ENV_OUT:-$SCRIPT_DIR/fabric_connection.env}"
# module1/split3/hlf → module1/split3 → module1 → project root
PROJECT_DIR="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
FABRIC_SAMPLES_DIR="$PROJECT_DIR/fabric-samples"
TEST_NETWORK_DIR="$FABRIC_SAMPLES_DIR/test-network"
CC_SRC="$SCRIPT_DIR/chaincode/trust_registry"
CHAINCODE_STAGE_DIR=""

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'; BOLD='\033[1m'
log()  { echo -e "${GREEN}[✓]${NC} $1"; }
warn() { echo -e "${YELLOW}[!]${NC} $1"; }
err()  { echo -e "${RED}[✗]${NC} $1" >&2; exit 1; }
step() { echo -e "\n${BLUE}${BOLD}━━ $1 ━━${NC}"; }

docker_info_ok() {
    # docker info can be slow on WSL when CLI plugins initialize; prefer
    # a lightweight server check first and avoid hard timeout false negatives.
    if docker version --format '{{.Server.Version}}' >/dev/null 2>&1; then
        return 0
    fi

    DOCKER_CLI_HINTS=false docker info >/dev/null 2>&1
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
    # Reduce daemon pressure before retrying chaincode builds.
    docker builder prune -af >/dev/null 2>&1 || true
    docker system prune -f >/dev/null 2>&1 || true
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

prepare_chaincode_source() {
    CHAINCODE_STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/batfl-chaincode.XXXXXX")"
    rm -rf "${CHAINCODE_STAGE_DIR:?}/$CC_NAME"
    cp -a "$SCRIPT_DIR/chaincode/trust_registry" "$CHAINCODE_STAGE_DIR/$CC_NAME"
    CC_SRC="$CHAINCODE_STAGE_DIR/$CC_NAME"
    log "Preparing chaincode source and vendoring Go dependencies (this can take a few minutes)..."
    (
        cd "$CC_SRC"
        export GO111MODULE=on
        # Vendor deps into chaincode package so peer-side Docker builds do not
        # need external network/DNS access to proxy.golang.org.
        export GOPROXY=https://proxy.golang.org,direct
        export GO111MODULE=on
        if command -v timeout >/dev/null 2>&1; then
            timeout 300 go mod tidy || err "go mod tidy failed/timed out while preparing chaincode dependencies"
            timeout 300 go mod download || err "go mod download failed/timed out while preparing chaincode dependencies"
            timeout 300 go mod vendor || err "go mod vendor failed/timed out while preparing chaincode dependencies"
        else
            go mod tidy || err "go mod tidy failed while preparing chaincode dependencies"
            go mod download || err "go mod download failed while preparing chaincode dependencies"
            go mod vendor || err "go mod vendor failed while preparing chaincode dependencies"
        fi
    )
    trap 'rm -rf "$CHAINCODE_STAGE_DIR"' EXIT
    log "Staged chaincode source at $CC_SRC"
}

step "Checking prerequisites"
command -v docker &>/dev/null || err "Docker not found. Install Docker Desktop."
docker_info_ok || err "Docker not running or not responsive. Start Docker Desktop and ensure WSL integration is enabled."
wait_for_docker 15 2 || err "Docker daemon not healthy. Ensure Docker Desktop is fully started and WSL integration is enabled."
command -v git    &>/dev/null || err "Git not found."
command -v curl   &>/dev/null || err "curl not found."
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
if [ ! -x "$FABRIC_SAMPLES_DIR/bin/peer" ] || ! "$FABRIC_SAMPLES_DIR/bin/peer" version >/dev/null 2>&1; then
    cd "$FABRIC_SAMPLES_DIR"
    warn "Fabric binaries missing or not executable in current shell — downloading fresh binaries..."
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
    [ -x "$FABRIC_SAMPLES_DIR/bin/$BIN" ] || err "Binary $BIN missing or not executable. Delete fabric-samples/ and retry."
done
if ! peer version >/dev/null 2>&1; then
    err "Fabric peer binary is present but not runnable in this shell. Re-run setup.sh from WSL so Linux binaries are installed."
fi
log "Fabric binaries verified"

prepare_chaincode_source

step "Starting test-network and creating channel"

# Pre-flight Docker health check
warn "Running Docker health check before network startup..."
if ! docker run --rm alpine echo "Docker OK" >/dev/null 2>&1; then
    err "Docker not responding properly. Verify Docker Desktop is running and WSL integration is enabled."
fi
docker system prune -f >/dev/null 2>&1 || true
log "Docker health OK"

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

NETWORK_CMD=(./network.sh up createChannel -c "$CHANNEL_NAME" -s "$STATE_DB")
if [ "$USE_CA" = "true" ]; then
    NETWORK_CMD+=( -ca )
fi
if [ "$CONSENSUS_MODE" = "bft" ]; then
    NETWORK_CMD+=( -bft )
fi
echo "Running: ${NETWORK_CMD[*]}"
create_channel_with_retry() {
    local attempt=1
    local max_attempts=3
    while [ "$attempt" -le "$max_attempts" ]; do
        if "${NETWORK_CMD[@]}" -r 10 -d 3; then
            return 0
        fi
        warn "network.sh createChannel failed (attempt ${attempt}/${max_attempts}). Cleaning up and retrying..."
        ./network.sh down 2>/dev/null || true
        sleep 4
        attempt=$((attempt + 1))
    done
    return 1
}

if ! create_channel_with_retry; then
    err "network.sh failed to create channel after retries. Check Docker logs: docker logs $(docker ps -a | grep 'fabric' | tail -1 | awk '{print $1}')"
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
CHANNEL_ARTIFACTS_DIR="$PROJECT_DIR/channel-artifacts"
mkdir -p "$CHANNEL_ARTIFACTS_DIR"
block_file=$(find "$TEST_NETWORK_DIR/channel-artifacts" -maxdepth 1 -type f -name '*.block' -print -quit)
if [ -n "$block_file" ]; then
    cp "$block_file" "$CHANNEL_ARTIFACTS_DIR/mychannel.block"
fi

write_connection_config() {
    mkdir -p "$(dirname "$FABRIC_ENV_OUT_FILE")"
    cat > "$FABRIC_ENV_OUT_FILE" << ENV
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
    log "Config written: $FABRIC_ENV_OUT_FILE"
}

write_connection_config

step "Building and deploying trust-registry chaincode"
export CORE_PEER_TLS_ENABLED=true
ORDERER_CA="$PROJECT_DIR/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem"
PEER1_TLS="$PROJECT_DIR/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
PEER2_TLS="$PROJECT_DIR/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"

DEPLOYED_WITH_NETWORK_SH=false
if [ "${BATFL_TRY_NETWORK_DEPLOYCC:-false}" = "true" ]; then
    cd "$TEST_NETWORK_DIR"
    if ./network.sh deployCC -c "$CHANNEL_NAME" -ccn "$CC_NAME" -ccp "$CC_SRC" -ccl go -ccv "$CC_VERSION" -ccs "$CC_SEQUENCE"; then
        log "Chaincode deployed via network.sh deployCC"
        DEPLOYED_WITH_NETWORK_SH=true
    else
        warn "network.sh deployCC failed; falling back to manual lifecycle flow"
        DEPLOYED_WITH_NETWORK_SH=false
    fi
    cd "$PROJECT_DIR"
else
    warn "Skipping network.sh deployCC (set BATFL_TRY_NETWORK_DEPLOYCC=true to enable). Using manual lifecycle flow directly."
fi

if [ "$DEPLOYED_WITH_NETWORK_SH" = true ]; then
    step "Smoke test"
    cd "$TEST_NETWORK_DIR"

    set_org1() {
        export CORE_PEER_LOCALMSPID="Org1MSP"
        export CORE_PEER_TLS_ROOTCERT_FILE="$PEER1_TLS"
        export CORE_PEER_MSPCONFIGPATH="$PROJECT_DIR/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
        export CORE_PEER_ADDRESS="localhost:7051"
    }

    set_org1
    peer chaincode invoke \
        -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
        --tls --cafile "$ORDERER_CA" -C "$CHANNEL_NAME" -n "$CC_NAME" \
        --peerAddresses localhost:7051 --tlsRootCertFiles "$PEER1_TLS" \
        --peerAddresses localhost:9051 --tlsRootCertFiles "$PEER2_TLS" \
        -c '{"function":"AppendEvent","Args":["SMOKE","0","setup","{}"]}' 2>&1
    sleep 5
    RESULT=$(peer chaincode query -C "$CHANNEL_NAME" -n "$CC_NAME" \
        -c '{"function":"ExportAuditTrail","Args":[]}' 2>&1)
    if echo "$RESULT" | python3 -c "import json,sys;d=json.load(sys.stdin);assert isinstance(d,list)" 2>/dev/null; then
        log "Smoke test PASSED ✅"
    else
        warn "Smoke test inconclusive — retry in 10s: peer chaincode query -C mychannel -n trust-registry -c '{\"function\":\"ExportAuditTrail\",\"Args\":[]}'"
    fi
    cd "$PROJECT_DIR"
else

if command -v go &>/dev/null; then
    cd "$CC_SRC"
    # Rebuild vendor folder in stage dir to keep dependency tree deterministic
    # while still allowing offline peer-side chaincode installation.
    if [ -d vendor ]; then
        chmod -R u+w vendor 2>/dev/null || true
        rm -rf vendor 2>/dev/null || true
    fi
    export GOPROXY=https://proxy.golang.org,direct
    export GO111MODULE=on
    go mod download 2>&1 | tail -5 || true
    go mod vendor 2>&1 | tail -5 || true
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

# Pre-install cleanup to reduce probability of docker.sock broken-pipe during peer image build.
recover_docker_socket
    if [ "${BATFL_RESTART_PEERS_ON_INSTALL_ERROR:-false}" = "true" ]; then
        restart_peer_container "Org1" || true
        restart_peer_container "Org2" || true
    fi

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
    local max_attempts=6
    local backoff=15
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

        if echo "$output" | grep -Eqi "timeout expired while executing transaction|keepalive ping failed|rpc error: code = Unavailable|failed to get chaincode package|no such file or directory|context deadline exceeded"; then
            warn "Detected peer/build runtime instability while installing on ${org_name}."
            recover_docker_socket
            if [ "${BATFL_RESTART_PEERS_ON_INSTALL_ERROR:-false}" = "true" ]; then
                restart_peer_container "$org_name" || true
            fi
            package_chaincode || true
            sleep 15  # Increased from 10 to give peer more time to recover
        fi

        warn "Chaincode install failed on ${org_name} (attempt ${attempts}/${max_attempts}). Waiting before retry..."
        # Exponential backoff with jitter: backoff * 2^(attempts-1)
        local sleep_time=$((backoff * (2 ** (attempts - 1))))
        warn "Waiting ${sleep_time} seconds before attempt $((attempts + 1))..."
        sleep "$sleep_time"
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
    --connTimeout 20s \
    --channelID $CHANNEL_NAME --name $CC_NAME --version $CC_VERSION \
    --package-id $CC_PACKAGE_ID --sequence $CC_SEQUENCE --tls --cafile $ORDERER_CA"

approve_chaincode() {
    local org_name="$1"
    local set_fn="$2"
    local attempts=1
    local max_attempts=5
    local backoff=5
    while [ "$attempts" -le "$max_attempts" ]; do
        if "$set_fn" && eval "peer lifecycle chaincode approveformyorg $APPROVE_ARGS"; then
            log "${org_name} approved"
            return 0
        fi
        warn "Chaincode approval failed on ${org_name} (attempt ${attempts}/${max_attempts}). Retrying..."
        sleep "$backoff"
        backoff=$((backoff + 5))
        attempts=$((attempts + 1))
    done
    err "Chaincode approval failed on ${org_name} after ${max_attempts} attempts. Check orderer logs."
}

approve_chaincode "Org1" set_org1
approve_chaincode "Org2" set_org2

set_org1
commit_attempts=1
commit_max_attempts=5
commit_backoff=5
while [ "$commit_attempts" -le "$commit_max_attempts" ]; do
    if peer lifecycle chaincode commit \
        -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
        --connTimeout 20s \
        --channelID "$CHANNEL_NAME" --name "$CC_NAME" --version "$CC_VERSION" \
        --sequence $CC_SEQUENCE --tls --cafile "$ORDERER_CA" \
        --peerAddresses localhost:7051 --tlsRootCertFiles "$PEER1_TLS" \
        --peerAddresses localhost:9051 --tlsRootCertFiles "$PEER2_TLS"; then
        log "Chaincode committed"
        break
    fi
    warn "Chaincode commit failed (attempt ${commit_attempts}/${commit_max_attempts}). Retrying..."
    sleep "$commit_backoff"
    commit_backoff=$((commit_backoff + 5))
    commit_attempts=$((commit_attempts + 1))
done
if [ "$commit_attempts" -gt "$commit_max_attempts" ]; then
    err "Chaincode commit failed after ${commit_max_attempts} attempts. Check orderer logs."
fi
sleep 5

step "Smoke test"
set_org1
peer chaincode invoke \
    -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com \
    --tls --cafile "$ORDERER_CA" -C "$CHANNEL_NAME" -n "$CC_NAME" \
    --peerAddresses localhost:7051 --tlsRootCertFiles "$PEER1_TLS" \
    --peerAddresses localhost:9051 --tlsRootCertFiles "$PEER2_TLS" \
    -c '{"function":"AppendEvent","Args":["SMOKE","0","setup","{}"]}' 2>&1
sleep 5
RESULT=$(peer chaincode query -C "$CHANNEL_NAME" -n "$CC_NAME" \
    -c '{"function":"ExportAuditTrail","Args":[]}' 2>&1)
if echo "$RESULT" | python3 -c "import json,sys;d=json.load(sys.stdin);assert isinstance(d,list)" 2>/dev/null; then
    log "Smoke test PASSED ✅"
else
    warn "Smoke test inconclusive — retry in 10s: peer chaincode query -C mychannel -n trust-registry -c '{\"function\":\"ExportAuditTrail\",\"Args\":[]}'"
fi
fi

write_connection_config

echo ""
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD}  Hyperledger Fabric network is ready!${NC}"
echo -e "${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo "  Next: pip install hf-fabric-gateway grpcio protobuf"
echo "  Then: cd ../../.. && python module1/split3/split3_main.py --demo --blockchain fabric"
echo "  Stop: ./teardown.sh"
