# BATFL — Blockchain-Based Dynamic Trust Modeling for Federated Fraud Detection

**Final Year Project | Supervised by Dr. V. Govindasamy**

---

## What This Project Does

BATFL trains a federated fraud detection model across multiple bank nodes without
any bank sharing raw transaction data. Each bank trains locally; only model weights
travel over the network. A dynamic trust scoring engine detects malicious participants
in real time and records every governance decision permanently on a blockchain.

**Two blockchain implementations are included and can run side-by-side:**
- **Ganache/Ethereum** — Solidity smart contract, real Ethereum transactions
- **Hyperledger Fabric** — Go chaincode, permissioned enterprise blockchain

---

## Project Structure

```
BATFL/
├── requirements.txt
├── data/                         ← Put creditcard.csv here
└── module1/
    ├── common/                   ← Shared FL components
    │   ├── data_partition.py     ← Dirichlet non-IID partitioning + SMOTE
    │   ├── local_models.py       ← FraudDNN + LogisticFraudModel
    │   ├── flower_client.py      ← BankFederatedClient
    │   ├── fedavg_strategy.py    ← FedAvg baseline strategy
    │   ├── trust_scoring.py      ← Dynamic trust scorer
    │   ├── trust_weighted_strategy.py ← Trust-weighted aggregation
    │   ├── attack_simulator.py   ← Label-flip / gradient-scale attacks
    │   └── governance_bridge.py  ← Module 1 ↔ Module 2 bridge
    │
    ├── split1/main.py            ← FedAvg baseline
    ├── split2/main.py            ← Trust-weighted FL + attacks + blockchain
    │
    ├── split3/
    │   ├── governance.py         ← GovernanceEngine (core)
    │   ├── model_hasher.py       ← SHA-256 hash chain
    │   ├── blockchain_sim.py     ← In-memory simulation
    │   ├── eth_gateway.py        ← Ganache/Ethereum gateway
    │   ├── fabric_gateway.py     ← Gateway factory (routes all 3 backends)
    │   ├── hlf_gateway.py        ← Hyperledger Fabric gateway
    │   ├── split3_main.py        ← Split 3 CLI
    │   ├── contracts/
    │   │   └── ModelRegistry.sol ← Solidity contract (Ganache)
    │   └── hlf/
    │       ├── setup.sh          ← Fabric network setup (1 command)
    │       ├── teardown.sh
    │       └── chaincode/trust_registry/
    │           ├── trust_registry.go  ← Go chaincode (Fabric)
    │           └── go.mod
    │
    ├── dashboard_server.py       ← Live browser dashboard
    └── run_client.py             ← Remote client (multi-machine)
```

---

## Step 1 — Install

```bash
pip install -r requirements.txt
```

---

## Step 2 — Run (no dataset, no blockchain setup needed)

```bash
cd BATFL

# Split 1: FedAvg baseline
python -m module1.split1.main --synthetic

# Split 2 (recommended): Modern Flower App runtime
# Windows PowerShell helper:
./module1/split2/run_flwr.ps1 -NumClients 5 -NumRounds 20 -Model dnn

# IMPORTANT: set Flower local topology to match client count
# File: %USERPROFILE%/.flwr/config.toml
#   [superlink.local]
#   options.num-supernodes = 5

# Split 2 (CLI form):
flwr run . --run-config "num_clients=5 num_rounds=20 model='dnn' use_synthetic=true attack='none' blockchain_enabled=false"

# Split 2: With attack detection
./module1/split2/run_flwr.ps1 -NumClients 5 -NumRounds 20 -Model dnn -Attack label_flip -Malicious "1"

# Split 3: Governance (simulation, no external deps)
python module1/split3/split3_main.py --demo --blockchain simulation
```

---

## Step 3 — Live Dashboard

Run in a second terminal while Split 2 is running:

```bash
python module1/dashboard_server.py --log logs_split2/trust_training_log.json
# Open browser: http://localhost:5000
```

---

## Blockchain Backends

### Backend 1: Simulation (default — no setup)
```bash
python module1/split3/split3_main.py --demo --blockchain simulation
```
Full blockchain semantics (immutable ledger, hash chain, tamper detection,
audit trail) simulated in memory. Works anywhere with no dependencies.

---

### Backend 2: Ganache / Ethereum (real blockchain — 5 min setup)

**Setup:**
```bash
# 1. Install Node.js from https://nodejs.org
npm install -g ganache

# 2. Start Ganache (keep this terminal open)
ganache --deterministic --port 8545
```

**Run:**
```bash
# Split 2 with live Ganache governance
python -m module1.split2.main --synthetic --blockchain ganache

# Split 3 standalone with Ganache
python module1/split3/split3_main.py --demo --blockchain ganache
```

Each FL round creates a real Ethereum transaction mined into a real block.
Transaction hashes are printed and verifiable in Ganache's output.

---

### Backend 3: Hyperledger Fabric (enterprise blockchain — Docker required)

**Requirements:** Docker Desktop with WSL integration enabled (≥4 GB RAM), Git, curl, Go 1.21+

**Setup:**
```bash
cd module1/split3/hlf
chmod +x setup.sh teardown.sh
./setup.sh
# Takes ~5–10 minutes on first run (downloads Docker images + binaries)
```

The Fabric backend uses the peer CLI through WSL, so no additional Python
Fabric SDK is required.

**Run:**
```bash
# Back to project root
cd ../../..
python module1/split3/split3_main.py --demo --blockchain fabric
```

**Windows one-command Option 3 pipeline (recommended):**
```powershell
powershell -ExecutionPolicy Bypass -File module1/split3/hlf/run_option3.ps1 \
  -NumClients 2 -NumRounds 20 -Model logistic
```

For a clean rerun, add `-Reset` to the enterprise wrapper:

```powershell
powershell -ExecutionPolicy Bypass -File run_option3_enterprise.ps1 -Reset -NumClients 2 -NumRounds 20 -Model logistic -Consensus bft
```

Or run standalone cleanup without starting training:

```powershell
powershell -ExecutionPolicy Bypass -File reset_enterprise.ps1
# Optional: reset and immediately re-setup Fabric
powershell -ExecutionPolicy Bypass -File reset_enterprise.ps1 -SetupAfterReset -Consensus bft
```

This runs the full Option 3 flow end-to-end:
1. Bootstraps Fabric network + deploys chaincode
2. Runs Split 2 with live Fabric commits (`blockchain_backend='fabric'`)
3. Runs Split 3 Fabric governance audit on the generated trust log

**Stop:**
```bash
cd module1/split3/hlf && ./teardown.sh
```

---

## Full Demo Scenario (for presentation)

```bash
# Terminal 1: Start FL training with attack + live Ganache governance
python -m module1.split2.main \
    --synthetic \
    --rounds 20 \
    --attack combined \
    --malicious 1 3 \
    --scale_factor 10 \
    --blockchain ganache

# Terminal 2: Live dashboard
python module1/dashboard_server.py --log logs_split2/trust_training_log.json

# Browser: http://localhost:5000
# Watch trust scores drop for clients 1 and 3 in real time

# After training — run full governance audit
python module1/split3/split3_main.py \
    --trust_log logs_split2/trust_training_log.json \
    --blockchain ganache \
    --tamper_round 5
```

---

## All Command Options

### Split 1
```
python -m module1.split1.main [OPTIONS]
  --synthetic          Use built-in synthetic data
  --data_path PATH     Path to creditcard.csv
  --num_clients INT    Number of bank nodes (default: 5)
  --rounds INT         Federation rounds (default: 20)
  --model dnn|logistic Model type (default: dnn)
  --alpha FLOAT        Dirichlet alpha for non-IID (default: 1.0)
```

### Split 2
```
python -m module1.split2.main [OPTIONS]
  --synthetic / --data_path PATH
  --num_clients INT    (default: 5)
  --rounds INT         (default: 20)
  --attack none|label_flip|gradient_scale|combined|backdoor|sign_flip|model_replacement|sybil
  --malicious INT...   Client IDs to attack
  --scale_factor FLOAT Gradient scale multiplier (default: 5.0)
  --attack_start INT   First attack round (default: 1)
  --attack_end INT     Last attack round (optional)
  --attack_intensity FLOAT  Attack strength multiplier (default: 1.0)
  --threshold_mode auto|fixed
  --decision_threshold FLOAT
  --threshold_beta FLOAT      F-beta used for auto threshold tuning
  --event_storage jsonl|sqlite
  --blockchain simulation|ganache|fabric   (default: simulation)
  --governance_policy PATH    JSON/YAML governance policy
  --seed INT                  Reproducibility seed
  --comparison_report         Build FedAvg vs TrustWeighted comparison report
  --comparison_dirs DIR...    Run dirs for comparison (default: logs_split1 logs_split2)
  --no_blockchain      Disable Module 2 entirely
```

Each Split 2 and Split 1 run now writes a reproducibility manifest:

```bash
logs_split2/run_manifest.json
logs_split1/run_manifest.json
```

The manifest includes dataset info, hyperparameters, seed, backend, Flower version, and git commit SHA.

Split 2 now supports a typed, versioned event schema with strict validation and migration support.
Each round event stores:

```bash
schema_name=batfl.round-log
schema_version=1
```

Storage backends:

```bash
--event_storage jsonl   # default: round_events.jsonl
--event_storage sqlite  # recommended: round_events.db
```

Deterministic partition caching is now checksum-based and portable:
- cache reuse happens only when dataset fingerprint + partition params match
- reruns with matching inputs reuse identical client partitions

With `--comparison_report`, Split 2 writes an automated baseline report in the log directory:

```bash
logs_split2/baseline_comparison.csv
logs_split2/baseline_comparison.md
logs_split2/baseline_comparison.png
```

Direct Fabric run from repo root (real dataset):
```bash
python -m module1.split2.main --data_path ../data/creditcard.csv --blockchain fabric --attack label_flip --malicious 1
```

Optional: force enterprise consensus mode when needed:
```bash
# Windows PowerShell
$env:BATFL_ENTERPRISE_CONSENSUS = "raft"   # or "bft"
python -m module1.split2.main --data_path ../data/creditcard.csv --blockchain fabric --attack label_flip --malicious 1
```

### Split 3
```
python module1/split3/split3_main.py [OPTIONS]
  --demo               Generate synthetic trust log
  --trust_log PATH     Path to trust_training_log.json
  --blockchain simulation|ganache|fabric
  --policy PATH        Governance policy JSON/YAML
  --tamper_round INT   Inject tamper at round N (demo)
  --output_dir PATH    Where to write reports
```

## Governance Security Controls

- Replay protection: every RoundEvent now carries run_id and event_sequence, and Split 3 rejects mixed-run events and duplicated sequence numbers.
- Key rotation: attestation signatures use BATFL_ATTESTATION_KEY_ID, and unknown key IDs are rejected during verification.
- Optional stronger signatures: set BATFL_ATTESTATION_ALGO=ED25519 and provide Ed25519 key material.
- Policy-driven quarantine: pass a policy file via --governance_policy (Split 2) or --policy (Split 3).

Example policy file:

```bash
module1/split3/governance_policy.example.json
```

HMAC key rotation environment variables:

```bash
BATFL_ATTESTATION_KEY_ID=coordinator-v2
BATFL_ATTESTATION_KEYS_JSON={"coordinator-v1":"old-secret","coordinator-v2":"new-secret"}
```

Optional Ed25519 environment variables:

```bash
BATFL_ATTESTATION_ALGO=ED25519
BATFL_ATTESTATION_KEY_ID=coordinator-v2
BATFL_ATTESTATION_ED25519_PRIVATE_KEY_B64=<base64-32-byte-private-key>
BATFL_ATTESTATION_ED25519_PUBLIC_KEYS_JSON={"coordinator-v2":"<base64-32-byte-public-key>"}
```

### Dashboard
```

Dashboard now also shows:
- effective backend evidence
- per-round commit/audit TX timeline
- quarantine timeline + policy reasons
- tamper banner when chain integrity breaks

### End-to-End Verification

```bash
python -m module1.verify_run --run_dir logs_split2
```

Checks performed:
- round event schema validation + migration
- signature chain verification
- governance artifacts present
- hash chain intact
- backend matches manifest expectation

### Single Command Demo Launcher

```bash
python -m module1.demo_launcher --run_dir logs_split2 --rounds 3 --num_clients 5 --event_storage sqlite
```

This runs:
1. Split 2 training
2. Split 3 audit (when blockchain enabled)
3. verify_run integrity checks
4. dashboard launch in browser
5. report bundle zip generation under the run dir

### Architecture and Threat Model Docs

- `docs/architecture/ARCHITECTURE.md`
- `docs/architecture/THREAT_MODEL.md`

### CLI and Packaging

The package now exposes entry points:

```bash
batfl demo -- --run_dir logs_split2 --rounds 3 --num_clients 5 --event_storage sqlite
batfl-verify --run_dir logs_split2
batfl train-split2 -- --synthetic --rounds 3 --num_clients 5 --blockchain fabric
batfl verify -- --run_dir logs_split2
batfl dashboard -- --log logs_split2/trust_training_log.json --port 5000
```
python module1/dashboard_server.py [OPTIONS]
  --log PATH           Path to trust_training_log.json
  --port INT           HTTP port (default: 5000)
  --host STR           Host (default: 127.0.0.1)
```

---

## Blockchain Comparison Table

| | Simulation | Ganache/Ethereum | Hyperledger Fabric |
|---|---|---|---|
| Setup time | 0 min | 5 min | 10 min |
| External deps | None | Node.js + ganache | Docker + Go |
| Transaction type | In-memory | Real ETH txn | Real Fabric txn |
| Smart contract | Python class | Solidity .sol | Go chaincode |
| Permissioned | No | No | Yes (MSP) |
| GDPR fit | Demo only | Limited | ✅ Production |
| Best for | Quick demo | Real blockchain demo | Enterprise demo |

---

## Troubleshooting

**ImportError: No module named 'flwr'**
```bash
pip install flwr>=1.5.0
```

**Ganache not reachable**
```bash
ganache --deterministic --port 8545   # keep running in a separate terminal
```

**Fabric: "fabric_connection.env not found"**
```bash
cd module1/split3/hlf && ./setup.sh
```
If this happens, also verify Docker Desktop is running and WSL integration is enabled.

**Fabric setup: git clone fails**
The script tries multiple tags (v2.5.4 → v2.5.3 → v2.5.0 → main).
If all fail, check your internet connection and retry.

**Fabric: smoke test fails but setup completed**
Wait 15 seconds and query manually:
```bash
export PATH="./fabric-samples/bin:$PATH"
export FABRIC_CFG_PATH="./fabric-samples/config/"
export CORE_PEER_TLS_ENABLED=true
export CORE_PEER_LOCALMSPID=Org1MSP
export CORE_PEER_ADDRESS=localhost:7051
export CORE_PEER_TLS_ROOTCERT_FILE=./organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt
export CORE_PEER_MSPCONFIGPATH=./organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp
peer chaincode query -C mychannel -n trust-registry -c '{"function":"GetModel","Args":["1"]}'
```

**Fabric: `jq: command not found`**
```bash
sudo apt-get update && sudo apt-get install -y jq
```
`jq` is a system package used by the Fabric scripts. Installing `jq` with `pip` will not fix this error.

**Fabric: `cannot load client cert for consenter` / crypto not found**
This error means the Fabric network bootstrap (credential generation) failed, usually during `configtxgen`.

First, clean up and check Docker:
```bash
cd module1/split3/hlf
./teardown.sh
docker ps -a  # check for stalled containers
docker rm -f $(docker ps -a | grep fabric | awk '{print $1}')  # remove Fabric containers
```

Then retry:
```bash
./setup.sh
```

If it still fails, enable debug logs:
```bash
cd fabric-samples/test-network
./network.sh up createChannel -c mychannel -ca -s leveldb   # verbose output
```

If verbose output shows `openssl` or `cryptogen` errors, you may be missing system tools. Verify:
```bash
command -v openssl && command -v jq && command -v curl
```

**Complete Fabric reset**
```bash
cd module1/split3/hlf && ./teardown.sh && ./setup.sh
```
