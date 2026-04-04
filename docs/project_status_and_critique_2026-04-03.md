
# BATFL Project Status and Workflow Critique (Evidence-Based)

Date: 2026-04-03
Scope: Assessment based only on files and artifacts currently present in this repository.

## 1) What Is Completed

- End-to-end project structure for all 3 modules exists.
  - Evidence: `module1/split1/main.py`, `module1/split2/main.py`, `module1/split3/split3_main.py`, plus shared components under `module1/common/`.

- Split 1 baseline FL pipeline is implemented.
  - Evidence: `module1/split1/main.py` includes orchestration, partition cache, client subprocesses, Flower server, and curve plotting.

- Split 2 trust-weighted pipeline with attack simulation is implemented.
  - Evidence: `module1/split2/main.py` includes trust-weighted strategy wiring, attack simulator wiring, per-round logging, and plotting.

- Split 3 governance pipeline is implemented for simulation/Ganache/Fabric modes.
  - Evidence: `module1/split3/governance.py`, `module1/split3/fabric_gateway.py`, `module1/split3/eth_gateway.py`, `module1/split3/hlf_gateway.py`.

- Dashboard server for live log monitoring exists and runs.
  - Evidence: `module1/dashboard_server.py`.
  - Additional runtime evidence from workspace context: dashboard command has been run with exit code 0 in at least one terminal session.

- Training and governance artifacts have been generated (not just code).
  - Evidence: `logs_split2/trust_training_log.json`, `logs_split2/split2_training_curves.png`, `governance_output/governance_report.json`, `governance_output/hash_chain.json`.

- Governance report indicates processed rounds and successful chain verification for that run.
  - Evidence: `governance_output/governance_report.json` summary shows `total_rounds: 18`, `chain_intact: true`.

- Fabric setup scripts exist and were improved in this workspace session for prerequisites/port conflicts.
  - Evidence: `module1/split3/hlf/setup.sh` includes explicit `jq` prerequisite check.
  - Evidence: `fabric-samples/test-network/compose/compose-ca.yaml` host port mappings were changed to avoid occupied defaults.

- README is detailed and runnable for primary flows.
  - Evidence: `README.md` includes setup/run commands for simulation, Ganache, and Fabric, plus troubleshooting entries (including `jq` guidance).

## 2) What Still Needs To Be Completed

- Automated test suite is missing.
  - Evidence: no project-level Python tests discovered (`**/test*.py`), no `pytest.ini`, no Python project test config file.
  - Impact: behavior is validated mainly through manual runs and generated artifacts.

- TODO tracking appears stale/incomplete relative to current repo state.
  - Evidence: `TODO.md` still has unchecked items (testing/verification steps) while artifacts in `logs_split2/` and `governance_output/` indicate runs have already occurred.
  - Impact: project status communication can be misleading.

- Dependency and environment hardening is still needed for reproducible Fabric setup.
  - Evidence: Fabric setup required manual troubleshooting around missing OS package (`jq`), host-port conflicts, and silent certificate generation failures.
  - Impact: setup may still fail on new machines unless environment assumptions are standardized.
  - Fix applied in this session: Added crypto verification check to `setup.sh` to fail fast if certificate generation didn't complete.

- Explicit backend verification is missing in runtime output contract.
  - Evidence: gateway factory in `module1/split3/fabric_gateway.py` can silently fall back to simulation on Fabric/Ganache errors.
  - Impact: a run requested as Fabric/Ganache may complete in simulation mode without strict failure.

- Consistent command examples need final normalization.
  - Evidence: code comments in some files still include legacy module-run patterns (for example in split file headers), while README uses root-level `python -m module1...` format.
  - Impact: user confusion when copying commands from different places.

## 3) Conceptual Workflow Critique (No Hallucination)

These are conceptual issues inferred from code/docs/artifacts in this repo, not speculation.

- The workflow currently mixes "demo reliability" and "production trust guarantees" in the same control path.
  - Evidence: `module1/split3/governance.py` states round hashing is derived from log-provided `model_hash` via synthetic params when actual parameters are unavailable.
  - Why this is conceptually problematic: if governance integrity is meant to attest real model updates, deriving chain entries from post-hoc log fields weakens the provenance model unless the source of `model_hash` is itself strongly trusted.

- Silent fallback from real blockchain backends to simulation weakens assurance semantics.
  - Evidence: `module1/split3/fabric_gateway.py` catches backend failures and returns simulation gateway automatically.
  - Why this is conceptually problematic: when a user asks for Fabric/Ganache, fallback-to-simulation can preserve pipeline continuity but break the meaning of "on-chain governance" unless surfaced as a hard error or explicit run status.

- Governance and training are loosely coupled through log files rather than an explicit immutable event interface.
  - Evidence: Split 3 consumes `trust_training_log.json` and processes after training (`process_trust_log`), with optional live use.
  - Why this is conceptually problematic: log-file mediation can be edited/replayed unless protected; governance claims become dependent on file custody rather than direct event signing/attestation.

- Operational fragility in environment setup is too central to the workflow.
  - Evidence: recent failures were caused by missing `jq` and Docker host port allocations for Fabric CA.
  - Why this is conceptually problematic: when environment friction dominates, workflow correctness and reproducibility depend on ad-hoc fixes rather than standardized deployment contracts.

- Status reporting logic in UI is partially hard-coded.
  - Evidence: dashboard code uses fixed assumptions (for example, total rounds/status thresholds are not fully derived from runtime configuration).
  - Why this is conceptually problematic: displayed completion/progress can diverge from actual experiment config, reducing trust in observability.

## 4) Recommended Next Completion Targets (Priority Order)

1. Add automated tests for:
   - data loading/partitioning,
   - trust scoring behavior,
   - governance chain verification and tamper detection,
   - gateway selection behavior (including fail-fast expectations).

2. Decide backend-failure policy explicitly:
   - either fail hard when Fabric/Ganache is requested but unavailable,
   - or keep fallback but emit explicit machine-readable run status (`requested_backend`, `effective_backend`, `fallback_reason`).

3. Replace or augment log-file coupling with signed round events (or hash commitments generated at source) to strengthen provenance.

4. Make dashboard progress/status fully runtime-driven from log metadata or CLI config (no fixed round assumptions).

5. Update `TODO.md` to reflect true current state and remaining acceptance criteria.

---
This report intentionally avoids claims that are not directly supported by files/artifacts present in this repository.

