"""
split3/split3_main.py
---------------------
Split 3 — Blockchain Governance Layer — CLI entry point.

Usage:
  # Demo mode (no Split 2 needed):
  python module1/split3/split3_main.py --demo --blockchain simulation
  python module1/split3/split3_main.py --demo --blockchain ganache
  python module1/split3/split3_main.py --demo --blockchain fabric

  # With real Split 2 output:
  python module1/split3/split3_main.py \
      --trust_log logs_split2/trust_training_log.json \
      --blockchain simulation

  # With tamper detection demo:
  python module1/split3/split3_main.py --demo --tamper_round 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

# Add split3/ to path so governance.py and siblings resolve
_SPLIT3_DIR = os.path.dirname(os.path.abspath(__file__))
if _SPLIT3_DIR not in sys.path:
    sys.path.insert(0, _SPLIT3_DIR)

from governance import GovernanceConfig, GovernanceEngine
from model_hasher import ModelHasher


# =============================================================================
# DEMO TRUST LOG GENERATOR
# =============================================================================

def generate_demo_trust_log(
    num_rounds: int = 10,
    num_clients: int = 5,
    malicious_client: int = 2,
    output_path: str = "./demo_trust_log.json",
) -> str:
    """Generate a synthetic trust_training_log.json for demo / testing."""
    rng = np.random.default_rng(42)
    log = []

    for rnd in range(1, num_rounds + 1):
        base_f1 = min(0.65 + rnd * 0.025 + rng.normal(0, 0.005), 0.92)
        trust_scores, anomaly_scores, flagged, trusted = {}, {}, [], []

        for cid in range(num_clients):
            if cid == malicious_client and rnd > 4:
                anomaly = float(rng.uniform(0.6, 0.95))
                trust   = max(0.05, 1.0 - anomaly - rng.uniform(0, 0.1))
                flagged.append(cid)
            else:
                anomaly = float(rng.uniform(0.0, 0.15))
                trust   = min(1.0, 0.85 + rng.uniform(0, 0.15))
                trusted.append(cid)
            trust_scores[str(cid)]   = round(trust, 4)
            anomaly_scores[str(cid)] = round(anomaly, 4)

        model_hash = hashlib.sha256(f"round_{rnd}_f1_{base_f1:.4f}".encode()).hexdigest()
        log.append({
            "round":            rnd,
            "timestamp":        time.time(),
            "model_hash":       model_hash,
            "trusted_clients":  trusted,
            "flagged_clients":  flagged,
            "trust_scores":     trust_scores,
            "anomaly_scores":   anomaly_scores,
            "cos_similarities": {str(c): round(float(rng.uniform(0.7, 1.0)), 4) for c in range(num_clients)},
            "euc_distances":    {str(c): round(float(rng.uniform(0.0, 0.5)), 4) for c in range(num_clients)},
            "global_f1":        round(base_f1, 6),
            "global_auc":       round(min(base_f1 + 0.10, 0.99), 6),
        })

    with open(output_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[Demo] Trust log: {output_path}")
    print(f"       {num_rounds} rounds, {num_clients} clients, "
          f"client {malicious_client} goes malicious after round 4")
    return output_path


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="BATFL Split 3 — Blockchain Governance Layer"
    )
    p.add_argument("--trust_log",   type=str, default=None,
                   help="Path to trust_training_log.json from Split 2")
    p.add_argument("--output_dir",  type=str, default="./governance_output")
    p.add_argument("--blockchain",  type=str, default="fabric",
                   choices=["simulation", "ganache", "fabric"],
                   help="Blockchain backend (default: fabric)")
    p.add_argument("--tamper_round",type=int, default=None,
                   help="Inject tamper at this round to demo detection")
    p.add_argument("--demo",        action="store_true",
                   help="Generate synthetic trust log and run")
    p.add_argument("--demo_rounds", type=int, default=10)
    p.add_argument("--anomaly_threshold",  type=float, default=0.5)
    p.add_argument("--quarantine_after",   type=int,   default=3)
    p.add_argument("--verify_every",       type=int,   default=5)
    p.add_argument("--policy", type=str, default=None,
                   help="Path to governance policy JSON/YAML")
    p.add_argument("--privacy_policy", type=str, default=None,
                   help="Path to privacy policy JSON/YAML")
    p.add_argument("--enforce_privacy_policy", action="store_true",
                   help="Fail run when privacy policy violations are detected")
    p.add_argument("--audit_chain", action="store_true",
                   help="Read-only audit from blockchain attestations (ignores trust_log)")
    p.add_argument("--allow_unsigned_events", action="store_true",
                   help="Allow legacy trust logs without signed round_event records")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 65)
    print("  SPLIT 3 — BLOCKCHAIN GOVERNANCE LAYER")
    print(f"  Backend  : {args.blockchain.upper()}")
    print("=" * 65)

    # ── Configure governance engine ───────────────────────────────────────────
    backend = args.blockchain.lower()
    config = GovernanceConfig(
        anomaly_threshold      = args.anomaly_threshold,
        consecutive_flag_limit = args.quarantine_after,
        verify_chain_every_n   = args.verify_every,
        use_simulation         = (backend == "simulation"),
        use_fabric             = (backend == "fabric"),
        expected_backend       = backend,
        require_backend_match  = (backend != "simulation"),
        fail_on_commit_error   = True,
        require_verified_round_events = (not args.allow_unsigned_events and not args.demo),
        policy_path            = args.policy,
        privacy_policy_path    = args.privacy_policy,
        enforce_privacy_policy = args.enforce_privacy_policy,
        output_dir             = args.output_dir,
    )

    engine = GovernanceEngine(config)

    # ── Read-only blockchain audit mode ───────────────────────────────────────
    if args.audit_chain:
        t0 = time.time()
        audit = engine.audit_blockchain_attestations()
        elapsed = time.time() - t0
        print(f"\n[Main] Blockchain audit completed in {elapsed:.2f}s")
        print(f"  Rounds discovered: {audit['round_count']}")
        print(f"  Verified rounds:   {audit['verified_count']}")
        print(f"  Failed rounds:     {audit['failed_count']}")
        print(f"  Overall status:    {'PASS' if audit['all_verified'] else 'FAIL'}")
        return

    # ── Trust log (legacy replay mode) ──────────────────────────────────────
    if args.demo:
        trust_log_path = "./demo_trust_log.json"
        generate_demo_trust_log(num_rounds=args.demo_rounds, output_path=trust_log_path)
    elif args.trust_log is None:
        print("\n[Error] Provide --trust_log or use --demo")
        sys.exit(1)
    else:
        trust_log_path = args.trust_log

    if not os.path.exists(trust_log_path):
        print(f"\n[Error] Trust log not found: {trust_log_path}")
        sys.exit(1)

    # ── Run ───────────────────────────────────────────────────────────────────
    t0 = time.time()
    report = engine.process_trust_log(trust_log_path)
    elapsed = time.time() - t0

    # ── Tamper simulation demo ────────────────────────────────────────────────
    if args.tamper_round is not None:
        result = engine.run_tamper_simulation(round_to_tamper=args.tamper_round)
        print(f"\n  Tamper simulation: {result['detection_message']}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[Main] Processing time: {elapsed:.2f}s")
    print(f"\n  Summary:")
    print(f"    Rounds processed:    {report.total_rounds}")
    print(f"    Hash chain intact:   {'✅ YES' if report.chain_intact else '🚨 BROKEN'}")
    print(f"    Tamper events:       {report.tamper_events}")
    print(f"    Quarantined clients: {report.quarantined_clients}")
    print(f"    Best Global F1:      {report.best_f1:.4f} @ Round {report.best_round}")
    print(f"\n  Reports → {args.output_dir}/")


if __name__ == "__main__":
    main()
