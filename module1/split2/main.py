# =============================================================================
# FILE: split2/main.py
# PURPOSE: Split 2 — Trust-Weighted Aggregation + Attack Simulation + Blockchain
#
# What Split 2 adds on top of Split 1:
#   - TrustWeightedFedAvg replaces InstrumentedFedAvg
#   - AttackSimulator injects label-flip / gradient-scale attacks
#   - Per-round trust scores logged to JSON
#   - Model SHA-256 hash logged each round (→ Split 3 blockchain audit trail)
#   - Optional live blockchain governance (simulation / ganache / fabric)
#
# Usage:
#   python -m split2.main --synthetic
#   python -m split2.main --synthetic --blockchain ganache
#   python -m split2.main --synthetic --blockchain fabric
#   python -m split2.main --data_path ../data/creditcard.csv --attack label_flip --malicious 1
#   python -m split2.main --synthetic --attack combined --malicious 1 3 --blockchain simulation
# =============================================================================

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODULE_ROOT not in sys.path:
    sys.path.insert(0, _MODULE_ROOT)

try:
    from module1.common.data_partition import (
        load_dataset,
        dirichlet_partition,
        make_synthetic_data,
        save_partitions,
        load_partition,
    )
    from module1.common.trust_weighted_strategy import get_trust_strategy
    from module1.common.attack_simulator import AttackSimulator
    from module1.common.flower_client import BankFederatedClient
    from module1.common.governance_bridge import build_governance_engine
except ImportError:
    from common.data_partition import (
        load_dataset,
        dirichlet_partition,
        make_synthetic_data,
        save_partitions,
        load_partition,
    )
    from common.trust_weighted_strategy import get_trust_strategy
    from common.attack_simulator import AttackSimulator
    from common.flower_client import BankFederatedClient
    from common.governance_bridge import build_governance_engine

SERVER_ADDRESS = "127.0.0.1:8081"


# =============================================================================
# MULTI-ATTACK WRAPPER
# =============================================================================

class MultiAttackSimulator:
    """Applies different attack types to different client subsets."""

    def __init__(self, flip_clients, scale_clients, scale_factor=5.0, attack_start_round=1):
        self.flip_attacker = AttackSimulator(
            attack_type="label_flip" if flip_clients else "none",
            malicious_clients=flip_clients,
            attack_start_round=attack_start_round,
        )
        self.scale_attacker = AttackSimulator(
            attack_type="gradient_scale" if scale_clients else "none",
            malicious_clients=scale_clients,
            scale_factor=scale_factor,
            attack_start_round=attack_start_round,
        )
        print(f"[MultiAttack] label_flip={flip_clients} | gradient_scale={scale_clients} (×{scale_factor})")

    def set_round(self, round_num):
        self.flip_attacker.set_round(round_num)
        self.scale_attacker.set_round(round_num)

    def is_malicious(self, client_id):
        return self.flip_attacker.is_malicious(client_id) or self.scale_attacker.is_malicious(client_id)

    def poison_data(self, client_id, X, y):
        return self.flip_attacker.poison_data(client_id, X, y)

    def poison_params(self, client_id, params, global_params):
        return self.scale_attacker.poison_params(client_id, params, global_params)

    def get_attack_summary(self):
        return {"flip_clients": self.flip_attacker.malicious_clients,
                "scale_clients": self.scale_attacker.malicious_clients}


# =============================================================================
# PLOTTING
# =============================================================================

def plot_training_curves(log_path: str, save_dir: str) -> None:
    if not os.path.exists(log_path):
        print("[Plot] Log not found — skipping.")
        return
    with open(log_path) as f:
        logs = json.load(f)
    if not logs:
        return

    rounds  = [l["round"]                        for l in logs]
    f1s     = [l.get("global_f1", 0)             for l in logs]
    aucs    = [l.get("global_auc", 0)            for l in logs]
    recalls = [l.get("global_recall", 0)         for l in logs]
    flagged = [len(l.get("flagged_clients", [])) for l in logs]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle("Split 2 — Trust-Weighted FL + Attack Detection", fontsize=13, fontweight="bold")

    for ax, (vals, title, color) in zip(axes[:3], [
        (f1s,     "Global F1",     "#2196F3"),
        (aucs,    "Global AUC",    "#4CAF50"),
        (recalls, "Global Recall", "#FF9800"),
    ]):
        ax.plot(rounds, vals, marker="o", linewidth=2, color=color, markersize=6)
        ax.fill_between(rounds, vals, alpha=0.12, color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Round"); ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.05); ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

    axes[3].bar(rounds, flagged, color="#E53935", alpha=0.8)
    axes[3].set_title("Flagged Clients / Round", fontsize=11)
    axes[3].set_xlabel("Round"); axes[3].set_ylabel("Count")
    axes[3].set_ylim(0, max(flagged or [1]) + 1)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "split2_training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved -> {out}")


# =============================================================================
# CLIENT SUBPROCESS ENTRY POINT
# =============================================================================

def _run_as_client(cid, cache, model_type, use_smote, server_address,
                   label_flip_clients=None, flip_fraction=1.0, attack_start_round=1):
    import flwr as fl

    partition = load_partition(cache, cid)
    client = BankFederatedClient(
        client_id          = cid,
        X_train            = partition["X_train"],
        y_train            = partition["y_train"],
        X_test             = partition["X_test"],
        y_test             = partition["y_test"],
        model_type         = model_type,
        use_smote          = use_smote,
        is_label_flip      = cid in set(label_flip_clients or []),
        flip_fraction      = flip_fraction,
        attack_start_round = attack_start_round,
    )

    max_attempts = 90
    for attempt in range(max_attempts):
        try:
            fl.client.start_client(server_address=server_address, client=client.to_client())
            break
        except Exception as exc:
            if attempt < max_attempts - 1:
                time.sleep(1.0)
            else:
                print(f"  [Bank {cid:02d}] Failed to connect after {max_attempts}s: {exc}")
                sys.exit(1)


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="Split 2 — Trust-Weighted FL + Attack Simulation + Blockchain Governance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data
    p.add_argument("--data_path",    type=str,   default=None)
    p.add_argument("--synthetic",    action="store_true",
                   help="Use synthetic data (no CSV needed)")
    p.add_argument("--max_samples",  type=int,   default=None)

    # FL
    p.add_argument("--num_clients",  type=int,   default=5)
    p.add_argument("--rounds",       type=int,   default=20)
    p.add_argument("--model",        type=str,   default="dnn", choices=["dnn", "logistic"])
    p.add_argument("--alpha",        type=float, default=1.0)
    p.add_argument("--fraction_fit", type=float, default=1.0)
    p.add_argument("--no_smote",     action="store_true")
    p.add_argument("--log_dir",      type=str,   default="logs_split2")
    p.add_argument("--port",         type=int,   default=8081)

    # Attack
    p.add_argument("--attack",       type=str,   default="none",
                   choices=["none", "label_flip", "gradient_scale", "combined"])
    p.add_argument("--malicious",    type=int,   nargs="+", default=[1])
    p.add_argument("--gs_clients",   type=int,   nargs="+", default=None)
    p.add_argument("--scale_factor", type=float, default=5.0)
    p.add_argument("--attack_start", type=int,   default=1)

    # Blockchain (Module 2)
    p.add_argument("--blockchain",   type=str,   default="simulation",
                   choices=["simulation", "ganache", "fabric"],
                   help="Blockchain backend for live governance during training (default: simulation)")
    p.add_argument("--no_blockchain", action="store_true",
                   help="Disable Module 2 entirely (Module 1 only mode)")

    # Hidden subprocess flags
    p.add_argument("--_client_mode",   action="store_true",  help=argparse.SUPPRESS)
    p.add_argument("--_cid",     type=int,   default=-1,            help=argparse.SUPPRESS)
    p.add_argument("--_cache",   type=str,   default="",            help=argparse.SUPPRESS)
    p.add_argument("--_model",   type=str,   default="dnn",         help=argparse.SUPPRESS)
    p.add_argument("--_smote",   type=str,   default="true",        help=argparse.SUPPRESS)
    p.add_argument("--_server",  type=str,   default="127.0.0.1:8081", help=argparse.SUPPRESS)
    p.add_argument("--_flip_clients",  type=str,   default="",      help=argparse.SUPPRESS)
    p.add_argument("--_flip_fraction", type=float, default=1.0,     help=argparse.SUPPRESS)
    return p


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = build_parser().parse_args()

    # ── CLIENT MODE ───────────────────────────────────────────────────────────
    if args._client_mode:
        flip_clients_sub = []
        if args._flip_clients:
            try:
                flip_clients_sub = [int(x) for x in args._flip_clients.split(",") if x.strip()]
            except ValueError:
                pass
        _run_as_client(
            cid                = args._cid,
            cache              = args._cache,
            model_type         = args._model,
            use_smote          = (args._smote.lower() == "true"),
            server_address     = args._server,
            label_flip_clients = flip_clients_sub,
            flip_fraction      = args._flip_fraction,
            attack_start_round = args.attack_start,
        )
        return

    # ── ORCHESTRATOR MODE ─────────────────────────────────────────────────────
    import flwr as fl

    server_address = f"127.0.0.1:{args.port}"
    os.makedirs(args.log_dir, exist_ok=True)

    sep = "=" * 65
    print(f"\n{sep}")
    print("  SPLIT 2 — TRUST-WEIGHTED FEDERATED LEARNING")
    print(f"  Model    : {args.model.upper()} | Clients: {args.num_clients} | Rounds: {args.rounds}")
    print(f"  Attack   : {args.attack.upper()} | Malicious: {args.malicious}")
    print(f"  Blockchain: {args.blockchain.upper() if not args.no_blockchain else 'DISABLED'}")
    print(f"{sep}\n")

    # ── Step 1: Attack simulator ──────────────────────────────────────────────
    flip_clients_for_subprocess = []
    scale_clients_for_server    = []

    if args.attack == "label_flip":
        flip_clients_for_subprocess = args.malicious
    elif args.attack == "gradient_scale":
        scale_clients_for_server    = args.malicious
    elif args.attack == "combined":
        flip_clients_for_subprocess = args.malicious
        scale_clients_for_server    = args.malicious
    elif args.gs_clients is not None:
        flip_clients_for_subprocess = args.malicious if args.attack == "label_flip" else []
        scale_clients_for_server    = args.gs_clients

    if args.gs_clients is not None and args.attack in ("label_flip", "none"):
        attacker = MultiAttackSimulator(
            flip_clients       = [],
            scale_clients      = args.gs_clients,
            scale_factor       = args.scale_factor,
            attack_start_round = args.attack_start,
        )
    else:
        server_attack_type = "gradient_scale" if scale_clients_for_server else "none"
        attacker = AttackSimulator(
            attack_type        = server_attack_type,
            malicious_clients  = scale_clients_for_server,
            scale_factor       = args.scale_factor,
            attack_start_round = args.attack_start,
        )

    # ── Step 2: Blockchain governance engine ──────────────────────────────────
    governance_engine = None
    if not args.no_blockchain:
        governance_engine = build_governance_engine(
            log_dir   = args.log_dir,
            backend   = args.blockchain,
            enabled   = True,
            strict    = True,
        )

    # ── Step 3: Load data ─────────────────────────────────────────────────────
    if args.synthetic or args.data_path is None:
        if not args.synthetic:
            print("[Main] No --data_path given — using synthetic data.\n")
        X, y = make_synthetic_data()
    else:
        X, y = load_dataset(args.data_path)

    # ── Step 4: Partition data ────────────────────────────────────────────────
    partitions = dirichlet_partition(
        X, y,
        num_clients = args.num_clients,
        alpha       = args.alpha,
        max_samples = args.max_samples,
    )

    # ── Step 5: Build trust-weighted strategy ─────────────────────────────────
    strategy = get_trust_strategy(
        num_clients      = args.num_clients,
        fraction_fit     = args.fraction_fit,
        log_dir          = args.log_dir,
        attack_simulator = attacker,
        governance_engine = governance_engine,
    )

    # ── Step 6: Run modern Flower simulation runtime ──────────────────────────
    partition_map = {int(p["client_id"]): p for p in partitions}

    def client_fn(context):
        cid = int(context.node_config.get("partition-id", context.node_id))
        part = partition_map[cid]
        client = BankFederatedClient(
            client_id          = cid,
            X_train            = part["X_train"],
            y_train            = part["y_train"],
            X_test             = part["X_test"],
            y_test             = part["y_test"],
            model_type         = args.model,
            use_smote          = not args.no_smote,
            is_label_flip      = cid in set(flip_clients_for_subprocess),
            flip_fraction      = 1.0,
            attack_start_round = args.attack_start,
        )
        return client.to_client()

    print("\n[Main] Starting Flower simulation runtime...\n")

    # ── Step 7: Start Flower simulation ───────────────────────────────────────
    try:
        fl.simulation.start_simulation(
            client_fn      = client_fn,
            num_clients    = args.num_clients,
            config         = fl.server.ServerConfig(num_rounds=args.rounds),
            strategy       = strategy,
            client_resources = {"num_cpus": 1.0},
            ray_init_args  = {
                "ignore_reinit_error": True,
                "include_dashboard": False,
            },
        )
    except Exception as exc:
        print(f"\n[Main] Simulation error: {exc}")
        raise

    print("\n[Main] All rounds complete.")

    strategy.print_summary()

    log_path = os.path.join(args.log_dir, "trust_training_log.json")
    plot_training_curves(log_path, args.log_dir)

    print(f"\n[Main] ✅ Split 2 complete.")
    print(f"  Log  → {log_path}")
    print(f"  Plot → {os.path.join(args.log_dir, 'split2_training_curves.png')}")
    if not args.no_blockchain:
        print(f"\n  Governance output → {args.log_dir}/governance_output/")
    print(f"\n  Next: python module1/split3/split3_main.py --trust_log {log_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
