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
import random
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
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
        load_partition,
    )
    from module1.common.partition_cache import (
        build_partition_spec,
        get_partition_cache_paths,
        load_partition_cache_if_match,
        save_partition_cache,
    )
    from module1.common.trust_weighted_strategy import get_trust_strategy
    from module1.common.attack_simulator import AttackSimulator
    from module1.common.flower_client import BankFederatedClient
    from module1.common.governance_bridge import build_governance_engine
    from module1.common.experiment_tracking import write_run_manifest, write_baseline_comparison_report, summarize_run
except ImportError:
    from common.data_partition import (
        load_dataset,
        dirichlet_partition,
        make_synthetic_data,
        load_partition,
    )
    from common.partition_cache import (
        build_partition_spec,
        get_partition_cache_paths,
        load_partition_cache_if_match,
        save_partition_cache,
    )
    from common.trust_weighted_strategy import get_trust_strategy
    from common.attack_simulator import AttackSimulator
    from common.flower_client import BankFederatedClient
    from common.governance_bridge import build_governance_engine
    from common.experiment_tracking import write_run_manifest, write_baseline_comparison_report, summarize_run

SERVER_ADDRESS = "127.0.0.1:8081"


def _to_bool_env(value: str, default: bool) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _to_wsl_path(path_text: str) -> str:
    normalized = os.path.abspath(path_text).replace("\\", "/")
    if len(normalized) >= 2 and normalized[1] == ":":
        drive = normalized[0].lower()
        rest = normalized[2:]
        if rest.startswith("/"):
            rest = rest[1:]
        return f"/mnt/{drive}/{rest}"
    return normalized


def _ensure_fabric_ready() -> None:
    """Preflight Fabric for direct Split 2 runs on Windows/WSL.

    Controlled by env vars:
      BATFL_SKIP_FABRIC_PREFLIGHT=true  -> skip all checks
      BATFL_AUTO_FABRIC_SETUP=false     -> do not auto-run enterprise setup on failed preflight
      BATFL_ENTERPRISE_CONSENSUS=bft|raft
    """
    if _to_bool_env(os.getenv("BATFL_SKIP_FABRIC_PREFLIGHT"), False):
        print("[Main] BATFL_SKIP_FABRIC_PREFLIGHT=true -> skipping Fabric preflight.")
        return

    repo_root = os.path.dirname(_MODULE_ROOT)
    enterprise_dir = os.path.join(repo_root, "module1", "split3", "hlf_enterprise")
    env_file = os.path.join(enterprise_dir, "fabric_connection.env")
    health_sh = os.path.join(enterprise_dir, "health_check.sh")
    setup_sh = os.path.join(enterprise_dir, "setup.sh")
    consensus_env = os.getenv("BATFL_ENTERPRISE_CONSENSUS", "auto").strip().lower()

    def _detect_consensus_mode() -> str:
        if consensus_env in {"raft", "bft"}:
            return consensus_env
        try:
            probe = subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                check=False,
                capture_output=True,
                text=True,
            )
            names = set((probe.stdout or "").splitlines())
            # BFT profile has additional orderers (orderer2/3/4).
            if "orderer2.example.com" in names or "orderer3.example.com" in names or "orderer4.example.com" in names:
                return "bft"
        except Exception:
            pass
        return "raft"

    consensus = _detect_consensus_mode()
    fallback_consensus = "raft" if consensus == "bft" else "bft"

    if not os.path.exists(health_sh) or not os.path.exists(setup_sh):
        raise RuntimeError(
            "Fabric enterprise scripts not found. Expected under module1/split3/hlf_enterprise/."
        )

    if os.path.exists(env_file):
        os.environ["BATFL_FABRIC_ENV_FILE"] = env_file

    wsl_health = _to_wsl_path(health_sh)

    def _run_health_check(mode: str) -> bool:
        health_cmd = f"chmod +x '{wsl_health}' ; '{wsl_health}' '{mode}'"
        print(f"[Main] Running Fabric preflight health check ({mode.upper()})...", flush=True)
        try:
            subprocess.run(["wsl", "bash", "-lc", health_cmd], check=True)
            os.environ["BATFL_ENTERPRISE_CONSENSUS"] = mode
            print(f"[Main] Fabric preflight passed ({mode.upper()}).", flush=True)
            return True
        except subprocess.CalledProcessError:
            return False

    if _run_health_check(consensus) or _run_health_check(fallback_consensus):
        return

    if not _to_bool_env(os.getenv("BATFL_AUTO_FABRIC_SETUP"), True):
        raise RuntimeError(
            "Fabric preflight failed and BATFL_AUTO_FABRIC_SETUP=false. "
            "Run module1/split3/hlf_enterprise/setup.sh, then retry."
        )

    wsl_setup = _to_wsl_path(setup_sh)
    setup_cmd = (
        f"export BATFL_ENTERPRISE_CONSENSUS='{consensus}' ; "
        f"chmod +x '{wsl_setup}' ; '{wsl_setup}'"
    )

    print(f"[Main] Fabric preflight failed, running enterprise setup ({consensus.upper()})...", flush=True)
    subprocess.run(["wsl", "bash", "-lc", setup_cmd], check=True)

    if os.path.exists(env_file):
        os.environ["BATFL_FABRIC_ENV_FILE"] = env_file

    if _run_health_check(consensus) or _run_health_check(fallback_consensus):
        print("[Main] Fabric setup + preflight completed.", flush=True)
        return

    raise RuntimeError(
        "Fabric setup completed but health check still failed for both RAFT and BFT modes."
    )


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
                   label_flip_clients=None, flip_fraction=1.0, attack_start_round=1,
                   attack_end_round=None, attack_type="none", attack_intensity=1.0,
                   trigger_feature_count=3, trigger_value=5.0, backdoor_target_label=1,
                   threshold_mode="auto", decision_threshold=0.5, threshold_beta=1.5):
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
        attack_end_round   = attack_end_round,
        attack_type        = attack_type,
        attack_intensity   = attack_intensity,
        trigger_feature_count = trigger_feature_count,
        trigger_value      = trigger_value,
        backdoor_target_label = backdoor_target_label,
        threshold_mode     = threshold_mode,
        decision_threshold = decision_threshold,
        threshold_beta     = threshold_beta,
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
                   choices=["none", "label_flip", "gradient_scale", "combined", "backdoor", "sign_flip", "model_replacement", "sybil"])
    p.add_argument("--malicious",    type=int,   nargs="+", default=[1])
    p.add_argument("--gs_clients",   type=int,   nargs="+", default=None)
    p.add_argument("--scale_factor", type=float, default=5.0)
    p.add_argument("--attack_start", type=int,   default=1)
    p.add_argument("--attack_end",   type=int,   default=None,
                   help="Optional attack stop round (inclusive)")
    p.add_argument("--attack_intensity", type=float, default=1.0,
                   help="Attack intensity multiplier")
    p.add_argument("--trigger_feature_count", type=int, default=3,
                   help="Backdoor trigger feature count")
    p.add_argument("--trigger_value", type=float, default=5.0,
                   help="Backdoor trigger value")
    p.add_argument("--backdoor_target_label", type=int, default=1,
                   help="Backdoor target class label")

    # Evaluation / threshold tuning
    p.add_argument("--threshold_mode", type=str, default="auto", choices=["auto", "fixed"],
                   help="Decision-threshold mode for fraud prediction")
    p.add_argument("--decision_threshold", type=float, default=0.5,
                   help="Fixed threshold when --threshold_mode=fixed")
    p.add_argument("--threshold_beta", type=float, default=1.5,
                   help="F-beta weight used in auto threshold tuning")

    # Experiment tracking
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--comparison_report", action="store_true",
                   help="Generate baseline comparison report from available runs")
    p.add_argument("--comparison_dirs", nargs="+", default=["logs_split1", "logs_split2"],
                   help="Run directories used for comparison report")

    # Blockchain (Module 2)
    p.add_argument("--blockchain",   type=str,   default="simulation",
                   choices=["simulation", "ganache", "fabric"],
                   help="Blockchain backend for live governance during training (default: simulation)")
    p.add_argument("--no_blockchain", action="store_true",
                   help="Disable Module 2 entirely (Module 1 only mode)")
    p.add_argument("--governance_policy", type=str, default=None,
                   help="Path to governance policy JSON/YAML file")
    p.add_argument("--event_storage", type=str, default="jsonl", choices=["jsonl", "sqlite"],
                   help="Round event storage backend")

    # Hidden subprocess flags
    p.add_argument("--_client_mode",   action="store_true",  help=argparse.SUPPRESS)
    p.add_argument("--_cid",     type=int,   default=-1,            help=argparse.SUPPRESS)
    p.add_argument("--_cache",   type=str,   default="",            help=argparse.SUPPRESS)
    p.add_argument("--_model",   type=str,   default="dnn",         help=argparse.SUPPRESS)
    p.add_argument("--_smote",   type=str,   default="true",        help=argparse.SUPPRESS)
    p.add_argument("--_server",  type=str,   default="127.0.0.1:8081", help=argparse.SUPPRESS)
    p.add_argument("--_flip_clients",  type=str,   default="",      help=argparse.SUPPRESS)
    p.add_argument("--_flip_fraction", type=float, default=1.0,     help=argparse.SUPPRESS)
    p.add_argument("--_attack_type", type=str, default="none", help=argparse.SUPPRESS)
    p.add_argument("--_attack_intensity", type=float, default=1.0, help=argparse.SUPPRESS)
    p.add_argument("--_attack_end", type=int, default=-1, help=argparse.SUPPRESS)
    p.add_argument("--_trigger_feature_count", type=int, default=3, help=argparse.SUPPRESS)
    p.add_argument("--_trigger_value", type=float, default=5.0, help=argparse.SUPPRESS)
    p.add_argument("--_backdoor_target_label", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--_threshold_mode", type=str, default="auto", help=argparse.SUPPRESS)
    p.add_argument("--_decision_threshold", type=float, default=0.5, help=argparse.SUPPRESS)
    p.add_argument("--_threshold_beta", type=float, default=1.5, help=argparse.SUPPRESS)
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
            attack_end_round   = None if args._attack_end < 0 else args._attack_end,
            attack_type        = args._attack_type,
            attack_intensity   = args._attack_intensity,
            trigger_feature_count = args._trigger_feature_count,
            trigger_value      = args._trigger_value,
            backdoor_target_label = args._backdoor_target_label,
            threshold_mode     = args._threshold_mode,
            decision_threshold = args._decision_threshold,
            threshold_beta     = args._threshold_beta,
        )
        return

    # ── ORCHESTRATOR MODE ─────────────────────────────────────────────────────
    import flwr as fl

    np.random.seed(args.seed)
    random.seed(args.seed)

    server_address = f"127.0.0.1:{args.port}"
    os.makedirs(args.log_dir, exist_ok=True)

    sep = "=" * 65
    print(f"\n{sep}")
    print("  SPLIT 2 — TRUST-WEIGHTED FEDERATED LEARNING")
    print(f"  Model    : {args.model.upper()} | Clients: {args.num_clients} | Rounds: {args.rounds}")
    print(f"  Attack   : {args.attack.upper()} | Malicious: {args.malicious} | schedule={args.attack_start}..{args.attack_end if args.attack_end is not None else 'end'}")
    print(f"  Blockchain: {args.blockchain.upper() if not args.no_blockchain else 'DISABLED'}")
    print(f"{sep}\n")

    # ── Step 1: Attack simulator ──────────────────────────────────────────────
    flip_clients_for_subprocess = []
    scale_clients_for_server    = []

    if args.attack == "label_flip":
        flip_clients_for_subprocess = args.malicious
    elif args.attack == "backdoor":
        flip_clients_for_subprocess = []
    elif args.attack == "gradient_scale":
        scale_clients_for_server    = args.malicious
    elif args.attack in ("sign_flip", "model_replacement", "sybil"):
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
        server_attack_type = args.attack if scale_clients_for_server else "none"
        attacker = AttackSimulator(
            attack_type        = server_attack_type,
            malicious_clients  = scale_clients_for_server,
            scale_factor       = args.scale_factor,
            attack_start_round = args.attack_start,
            attack_end_round   = args.attack_end,
            attack_intensity   = args.attack_intensity,
            trigger_feature_count = args.trigger_feature_count,
            trigger_value      = args.trigger_value,
            backdoor_target_label = args.backdoor_target_label,
        )

    # ── Step 2: Blockchain governance engine ──────────────────────────────────
    governance_engine = None
    if not args.no_blockchain:
        if args.blockchain == "fabric":
            _ensure_fabric_ready()
        governance_engine = build_governance_engine(
            log_dir   = args.log_dir,
            backend   = args.blockchain,
            enabled   = True,
            strict    = True,
            policy_path = args.governance_policy,
        )

    # ── Step 3: Load data ─────────────────────────────────────────────────────
    if args.synthetic or args.data_path is None:
        if not args.synthetic:
            print("[Main] No --data_path given — using synthetic data.\n")
        X, y = make_synthetic_data()
    else:
        X, y = load_dataset(args.data_path)

    # ── Step 4: Partition data ────────────────────────────────────────────────
    cache_path, _ = get_partition_cache_paths(args.log_dir, ".partition_cache_flwr_app.npz")
    spec = build_partition_spec(
        X=X,
        y=y,
        num_clients=args.num_clients,
        alpha=args.alpha,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    partitions = load_partition_cache_if_match(cache_path, spec, args.num_clients)
    if partitions is None:
        partitions = dirichlet_partition(
            X, y,
            num_clients = args.num_clients,
            alpha       = args.alpha,
            max_samples = args.max_samples,
        )
        save_partition_cache(partitions, cache_path, spec)
        print(f"[Main] Partition cache rebuilt -> {cache_path}")
    else:
        print(f"[Main] Reusing deterministic partition cache -> {cache_path}")

    # ── Step 5: Build trust-weighted strategy ─────────────────────────────────
    strategy = get_trust_strategy(
        num_clients      = args.num_clients,
        fraction_fit     = args.fraction_fit,
        log_dir          = args.log_dir,
        attack_simulator = attacker,
        governance_engine = governance_engine,
        event_storage_backend = args.event_storage,
    )

    # ── Step 6: Run modern Flower simulation runtime ──────────────────────────
    partition_map = {int(p["client_id"]): p for p in partitions}

    def client_fn(context):
        cid = int(context.node_config.get("partition-id", context.node_id))
        part = partition_map[cid]
        malicious_set = set(int(x) for x in args.malicious)
        effective_attack_type = args.attack if cid in malicious_set else "none"
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
            attack_end_round   = args.attack_end,
            attack_type        = effective_attack_type,
            attack_intensity   = args.attack_intensity,
            trigger_feature_count = args.trigger_feature_count,
            trigger_value      = args.trigger_value,
            backdoor_target_label = args.backdoor_target_label,
            threshold_mode     = args.threshold_mode,
            decision_threshold = args.decision_threshold,
            threshold_beta     = args.threshold_beta,
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

    repo_root = os.path.dirname(_MODULE_ROOT)
    run_id = os.getenv("BATFL_RUN_ID", f"run-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")
    dataset_meta = {
        "source": "synthetic" if (args.synthetic or args.data_path is None) else args.data_path,
        "num_samples": int(len(y)),
        "num_features": int(X.shape[1]),
        "fraud_rate": float(np.mean(y)),
    }
    run_config = {
        "strategy": "trust_weighted",
        "model": args.model,
        "num_clients": args.num_clients,
        "rounds": args.rounds,
        "alpha": args.alpha,
        "fraction_fit": args.fraction_fit,
        "threshold_mode": args.threshold_mode,
        "decision_threshold": args.decision_threshold,
        "threshold_beta": args.threshold_beta,
        "event_storage": args.event_storage,
        "attack": args.attack,
        "attack_start": args.attack_start,
        "attack_end": args.attack_end,
        "attack_intensity": args.attack_intensity,
        "malicious_clients": args.malicious,
        "scale_factor": args.scale_factor,
        "blockchain": ("disabled" if args.no_blockchain else args.blockchain),
        "seed": args.seed,
    }
    runtime_meta = {
        "flower_server": server_address,
        "flower_version": str(getattr(fl, "__version__", "unknown")),
        "backend": args.blockchain,
        "governance_policy": args.governance_policy,
    }
    manifest_path = write_run_manifest(
        repo_root=repo_root,
        log_dir=args.log_dir,
        run_config=run_config,
        dataset_meta=dataset_meta,
        runtime_meta=runtime_meta,
        governance_output_dir=(os.path.join(args.log_dir, "governance_output") if not args.no_blockchain else None),
    )
    print(f"  Manifest → {manifest_path}")

    if args.comparison_report:
        summaries = []
        for run_dir in args.comparison_dirs:
            trust_log = os.path.join(run_dir, "trust_training_log.json")
            split1_log = os.path.join(run_dir, "training_log.json")
            manifest = os.path.join(run_dir, "run_manifest.json")
            if os.path.exists(trust_log):
                summaries.append(summarize_run(trust_log, manifest_path=manifest, label=os.path.basename(run_dir)))
            elif os.path.exists(split1_log):
                summaries.append(summarize_run(split1_log, manifest_path=manifest, label=os.path.basename(run_dir)))
        if summaries:
            report_paths = write_baseline_comparison_report(
                summaries=summaries,
                output_dir=args.log_dir,
                prefix="baseline_comparison",
            )
            print(f"  Comparison CSV → {report_paths['csv']}")
            print(f"  Comparison Plot → {report_paths['plot']}")
    print(f"\n  Next: python module1/split3/split3_main.py --trust_log {log_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
