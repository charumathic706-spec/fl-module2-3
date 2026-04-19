from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from flwr.client import ClientApp
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents, ServerConfig

from module1.common.attack_simulator import AttackSimulator
from module1.common.data_partition import (
    dirichlet_partition,
    load_dataset,
    make_synthetic_data,
)
from module1.common.flower_client import BankFederatedClient
from module1.common.governance_bridge import build_governance_engine
from module1.common.trust_weighted_strategy import get_trust_strategy


_PARTITION_CACHE: Dict[Tuple, Dict[int, Dict]] = {}


def _to_bool(v, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _csv_ints(text: str | None) -> List[int]:
    if not text:
        return []
    out = []
    for token in str(text).split(","):
        t = token.strip()
        if t:
            out.append(int(t))
    return out


def _load_partitions(run_config: Dict) -> Dict[int, Dict]:
    num_clients = int(run_config.get("num_clients", 5))
    alpha = float(run_config.get("alpha", 1.0))
    max_samples_val = run_config.get("max_samples", "")
    max_samples = int(max_samples_val) if str(max_samples_val).strip() else None
    use_synthetic = _to_bool(run_config.get("use_synthetic", True), True)
    data_path = str(run_config.get("data_path", "")).strip()

    key = (num_clients, alpha, max_samples, use_synthetic, data_path)
    if key in _PARTITION_CACHE:
        return _PARTITION_CACHE[key]

    if use_synthetic or not data_path:
        X, y = make_synthetic_data()
    else:
        X, y = load_dataset(data_path)

    partitions = dirichlet_partition(
        X,
        y,
        num_clients=num_clients,
        alpha=alpha,
        max_samples=max_samples,
    )
    part_map = {int(p["client_id"]): p for p in partitions}
    _PARTITION_CACHE[key] = part_map
    return part_map


def _partitions_cache_path(cfg: Dict) -> str:
    log_dir = str(cfg.get("log_dir", "logs_split2"))
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(log_dir, ".partition_cache_flwr_app.npz")


def _save_partition_cache(partitions: Dict[int, Dict], path: str) -> None:
    arrays = {}
    for cid, p in partitions.items():
        arrays[f"{cid}_X_train"] = p["X_train"]
        arrays[f"{cid}_y_train"] = p["y_train"]
        arrays[f"{cid}_X_test"] = p["X_test"]
        arrays[f"{cid}_y_test"] = p["y_test"]
    np.savez_compressed(path, **arrays)


def _load_partition_from_cache(path: str, cid: int) -> Dict:
    d = np.load(path)
    return {
        "client_id": cid,
        "X_train": d[f"{cid}_X_train"],
        "y_train": d[f"{cid}_y_train"],
        "X_test": d[f"{cid}_X_test"],
        "y_test": d[f"{cid}_y_test"],
    }


def _build_attacker(run_config: Dict):
    attack = str(run_config.get("attack", "none")).strip().lower()
    malicious = _csv_ints(str(run_config.get("malicious", "1")))
    scale_factor = float(run_config.get("scale_factor", 5.0))
    attack_start = int(run_config.get("attack_start", 1))

    scale_clients = malicious if attack in {"gradient_scale", "combined"} else []
    server_attack_type = "gradient_scale" if scale_clients else "none"

    return AttackSimulator(
        attack_type=server_attack_type,
        malicious_clients=scale_clients,
        scale_factor=scale_factor,
        attack_start_round=attack_start,
    )


def _build_server_components(context: Context) -> ServerAppComponents:
    cfg = context.run_config

    num_clients = int(cfg.get("num_clients", 5))
    rounds = int(cfg.get("num_rounds", 20))
    fraction_fit = float(cfg.get("fraction_fit", 1.0))
    log_dir = str(cfg.get("log_dir", "logs_split2"))

    blockchain_enabled = _to_bool(cfg.get("blockchain_enabled", True), True)
    blockchain_backend = str(cfg.get("blockchain_backend", "fabric"))
    privacy_policy_path = str(cfg.get("privacy_policy", "")).strip() or None
    enforce_privacy_policy = _to_bool(cfg.get("enforce_privacy_policy", False), False)

    governance_engine = None
    if blockchain_enabled:
        governance_engine = build_governance_engine(
            log_dir=log_dir,
            backend=blockchain_backend,
            enabled=True,
            strict=True,
            privacy_policy_path=privacy_policy_path,
            enforce_privacy_policy=enforce_privacy_policy,
        )

    strategy = get_trust_strategy(
        num_clients=num_clients,
        fraction_fit=fraction_fit,
        log_dir=log_dir,
        attack_simulator=_build_attacker(cfg),
        governance_engine=governance_engine,
    )

    # Precompute and persist partitions once for all ClientApp actors.
    part_map = _load_partitions(cfg)
    cache_path = _partitions_cache_path(cfg)
    _save_partition_cache(part_map, cache_path)

    return ServerAppComponents(
        strategy=strategy,
        config=ServerConfig(num_rounds=rounds),
    )


def _client_fn(context: Context):
    cfg = context.run_config

    num_clients = int(cfg.get("num_clients", 5))
    model = str(cfg.get("model", "dnn"))
    use_smote = not _to_bool(cfg.get("no_smote", False), False)
    attack = str(cfg.get("attack", "none")).strip().lower()
    malicious = _csv_ints(str(cfg.get("malicious", "1")))
    attack_start = int(cfg.get("attack_start", 1))

    partition_id = int(context.node_config.get("partition-id", context.node_id))
    if partition_id >= num_clients:
        raise ValueError(
            f"partition-id {partition_id} is out of range for num_clients={num_clients}. "
            "Set Flower supernodes equal to num_clients."
        )
    cid = partition_id

    cache_path = _partitions_cache_path(cfg)
    if Path(cache_path).exists():
        part = _load_partition_from_cache(cache_path, cid)
    else:
        # Fallback in case cache is missing (first actor race / manual runs).
        part = _load_partitions(cfg)[cid]

    is_label_flip = attack in {"label_flip", "combined"} and (cid in set(malicious))

    client = BankFederatedClient(
        client_id=cid,
        X_train=part["X_train"],
        y_train=part["y_train"],
        X_test=part["X_test"],
        y_test=part["y_test"],
        model_type=model,
        use_smote=use_smote,
        is_label_flip=is_label_flip,
        flip_fraction=1.0,
        attack_start_round=attack_start,
    )
    return client.to_client()


server_app = ServerApp(server_fn=_build_server_components)
client_app = ClientApp(client_fn=_client_fn)
