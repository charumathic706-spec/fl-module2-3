from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from module1.common.data_partition import save_partitions, load_partition
except ImportError:
    from common.data_partition import save_partitions, load_partition


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dataset_fingerprint(X: np.ndarray, y: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(X.shape).encode("utf-8"))
    h.update(str(y.shape).encode("utf-8"))
    h.update(str(X.dtype).encode("utf-8"))
    h.update(str(y.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(X).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()


def build_partition_spec(
    *,
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    max_samples: int | None,
    seed: int,
) -> Dict[str, Any]:
    return {
        "dataset_fingerprint": _dataset_fingerprint(X, y),
        "dataset_shape": [int(X.shape[0]), int(X.shape[1])],
        "label_count": int(len(y)),
        "label_sum": int(np.sum(y)),
        "num_clients": int(num_clients),
        "alpha": float(alpha),
        "max_samples": None if max_samples is None else int(max_samples),
        "seed": int(seed),
    }


def partition_spec_checksum(spec: Dict[str, Any]) -> str:
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return _sha256_text(canonical)


def get_partition_cache_paths(log_dir: str, filename: str = ".partition_cache.npz") -> Tuple[str, str]:
    npz_path = os.path.join(log_dir, filename)
    meta_path = npz_path + ".meta.json"
    return npz_path, meta_path


def save_partition_cache(partitions: List[Dict[str, Any]], cache_path: str, spec: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    save_partitions(partitions, cache_path)
    checksum = partition_spec_checksum(spec)
    with open(cache_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"spec": spec, "checksum": checksum}, f, indent=2)


def load_partition_cache_if_match(
    cache_path: str,
    expected_spec: Dict[str, Any],
    num_clients: int,
) -> List[Dict[str, Any]] | None:
    meta_path = cache_path + ".meta.json"
    if not (os.path.exists(cache_path) and os.path.exists(meta_path)):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    expected_checksum = partition_spec_checksum(expected_spec)
    if str(meta.get("checksum", "")) != expected_checksum:
        return None

    out: List[Dict[str, Any]] = []
    for cid in range(int(num_clients)):
        out.append(load_partition(cache_path, cid))
    return out
