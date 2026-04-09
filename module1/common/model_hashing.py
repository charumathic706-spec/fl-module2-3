from __future__ import annotations

import hashlib
import struct
from typing import List, Tuple

import numpy as np


def hash_model_parameters_canonical(params: List[np.ndarray]) -> Tuple[str, int, int]:
    """Canonical SHA-256 hash for model parameter tensors.

    Canonicalization rules:
    - dtype normalized to float32
    - tensor order preserved as provided
    - each tensor prefixed with shape descriptor (big-endian uints)
    """
    hasher = hashlib.sha256()
    total_bytes = 0

    for arr in params:
        arr32 = np.asarray(arr, dtype=np.float32)
        shape_tag = struct.pack(f">{len(arr32.shape)}I", *arr32.shape)
        data = arr32.flatten().tobytes()
        hasher.update(shape_tag)
        hasher.update(data)
        total_bytes += len(data)

    return hasher.hexdigest(), len(params), total_bytes
