"""
common/governance_bridge.py
---------------------------
Bridges Module 1 (split2) with Module 2 (split3) blockchain governance.

Three backends:
    simulation — blockchain_sim.py   (default, no external deps)
    ganache    — eth_gateway.py      (real Ethereum via Ganache)
    fabric     — hlf_gateway.py      (real Hyperledger Fabric)

Usage:
    from common.governance_bridge import build_governance_engine
    engine = build_governance_engine(backend="simulation")
    engine = build_governance_engine(backend="ganache")
    engine = build_governance_engine(backend="fabric")
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from typing import Optional


def _find_split3() -> Optional[Path]:
    """Locate split3/ containing governance.py."""
    this = Path(__file__).resolve()
    for p in [
        this.parent.parent / "split3",           # module1/split3
        this.parent.parent.parent / "split3",    # project/split3
        Path(os.getcwd()) / "split3",
        Path(os.getcwd()).parent / "split3",
    ]:
        if (p / "governance.py").exists():
            return p
    return None


def build_governance_engine(
    log_dir: str = "logs_split2",
    governance_output: str = "governance_output",
    anomaly_threshold: float = 0.5,
    consecutive_flag_limit: int = 3,
    backend: str = "simulation",
    enabled: bool = True,
    strict: bool = True,
    policy_path: Optional[str] = None,
    privacy_policy_path: Optional[str] = None,
    enforce_privacy_policy: bool = False,
):
    """
    Build and return a GovernanceEngine for Module 2 blockchain integration.

    Args:
        log_dir:                Where split2 writes trust_training_log.json
        governance_output:      Sub-directory for governance reports
        anomaly_threshold:      Anomaly score above which clients are flagged
        consecutive_flag_limit: Flagged rounds before quarantine
        backend:                "simulation" | "ganache" | "fabric"
        enabled:                False = return None (Module 1 only mode)
    """
    if not enabled:
        print("[Module 2] Governance disabled — Module 1 only mode.")
        return None

    split3_path = _find_split3()
    if split3_path is None:
        msg = "[Module 2] split3/ not found."
        if strict:
            raise RuntimeError(msg + " Strict mode is enabled, refusing to disable governance.")
        print(msg + " Governance disabled.")
        return None

    if str(split3_path) not in sys.path:
        sys.path.insert(0, str(split3_path))

    try:
        from governance import GovernanceEngine, GovernanceConfig
    except ImportError as e:
        if strict:
            raise RuntimeError(
                f"[Module 2] Cannot import GovernanceEngine in strict mode: {e}"
            ) from e
        print(f"[Module 2] WARNING: Cannot import GovernanceEngine: {e}")
        return None

    output_dir = os.path.join(log_dir, governance_output)
    os.makedirs(output_dir, exist_ok=True)

    backend = backend.lower()
    config = GovernanceConfig(
        anomaly_threshold       = anomaly_threshold,
        consecutive_flag_limit  = consecutive_flag_limit,
        use_simulation          = (backend == "simulation"),
        use_fabric              = (backend == "fabric"),
        expected_backend        = backend,
        require_backend_match   = strict and (backend != "simulation"),
        fail_on_commit_error    = strict,
        require_verified_round_events = strict,
        policy_path             = policy_path,
        privacy_policy_path     = privacy_policy_path,
        enforce_privacy_policy  = enforce_privacy_policy,
        output_dir              = output_dir,
    )

    engine = GovernanceEngine(config=config)
    print(f"[Module 2] GovernanceEngine ready | backend={backend.upper()} | output={output_dir}")
    return engine
