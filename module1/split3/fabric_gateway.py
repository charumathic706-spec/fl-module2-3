"""
split3/fabric_gateway.py
------------------------
Gateway factory for BATFL Module 2 blockchain layer.

Routes to one of three backends:
    simulation  — SimBlockchainGateway  (blockchain_sim.py)  default
    ganache     — EthBlockchainGateway  (eth_gateway.py)     real Ethereum
    fabric      — HLFGateway            (hlf_gateway.py)     Hyperledger Fabric

Called by governance.py as:
    create_gateway(use_simulation=True/False, use_fabric=True/False)
"""
from __future__ import annotations
import logging, os, sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def create_gateway(
    use_simulation: bool = True,
    use_fabric: bool = False,
    org_msp: str = "Org1MSP",
    allow_fallback: bool = True,
    **kwargs,
) -> Any:
    """
    Factory — returns the correct gateway instance.

    Priority:
        use_fabric=True               → HLFGateway (Hyperledger Fabric)
        use_simulation=False          → EthBlockchainGateway (Ganache)
        use_simulation=True (default) → SimBlockchainGateway (in-memory)
    """
    # Ensure split3/ is on sys.path so sibling modules resolve
    _ensure_on_path()

    if use_fabric:
        return _get_fabric(org_msp, allow_fallback=allow_fallback, **kwargs)
    elif not use_simulation:
        return _get_ganache(org_msp, allow_fallback=allow_fallback)
    else:
        return _get_sim(org_msp)


# ── Backend loaders ───────────────────────────────────────────────────────────

def _get_sim(org_msp: str) -> Any:
    from blockchain_sim import SimBlockchainGateway
    logger.info("[Gateway] Mode: SIMULATION")
    return SimBlockchainGateway(org_msp=org_msp)


def _get_ganache(org_msp: str, allow_fallback: bool = True) -> Any:
    try:
        from eth_gateway import EthBlockchainGateway
        logger.info("[Gateway] Mode: GANACHE/ETHEREUM")
        return EthBlockchainGateway(org_msp=org_msp)
    except ImportError as e:
        if not allow_fallback:
            raise
        logger.warning(f"[Gateway] Ganache SDK missing ({e}) — using simulation.\n"
                       "  Fix: pip install web3 py-solc-x  +  npm install -g ganache")
        return _get_sim(org_msp)
    except Exception as e:
        if not allow_fallback:
            raise
        logger.warning(f"[Gateway] Ganache error ({e}) — using simulation.\n"
                       "  Is Ganache running?  ganache --deterministic --port 8545")
        return _get_sim(org_msp)


def _get_fabric(org_msp: str, allow_fallback: bool = True, **kwargs) -> Any:
    try:
        from hlf_gateway import HLFGateway, HLF_AVAILABLE
        if not HLF_AVAILABLE:
            if not allow_fallback:
                raise ImportError("hf-fabric-gateway not installed")
            logger.warning("[Gateway] hf-fabric-gateway not installed — using simulation.\n"
                           "  Fix: pip install hf-fabric-gateway grpcio protobuf")
            return _get_sim(org_msp)
        logger.info("[Gateway] Mode: HYPERLEDGER FABRIC")
        return HLFGateway(msp_id=org_msp, **kwargs)
    except FileNotFoundError as e:
        if not allow_fallback:
            raise
        logger.warning(f"[Gateway] Fabric network not ready: {e}\n"
                       "  Fix: cd module1/split3/hlf && ./setup.sh")
        return _get_sim(org_msp)
    except ImportError as e:
        if not allow_fallback:
            raise
        logger.warning(f"[Gateway] Fabric SDK missing: {e}\n"
                       "  Fix: pip install hf-fabric-gateway grpcio protobuf")
        return _get_sim(org_msp)
    except Exception as e:
        if not allow_fallback:
            raise
        logger.warning(f"[Gateway] Fabric error: {e} — using simulation.")
        return _get_sim(org_msp)


def _ensure_on_path() -> None:
    """Add split3/ directory to sys.path."""
    p = str(Path(__file__).resolve().parent)
    if p not in sys.path:
        sys.path.insert(0, p)
