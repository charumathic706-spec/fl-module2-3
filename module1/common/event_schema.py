from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

EVENT_SCHEMA_NAME = "batfl.round-log"
EVENT_SCHEMA_VERSION = 1


class EventValidationError(ValueError):
    """Raised when a round event payload fails strict validation."""


@dataclass(frozen=True)
class RoundEventRecord:
    schema_name: str
    schema_version: int
    round: int
    timestamp: str
    model_hash: str
    trusted_clients: List[int]
    flagged_clients: List[int]
    quarantined_clients: List[int]
    trust_weights: Dict[str, float]
    anomaly_scores: Dict[str, float]
    trust_scores: Dict[str, float]
    global_f1: float
    global_auc: float
    global_pr_auc: float
    blockchain_tx_id: str
    audit_tx_id: str

    @staticmethod
    def _validate_int_list(name: str, value: Any) -> List[int]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise EventValidationError(f"{name} must be a list")
        out: List[int] = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError) as exc:
                raise EventValidationError(f"{name} contains non-integer value: {item}") from exc
        return sorted(set(out))

    @staticmethod
    def _validate_float_map(name: str, value: Any) -> Dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise EventValidationError(f"{name} must be a dict")
        out: Dict[str, float] = {}
        for k, v in value.items():
            try:
                out[str(k)] = float(v)
            except (TypeError, ValueError) as exc:
                raise EventValidationError(f"{name} has non-float value for key {k}") from exc
        return out

    @classmethod
    def migrate_to_current(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate older payloads to current schema version."""
        schema_version = int(payload.get("schema_version", 0) or 0)

        # v0 (legacy): no schema fields and no quarantined_clients.
        if schema_version <= 0:
            upgraded = dict(payload)
            upgraded["schema_name"] = EVENT_SCHEMA_NAME
            upgraded["schema_version"] = EVENT_SCHEMA_VERSION
            upgraded.setdefault("quarantined_clients", payload.get("quarantined_clients", []))
            upgraded.setdefault("global_pr_auc", payload.get("global_pr_auc", 0.0))
            upgraded.setdefault("blockchain_tx_id", payload.get("blockchain_tx_id", ""))
            upgraded.setdefault("audit_tx_id", payload.get("audit_tx_id", ""))
            return upgraded

        if schema_version == EVENT_SCHEMA_VERSION:
            return payload

        raise EventValidationError(
            f"Unsupported schema version {schema_version}; expected <= {EVENT_SCHEMA_VERSION}"
        )

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "RoundEventRecord":
        if not isinstance(payload, dict):
            raise EventValidationError("Round event payload must be a dict")

        data = cls.migrate_to_current(payload)
        schema_name = str(data.get("schema_name", "")).strip()
        schema_version = int(data.get("schema_version", 0) or 0)
        if schema_name != EVENT_SCHEMA_NAME:
            raise EventValidationError(
                f"schema_name mismatch: got '{schema_name}', expected '{EVENT_SCHEMA_NAME}'"
            )
        if schema_version != EVENT_SCHEMA_VERSION:
            raise EventValidationError(
                f"schema_version mismatch: got {schema_version}, expected {EVENT_SCHEMA_VERSION}"
            )

        round_num = int(data.get("round", 0) or 0)
        if round_num <= 0:
            raise EventValidationError("round must be >= 1")

        timestamp = str(data.get("timestamp", "")).strip()
        if not timestamp:
            raise EventValidationError("timestamp is required")

        model_hash = str(data.get("model_hash", "")).strip()
        if len(model_hash) != 64:
            raise EventValidationError("model_hash must be 64 hex chars")

        trusted = cls._validate_int_list("trusted_clients", data.get("trusted_clients", []))
        flagged = cls._validate_int_list("flagged_clients", data.get("flagged_clients", []))
        quarantined = cls._validate_int_list("quarantined_clients", data.get("quarantined_clients", []))

        trust_weights = cls._validate_float_map("trust_weights", data.get("trust_weights", {}))
        anomaly_scores = cls._validate_float_map("anomaly_scores", data.get("anomaly_scores", {}))
        trust_scores = cls._validate_float_map("trust_scores", data.get("trust_scores", {}))

        return cls(
            schema_name=schema_name,
            schema_version=schema_version,
            round=round_num,
            timestamp=timestamp,
            model_hash=model_hash,
            trusted_clients=trusted,
            flagged_clients=flagged,
            quarantined_clients=quarantined,
            trust_weights=trust_weights,
            anomaly_scores=anomaly_scores,
            trust_scores=trust_scores,
            global_f1=float(data.get("global_f1", 0.0)),
            global_auc=float(data.get("global_auc", 0.0)),
            global_pr_auc=float(data.get("global_pr_auc", 0.0)),
            blockchain_tx_id=str(data.get("blockchain_tx_id", "")),
            audit_tx_id=str(data.get("audit_tx_id", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_name": self.schema_name,
            "schema_version": self.schema_version,
            "round": self.round,
            "timestamp": self.timestamp,
            "model_hash": self.model_hash,
            "trusted_clients": list(self.trusted_clients),
            "flagged_clients": list(self.flagged_clients),
            "quarantined_clients": list(self.quarantined_clients),
            "trust_weights": dict(self.trust_weights),
            "anomaly_scores": dict(self.anomaly_scores),
            "trust_scores": dict(self.trust_scores),
            "global_f1": float(self.global_f1),
            "global_auc": float(self.global_auc),
            "global_pr_auc": float(self.global_pr_auc),
            "blockchain_tx_id": self.blockchain_tx_id,
            "audit_tx_id": self.audit_tx_id,
        }
