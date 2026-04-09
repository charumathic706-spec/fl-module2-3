from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any, Dict, List, Optional, Tuple


ROUND_EVENT_SCHEMA = "batfl.round-event.v1"
ROUND_EVENT_ALGO = "HMAC-SHA256"
ROUND_EVENT_GENESIS = "0" * 64


def _normalize_score_map(raw: Dict[str, Any]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for k, v in (raw or {}).items():
        normalized[str(k)] = round(float(v), 8)
    return dict(sorted(normalized.items(), key=lambda kv: kv[0]))


def build_round_event_payload(
    round_log: Dict[str, Any],
    run_id: str,
    event_sequence: int,
) -> Dict[str, Any]:
    return {
        "schema": ROUND_EVENT_SCHEMA,
        "run_id": str(run_id),
        "event_sequence": int(event_sequence),
        "round": int(round_log["round"]),
        "timestamp": str(round_log.get("timestamp", "")),
        "model_hash": str(round_log.get("model_hash", "")),
        "global_f1": round(float(round_log.get("global_f1", 0.0)), 6),
        "global_auc": round(float(round_log.get("global_auc", 0.0)), 6),
        "trusted_clients": sorted(int(c) for c in (round_log.get("trusted_clients", []) or [])),
        "flagged_clients": sorted(int(c) for c in (round_log.get("flagged_clients", []) or [])),
        "trust_scores": _normalize_score_map(round_log.get("trust_scores", {})),
        "anomaly_scores": _normalize_score_map(round_log.get("anomaly_scores", {})),
    }


def _event_material(payload: Dict[str, Any], prev_event_hash: str) -> str:
    material = {
        "prev_event_hash": prev_event_hash,
        "payload": payload,
    }
    return json.dumps(material, sort_keys=True, separators=(",", ":"))


def compute_round_event_hash(payload: Dict[str, Any], prev_event_hash: str) -> str:
    return hashlib.sha256(_event_material(payload, prev_event_hash).encode("utf-8")).hexdigest()


def sign_round_event_hash(event_hash: str, key_bytes: bytes) -> str:
    return hmac.new(key_bytes, event_hash.encode("utf-8"), hashlib.sha256).hexdigest()


def create_signed_round_event(
    round_log: Dict[str, Any],
    prev_event_hash: str,
    key_bytes: bytes,
    key_id: str,
    run_id: str,
    event_sequence: int,
) -> Dict[str, Any]:
    payload = build_round_event_payload(
        round_log=round_log,
        run_id=run_id,
        event_sequence=event_sequence,
    )
    event_hash = compute_round_event_hash(payload, prev_event_hash)
    signature = sign_round_event_hash(event_hash, key_bytes)
    return {
        "schema": ROUND_EVENT_SCHEMA,
        "signing_algo": ROUND_EVENT_ALGO,
        "key_id": key_id,
        "prev_event_hash": prev_event_hash,
        "event_hash": event_hash,
        "signature": signature,
        "payload": payload,
    }


def verify_signed_round_event(
    event: Dict[str, Any],
    expected_prev_hash: str,
    key_bytes: bytes,
    expected_run_id: Optional[str] = None,
    expected_event_sequence: Optional[int] = None,
) -> Tuple[bool, str]:
    if not isinstance(event, dict):
        return False, "round_event missing or invalid type"
    if event.get("schema") != ROUND_EVENT_SCHEMA:
        return False, "round_event schema mismatch"
    if event.get("signing_algo") != ROUND_EVENT_ALGO:
        return False, "round_event signing algorithm mismatch"

    prev_event_hash = str(event.get("prev_event_hash", ""))
    if prev_event_hash != expected_prev_hash:
        return False, "round_event chain link mismatch"

    payload = event.get("payload", {})
    payload_run_id = str(payload.get("run_id", "")).strip()
    if not payload_run_id:
        return False, "round_event run_id missing"

    payload_event_sequence = payload.get("event_sequence")
    try:
        payload_event_sequence = int(payload_event_sequence)
    except (TypeError, ValueError):
        return False, "round_event event_sequence missing or invalid"

    if expected_run_id is not None and payload_run_id != expected_run_id:
        return False, "round_event run_id mismatch"

    if expected_event_sequence is not None and payload_event_sequence != expected_event_sequence:
        return False, "round_event event_sequence mismatch"

    expected_hash = compute_round_event_hash(payload, prev_event_hash)
    given_hash = str(event.get("event_hash", ""))
    if not hmac.compare_digest(given_hash, expected_hash):
        return False, "round_event hash mismatch"

    expected_sig = sign_round_event_hash(given_hash, key_bytes)
    given_sig = str(event.get("signature", ""))
    if not hmac.compare_digest(given_sig, expected_sig):
        return False, "round_event signature mismatch"

    return True, "ok"


def verify_round_event_payload_matches_log(event: Dict[str, Any], round_log: Dict[str, Any]) -> Tuple[bool, str]:
    payload = event.get("payload", {})
    run_id = str(payload.get("run_id", "")).strip()
    if not run_id:
        return False, "round_event run_id missing"

    try:
        event_sequence = int(payload.get("event_sequence"))
    except (TypeError, ValueError):
        return False, "round_event event_sequence missing or invalid"

    expected_payload = build_round_event_payload(
        round_log,
        run_id=run_id,
        event_sequence=event_sequence,
    )
    canonical_event_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    canonical_expected_payload = json.dumps(expected_payload, sort_keys=True, separators=(",", ":"))
    if canonical_event_payload != canonical_expected_payload:
        return False, "round_event payload does not match trust log entry"
    return True, "ok"


def verify_round_event_chain(
    events: List[Dict[str, Any]],
    key_bytes: bytes,
    genesis_hash: str = ROUND_EVENT_GENESIS,
) -> Tuple[bool, str]:
    prev_hash = genesis_hash
    expected_run_id: Optional[str] = None
    expected_event_sequence = 1
    for idx, event in enumerate(events, start=1):
        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        if expected_run_id is None:
            expected_run_id = str(payload.get("run_id", "")).strip() or None
        ok, reason = verify_signed_round_event(
            event,
            prev_hash,
            key_bytes,
            expected_run_id=expected_run_id,
            expected_event_sequence=expected_event_sequence,
        )
        if not ok:
            return False, f"event #{idx} verification failed: {reason}"
        prev_hash = str(event.get("event_hash", ""))
        expected_event_sequence += 1
    return True, "ok"
