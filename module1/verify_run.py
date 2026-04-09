from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

from module1.common.event_schema import RoundEventRecord
from module1.common.round_event_security import ROUND_EVENT_GENESIS, verify_round_event_chain


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def _norm_backend(value: str) -> str:
    v = str(value or "").strip().lower()
    aliases = {
        "": "",
        "none": "disabled",
        "false": "disabled",
    }
    return aliases.get(v, v)


def _is_blockchain_enabled(backend: str) -> bool:
    return _norm_backend(backend) not in {"", "disabled"}


def verify_run(run_dir: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    trust_log_path = os.path.join(run_dir, "trust_training_log.json")
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    gov_dir = os.path.join(run_dir, "governance_output")
    gov_report_path = os.path.join(gov_dir, "governance_report.json")
    hash_chain_path = os.path.join(gov_dir, "hash_chain.json")

    if not os.path.exists(trust_log_path):
        errors.append(f"Missing trust log: {trust_log_path}")
        return False, errors

    trust_log = _read_json(trust_log_path)
    if not isinstance(trust_log, list) or not trust_log:
        errors.append("trust_training_log.json is empty or invalid")
        return False, errors
    _ok(f"Trust log found with {len(trust_log)} rounds")

    manifest = _read_json(manifest_path) if os.path.exists(manifest_path) else {}
    run_cfg = manifest.get("run_config", {}) if isinstance(manifest, dict) else {}
    expected_backend = _norm_backend(str(run_cfg.get("blockchain", "")))
    expected_rounds = int(run_cfg.get("rounds", 0) or 0)
    expected_event_storage = str(run_cfg.get("event_storage", "")).strip().lower()

    if expected_rounds > 0:
        actual_rounds = len(trust_log)
        if actual_rounds != expected_rounds:
            errors.append(
                f"Round count mismatch: manifest expects {expected_rounds}, trust log has {actual_rounds}"
            )
        else:
            _ok(f"Round count matches manifest: {actual_rounds}")

    observed_rounds: List[int] = []

    # Strict schema validation with migration support
    for idx, rec in enumerate(trust_log, start=1):
        try:
            RoundEventRecord.from_payload(rec)
            observed_rounds.append(int(rec.get("round", 0) or 0))
        except Exception as exc:
            errors.append(f"Round {idx} schema validation failed: {exc}")
            break
    if not errors:
        _ok("Round event schema validation passed")

    if observed_rounds:
        expected_sequence = list(range(1, len(observed_rounds) + 1))
        if observed_rounds != expected_sequence:
            errors.append(
                f"Round numbering is not contiguous from 1..N: observed {observed_rounds}"
            )
        else:
            _ok("Round numbering is contiguous (1..N)")

    if expected_event_storage == "sqlite":
        db_path = os.path.join(run_dir, "round_events.db")
        if not os.path.exists(db_path):
            errors.append(f"Expected sqlite storage artifact missing: {db_path}")
        else:
            _ok("SQLite event storage artifact present")
    elif expected_event_storage == "jsonl":
        jsonl_path = os.path.join(run_dir, "round_events.jsonl")
        if not os.path.exists(jsonl_path):
            errors.append(f"Expected jsonl storage artifact missing: {jsonl_path}")
        else:
            _ok("JSONL event storage artifact present")

    # Signature chain validation
    key = os.getenv("BATFL_ROUND_EVENT_KEY", "batfl-dev-round-event-key").encode("utf-8")
    events: List[Dict[str, Any]] = []
    event_rounds: List[int] = []
    for rec in trust_log:
        ev = rec.get("round_event")
        if isinstance(ev, dict):
            events.append(ev)
            payload = ev.get("payload", {}) if isinstance(ev, dict) else {}
            event_rounds.append(int(payload.get("round", rec.get("round", 0)) or 0))
    if len(events) != len(trust_log):
        errors.append(
            f"Round event coverage mismatch: {len(events)} signed events for {len(trust_log)} rounds"
        )
    elif not events:
        errors.append("No round_event entries found in trust log")
    else:
        ok, reason = verify_round_event_chain(events, key_bytes=key, genesis_hash=ROUND_EVENT_GENESIS)
        if not ok:
            errors.append(f"Round event signature chain invalid: {reason}")
        else:
            _ok("Round event signatures and chain link are valid")

    if event_rounds and observed_rounds and event_rounds != observed_rounds:
        errors.append(
            f"Round event payload rounds mismatch trust log rounds: events={event_rounds}, logs={observed_rounds}"
        )
    elif event_rounds:
        _ok("Round event payload rounds align with trust log rounds")

    # Governance artifact existence
    gov_exists = os.path.exists(gov_report_path) and os.path.exists(hash_chain_path)
    if gov_exists:
        _ok("Governance outputs exist (governance_report.json + hash_chain.json)")
    elif _is_blockchain_enabled(expected_backend):
        errors.append("Governance outputs missing under run_dir/governance_output")
    else:
        _ok("Governance output check skipped (manifest backend disabled/empty)")

    if gov_exists:
        gov_report = _read_json(gov_report_path)
        summary = gov_report.get("summary", {}) if isinstance(gov_report, dict) else {}
        chain_intact = bool(summary.get("chain_intact", False))
        if chain_intact:
            _ok("Governance hash chain intact")
        else:
            errors.append("Governance chain_intact is false")

        actual_backend = _norm_backend(str(summary.get("backend_used", "")))
        if _is_blockchain_enabled(expected_backend):
            if actual_backend != expected_backend:
                errors.append(
                    f"Backend mismatch: manifest expects '{expected_backend}', governance reports '{actual_backend}'"
                )
            else:
                _ok(f"Backend matches manifest: {actual_backend}")
        else:
            _ok("Backend match check skipped (manifest backend disabled/empty)")

        gov_records = []
        if isinstance(gov_report, dict):
            if isinstance(gov_report.get("records"), list):
                gov_records = gov_report.get("records", [])
            elif isinstance(gov_report.get("round_records"), list):
                gov_records = gov_report.get("round_records", [])
        if isinstance(gov_records, list) and gov_records:
            if _is_blockchain_enabled(expected_backend) and len(gov_records) < len(trust_log):
                errors.append(
                    f"Governance records incomplete: {len(gov_records)} records for {len(trust_log)} rounds"
                )
            else:
                _ok(f"Governance records available: {len(gov_records)}")

            trust_by_round: Dict[int, Dict[str, Any]] = {
                int(r.get("round", 0) or 0): r for r in trust_log
            }
            for rec in gov_records:
                round_num = int(rec.get("round", 0) or 0)
                trust_rec = trust_by_round.get(round_num)
                if trust_rec is None:
                    continue
                trust_tx = str(trust_rec.get("blockchain_tx_id", ""))
                gov_tx = str(rec.get("blockchain_tx_id", ""))
                trust_audit = str(trust_rec.get("audit_tx_id", ""))
                gov_audit = str(rec.get("audit_tx_id", ""))
                if trust_tx and gov_tx and trust_tx != gov_tx:
                    errors.append(
                        f"Round {round_num} blockchain_tx_id mismatch between trust log and governance report"
                    )
                if trust_audit and gov_audit and trust_audit != gov_audit:
                    errors.append(
                        f"Round {round_num} audit_tx_id mismatch between trust log and governance report"
                    )
        elif _is_blockchain_enabled(expected_backend):
            errors.append("Governance report has no records[] entries")

    return (len(errors) == 0), errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify BATFL run artifacts and integrity")
    parser.add_argument("--run_dir", type=str, required=True, help="Run directory (e.g. logs_split2)")
    args = parser.parse_args()

    ok, errors = verify_run(args.run_dir)
    if ok:
        print("\nVerification summary: PASS")
        return

    print("\nVerification summary: FAIL")
    for err in errors:
        _fail(err)
    sys.exit(1)


if __name__ == "__main__":
    main()
