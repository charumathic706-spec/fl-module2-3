import json

from module1.common.event_schema import EVENT_SCHEMA_NAME, EVENT_SCHEMA_VERSION, RoundEventRecord
from module1.common.event_storage import JsonlEventStorage, SqliteEventStorage


def _sample_round_record():
    return {
        "schema_name": EVENT_SCHEMA_NAME,
        "schema_version": EVENT_SCHEMA_VERSION,
        "round": 1,
        "timestamp": "2026-04-09T00:00:00+00:00",
        "model_hash": "a" * 64,
        "trusted_clients": [0, 2],
        "flagged_clients": [1],
        "quarantined_clients": [],
        "trust_weights": {"0": 0.6, "1": 0.0, "2": 0.4},
        "anomaly_scores": {"0": 0.1, "1": 0.9, "2": 0.2},
        "trust_scores": {"0": 0.95, "1": 0.2, "2": 0.9},
        "global_f1": 0.8,
        "global_auc": 0.9,
        "global_pr_auc": 0.75,
        "blockchain_tx_id": "",
        "audit_tx_id": "",
    }


def test_schema_migration_from_legacy_payload():
    legacy = {
        "round": 1,
        "timestamp": "2026-04-09T00:00:00+00:00",
        "model_hash": "a" * 64,
        "trusted_clients": [0],
        "flagged_clients": [],
        "trust_weights": {},
        "anomaly_scores": {},
        "trust_scores": {},
        "global_f1": 0.0,
        "global_auc": 0.0,
    }
    rec = RoundEventRecord.from_payload(legacy)
    assert rec.schema_name == EVENT_SCHEMA_NAME
    assert rec.schema_version == EVENT_SCHEMA_VERSION


def test_jsonl_and_sqlite_storage_roundtrip(tmp_path):
    record = _sample_round_record()

    jsonl = JsonlEventStorage(str(tmp_path / "events.jsonl"))
    jsonl.append_round_event(record)
    out_jsonl = jsonl.read_all_events()
    assert len(out_jsonl) == 1
    assert out_jsonl[0]["round"] == 1

    sqlite = SqliteEventStorage(str(tmp_path / "events.db"))
    sqlite.append_round_event(record)
    out_sqlite = sqlite.read_all_events()
    assert len(out_sqlite) == 1
    assert out_sqlite[0]["model_hash"] == "a" * 64
