import json

import module1.verify_run as verify_module


def test_verify_run_passes_on_valid_artifacts(tmp_path, monkeypatch):
    run_dir = tmp_path / "logs_split2"
    gov_dir = run_dir / "governance_output"
    gov_dir.mkdir(parents=True)

    round_event = {
        "schema": "batfl.round-event.v1",
        "signing_algo": "HMAC-SHA256",
        "key_id": "coordinator-round-event",
        "prev_event_hash": "0" * 64,
        "event_hash": "f" * 64,
        "signature": "f" * 64,
        "payload": {
            "schema": "batfl.round-event.v1",
            "run_id": "run-test",
            "event_sequence": 1,
            "round": 1,
            "timestamp": "2026-04-09T00:00:00+00:00",
            "model_hash": "a" * 64,
            "global_f1": 0.8,
            "global_auc": 0.9,
            "trusted_clients": [0],
            "flagged_clients": [],
            "trust_scores": {},
            "anomaly_scores": {},
        },
    }

    trust_log = [{
        "schema_name": "batfl.round-log",
        "schema_version": 1,
        "round": 1,
        "timestamp": "2026-04-09T00:00:00+00:00",
        "model_hash": "a" * 64,
        "trusted_clients": [0],
        "flagged_clients": [],
        "quarantined_clients": [],
        "trust_weights": {},
        "anomaly_scores": {},
        "trust_scores": {},
        "global_f1": 0.8,
        "global_auc": 0.9,
        "global_pr_auc": 0.7,
        "blockchain_tx_id": "",
        "audit_tx_id": "",
        "round_event": round_event,
    }]

    (run_dir / "trust_training_log.json").write_text(json.dumps(trust_log), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"run_config": {"blockchain": "disabled"}}),
        encoding="utf-8",
    )
    (gov_dir / "governance_report.json").write_text(
        json.dumps({"summary": {"chain_intact": True, "backend_used": "simulation"}}),
        encoding="utf-8",
    )
    (gov_dir / "hash_chain.json").write_text(json.dumps([]), encoding="utf-8")

    # For this unit test we only check command plumbing, not crypto-level signature generation.
    monkeypatch.setattr(verify_module, "verify_round_event_chain", lambda *args, **kwargs: (True, "ok"))
    ok, errors = verify_module.verify_run(str(run_dir))
    assert ok, errors
