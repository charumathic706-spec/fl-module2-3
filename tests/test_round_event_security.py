import numpy as np

from module1.common.round_event_security import (
    ROUND_EVENT_GENESIS,
    create_signed_round_event,
    verify_signed_round_event,
    verify_round_event_payload_matches_log,
)


def _sample_round_log(round_num: int = 1):
    return {
        "round": round_num,
        "timestamp": "2026-04-09T10:00:00+00:00",
        "model_hash": "a" * 64,
        "trusted_clients": [0, 2, 3],
        "flagged_clients": [1],
        "trust_scores": {"0": 0.98, "1": 0.2, "2": 0.95, "3": 0.92},
        "anomaly_scores": {"0": 0.05, "1": 0.91, "2": 0.08, "3": 0.09},
        "global_f1": 0.88,
        "global_auc": 0.95,
    }


def test_signed_round_event_verifies():
    key = b"test-round-event-key"
    round_log = _sample_round_log(1)
    event = create_signed_round_event(
        round_log,
        ROUND_EVENT_GENESIS,
        key,
        "test-key",
        run_id="run-test-1",
        event_sequence=1,
    )

    ok, reason = verify_signed_round_event(
        event,
        ROUND_EVENT_GENESIS,
        key,
        expected_run_id="run-test-1",
        expected_event_sequence=1,
    )
    assert ok, reason

    ok, reason = verify_round_event_payload_matches_log(event, round_log)
    assert ok, reason


def test_tampered_round_log_payload_is_rejected():
    key = b"test-round-event-key"
    round_log = _sample_round_log(1)
    event = create_signed_round_event(
        round_log,
        ROUND_EVENT_GENESIS,
        key,
        "test-key",
        run_id="run-test-1",
        event_sequence=1,
    )

    tampered_log = dict(round_log)
    tampered_log["flagged_clients"] = []

    ok, _ = verify_round_event_payload_matches_log(event, tampered_log)
    assert not ok
