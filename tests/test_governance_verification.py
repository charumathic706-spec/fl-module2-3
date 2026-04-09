import pytest

from module1.common.round_event_security import ROUND_EVENT_GENESIS, create_signed_round_event
from module1.split3.governance import GovernanceConfig, GovernanceEngine


class FakeGatewayBase:
    def __init__(self):
        self._alerts = []
        self._audit = []

    def append_audit_event(self, event_type, round_num, data, actor="FederatedCoordinator"):
        self._audit.append({"event_type": event_type, "round": round_num, "data": data, "actor": actor})
        return "audit_tx_1"

    def register_model(self, **kwargs):
        return "tx_1", True

    def raise_tamper_alert(self, round_num, alert_type, detail, severity="HIGH"):
        self._alerts.append({"round": round_num, "type": alert_type, "detail": detail, "severity": severity})
        return "alert_tx_1", {"round": round_num, "alert_type": alert_type}

    def get_tamper_alerts(self):
        return list(self._alerts)

    def get_audit_trail(self):
        return list(self._audit)

    def get_block_count(self):
        return 0

    def get_all_model_records(self):
        return []


class SimGateway(FakeGatewayBase):
    pass


SimGateway.__name__ = "SimBlockchainGateway"


@pytest.fixture
def signed_round_log(monkeypatch):
    monkeypatch.setenv("BATFL_ROUND_EVENT_KEY", "test-round-event-key")
    log = {
        "round": 1,
        "timestamp": "2026-04-09T10:00:00+00:00",
        "model_hash": "a" * 64,
        "trusted_clients": [0, 2, 3],
        "flagged_clients": [1],
        "trust_scores": {"0": 0.98, "1": 0.2, "2": 0.95, "3": 0.92},
        "anomaly_scores": {"0": 0.05, "1": 0.91, "2": 0.08, "3": 0.09},
        "global_f1": 0.88,
        "global_auc": 0.95,
    }
    log["round_event"] = create_signed_round_event(
        round_log=log,
        prev_event_hash=ROUND_EVENT_GENESIS,
        key_bytes=b"test-round-event-key",
        key_id="test-key",
        run_id="run-test-1",
        event_sequence=1,
    )
    return log


def test_backend_mismatch_fails(monkeypatch, tmp_path):
    import module1.split3.governance as gov

    monkeypatch.setattr(gov, "create_gateway", lambda **kwargs: SimGateway())

    cfg = GovernanceConfig(
        use_simulation=False,
        use_fabric=True,
        expected_backend="fabric",
        require_backend_match=True,
        output_dir=str(tmp_path),
    )

    with pytest.raises(RuntimeError, match="Requested backend 'fabric' but active backend is 'simulation'"):
        GovernanceEngine(cfg)


def test_tampered_log_entry_rejected(monkeypatch, tmp_path, signed_round_log):
    import module1.split3.governance as gov

    monkeypatch.setattr(gov, "create_gateway", lambda **kwargs: SimGateway())

    cfg = GovernanceConfig(
        use_simulation=True,
        use_fabric=False,
        require_backend_match=False,
        require_verified_round_events=True,
        output_dir=str(tmp_path),
    )
    engine = GovernanceEngine(cfg)

    tampered = dict(signed_round_log)
    tampered["flagged_clients"] = []

    with pytest.raises(RuntimeError, match="Round event payload mismatch"):
        engine.process_round(tampered)


def test_unknown_attestation_key_is_rejected(monkeypatch, tmp_path):
    import module1.split3.governance as gov

    monkeypatch.setattr(gov, "create_gateway", lambda **kwargs: SimGateway())
    monkeypatch.setenv("BATFL_ATTESTATION_KEYS_JSON", '{"known-key":"known-secret"}')
    monkeypatch.setenv("BATFL_ATTESTATION_KEY_ID", "known-key")

    cfg = GovernanceConfig(
        use_simulation=True,
        use_fabric=False,
        require_backend_match=False,
        require_verified_round_events=True,
        output_dir=str(tmp_path),
    )
    engine = GovernanceEngine(cfg)

    payload = {
        "schema": "batfl.round-attestation.v1",
        "round": 1,
        "model_hash": "a" * 64,
        "block_hash": "b" * 64,
        "prev_block_hash": "0" * 64,
        "global_f1": 0.9,
        "global_auc": 0.95,
        "trusted_clients": [0, 2],
        "flagged_clients": [1],
        "trust_scores": {"0": 0.98, "1": 0.2},
        "anomaly_scores": {"0": 0.05, "1": 0.91},
        "attestation_algo": "HMAC-SHA256",
        "attestation_key_id": "unknown-key",
        "attestation_signature": "deadbeef",
    }
    ok, reason = engine._verify_attestation_signature_with_reason(payload)
    assert not ok
    assert "unknown attestation key_id" in reason


def test_replay_duplicate_sequence_rejected(monkeypatch, tmp_path):
    import module1.split3.governance as gov

    monkeypatch.setattr(gov, "create_gateway", lambda **kwargs: SimGateway())
    monkeypatch.setenv("BATFL_ROUND_EVENT_KEY", "test-round-event-key")

    cfg = GovernanceConfig(
        use_simulation=True,
        use_fabric=False,
        require_backend_match=False,
        require_verified_round_events=True,
        output_dir=str(tmp_path),
    )
    engine = GovernanceEngine(cfg)

    base = {
        "timestamp": "2026-04-09T10:00:00+00:00",
        "model_hash": "a" * 64,
        "trusted_clients": [0, 2, 3],
        "flagged_clients": [1],
        "trust_scores": {"0": 0.98, "1": 0.2, "2": 0.95, "3": 0.92},
        "anomaly_scores": {"0": 0.05, "1": 0.91, "2": 0.08, "3": 0.09},
        "global_f1": 0.88,
        "global_auc": 0.95,
    }

    round1 = dict(base)
    round1["round"] = 1
    round1["round_event"] = create_signed_round_event(
        round_log=round1,
        prev_event_hash=ROUND_EVENT_GENESIS,
        key_bytes=b"test-round-event-key",
        key_id="test-key",
        run_id="run-test-1",
        event_sequence=1,
    )
    engine.process_round(round1)

    round2 = dict(base)
    round2["round"] = 2
    round2["round_event"] = create_signed_round_event(
        round_log=round2,
        prev_event_hash=round1["round_event"]["event_hash"],
        key_bytes=b"test-round-event-key",
        key_id="test-key",
        run_id="run-test-1",
        event_sequence=1,
    )

    with pytest.raises(RuntimeError, match="duplicated round_event sequence"):
        engine.process_round(round2)


def test_policy_violation_can_quarantine_flagged(monkeypatch, tmp_path, signed_round_log):
    import module1.split3.governance as gov

    monkeypatch.setattr(gov, "create_gateway", lambda **kwargs: SimGateway())

    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        """
{
  "quarantine": {
    "consecutive_flag_limit": 5,
    "on_policy_violation": "quarantine_flagged"
  },
  "round_requirements": {
    "min_global_f1": 0.99,
    "min_required_clients": 3
  }
}
""".strip(),
        encoding="utf-8",
    )

    cfg = GovernanceConfig(
        use_simulation=True,
        use_fabric=False,
        require_backend_match=False,
        require_verified_round_events=True,
        policy_path=str(policy_path),
        output_dir=str(tmp_path),
    )
    engine = GovernanceEngine(cfg)
    rec = engine.process_round(dict(signed_round_log))

    assert rec.policy_violations
    assert 1 in rec.quarantined_clients
