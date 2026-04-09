"""
governance.py
-------------
Blockchain governance layer for the Federated Fraud Detection framework.

Responsibilities:
  1. Ingest trust_training_log.json from Split 2
  2. Hash every model update and build the hash chain via ModelHasher
  3. Register each round on the blockchain (sim or real Fabric)
  4. Detect tamper events and raise on-chain alerts
  5. Produce an immutable audit trail (GDPR Article 30, PCI-DSS Req 10)
  6. Enforce governance rules:
       - Reject rounds with broken hash chains
       - Flag clients with anomaly_score > threshold
       - Auto-quarantine clients flagged ≥ N consecutive rounds

Input:  trust_training_log.json  (written by Split 2)
Output: governance_report.json   (hash chain + audit trail + tamper events)
        hash_chain.json           (exportable hash chain)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    # Package import path (tests / module imports).
    from module1.split3.model_hasher import ModelHasher, TamperReport, simulate_tamper
    from module1.split3.fabric_gateway import create_gateway
except ImportError:
    # Script-style execution from split3/ directory.
    from model_hasher import ModelHasher, TamperReport, simulate_tamper
    from fabric_gateway import create_gateway

try:
    from module1.common.round_event_security import (
        ROUND_EVENT_GENESIS,
        verify_signed_round_event,
        verify_round_event_payload_matches_log,
    )
except ImportError:
    from common.round_event_security import (
        ROUND_EVENT_GENESIS,
        verify_signed_round_event,
        verify_round_event_payload_matches_log,
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Governance configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class GovernanceConfig:
    # Tamper detection
    anomaly_threshold:         float = 0.5     # above this → flag client
    consecutive_flag_limit:    int   = 3        # auto-quarantine after N rounds
    min_f1_threshold:          float = 0.30    # rounds below this are suspicious

    # Hash chain
    verify_chain_every_n:      int   = 5        # re-verify full chain every N rounds

    # Blockchain
    use_simulation:            bool  = True
    use_fabric:                bool  = False
    org_msp:                   str   = "Org1MSP"
    expected_backend:          Optional[str] = None  # simulation | ganache | fabric
    require_backend_match:     bool  = False
    fail_on_commit_error:      bool  = True
    require_verified_round_events: bool = True
    policy_path:               Optional[str] = None

    # Audit trail actor label
    audit_actor:               str   = "FederatedCoordinator"

    # Output paths
    output_dir:                str   = "./governance_output"
    report_filename:           str   = "governance_report.json"
    hash_chain_filename:       str   = "hash_chain.json"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Governance report structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class RoundGovernanceRecord:
    round_num:           int
    chain_round:         int
    model_hash:          str
    block_hash:          str
    chain_intact:        bool
    flagged_clients:     List[int]
    trusted_clients:     List[int]
    quarantined_clients: List[int]
    tamper_alerts:       List[str]
    global_f1:           float
    global_auc:          float
    blockchain_tx_id:    str
    audit_tx_id:         str
    timestamp:           float = field(default_factory=time.time)
    round_event_verified: bool = False
    round_event_reason:   str = ""
    attestation_verified: bool = False
    attestation_key_id:   str = ""
    attestation_algo:     str = ""
    attestation_signature: str = ""
    policy_violations:    List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "round":               self.round_num,
            "chain_round":         self.chain_round,
            "timestamp":           self.timestamp,
            "model_hash":          self.model_hash,
            "block_hash":          self.block_hash,
            "chain_intact":        self.chain_intact,
            "flagged_clients":     self.flagged_clients,
            "trusted_clients":     self.trusted_clients,
            "quarantined_clients": self.quarantined_clients,
            "tamper_alerts":       self.tamper_alerts,
            "global_f1":           round(self.global_f1, 6),
            "global_auc":          round(self.global_auc, 6),
            "blockchain_tx_id":    self.blockchain_tx_id,
            "audit_tx_id":         self.audit_tx_id,
            "round_event_verified": self.round_event_verified,
            "round_event_reason":   self.round_event_reason,
            "attestation_verified": self.attestation_verified,
            "attestation_key_id":   self.attestation_key_id,
            "attestation_algo":     self.attestation_algo,
            "attestation_signature": self.attestation_signature,
            "policy_violations": self.policy_violations,
        }


@dataclass
class GovernanceReport:
    total_rounds:        int
    chain_intact:        bool
    tamper_events:       int
    quarantined_clients: List[int]
    backend_used:        str
    backend_source:      str
    best_f1:             float
    best_round:          int
    round_records:       List[RoundGovernanceRecord] = field(default_factory=list)
    tamper_report:       Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "summary": {
                "total_rounds":        self.total_rounds,
                "chain_intact":        self.chain_intact,
                "tamper_events":       self.tamper_events,
                "quarantined_clients": self.quarantined_clients,
                "backend_used":        self.backend_used,
                "backend_source":      self.backend_source,
                "best_f1":             round(self.best_f1, 6),
                "best_round":          self.best_round,
            },
            "tamper_report":  self.tamper_report,
            "round_records":  [r.to_dict() for r in self.round_records],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GovernanceEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GovernanceEngine:
    """
    Core governance orchestrator.

    Takes trust_training_log.json entries one at a time (or all at once)
    and for each round:
      1. Derives a deterministic model hash from the log entry's model_hash
         (in production, the actual model params are passed directly)
      2. Extends the local hash chain
      3. Registers the block on the blockchain
      4. Checks anomaly scores → flags / quarantines clients
      5. Raises tamper alerts when chain breaks
      6. Appends GDPR-compliant audit events
    """

    def __init__(self, config: Optional[GovernanceConfig] = None) -> None:
        self.cfg     = config or GovernanceConfig()
        self.hasher  = ModelHasher()
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self.gateway = create_gateway(
            use_simulation=self.cfg.use_simulation,
            use_fabric=self.cfg.use_fabric,
            org_msp=self.cfg.org_msp,
            allow_fallback=not self.cfg.require_backend_match,
            deployment_file=os.path.join(self.cfg.output_dir, "eth_deployment.json"),
        )

        self._active_backend = self._detect_active_backend()
        expected_backend = self.cfg.expected_backend or (
            "fabric" if self.cfg.use_fabric else
            ("simulation" if self.cfg.use_simulation else "ganache")
        )
        if self.cfg.require_backend_match and self._active_backend != expected_backend:
            raise RuntimeError(
                f"Requested backend '{expected_backend}' but active backend is '{self._active_backend}'. "
                f"Refusing to continue in strict mode."
            )
        self._round_records:         List[RoundGovernanceRecord] = []
        self._quarantine_tracker:    Dict[int, int] = {}   # client_id → consecutive flags
        self._quarantined_clients:   set            = set()
        self._best_f1:               float          = 0.0
        self._best_round:            int            = 0
        self._last_global_auc:       Optional[float] = None
        self._blockchain_round_offset: int          = self._compute_blockchain_round_offset()
        self._seed_hasher_from_blockchain()
        self._policy:                Dict[str, Any] = self._load_policy(self.cfg.policy_path)
        self._attest_algo:           str            = str(os.getenv("BATFL_ATTESTATION_ALGO", "HMAC-SHA256")).strip().upper()
        self._attest_key_id:         str            = str(os.getenv("BATFL_ATTESTATION_KEY_ID", "coordinator-default")).strip()
        self._attest_hmac_keys:      Dict[str, bytes] = self._load_hmac_attestation_keys()
        self._attest_ed25519_private_by_id, self._attest_ed25519_public_by_id = self._load_ed25519_attestation_keys()
        self._round_event_key:       bytes          = os.getenv("BATFL_ROUND_EVENT_KEY", "batfl-dev-round-event-key").encode("utf-8")
        self._last_round_event_hash: str            = ROUND_EVENT_GENESIS
        self._round_event_run_id:    Optional[str]  = None
        self._seen_round_event_sequences: Set[int]  = set()
        self._next_round_event_sequence: int        = 1

        self._validate_attestation_key_state()

        print(f"\n[Governance] Engine initialised")
        print(f"  Mode:      {'SIMULATION' if self.cfg.use_simulation else ('FABRIC' if self.cfg.use_fabric else 'GANACHE')}")
        print(f"  Output:    {self.cfg.output_dir}")
        print(f"  Attest:    {self._attest_algo} (active key_id={self._attest_key_id})")
        if self.cfg.policy_path:
            print(f"  Policy:    {self.cfg.policy_path}")
        if self._blockchain_round_offset > 0:
            print(f"  Chain offset: on-chain rounds already exist (offset={self._blockchain_round_offset})")

    # ── Public API ────────────────────────────────────────────────────────────

    def process_trust_log(self, trust_log_path: str) -> GovernanceReport:
        """
        Process a complete trust_training_log.json file from Split 2.
        Iterates through all rounds and applies governance to each.

        Args:
            trust_log_path: Path to trust_training_log.json

        Returns:
            GovernanceReport with full audit trail and hash chain
        """
        print(f"\n[Governance] Loading trust log: {trust_log_path}")
        with open(trust_log_path, "r") as f:
            trust_log = json.load(f)

        print(f"  Rounds to process: {len(trust_log)}")
        print(f"{'─'*60}")

        for entry in trust_log:
            self.process_round(entry)

        return self._finalise()

    def process_round(self, log_entry: Dict) -> RoundGovernanceRecord:
        """
        Process a single round's trust log entry.
        Can be called incrementally during live training.

        Args:
            log_entry:  One element from trust_training_log.json

        Returns:
            RoundGovernanceRecord
        """
        rnd         = log_entry["round"]
        model_hash  = log_entry.get("model_hash", "")
        global_f1   = log_entry.get("global_f1", 0.0)
        global_auc  = log_entry.get("global_auc", 0.0)
        trust_scores    = log_entry.get("trust_scores", {})
        anomaly_scores  = log_entry.get("anomaly_scores", {})
        flagged         = [int(c) for c in log_entry.get("flagged_clients", [])]
        trusted         = [int(c) for c in log_entry.get("trusted_clients", [])]
        chain_rnd       = rnd + self._blockchain_round_offset

        print(f"\n  [Round {rnd:02d}] F1={global_f1:.4f}  "
              f"trusted={trusted}  flagged={flagged}")
        if chain_rnd != rnd:
            print(f"    [ChainMap] local round {rnd} -> on-chain round {chain_rnd}")

        round_event_ok, round_event_reason = self._verify_round_event(log_entry)
        policy_violations = self._evaluate_round_policy(
            round_num=rnd,
            log_entry=log_entry,
            trusted=trusted,
            flagged=flagged,
            global_f1=float(global_f1),
            global_auc=float(global_auc),
        )

        # 1. Build hash chain entry from Split 2's logged model hash.
        #    This preserves end-to-end model identity from training to governance.
        if not model_hash:
            raise ValueError(
                f"Missing model_hash at round {rnd}. Cannot build governance chain."
            )
        hash_record = self.hasher.append_external_model_hash(
            round_num=rnd,
            model_hash=model_hash,
            param_count=int(log_entry.get("param_count", 0)),
            total_bytes=int(log_entry.get("total_bytes", 0)),
            timestamp=time.time(),
        )

        # 2. Check governance rules → quarantine decisions
        newly_quarantined = self._apply_governance_rules(
            rnd,
            flagged,
            anomaly_scores,
            policy_violations,
        )
        quarantined_now = list(self._quarantined_clients)

        # 3. Build signed attestation payload and commit it on-chain first.
        attestation_payload = self._build_attestation_payload(
            round_num=rnd,
            model_hash=hash_record.model_hash,
            block_hash=hash_record.block_hash,
            prev_block_hash=hash_record.prev_block_hash,
            global_f1=global_f1,
            global_auc=global_auc,
            trusted=trusted,
            flagged=flagged,
            trust_scores=trust_scores,
            anomaly_scores=anomaly_scores,
        )
        provided_sig = str(log_entry.get("attestation_signature", "")).strip()
        attestation_verified = False
        if provided_sig:
            attestation_payload["attestation_signature"] = provided_sig
            sig_ok, sig_reason = self._verify_attestation_signature_with_reason(attestation_payload)
            attestation_payload["signature_verified"] = sig_ok
            attestation_verified = bool(attestation_payload["signature_verified"])
            if not attestation_payload["signature_verified"]:
                raise RuntimeError(
                    f"Invalid attestation signature at round {rnd}: {sig_reason}. Refusing commit."
                )
        else:
            signature = self._sign_attestation_payload(attestation_payload)
            attestation_payload["attestation_signature"] = signature
            attestation_payload["signature_verified"] = True
            attestation_verified = True

        attestation_tx = self.gateway.append_audit_event(
            event_type="ROUND_ATTESTED",
            round_num=chain_rnd,
            data=attestation_payload,
            actor=self.cfg.audit_actor,
        )

        # 4. Register on blockchain model registry
        tx_id, ok = self.gateway.register_model(
            round_num=chain_rnd,
            model_hash=hash_record.model_hash,
            block_hash=hash_record.block_hash,
            prev_block_hash=hash_record.prev_block_hash,
            global_f1=global_f1,
            global_auc=global_auc,
            trusted_clients=trusted,
            flagged_clients=flagged,
            param_count=hash_record.param_count,
            total_bytes=hash_record.total_bytes,
        )
        if self.cfg.fail_on_commit_error and not ok:
            raise RuntimeError(
                f"Blockchain commit failed at local round {rnd} (on-chain round {chain_rnd}). tx={tx_id}. "
                "Stopping to preserve governance integrity."
            )

        # 5. Periodic chain verification
        tamper_alerts: List[str] = []
        chain_intact = True
        if rnd % self.cfg.verify_chain_every_n == 0 or rnd == 1:
            report = self.hasher.verify_chain()
            chain_intact = report.is_intact
            if not chain_intact:
                for tampered_rnd in report.tampered_rounds:
                    alert_msg = (f"HASH_CHAIN_BREAK at round {tampered_rnd}: "
                                 f"{report.details}")
                    tamper_alerts.append(alert_msg)
                    self.gateway.raise_tamper_alert(
                        round_num=tampered_rnd + self._blockchain_round_offset,
                        alert_type="HASH_CHAIN_BREAK",
                        detail=str(report.details),
                        severity="CRITICAL",
                    )

        # 6. Raise alerts for newly quarantined clients
        for cid in newly_quarantined:
            alert_msg = (f"CLIENT_QUARANTINED: client {cid} "
                         f"flagged {self.cfg.consecutive_flag_limit} consecutive rounds")
            tamper_alerts.append(alert_msg)
            self.gateway.raise_tamper_alert(
                round_num=chain_rnd,
                alert_type="CLIENT_QUARANTINED",
                detail=alert_msg,
                severity="HIGH",
            )

        # 7. Audit trail event
        audit_tx = self.gateway.append_audit_event(
            event_type="ROUND_COMMITTED",
            round_num=chain_rnd,
            data={
                "model_hash":          hash_record.model_hash,
                "block_hash":          hash_record.block_hash,
                "local_round":         rnd,
                "chain_round":         chain_rnd,
                "attestation_tx_id":    attestation_tx,
                "attestation_key_id":   attestation_payload["attestation_key_id"],
                "attestation_algo":     attestation_payload["attestation_algo"],
                "attestation_signature": attestation_payload["attestation_signature"],
                "global_f1":           global_f1,
                "global_auc":          global_auc,
                "trusted_clients":     trusted,
                "flagged_clients":     flagged,
                "quarantined_clients": quarantined_now,
                "policy_violations":   policy_violations,
                "chain_intact":        chain_intact,
            },
            actor=self.cfg.audit_actor,
        )

        # 7. Track best F1
        if global_f1 > self._best_f1:
            self._best_f1   = global_f1
            self._best_round = rnd

        record = RoundGovernanceRecord(
            round_num=rnd,
            chain_round=chain_rnd,
            model_hash=hash_record.model_hash,
            block_hash=hash_record.block_hash,
            chain_intact=chain_intact,
            flagged_clients=flagged,
            trusted_clients=trusted,
            quarantined_clients=quarantined_now,
            tamper_alerts=tamper_alerts,
            global_f1=global_f1,
            global_auc=global_auc,
            blockchain_tx_id=tx_id,
            audit_tx_id=audit_tx,
            round_event_verified=bool(round_event_ok),
            round_event_reason=str(round_event_reason),
            attestation_verified=attestation_verified,
            attestation_key_id=str(attestation_payload["attestation_key_id"]),
            attestation_algo=str(attestation_payload["attestation_algo"]),
            attestation_signature=str(attestation_payload["attestation_signature"]),
            policy_violations=policy_violations,
        )
        self._round_records.append(record)
        self._last_global_auc = float(global_auc)
        # Persist incremental governance artifacts so live dashboards can read
        # commit/audit telemetry without waiting for end-of-run finalization.
        self.export_reports()
        return record

    def _compute_blockchain_round_offset(self) -> int:
        """Return max on-chain round to avoid round-number collisions on reruns."""
        try:
            if not hasattr(self.gateway, "get_all_model_records"):
                return 0
            records = self.gateway.get_all_model_records() or []
            rounds: List[int] = []
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                raw = rec.get("round")
                if raw is None:
                    continue
                try:
                    rounds.append(int(raw))
                except Exception:
                    continue
            return max(rounds) if rounds else 0
        except Exception:
            return 0

    def _seed_hasher_from_blockchain(self) -> None:
        """Seed local hasher chain head from latest on-chain block hash for reruns."""
        if self._blockchain_round_offset <= 0:
            return
        try:
            latest = self._get_latest_model_record()
            if not latest:
                return
            prev_hash = (
                latest.get("blockHash")
                or latest.get("block_hash")
                or latest.get("blockhash")
            )
            if isinstance(prev_hash, str) and len(prev_hash) == 64:
                # Seed private chain head so next append_external_model_hash links correctly.
                self.hasher._last_block_hash = prev_hash  # type: ignore[attr-defined]
        except Exception:
            return

    def _get_latest_model_record(self) -> Optional[Dict[str, Any]]:
        if not hasattr(self.gateway, "get_all_model_records"):
            return None
        records = self.gateway.get_all_model_records() or []
        best: Optional[Tuple[int, Dict[str, Any]]] = None
        for rec in records:
            if not isinstance(rec, dict):
                continue
            raw = rec.get("round")
            if raw is None:
                continue
            try:
                r = int(raw)
            except Exception:
                continue
            if best is None or r > best[0]:
                best = (r, rec)
        return best[1] if best is not None else None

    def _verify_round_event(self, log_entry: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify signed RoundEvent chain and payload provenance for this round."""
        if not self.cfg.require_verified_round_events:
            return True, "unsigned events allowed"

        event = log_entry.get("round_event")
        if event is None:
            raise RuntimeError("Missing round_event in trust log entry. Refusing unsigned governance input.")

        payload = event.get("payload", {}) if isinstance(event, dict) else {}
        run_id = str(payload.get("run_id", "")).strip()
        if not run_id:
            raise RuntimeError("Round event missing run_id.")

        if self._round_event_run_id is None:
            self._round_event_run_id = run_id
        elif run_id != self._round_event_run_id:
            raise RuntimeError(
                f"Round event run_id mismatch: expected '{self._round_event_run_id}', got '{run_id}'."
            )

        try:
            event_sequence = int(payload.get("event_sequence"))
        except (TypeError, ValueError):
            raise RuntimeError("Round event missing/invalid event_sequence.")

        if event_sequence in self._seen_round_event_sequences:
            raise RuntimeError(f"Replay detected: duplicated round_event sequence {event_sequence}.")

        ok, reason = verify_signed_round_event(
            event=event,
            expected_prev_hash=self._last_round_event_hash,
            key_bytes=self._round_event_key,
            expected_run_id=self._round_event_run_id,
            expected_event_sequence=self._next_round_event_sequence,
        )
        if not ok:
            raise RuntimeError(f"Round event verification failed: {reason}")

        ok, reason = verify_round_event_payload_matches_log(event, log_entry)
        if not ok:
            raise RuntimeError(f"Round event payload mismatch: {reason}")

        self._last_round_event_hash = str(event.get("event_hash", ROUND_EVENT_GENESIS))
        self._seen_round_event_sequences.add(event_sequence)
        self._next_round_event_sequence += 1
        return True, "ok"

    def verify_attestation_signature(self, payload: Dict[str, Any]) -> bool:
        """Verify an attestation signature from canonical payload fields."""
        ok, _ = self._verify_attestation_signature_with_reason(payload)
        return ok

    def verify_round_from_blockchain(self, round_num: int) -> Dict[str, Any]:
        """
        Read attestation + model commit from blockchain and independently verify
        attestation signature and model-hash consistency.
        """
        model_record = None
        if hasattr(self.gateway, "get_model_record"):
            model_record = self.gateway.get_model_record(round_num)

        audit_events = self.gateway.get_audit_trail()
        attested_events = []
        for e in audit_events:
            try:
                event_round = int(e.get("round", -1))
            except Exception:
                continue
            event_type = e.get("event_type") or e.get("eventType")
            if event_round == int(round_num) and str(event_type) == "ROUND_ATTESTED":
                attested_events.append(e)
        if not attested_events:
            return {
                "round": round_num,
                "verified": False,
                "reason": "ROUND_ATTESTED event not found on chain",
            }

        latest = attested_events[-1]
        raw_payload = latest.get("data", {})
        if isinstance(raw_payload, str):
            try:
                payload = json.loads(raw_payload)
            except json.JSONDecodeError:
                payload = {}
        elif isinstance(raw_payload, dict):
            payload = dict(raw_payload)
        else:
            payload = {}
        sig_ok, sig_reason = self._verify_attestation_signature_with_reason(payload)

        if not model_record:
            return {
                "round": round_num,
                "verified": False,
                "reason": "MODEL record not found on chain",
                "signature_valid": bool(sig_ok),
                "signature_reason": sig_reason,
                "model_hash_match": False,
                "attestation_key_id": payload.get("attestation_key_id"),
                "attestation_algo": payload.get("attestation_algo"),
                "attestation_event_id": latest.get("event_id") or latest.get("id"),
                "blockchain_model_hash": None,
                "attested_model_hash": payload.get("model_hash"),
            }

        chain_model_hash = (
            model_record.get("model_hash")
            or model_record.get("modelHash")
            or model_record.get("stored_hash")
        )
        payload_model_hash = payload.get("model_hash")
        hash_match = bool(chain_model_hash and payload_model_hash and chain_model_hash == payload_model_hash)

        return {
            "round": round_num,
            "verified": bool(sig_ok and hash_match),
            "signature_valid": bool(sig_ok),
            "signature_reason": sig_reason,
            "model_hash_match": bool(hash_match),
            "attestation_key_id": payload.get("attestation_key_id"),
            "attestation_algo": payload.get("attestation_algo"),
            "attestation_event_id": latest.get("event_id") or latest.get("id"),
            "blockchain_model_hash": chain_model_hash,
            "attested_model_hash": payload_model_hash,
        }

    def get_committed_round_numbers(self) -> List[int]:
        """Return round numbers discoverable from blockchain state only."""
        rounds: List[int] = []
        if hasattr(self.gateway, "get_all_model_records"):
            records = self.gateway.get_all_model_records()
            for rec in records:
                rnd = rec.get("round")
                if rnd is not None:
                    rounds.append(int(rnd))
        elif hasattr(self.gateway, "get_block_count"):
            count = int(self.gateway.get_block_count())
            # Simulation includes genesis block; real backends return round count.
            if self._active_backend == "simulation":
                count = max(0, count - 1)
            rounds = list(range(1, count + 1))
        return sorted(set(rounds))

    def audit_blockchain_attestations(self) -> Dict[str, Any]:
        """Perform read-only attestation verification for all committed rounds."""
        rounds = self.get_committed_round_numbers()
        results = [self.verify_round_from_blockchain(rnd) for rnd in rounds]
        verified = [r for r in results if r.get("verified")]
        failed = [r for r in results if not r.get("verified")]
        return {
            "round_count": len(rounds),
            "verified_count": len(verified),
            "failed_count": len(failed),
            "all_verified": len(failed) == 0,
            "results": results,
        }

    def run_tamper_simulation(
        self,
        round_to_tamper: int = 5,
    ) -> Dict:
        """
        Inject a simulated tamper into the hash chain and verify detection.
        Returns a dict describing the tamper event and detection result.
        """
        print(f"\n[Governance] 🔬 Tamper simulation — injecting at round {round_to_tamper}")

        if not self.hasher.get_chain():
            return {"error": "No hash chain built yet. Run process_trust_log() first."}

        # Inject tamper
        tampered_hasher = simulate_tamper(self.hasher, round_to_tamper)

        # Verify — should detect
        report = tampered_hasher.verify_chain()
        detected = not report.is_intact

        result = {
            "tampered_round":      round_to_tamper,
            "detected":            detected,
            "tampered_rounds":     report.tampered_rounds,
            "broken_links":        report.broken_links,
            "detection_message":   report.summary(),
        }

        print(f"  Tamper {'DETECTED ✅' if detected else 'MISSED ❌'}")
        print(f"  {report.summary()}")

        # Log tamper simulation to audit trail
        self.gateway.append_audit_event(
            event_type="TAMPER_SIMULATION",
            round_num=round_to_tamper,
            data=result,
            actor="SecurityAudit",
        )

        return result

    def export_reports(self) -> Tuple[str, str]:
        """
        Write governance_report.json and hash_chain.json to output_dir.

        Returns:
            (governance_report_path, hash_chain_path)
        """
        report = self._build_report()

        report_path = os.path.join(self.cfg.output_dir, self.cfg.report_filename)
        chain_path  = os.path.join(self.cfg.output_dir, self.cfg.hash_chain_filename)

        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        with open(chain_path, "w") as f:
            json.dump(self.hasher.export_chain(), f, indent=2)

        print(f"\n[Governance] Reports written:")
        print(f"  → {report_path}")
        print(f"  → {chain_path}")
        return report_path, chain_path

    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        report = self._build_report()
        print(f"\n{'='*65}")
        print(f"  GOVERNANCE SUMMARY")
        print(f"{'='*65}")
        print(f"  Rounds processed:    {report.total_rounds}")
        print(f"  Backend:             {report.backend_used} ({report.backend_source})")
        print(f"  Hash chain intact:   {'✅ YES' if report.chain_intact else '🚨 BROKEN'}")
        print(f"  Tamper events:       {report.tamper_events}")
        print(f"  Quarantined clients: {report.quarantined_clients}")
        print(f"  Best Global F1:      {report.best_f1:.4f} (Round {report.best_round})")
        print(f"\n  Blockchain blocks:   {self.gateway.get_block_count()}")
        alerts = self.gateway.get_tamper_alerts()
        print(f"  On-chain alerts:     {len(alerts)}")
        audit  = self.gateway.get_audit_trail()
        print(f"  Audit events:        {len(audit)}")
        print(f"{'='*65}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_governance_rules(
        self,
        rnd:           int,
        flagged:       List[int],
        anomaly_scores: Dict[str, float],
        policy_violations: List[str],
    ) -> List[int]:
        """
        Update quarantine tracker and return newly quarantined clients.
        Rule: flag a client on-chain after consecutive_flag_limit rounds flagged.
        """
        # Increment flag counter for flagged clients
        for cid in flagged:
            self._quarantine_tracker[cid] = self._quarantine_tracker.get(cid, 0) + 1

        # Reset counter for clients NOT flagged this round
        all_clients = set(self._quarantine_tracker.keys())
        for cid in all_clients - set(flagged):
            self._quarantine_tracker[cid] = max(
                0, self._quarantine_tracker[cid] - 1
            )

        # Newly quarantined this round
        newly_quarantined = []
        for cid, count in self._quarantine_tracker.items():
            if (count >= self.cfg.consecutive_flag_limit and
                    cid not in self._quarantined_clients):
                self._quarantined_clients.add(cid)
                newly_quarantined.append(cid)
                print(f"    ⚠️  Client {cid} QUARANTINED after "
                      f"{count} consecutive flagged rounds")

        quarantine_policy = self._policy.get("quarantine", {})
        policy_action = str(quarantine_policy.get("on_policy_violation", "quarantine_flagged")).strip().lower()
        if policy_violations and policy_action == "quarantine_flagged":
            for cid in flagged:
                if cid in self._quarantined_clients:
                    continue
                self._quarantined_clients.add(cid)
                newly_quarantined.append(cid)
                print(
                    f"    ⚠️  Client {cid} QUARANTINED by policy violation "
                    f"(round {rnd}, violations={len(policy_violations)})"
                )

        return newly_quarantined

    def _evaluate_round_policy(
        self,
        round_num: int,
        log_entry: Dict[str, Any],
        trusted: List[int],
        flagged: List[int],
        global_f1: float,
        global_auc: float,
    ) -> List[str]:
        """Evaluate configurable governance policy for the current round."""
        violations: List[str] = []
        policy = self._policy.get("round_requirements", {})

        min_f1 = float(policy.get("min_global_f1", self.cfg.min_f1_threshold))
        if global_f1 < min_f1:
            violations.append(f"min_global_f1 violation: {global_f1:.6f} < {min_f1:.6f}")

        min_auc = float(policy.get("min_global_auc", 0.0))
        if global_auc < min_auc:
            violations.append(f"min_global_auc violation: {global_auc:.6f} < {min_auc:.6f}")

        min_required_clients = int(policy.get("min_required_clients", 0))
        participated = len(set(int(c) for c in (trusted or [])) | set(int(c) for c in (flagged or [])))
        if participated < min_required_clients:
            violations.append(
                f"required_client_participation violation: {participated} < {min_required_clients}"
            )

        max_false_positive_rate = float(policy.get("max_false_positive_rate", 1.0))
        fp = log_entry.get("global_fp")
        tn = log_entry.get("global_tn")
        if fp is not None and tn is not None:
            denom = float(fp) + float(tn)
            fpr = (float(fp) / denom) if denom > 0 else 0.0
            if fpr > max_false_positive_rate:
                violations.append(
                    f"max_false_positive_rate violation: {fpr:.6f} > {max_false_positive_rate:.6f}"
                )

        max_auc_drift = float(policy.get("max_auc_drift", 1.0))
        if self._last_global_auc is not None:
            drift = abs(global_auc - float(self._last_global_auc))
            if drift > max_auc_drift:
                violations.append(f"max_auc_drift violation: {drift:.6f} > {max_auc_drift:.6f}")

        if violations:
            print(f"    [Policy] Round {round_num} violations: {violations}")

        return violations

    def _finalise(self) -> GovernanceReport:
        """Called after processing all rounds to produce the final report."""
        self.print_summary()
        self.export_reports()
        return self._build_report()

    def _build_report(self) -> GovernanceReport:
        tamper_report_raw = self.hasher.verify_chain()
        return GovernanceReport(
            total_rounds=len(self._round_records),
            chain_intact=tamper_report_raw.is_intact,
            tamper_events=len(self.gateway.get_tamper_alerts()),
            quarantined_clients=sorted(self._quarantined_clients),
            backend_used=self._active_backend,
            backend_source=self.gateway.__class__.__name__,
            best_f1=self._best_f1,
            best_round=self._best_round,
            round_records=self._round_records,
            tamper_report=tamper_report_raw.__dict__,
        )

    def _detect_active_backend(self) -> str:
        """Infer active gateway backend from concrete gateway class name."""
        name = self.gateway.__class__.__name__
        if name == "HLFGateway":
            return "fabric"
        if name == "EthBlockchainGateway":
            return "ganache"
        if name == "SimBlockchainGateway":
            return "simulation"
        return "unknown"

    @staticmethod
    def _parse_json_env_dict(env_name: str) -> Dict[str, Any]:
        raw = os.getenv(env_name, "").strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in {env_name}: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"{env_name} must be a JSON object")
        return {str(k): v for k, v in parsed.items()}

    def _load_policy(self, policy_path: Optional[str]) -> Dict[str, Any]:
        policy = {
            "version": "batfl.governance-policy.v1",
            "quarantine": {
                "consecutive_flag_limit": int(self.cfg.consecutive_flag_limit),
                "on_policy_violation": "quarantine_flagged",
            },
            "round_requirements": {
                "min_global_f1": float(self.cfg.min_f1_threshold),
                "min_global_auc": 0.0,
                "max_false_positive_rate": 1.0,
                "max_auc_drift": 1.0,
                "min_required_clients": 0,
            },
        }
        if not policy_path:
            return policy

        with open(policy_path, "r", encoding="utf-8") as f:
            text = f.read()

        if policy_path.lower().endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "YAML policy requested but PyYAML is not installed. "
                    "Install dependency 'pyyaml'."
                ) from exc
            loaded = yaml.safe_load(text)
        else:
            loaded = json.loads(text)

        if not isinstance(loaded, dict):
            raise RuntimeError("Governance policy must be an object")

        quarantine = loaded.get("quarantine", {}) or {}
        round_requirements = loaded.get("round_requirements", {}) or {}
        if isinstance(quarantine, dict):
            policy["quarantine"].update(quarantine)
        if isinstance(round_requirements, dict):
            policy["round_requirements"].update(round_requirements)

        self.cfg.consecutive_flag_limit = int(
            policy["quarantine"].get("consecutive_flag_limit", self.cfg.consecutive_flag_limit)
        )
        self.cfg.min_f1_threshold = float(
            policy["round_requirements"].get("min_global_f1", self.cfg.min_f1_threshold)
        )
        return policy

    def _load_hmac_attestation_keys(self) -> Dict[str, bytes]:
        keyring_raw = self._parse_json_env_dict("BATFL_ATTESTATION_KEYS_JSON")
        keyring: Dict[str, bytes] = {}
        for key_id, key_val in keyring_raw.items():
            keyring[str(key_id)] = str(key_val).encode("utf-8")

        legacy_key = os.getenv("BATFL_ATTESTATION_KEY", "batfl-dev-attestation-key")
        if self._attest_key_id not in keyring:
            keyring[self._attest_key_id] = legacy_key.encode("utf-8")
        return keyring

    def _load_ed25519_attestation_keys(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        private_by_id: Dict[str, Any] = {}
        public_by_id: Dict[str, Any] = {}

        if self._attest_algo != "ED25519":
            return private_by_id, public_by_id

        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import (
                Ed25519PrivateKey,
                Ed25519PublicKey,
            )
        except ImportError as exc:
            raise RuntimeError(
                "ED25519 attestation requested but 'cryptography' is not installed."
            ) from exc

        pub_keyring_raw = self._parse_json_env_dict("BATFL_ATTESTATION_ED25519_PUBLIC_KEYS_JSON")
        for key_id, pub_b64 in pub_keyring_raw.items():
            pub_bytes = base64.b64decode(str(pub_b64).encode("utf-8"))
            public_by_id[key_id] = Ed25519PublicKey.from_public_bytes(pub_bytes)

        priv_keyring_raw = self._parse_json_env_dict("BATFL_ATTESTATION_ED25519_PRIVATE_KEYS_JSON")
        for key_id, priv_b64 in priv_keyring_raw.items():
            priv_bytes = base64.b64decode(str(priv_b64).encode("utf-8"))
            private_key = Ed25519PrivateKey.from_private_bytes(priv_bytes)
            private_by_id[key_id] = private_key
            if key_id not in public_by_id:
                public_by_id[key_id] = private_key.public_key()

        active_priv_b64 = os.getenv("BATFL_ATTESTATION_ED25519_PRIVATE_KEY_B64", "").strip()
        if active_priv_b64 and self._attest_key_id not in private_by_id:
            priv_bytes = base64.b64decode(active_priv_b64.encode("utf-8"))
            private_key = Ed25519PrivateKey.from_private_bytes(priv_bytes)
            private_by_id[self._attest_key_id] = private_key
            if self._attest_key_id not in public_by_id:
                public_by_id[self._attest_key_id] = private_key.public_key()

        return private_by_id, public_by_id

    def _validate_attestation_key_state(self) -> None:
        if self._attest_algo == "HMAC-SHA256":
            if self._attest_key_id not in self._attest_hmac_keys:
                raise RuntimeError(
                    f"Active attestation key_id '{self._attest_key_id}' not found in key registry"
                )
            return

        if self._attest_algo == "ED25519":
            if self._attest_key_id not in self._attest_ed25519_public_by_id:
                raise RuntimeError(
                    f"Active Ed25519 attestation key_id '{self._attest_key_id}' not found in public key registry"
                )
            if self._attest_key_id not in self._attest_ed25519_private_by_id:
                raise RuntimeError(
                    f"Active Ed25519 private key for key_id '{self._attest_key_id}' is missing"
                )
            return

        raise RuntimeError(f"Unsupported attestation algorithm: {self._attest_algo}")

    def _verify_attestation_signature_with_reason(self, payload: Dict[str, Any]) -> Tuple[bool, str]:
        given = str(payload.get("attestation_signature", "")).strip()
        if not given:
            return False, "attestation_signature missing"

        payload_algo = str(payload.get("attestation_algo", "")).strip().upper()
        if payload_algo != self._attest_algo:
            return False, f"attestation_algo mismatch: got {payload_algo}, expected {self._attest_algo}"

        key_id = str(payload.get("attestation_key_id", "")).strip()
        if not key_id:
            return False, "attestation_key_id missing"

        canonical = self._attestation_digest(payload)
        if payload_algo == "HMAC-SHA256":
            key_bytes = self._attest_hmac_keys.get(key_id)
            if key_bytes is None:
                return False, f"unknown attestation key_id '{key_id}'"
            expected = hmac.new(key_bytes, canonical.encode(), hashlib.sha256).hexdigest()
            return hmac.compare_digest(given, expected), "ok"

        if payload_algo == "ED25519":
            pub_key = self._attest_ed25519_public_by_id.get(key_id)
            if pub_key is None:
                return False, f"unknown attestation key_id '{key_id}'"
            try:
                sig_bytes = base64.b64decode(given.encode("utf-8"))
            except Exception:
                return False, "invalid base64 ed25519 signature"
            try:
                pub_key.verify(sig_bytes, canonical.encode("utf-8"))
                return True, "ok"
            except Exception:
                return False, "ed25519 signature mismatch"

        return False, f"unsupported attestation algorithm '{payload_algo}'"

    def _build_attestation_payload(
        self,
        round_num: int,
        model_hash: str,
        block_hash: str,
        prev_block_hash: str,
        global_f1: float,
        global_auc: float,
        trusted: List[int],
        flagged: List[int],
        trust_scores: Dict[str, Any],
        anomaly_scores: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = {
            "schema": "batfl.round-attestation.v1",
            "round": int(round_num),
            "model_hash": str(model_hash),
            "block_hash": str(block_hash),
            "prev_block_hash": str(prev_block_hash),
            "global_f1": round(float(global_f1), 6),
            "global_auc": round(float(global_auc), 6),
            "trusted_clients": sorted(int(c) for c in (trusted or [])),
            "flagged_clients": sorted(int(c) for c in (flagged or [])),
            "trust_scores": self._normalize_score_map(trust_scores),
            "anomaly_scores": self._normalize_score_map(anomaly_scores),
            "attestation_algo": self._attest_algo,
            "attestation_key_id": self._attest_key_id,
        }
        return payload

    def _sign_attestation_payload(self, payload: Dict[str, Any]) -> str:
        canonical = self._attestation_digest(payload)
        if self._attest_algo == "HMAC-SHA256":
            key_bytes = self._attest_hmac_keys.get(self._attest_key_id)
            if key_bytes is None:
                raise RuntimeError(f"Unknown active attestation key_id '{self._attest_key_id}'")
            return hmac.new(key_bytes, canonical.encode(), hashlib.sha256).hexdigest()

        if self._attest_algo == "ED25519":
            private_key = self._attest_ed25519_private_by_id.get(self._attest_key_id)
            if private_key is None:
                raise RuntimeError(f"Unknown active Ed25519 key_id '{self._attest_key_id}'")
            signature = private_key.sign(canonical.encode("utf-8"))
            return base64.b64encode(signature).decode("utf-8")

        raise RuntimeError(f"Unsupported attestation algorithm: {self._attest_algo}")

    @staticmethod
    def _normalize_score_map(raw: Dict[str, Any]) -> Dict[str, float]:
        normalized: Dict[str, float] = {}
        for k, v in (raw or {}).items():
            normalized[str(k)] = round(float(v), 8)
        return dict(sorted(normalized.items(), key=lambda kv: kv[0]))

    @staticmethod
    def _attestation_digest(payload: Dict[str, Any]) -> str:
        signable = {
            k: v
            for k, v in payload.items()
            if k not in {"attestation_signature", "signature_verified"}
        }
        return json.dumps(signable, sort_keys=True, separators=(",", ":"))
