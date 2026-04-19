"""
trust_weighted_strategy.py
--------------------------
Split 2's core deliverable: Trust-Weighted Adaptive Aggregation Strategy.

Replaces the standard FedAvg from Split 1 with a custom Flower Strategy
that integrates the trust scoring engine (trust_scoring.py) directly into
the aggregation step.

Per-round pipeline:
  1. Receive model updates from all clients (via Flower FitRes)
  2. Extract gradient deltas and flattened parameters from each client
  3. Run TrustScorer.score_round() -> per-client anomaly scores + trust weights
  4. Aggregate: global_params = ? w_i ? ?_i  (trust-weighted, not uniform)
  5. Flag malicious clients (w_i ? 0) and log to blockchain-ready metadata
  6. Log round results -> JSON for Split 3 blockchain audit trail
"""

from __future__ import annotations

import os
import json
import uuid
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone

import flwr as fl
from flwr.common import (
    Parameters, Scalar, Metrics,
    FitRes, EvaluateRes,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

try:
    from module1.common.trust_scoring import TrustScorer, AggregationResult
    from module1.common.fedavg_strategy import weighted_average  # reuse metric aggregation from Split 1
    from module1.common.model_hashing import hash_model_parameters_canonical
    from module1.common.round_event_security import (
        ROUND_EVENT_GENESIS,
        create_signed_round_event,
    )
    from module1.common.event_schema import EVENT_SCHEMA_NAME, EVENT_SCHEMA_VERSION, RoundEventRecord
    from module1.common.event_storage import create_event_storage
except ImportError:
    from common.trust_scoring import TrustScorer, AggregationResult
    from common.fedavg_strategy import weighted_average  # reuse metric aggregation from Split 1
    from common.model_hashing import hash_model_parameters_canonical
    from common.round_event_security import (
        ROUND_EVENT_GENESIS,
        create_signed_round_event,
    )
    from common.event_schema import EVENT_SCHEMA_NAME, EVENT_SCHEMA_VERSION, RoundEventRecord
    from common.event_storage import create_event_storage


# ??????????????????????????????????????????????????????????????????????????
# Trust-Weighted Aggregation Strategy
# ??????????????????????????????????????????????????????????????????????????

class TrustWeightedFedAvg(Strategy):
    """
    Custom Flower Strategy implementing trust-weighted adaptive aggregation.

    Key difference from standard FedAvg:
      FedAvg:       w_i = n_i / ? n_j          (proportional to data size)
      TrustWeighted: w_i = ?_i / ? ?_j         (proportional to trust score)
      Malicious:     w_i = ? ? 0               (near-zero weight)

    The global model update formula:
      ?_global(t+1) = ?_i  w_i(t) ? ?_i(t)
    where w_i(t) is derived from the dynamic trust score ?_i(t).
    """

    def __init__(
        self,
        num_clients:        int,
        fraction_fit:       float = 1.0,
        fraction_evaluate:  float = 1.0,
        # Trust scorer hyperparameters
        lambda_cosine:      float = 0.3,   # full-vector cos diluted by hidden layers
        lambda_distance:    float = 0.5,   # euc distance is the most reliable signal
        lambda_norm:        float = 0.2,
        gamma:              float = 0.85,  # faster trust decay (was 0.9) -- C1 neutralised quicker
        anomaly_threshold:  float = 0.40,  # label-flip scores ~0.70, well above threshold
        malicious_weight:   float = 1e-6,
        min_trust_floor:    float = 0.85,  # raised 0.75->0.85: data shows C0/C4 still eroding at R20 with 0.75
        # Logging
        log_dir:            str   = "logs_split2",
        # Attack simulator (injected from main.py for simulation)
        attack_simulator    = None,
        governance_engine   = None,
        event_storage_backend: str = "jsonl",
    ):
        self.num_clients       = num_clients
        self.fraction_fit      = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.log_dir           = log_dir
        self.attacker          = attack_simulator
        self.governance_engine = governance_engine

        # Trust scorer instance (maintains trust state across rounds)
        self.trust_scorer = TrustScorer(
            num_clients=num_clients,
            lambda_cosine=lambda_cosine,
            lambda_distance=lambda_distance,
            lambda_norm=lambda_norm,
            gamma=gamma,
            anomaly_threshold=anomaly_threshold,
            malicious_weight=malicious_weight,
            min_trust_floor=min_trust_floor,
        )

        # State tracking
        self.current_global_params: Optional[List[np.ndarray]] = None
        self.round_logs:  List[Dict] = []
        self._pending_round_logs: Dict[int, Dict] = {}
        self.round_event_log_path = os.path.join(log_dir, "round_events.jsonl")
        self.event_storage_backend = event_storage_backend
        self.event_storage = create_event_storage(event_storage_backend, log_dir)
        self._last_round_event_hash: str = ROUND_EVENT_GENESIS
        self._round_event_run_id: str = os.getenv("BATFL_RUN_ID", f"run-{uuid.uuid4().hex[:12]}")
        self._round_event_sequence: int = 0
        self._round_event_key_id: str = os.getenv("BATFL_ROUND_EVENT_KEY_ID", "coordinator-round-event")
        self._round_event_key: bytes = os.getenv("BATFL_ROUND_EVENT_KEY", "batfl-dev-round-event-key").encode("utf-8")
        self.best_f1:     float = 0.0
        self.best_round:  int   = 0

        os.makedirs(log_dir, exist_ok=True)
        if os.path.exists(self.round_event_log_path):
            os.remove(self.round_event_log_path)
        print(
            f"\n[Server] TrustWeightedFedAvg initialised | "
            f"{num_clients} clients | "
            f"anomaly_threshold={anomaly_threshold} | "
            f"?={gamma} | logs -> {log_dir}/\n"
        )

    # -- Flower Strategy interface ---------------------------------------------

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """No initial parameters -- clients initialise their own models."""
        return None

    def configure_fit(self, server_round, parameters, client_manager):
        """Select clients and send global model for local training."""
        if self.attacker:
            self.attacker.set_round(server_round)

        sample_size = max(1, int(self.num_clients * self.fraction_fit))
        clients     = client_manager.sample(num_clients=sample_size, min_num_clients=sample_size)
        config      = {"server_round": server_round}
        return [(client, fl.common.FitIns(parameters, config)) for client in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Select clients for federated evaluation."""
        sample_size = max(1, int(self.num_clients * self.fraction_evaluate))
        clients     = client_manager.sample(num_clients=sample_size, min_num_clients=sample_size)
        config      = {"server_round": server_round}
        return [(client, fl.common.EvaluateIns(parameters, config)) for client in clients]

    def evaluate(self, server_round, parameters):
        """Server-side evaluation -- not used (clients evaluate locally)."""
        return None

    # -- Core: Trust-Weighted Aggregation -------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results:      List[Tuple[ClientProxy, FitRes]],
        failures:     List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Main aggregation step -- replaces FedAvg weighted average with
        trust-weighted aggregation using gradient validation.
        """
        if not results:
            return None, {}

        print(f"\n{'='*65}")
        print(f"  ROUND {server_round} -- TRUST-WEIGHTED AGGREGATION")
        print(f"  Clients: {len(results)} responded | {len(failures)} failures")
        print(f"{'='*65}")

        # -- Extract parameters and metadata from each client ------------------
        client_params:     Dict[int, List[np.ndarray]] = {}
        client_flat_params: Dict[int, np.ndarray]       = {}
        client_gradients:  Dict[int, np.ndarray]        = {}
        client_n_samples:  Dict[int, int]               = {}

        global_flat = (
            np.concatenate([p.flatten() for p in self.current_global_params])
            if self.current_global_params else None
        )

        for proxy, fit_res in results:
            m   = fit_res.metrics
            cid = int(m.get("client_id", 0))
            params = parameters_to_ndarrays(fit_res.parameters)

            # -- Inject attack if applicable -----------------------------------
            if self.attacker and self.attacker.is_malicious(cid):
                if self.current_global_params:
                    params = self.attacker.poison_params(
                        cid, params, self.current_global_params
                    )

            client_params[cid]     = params
            client_n_samples[cid]  = fit_res.num_examples

            flat = np.concatenate([p.flatten() for p in params])
            client_flat_params[cid] = flat

            # Gradient delta = client params - global params
            if global_flat is not None and len(global_flat) == len(flat):
                client_gradients[cid] = flat - global_flat
            else:
                client_gradients[cid] = flat   # first round: use params as gradient proxy

            print(
                f"  Client {cid:02d} | samples={fit_res.num_examples} | "
                f"F1={m.get('train_f1', 0):.4f} | AUC={m.get('train_auc', 0):.4f}"
            )

        # -- Run trust scoring -------------------------------------------------
        global_flat_for_scoring = (
            global_flat if global_flat is not None
            else np.zeros_like(list(client_flat_params.values())[0])
        )

        trust_result = self.trust_scorer.score_round(
            round_num=server_round,
            client_gradients=client_gradients,
            client_params=client_flat_params,
            global_params=global_flat_for_scoring,
        )
        self.trust_scorer.print_round_report(trust_result)

        # -- Trust-weighted parameter aggregation ------------------------------
        aggregated_params = self._weighted_aggregate(
            client_params, trust_result.trust_weights
        )
        self.current_global_params = aggregated_params

        # -- Compute model hash for blockchain audit trail (Split 3) -----------
        model_hash = self._hash_params(aggregated_params)
        print(f"\n  [ModelHash] {model_hash[:16]}...  (-> Split 3 blockchain log)")

        # -- Save pending round metadata (finalized after evaluate) ------------
        self._pending_round_logs[server_round] = self._build_round_log(
            server_round, trust_result, model_hash,
            {cid: m for _, fit_res in results
             for cid, m in [(int(fit_res.metrics.get("client_id", 0)), fit_res.metrics)]}
        )

        aggregated_metrics = {
            "flagged_count":   float(len(trust_result.flagged_clients)),
            "trusted_count":   float(len(trust_result.trusted_clients)),
            "avg_trust":       float(np.mean([
                                   self.trust_scorer.trust_records[cid].trust_score
                                   for cid in client_params
                               ])),
            "min_anomaly":     float(min(trust_result.anomaly_scores.values())),
            "max_anomaly":     float(max(trust_result.anomaly_scores.values())),
        }

        return ndarrays_to_parameters(aggregated_params), aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results:      List[Tuple[ClientProxy, EvaluateRes]],
        failures:     List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients."""
        if not results:
            return None, {}

        print(f"\n  ROUND {server_round} -- EVALUATE")
        for _, eval_res in results:
            m   = eval_res.metrics
            cid = int(m.get("client_id", -1))
            trust = self.trust_scorer.trust_records.get(cid)
            flag  = "[ATTACK]" if (trust and trust.is_malicious) else "[OK]"
            print(
                f"  {flag} Client {cid:02d} | "
                f"F1={m.get('f1', 0):.4f} | AUC={m.get('auc_roc', 0):.4f} | "
                f"Recall={m.get('recall', 0):.4f}"
            )

        losses       = [eval_res.loss      for _, eval_res in results]
        n_samples    = [eval_res.num_examples for _, eval_res in results]
        metrics_list = [(n, eval_res.metrics) for n, (_, eval_res) in zip(n_samples, results)]

        weighted_loss    = sum(l * n for l, n in zip(losses, n_samples)) / max(sum(n_samples), 1)
        global_metrics   = weighted_average(metrics_list)

        global_f1  = global_metrics.get("f1", 0)
        global_auc = global_metrics.get("auc_roc", 0)
        global_pr_auc = global_metrics.get("pr_auc", 0)
        g_acc  = global_metrics.get("accuracy", 0)
        g_bal  = global_metrics.get("balanced_accuracy", 0)
        g_mcc  = global_metrics.get("mcc", 0)
        g_spec = global_metrics.get("specificity", 0)
        g_pre  = global_metrics.get("precision", 0)
        g_rec  = global_metrics.get("recall", 0)
        g_fraud_pre = global_metrics.get("fraud_precision", g_pre)
        g_fraud_rec = global_metrics.get("fraud_recall", g_rec)
        g_thresh = global_metrics.get("decision_threshold", 0.5)
        print(
            f"\n  [Global Eval R{server_round:02d}]"
            f"  F1={global_f1:.4f}  AUC={global_auc:.4f}  PR-AUC={global_pr_auc:.4f}"
            f"  Recall={g_rec:.4f}  Precision={g_pre:.4f}"
        )
        print(
            f"  [Metrics]     Accuracy={g_acc:.4f}  BalancedAcc={g_bal:.4f}"
            f"  MCC={g_mcc:.4f}  Specificity={g_spec:.4f}"
        )

        if global_f1 > self.best_f1:
            self.best_f1    = global_f1
            self.best_round = server_round
            print(f"  * New best model! F1={global_f1:.4f} at round {server_round}")

        # Finalize this round entry, then commit to blockchain before log flush.
        round_log = self._pending_round_logs.get(server_round)
        if round_log is not None:
            round_log.update({
                "global_f1":               global_f1,
                "global_auc":              global_auc,
                "global_pr_auc":           global_pr_auc,
                "global_recall":           global_metrics.get("recall", 0),
                "global_precision":        global_metrics.get("precision", 0),
                "global_fraud_recall":     g_fraud_rec,
                "global_fraud_precision":  g_fraud_pre,
                "global_decision_threshold": g_thresh,
                "global_accuracy":         global_metrics.get("accuracy", 0),
                "global_balanced_accuracy":global_metrics.get("balanced_accuracy", 0),
                "global_mcc":              global_metrics.get("mcc", 0),
                "global_specificity":      global_metrics.get("specificity", 0),
                "global_tp":               global_metrics.get("tp", 0),
                "global_fp":               global_metrics.get("fp", 0),
                "global_tn":               global_metrics.get("tn", 0),
                "global_fn":               global_metrics.get("fn", 0),
                "global_confusion_matrix": [
                    [int(global_metrics.get("tn", 0)), int(global_metrics.get("fp", 0))],
                    [int(global_metrics.get("fn", 0)), int(global_metrics.get("tp", 0))],
                ],
            })

            round_event = create_signed_round_event(
                round_log=round_log,
                prev_event_hash=self._last_round_event_hash,
                key_bytes=self._round_event_key,
                key_id=self._round_event_key_id,
                run_id=self._round_event_run_id,
                event_sequence=self._round_event_sequence + 1,
            )
            round_log["round_event"] = round_event
            self._append_round_event(round_event)
            self._last_round_event_hash = str(round_event.get("event_hash", ROUND_EVENT_GENESIS))
            self._round_event_sequence += 1

            # Boundary 2: sign+commit round attestation before next round begins.
            if self.governance_engine is not None:
                try:
                    gov_record = self.governance_engine.process_round(round_log)
                    round_log.update({
                        "blockchain_tx_id": gov_record.blockchain_tx_id,
                        "audit_tx_id": gov_record.audit_tx_id,
                        "quarantined_clients": gov_record.quarantined_clients,
                        "round_event_verified": gov_record.round_event_verified,
                        "round_event_reason": gov_record.round_event_reason,
                        "attestation_verified": gov_record.attestation_verified,
                        "attestation_key_id": gov_record.attestation_key_id,
                        "attestation_algo": gov_record.attestation_algo,
                        "attestation_signature": gov_record.attestation_signature,
                        "policy_violations": gov_record.policy_violations,
                        "privacy_violations": gov_record.privacy_violations,
                    })
                except Exception as _ge:
                    raise RuntimeError(
                        f"[Module 2] Governance failure at round {server_round}: {_ge}"
                    ) from _ge
            else:
                round_log.setdefault("blockchain_tx_id", "")
                round_log.setdefault("audit_tx_id", "")

            self.round_logs.append(round_log)
            self.event_storage.append_round_event(round_log)
            self._pending_round_logs.pop(server_round, None)
            self._flush_logs()
        else:
            print(f"  [Warn] Missing pending round log for round {server_round}; skipping flush.")

        return float(weighted_loss), global_metrics

    # -- Parameter aggregation -------------------------------------------------

    def _weighted_aggregate(
        self,
        client_params:  Dict[int, List[np.ndarray]],
        trust_weights:  Dict[int, float],
    ) -> List[np.ndarray]:
        """
        Perform trust-weighted parameter aggregation.
        ?_global = ?_i  w_i ? ?_i
        """
        # Get shape reference from first client
        first_params = next(iter(client_params.values()))
        aggregated   = [np.zeros_like(p, dtype=np.float64) for p in first_params]

        for cid, params in client_params.items():
            w = trust_weights.get(cid, 0.0)
            for i, p in enumerate(params):
                aggregated[i] += w * p.astype(np.float64)

        return [a.astype(np.float32) for a in aggregated]

    # -- Hashing (for Split 3 blockchain) -------------------------------------

    def _hash_params(self, params: List[np.ndarray]) -> str:
        """Compute canonical SHA-256 hash of model parameters for governance logging."""
        digest, _, _ = hash_model_parameters_canonical(params)
        return digest

    # -- Logging ---------------------------------------------------------------

    def _build_round_log(
        self,
        round_num:    int,
        result:       AggregationResult,
        model_hash:   str,
        client_fit_metrics: Dict,
    ) -> Dict:
        """Build per-round metadata to be finalized after evaluation metrics arrive."""
        trust_summary = self.trust_scorer.get_trust_summary()
        log = {
            "schema_name":     EVENT_SCHEMA_NAME,
            "schema_version":  EVENT_SCHEMA_VERSION,
            "round":           round_num,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "model_hash":      model_hash,
            "trusted_clients": result.trusted_clients,
            "flagged_clients": result.flagged_clients,
            "quarantined_clients": [],
            "trust_weights":   {str(k): v for k, v in result.trust_weights.items()},
            "anomaly_scores":  {str(k): v for k, v in result.anomaly_scores.items()},
            "cos_similarities": {str(k): v for k, v in result.cos_similarities.items()},
            "euc_distances":   {str(k): v for k, v in result.euc_distances.items()},
            "trust_scores":    {
                str(cid): info["trust_score"]
                for cid, info in trust_summary.items()
            },
            # Placeholder -- updated after evaluate_aggregate()
            "global_f1":       0.0,
            "global_auc":      0.0,
        }
        return log

    def _flush_logs(self) -> None:
        path = os.path.join(self.log_dir, "trust_training_log.json")
        with open(path, "w") as f:
            json.dump(self.round_logs, f, indent=2)

    def _append_round_event(self, event: Dict) -> None:
        with open(self.round_event_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True))
            f.write("\n")

    def validate_round_logs(self) -> None:
        for rec in self.round_logs:
            RoundEventRecord.from_payload(rec)

    def print_summary(self) -> None:
        """Final summary after all rounds — shows all evaluation metrics."""
        trust_summary = self.trust_scorer.get_trust_summary()
        last = self.round_logs[-1] if self.round_logs else {}

        print(f"\n{'='*65}")
        print(f"  SPLIT 2 FEDERATED TRAINING COMPLETE")
        print(f"  Total Rounds:          {len(self.round_logs)}")
        print(f"  Best F1 (round):       {self.best_f1:.4f}  (Round {self.best_round})")
        print(f"\n  === FINAL ROUND METRICS ===")
        print(f"  F1 Score:              {last.get('global_f1', 0):.4f}")
        print(f"  AUC-ROC:               {last.get('global_auc', 0):.4f}")
        print(f"  PR-AUC:                {last.get('global_pr_auc', 0):.4f}")
        print(f"  Recall (Sensitivity):  {last.get('global_recall', 0):.4f}")
        print(f"  Precision:             {last.get('global_precision', 0):.4f}")
        print(f"  Fraud Recall:          {last.get('global_fraud_recall', 0):.4f}")
        print(f"  Fraud Precision:       {last.get('global_fraud_precision', 0):.4f}")
        print(f"  Decision Threshold:    {last.get('global_decision_threshold', 0.5):.4f}")
        print(f"  Accuracy:              {last.get('global_accuracy', 0):.4f}  <- raw (misleading on imbalanced data)")
        print(f"  Balanced Accuracy:     {last.get('global_balanced_accuracy', 0):.4f}  <- correct for imbalanced")
        print(f"  MCC:                   {last.get('global_mcc', 0):.4f}  (range [-1,1], >0.5=good, >0.7=strong)")
        print(f"  Specificity:           {last.get('global_specificity', 0):.4f}")
        tp = int(last.get('global_tp', 0))
        fp = int(last.get('global_fp', 0))
        tn = int(last.get('global_tn', 0))
        fn = int(last.get('global_fn', 0))
        print(f"  TP / FP / TN / FN:     {tp} / {fp} / {tn} / {fn}")
        print(f"\n  === ATTACK DETECTION ===")
        flagged_total = sum(1 for info in trust_summary.values() if info["is_malicious"])
        print(f"  Clients flagged:       {flagged_total}")
        print(f"  Rounds flagged:        {sum(info['rounds_flagged'] for info in trust_summary.values())}")
        print(f"\n  === FINAL TRUST SCORES ===")
        for cid, info in sorted(trust_summary.items()):
            status = "[ATTACK] MALICIOUS" if info["is_malicious"] else "[OK]     trusted  "
            print(
                f"    Client {cid:02d} {status} | "
                f"tau={info['trust_score']:.4f} | "
                f"alpha={info['anomaly_score']:.4f} | "
                f"cos={info['cos_similarity']:+.4f} | "
                f"flagged={info['rounds_flagged']} rounds"
            )
        print(f"\n  Log: {self.log_dir}/trust_training_log.json")
        print(f"{'='*65}\n")

# =============================================================================
# FACTORY  — imported by split2/main.py as get_trust_strategy()
# =============================================================================

def get_trust_strategy(
    num_clients:        int,
    fraction_fit:       float = 1.0,
    log_dir:            str   = "logs_split2",
    attack_simulator          = None,
    governance_engine         = None,   # MODULE 2: pass GovernanceEngine here
    event_storage_backend: str = "jsonl",
    # TrustScorer hyperparameters (all have sensible defaults in TrustWeightedFedAvg)
    lambda_cosine:      float = 0.5,
    lambda_distance:    float = 0.3,
    lambda_norm:        float = 0.2,
    gamma:              float = 0.85,
    anomaly_threshold:  float = 0.40,
    malicious_weight:   float = 1e-6,
    min_trust_floor:    float = 0.70,
) -> TrustWeightedFedAvg:
    """
    Factory function — mirrors get_fedavg_strategy() in fedavg_strategy.py.

    Called by split2/main.py:
        strategy = get_trust_strategy(
            num_clients      = args.num_clients,
            fraction_fit     = args.fraction_fit,
            log_dir          = args.log_dir,
            attack_simulator = attacker,
        )

    Args:
        num_clients:       Total number of bank nodes.
        fraction_fit:      Fraction of clients selected per round (1.0 = all).
        log_dir:           Directory for trust_training_log.json and plots.
        attack_simulator:  AttackSimulator instance injected from main.py for
                           server-side gradient-scaling attack simulation.
                           Pass None for clean (no-attack) runs.
        lambda_cosine:     Weight for cosine similarity in anomaly score (0.5).
        lambda_distance:   Weight for euclidean distance penalty (0.3).
        lambda_norm:       Weight for gradient norm ratio (0.2).
        gamma:             EMA decay for trust score update (0.85).
        anomaly_threshold: alpha threshold to flag a client as malicious (0.40).
        malicious_weight:  Aggregation weight given to flagged clients (1e-6).
        min_trust_floor:   Minimum trust score for non-flagged clients (0.70).

    Returns:
        TrustWeightedFedAvg instance ready for fl.server.start_server().
    """
    return TrustWeightedFedAvg(
        num_clients       = num_clients,
        fraction_fit      = fraction_fit,
        fraction_evaluate = fraction_fit,
        lambda_cosine     = lambda_cosine,
        lambda_distance   = lambda_distance,
        lambda_norm       = lambda_norm,
        gamma             = gamma,
        anomaly_threshold = anomaly_threshold,
        malicious_weight  = malicious_weight,
        min_trust_floor   = min_trust_floor,
        log_dir           = log_dir,
        attack_simulator  = attack_simulator,
        governance_engine = governance_engine,
        event_storage_backend = event_storage_backend,
    )
