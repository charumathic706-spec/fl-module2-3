# =============================================================================
# FILE: flower_client.py
# PURPOSE: Defines BankFederatedClient — the Flower client that represents a
#          single financial institution (bank node) in the federated network.
#
#          What it does each round:
#            1. Receives the current global model parameters from the server
#            2. Loads those parameters into its local DNN
#            3. Trains the DNN on its OWN private local transaction data
#            4. Returns the updated parameters + training metrics to the server
#            5. On evaluate() calls: runs inference on local held-out test set
#
#          PRIVACY GUARANTEE: Raw transaction data NEVER leaves the client.
#          Only model weight tensors (float32 arrays) are transmitted.
#
#          Also exports make_client_fn() — a factory closure required by
#          fl.simulation.start_simulation() in main.py.
# =============================================================================

from __future__ import annotations

import numpy as np
import flwr as fl
from flwr.common import (
    Context,                     # NEW: required by updated Flower client_fn signature
    EvaluateIns, EvaluateRes,
    FitIns, FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from typing import Dict, List

try:
    from module1.common.local_models import get_model
    from module1.common.data_partition import apply_smote
except ImportError:
    from common.local_models import get_model
    from common.data_partition import apply_smote


# =============================================================================
# BANK FEDERATED CLIENT
# =============================================================================

class BankFederatedClient(fl.client.Client):
    """
    Flower client representing one bank node.

    Attributes:
        client_id  — integer identifier (0, 1, 2, ...)
        model      — local DNNFraudModel (or LogisticFraudModel)
        X_train / y_train — private local training data (never transmitted)
        X_test  / y_test  — private local test data (never transmitted)
    """

    def __init__(
        self,
        client_id:        int,
        X_train:          np.ndarray,
        y_train:          np.ndarray,
        X_test:           np.ndarray,
        y_test:           np.ndarray,
        model_type:       str   = "dnn",
        use_smote:        bool  = True,
        # FIX v11: label-flip attack parameters — client self-poisons its own
        # training labels when instructed. This is necessary because the
        # AttackSimulator lives on the SERVER process, but the client runs in a
        # SEPARATE subprocess and receives data before the server starts. The
        # only way to inject a label-flip at train time is for the client to
        # know it should flip its own labels.
        is_label_flip:    bool  = False,   # True if this client does label-flip
        flip_fraction:    float = 1.0,     # fraction of labels to flip
        attack_start_round: int = 1,       # round to start flipping
        attack_end_round: int | None = None,
        attack_type: str = "none",
        attack_intensity: float = 1.0,
        trigger_feature_count: int = 3,
        trigger_value: float = 5.0,
        backdoor_target_label: int = 1,
        threshold_mode: str = "auto",
        decision_threshold: float = 0.5,
        threshold_beta: float = 1.5,
    ):
        self.client_id         = client_id
        self.input_dim         = X_train.shape[1]
        self.is_label_flip     = is_label_flip
        self.flip_fraction     = flip_fraction
        self.attack_start_round = attack_start_round
        self.attack_end_round = attack_end_round
        self.attack_type = str(attack_type).lower().strip()
        self.attack_intensity = float(max(0.0, attack_intensity))
        self.trigger_feature_count = int(max(1, trigger_feature_count))
        self.trigger_value = float(trigger_value)
        self.backdoor_target_label = int(backdoor_target_label)
        self._current_round    = 0

        # Optionally apply SMOTE to balance the local training set
        if use_smote:
            self.X_train, self.y_train = apply_smote(X_train, y_train)
        else:
            self.X_train, self.y_train = X_train.copy(), y_train.copy()

        self.X_test     = X_test.copy()
        self.y_test     = y_test.copy()
        self.n_train    = len(self.X_train)
        self.n_test     = len(self.X_test)

        # Instantiate local model via factory
        self.model = get_model(
            model_type,
            self.input_dim,
            threshold_mode=threshold_mode,
            decision_threshold=decision_threshold,
            threshold_beta=threshold_beta,
        )

        attack_tag = " [LABEL-FLIP ATTACKER]" if is_label_flip else ""
        print(
            f"  [Bank {client_id:02d}] Ready | model={model_type} | "
            f"train={self.n_train:,} | test={self.n_test:,} | "
            f"fraud_train={int(self.y_train.sum())} ({self.y_train.mean()*100:.1f}%)"
            f"{attack_tag}"
        )

    def _is_attack_active(self) -> bool:
        if self._current_round < self.attack_start_round:
            return False
        if self.attack_end_round is not None and self._current_round > self.attack_end_round:
            return False
        return True

    def _apply_backdoor(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_poisoned = X.copy()
        y_poisoned = y.copy()
        fraction = min(1.0, max(0.0, 0.2 * self.attack_intensity))
        n_poison = max(1, int(len(y_poisoned) * fraction))
        idx = np.random.choice(len(y_poisoned), size=n_poison, replace=False)
        feature_count = min(self.trigger_feature_count, X_poisoned.shape[1])
        X_poisoned[idx, :feature_count] = X_poisoned[idx, :feature_count] + (self.trigger_value * self.attack_intensity)
        y_poisoned[idx] = self.backdoor_target_label
        print(
            f"  [Attack] Bank {self.client_id:02d} BACKDOOR | "
            f"poisoned {n_poison}/{len(y_poisoned)} | features={feature_count} | target={self.backdoor_target_label}"
        )
        return X_poisoned, y_poisoned

    # -------------------------------------------------------------------------
    # GET PARAMETERS — called by Flower during initialization
    # -------------------------------------------------------------------------

    def get_parameters(self, ins):
        """Return current local model parameters for server initialization."""
        try:
            params = self.model.get_params()
        except RuntimeError:
            # Some baselines (e.g., logistic) expose params only after first fit.
            y = self.y_train
            idx0 = np.where(y == 0)[0]
            idx1 = np.where(y == 1)[0]

            if len(idx0) > 0 and len(idx1) > 0:
                # Ensure bootstrap data contains both classes.
                seed_idx = [int(idx0[0]), int(idx1[0])]
                bootstrap_n = min(32, len(self.X_train))
                if bootstrap_n > 2:
                    remaining = np.random.choice(
                        len(self.X_train), size=bootstrap_n - 2, replace=False
                    )
                    seed_idx.extend(int(i) for i in remaining)
                seed_idx = np.array(seed_idx, dtype=int)
                self.model.fit(self.X_train[seed_idx], self.y_train[seed_idx])
            else:
                # Fallback: initialise with full local set (single-class edge case).
                self.model.fit(self.X_train, self.y_train)
            params = self.model.get_params()
        return fl.common.GetParametersRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(params),
        )

    # -------------------------------------------------------------------------
    # FIT — called by Flower server each training round
    # -------------------------------------------------------------------------

    def fit(self, ins: FitIns) -> FitRes:
        """
        1. Deserialise global parameters and load into local model.
        2. Train locally on private data (with optional label-flip poisoning).
        3. Serialise updated parameters and return to server with metrics.

        FIX v11: Label-flip attack is applied here at training time, not on the
        server. The server's AttackSimulator only handles gradient_scale (which
        acts on already-trained params). Label-flip must corrupt the CLIENT's
        local training labels before fit() runs, otherwise the gradient the
        server receives is NOT actually flipped.
        """
        # Extract round number from config (Flower passes it in configure_fit)
        self._current_round = int(ins.config.get("server_round", self._current_round + 1))
        print(f"  [Bank {self.client_id:02d}] fit() round {self._current_round}")

        # Load global model weights into local model
        global_params = parameters_to_ndarrays(ins.parameters)
        if global_params:
            try:
                self.model.set_params(global_params)
            except Exception as exc:
                print(f"  [Bank {self.client_id:02d}] Note: set_params skipped ({exc})")

        # FIX v11: Label-flip attack — poison labels BEFORE local training
        X_to_train = self.X_train
        y_to_train = self.y_train

        if self.attack_type == "backdoor" and self._is_attack_active():
            X_to_train, y_to_train = self._apply_backdoor(self.X_train, self.y_train)

        if self.is_label_flip and self._is_attack_active():
            y_to_train = self.y_train.copy()
            n_flip = max(1, int(len(y_to_train) * self.flip_fraction * max(self.attack_intensity, 1e-9)))
            flip_idx = np.random.choice(len(y_to_train), size=n_flip, replace=False)
            y_to_train[flip_idx] = 1 - y_to_train[flip_idx]
            n_flipped = int((y_to_train != self.y_train).sum())
            print(
                f"  [Attack] Bank {self.client_id:02d} LABEL-FLIP | "
                f"Flipped {n_flipped}/{len(y_to_train)} labels "
                f"({n_flipped/len(y_to_train)*100:.1f}%)"
            )

        # Local training
        self.model.fit(X_to_train, y_to_train)

        # Evaluate on training set for logging only
        train_metrics = self.model.evaluate(self.X_train, self.y_train)  # evaluate on TRUE labels

        print(
            f"  [Bank {self.client_id:02d}] fit() done | "
            f"F1={train_metrics['f1']:.4f} | AUC={train_metrics['auc_roc']:.4f}"
        )

        return FitRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(self.model.get_params()),
            num_examples=self.n_train,
            metrics={
                "client_id":    float(self.client_id),
                "train_f1":     float(train_metrics["f1"]),
                "train_auc":    float(train_metrics["auc_roc"]),
                "train_recall": float(train_metrics["recall"]),
                "n_fraud":      float(int(self.y_train.sum())),
            },
        )

    # -------------------------------------------------------------------------
    # EVALUATE — called by Flower server to assess global model quality
    # -------------------------------------------------------------------------

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        1. Load received global parameters into local model.
        2. Run inference on local held-out test set.
        3. Return loss + metrics to server.
        """
        print(f"  [Bank {self.client_id:02d}] evaluate() start")

        global_params = parameters_to_ndarrays(ins.parameters)
        if global_params:
            try:
                self.model.set_params(global_params)
            except Exception as exc:
                print(f"  [Bank {self.client_id:02d}] Note: eval set_params skipped ({exc})")

        metrics = self.model.evaluate(self.X_test, self.y_test)

        print(
            f"  [Bank {self.client_id:02d}] evaluate() done | "
            f"F1={metrics['f1']:.4f} | AUC={metrics['auc_roc']:.4f} | "
            f"Recall={metrics['recall']:.4f}"
        )

        # loss = 1 - F1  (higher F1 = lower loss)
        loss = float(1.0 - metrics["f1"])

        # FIX v11: Return full metric set so aggregate_evaluate() in
        # trust_weighted_strategy can log balanced_accuracy, mcc,
        # specificity, tp/fp/tn/fn into the trust_training_log.json.
        # Previously these were missing → logged as 0 in every round.
        return EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            loss=loss,
            num_examples=self.n_test,
            metrics={
                "client_id":          float(self.client_id),
                "f1":                 float(metrics["f1"]),
                "auc_roc":            float(metrics["auc_roc"]),
                "pr_auc":             float(metrics.get("pr_auc", 0.0)),
                "precision":          float(metrics["precision"]),
                "recall":             float(metrics["recall"]),
                "fraud_precision":    float(metrics.get("fraud_precision", 0.0)),
                "fraud_recall":       float(metrics.get("fraud_recall", 0.0)),
                "decision_threshold": float(metrics.get("decision_threshold", 0.5)),
                "accuracy":           float(metrics["accuracy"]),
                "balanced_accuracy":  float(metrics.get("balanced_accuracy", 0.0)),
                "mcc":                float(metrics.get("mcc", 0.0)),
                "specificity":        float(metrics.get("specificity", 0.0)),
                "tp":                 float(metrics.get("tp", 0.0)),
                "fp":                 float(metrics.get("fp", 0.0)),
                "tn":                 float(metrics.get("tn", 0.0)),
                "fn":                 float(metrics.get("fn", 0.0)),
            },
        )

    def to_client(self) -> "BankFederatedClient":
        """
        Required by fl.client.start_client() when using the gRPC transport.
        Flower calls to_client() on the object passed to start_client() to
        get the raw Client instance. Since BankFederatedClient already IS a
        fl.client.Client, we just return self.
        """
        return self


# =============================================================================
# CLIENT FACTORY — passed to fl.simulation.start_simulation()
# =============================================================================

def make_client_fn(
    partitions:         List[Dict],
    model_type:         str   = "dnn",
    use_smote:          bool  = True,
    # FIX v11: label-flip attack parameters forwarded to each client
    label_flip_clients: List[int] = None,
    flip_fraction:      float = 1.0,
    attack_start_round: int   = 1,
    attack_end_round:   int | None = None,
    attack_type:        str = "none",
    attack_intensity:   float = 1.0,
    trigger_feature_count: int = 3,
    trigger_value:      float = 5.0,
    backdoor_target_label: int = 1,
    threshold_mode:     str = "auto",
    decision_threshold: float = 0.5,
    threshold_beta:     float = 1.5,
):
    """
    Returns a closure client_fn(context: Context) → BankFederatedClient.

    FIXED: Flower's simulation engine now requires the new Context-based
    signature:  def client_fn(context: Context) -> Client

    FIX v11: label_flip_clients — list of client IDs that will self-poison
    their training labels. This fixes the label-flip attack which previously
    only existed on the server side (where it had no effect on training).

    Args:
        partitions:          Output of dirichlet_partition() from data_partition.py
        model_type:          "dnn" (recommended) | "logistic"
        use_smote:           Apply SMOTE oversampling on local training data
        label_flip_clients:  Client IDs that will flip their labels during training
        flip_fraction:       Fraction of labels to flip (default 1.0 = all)
        attack_start_round:  Round to begin attacking (default 1)
    """
    _flip_set = set(label_flip_clients or [])

    def client_fn(context: Context) -> BankFederatedClient:
        # Flower sets partition-id in node_config for each simulated client
        cid       = int(context.node_config["partition-id"])
        partition = partitions[cid]
        return BankFederatedClient(
            client_id          = cid,
            X_train            = partition["X_train"],
            y_train            = partition["y_train"],
            X_test             = partition["X_test"],
            y_test             = partition["y_test"],
            model_type         = model_type,
            use_smote          = use_smote,
            is_label_flip      = cid in _flip_set,
            flip_fraction      = flip_fraction,
            attack_start_round = attack_start_round,
            attack_end_round   = attack_end_round,
            attack_type        = attack_type,
            attack_intensity   = attack_intensity,
            trigger_feature_count = trigger_feature_count,
            trigger_value      = trigger_value,
            backdoor_target_label = backdoor_target_label,
            threshold_mode     = threshold_mode,
            decision_threshold = decision_threshold,
            threshold_beta     = threshold_beta,
        )

    return client_fn