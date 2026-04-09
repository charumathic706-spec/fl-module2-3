"""
attack_simulator.py
--------------------
Simulates adversarial poisoning attacks on federated clients.

Implements the two attack types specified in the project proposal:
  1. Label-Flipping Attack  -- malicious client flips fraud labels (0<->1)
                              during local training to corrupt the global model
  2. Gradient Scaling Attack -- malicious client amplifies its gradient update
                              by a large factor to dominate aggregation

These are injected at the client level -- the server's trust scoring
(trust_scoring.py) must detect and neutralise them.

Usage:
    attacker = AttackSimulator(attack_type="label_flip", malicious_clients=[1, 3])
    X_poisoned, y_poisoned = attacker.poison_data(client_id=1, X, y)
    params_poisoned = attacker.poison_params(client_id=1, params, global_params)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional


class AttackSimulator:
    """
    Injects adversarial behaviour into specified malicious clients.

    Supports:
      - "label_flip"       : Flip fraud/legit labels before local training
      - "gradient_scale"   : Amplify gradient updates after training
            - "backdoor"         : Add trigger pattern and force target label
            - "sign_flip"        : Reverse gradient direction
            - "model_replacement": Aggressive targeted parameter replacement
            - "sybil"            : Coordinate malicious clients to submit near-identical updates
      - "combined"         : Both attacks simultaneously
      - "none"             : No attack (clean baseline)
    """

    def __init__(
        self,
        attack_type:       str        = "label_flip",
        malicious_clients: List[int]  = None,
        flip_fraction:     float      = 1.0,    # Fraction of labels to flip
        scale_factor:      float      = 5.0,    # Gradient amplification factor
        attack_start_round: int       = 1,      # Round to begin attacking
        attack_end_round:   Optional[int] = None,
        attack_intensity:   float      = 1.0,
        trigger_value:      float      = 5.0,
        trigger_feature_count: int     = 3,
        backdoor_target_label: int     = 1,
    ):
        self.attack_type        = attack_type.lower()
        self.malicious_clients  = set(malicious_clients or [])
        self.flip_fraction      = flip_fraction
        self.scale_factor       = scale_factor
        self.attack_start_round = attack_start_round
        self.attack_end_round   = attack_end_round
        self.attack_intensity   = max(0.0, attack_intensity)
        self.trigger_value      = trigger_value
        self.trigger_feature_count = max(1, int(trigger_feature_count))
        self.backdoor_target_label = int(backdoor_target_label)
        self.current_round      = 0
        self._sybil_delta_cache: Optional[List[np.ndarray]] = None
        self._sybil_round_cache: int = -1

        valid_types = {
            "label_flip",
            "gradient_scale",
            "backdoor",
            "sign_flip",
            "model_replacement",
            "sybil",
            "combined",
            "none",
        }
        if self.attack_type not in valid_types:
            raise ValueError(f"attack_type must be one of {valid_types}")

        if self.malicious_clients:
            print(
                f"[AttackSimulator] [!]  Attack type: '{attack_type}' | "
                f"Malicious clients: {sorted(self.malicious_clients)} | "
                f"Scale factor: {scale_factor}x | "
                f"Schedule: rounds {attack_start_round}..{attack_end_round if attack_end_round is not None else 'end'} | "
                f"Intensity: {self.attack_intensity}"
            )
        else:
            print("[AttackSimulator] Clean run -- no malicious clients.")

    def set_round(self, round_num: int) -> None:
        self.current_round = round_num

    def _is_active_round(self) -> bool:
        if self.current_round < self.attack_start_round:
            return False
        if self.attack_end_round is not None and self.current_round > self.attack_end_round:
            return False
        return True

    def is_malicious(self, client_id: int) -> bool:
        return (
            client_id in self.malicious_clients
            and self._is_active_round()
        )

    # -- Attack 1: Label Flipping ----------------------------------------------

    def poison_data(
        self,
        client_id: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Label-flipping attack: flip fraud<->legit labels for malicious clients.

        Effect on gradient: reversed learning signal causes gradient to point
        in the OPPOSITE direction to the global gradient -> detectable via
        negative cosine similarity in trust_scoring.py.

        Args:
            client_id: Client being poisoned
            X:         Feature matrix (unchanged)
            y:         Labels to potentially flip

        Returns:
            (X, y_poisoned) -- X is unchanged, y may be flipped
        """
        if not self.is_malicious(client_id):
            return X, y

        if self.attack_type not in ("label_flip", "combined"):
            return X, y

        y_poisoned = y.copy()
        n_flip     = max(1, int(len(y) * self.flip_fraction * max(self.attack_intensity, 1e-9)))
        flip_idx   = np.random.choice(len(y), size=n_flip, replace=False)
        y_poisoned[flip_idx] = 1 - y_poisoned[flip_idx]  # flip 0<->1

        n_actual_flips = int((y_poisoned != y).sum())
        print(
            f"  [Attack] Client {client_id:02d} LABEL-FLIP | "
            f"Flipped {n_actual_flips}/{len(y)} labels "
            f"({n_actual_flips/len(y)*100:.1f}%)"
        )
        return X, y_poisoned

    def poison_backdoor_data(
        self,
        client_id: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Inject trigger pattern into features and relabel to backdoor target."""
        if not self.is_malicious(client_id):
            return X, y
        if self.attack_type not in ("backdoor", "combined"):
            return X, y

        X_poisoned = X.copy()
        y_poisoned = y.copy()

        poison_fraction = min(1.0, max(0.0, 0.2 * self.attack_intensity))
        n_poison = max(1, int(len(y_poisoned) * poison_fraction))
        idx = np.random.choice(len(y_poisoned), size=n_poison, replace=False)

        feature_count = min(self.trigger_feature_count, X_poisoned.shape[1])
        X_poisoned[idx, :feature_count] = X_poisoned[idx, :feature_count] + (self.trigger_value * self.attack_intensity)
        y_poisoned[idx] = self.backdoor_target_label

        print(
            f"  [Attack] Client {client_id:02d} BACKDOOR | "
            f"poisoned={n_poison}/{len(y_poisoned)} | "
            f"trigger_features={feature_count} | target={self.backdoor_target_label}"
        )
        return X_poisoned, y_poisoned

    # -- Attack 2: Gradient Scaling --------------------------------------------

    def poison_params(
        self,
        client_id:    int,
        params:       List[np.ndarray],
        global_params: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Gradient scaling attack: amplify the parameter update delta.

        delta_i = params_i - global_params
        poisoned = global_params + scale_factor * delta_i

        Effect: client's update dominates aggregation due to large norm
        -> detectable via norm_ratio >> 1 in trust_scoring.py.

        Args:
            client_id:    Client being poisoned
            params:       Client's post-training parameters
            global_params: Current global model parameters

        Returns:
            Poisoned parameters (amplified delta from global)
        """
        if not self.is_malicious(client_id):
            return params

        if self.attack_type in ("gradient_scale", "combined"):
            poisoned = []
            for p, g in zip(params, global_params):
                delta = p - g
                factor = self.scale_factor * max(self.attack_intensity, 1e-9)
                poisoned.append(g + factor * delta)

            orig_norm    = np.linalg.norm(np.concatenate([p.flatten() for p in params]))
            poison_norm  = np.linalg.norm(np.concatenate([p.flatten() for p in poisoned]))
            print(
                f"  [Attack] Client {client_id:02d} GRADIENT-SCALE | "
                f"factor={self.scale_factor * self.attack_intensity:.2f}x | "
                f"norm: {orig_norm:.2f} -> {poison_norm:.2f}"
            )
            return poisoned

        if self.attack_type == "sign_flip":
            poisoned = []
            for p, g in zip(params, global_params):
                delta = p - g
                factor = max(1.0, self.scale_factor * self.attack_intensity)
                poisoned.append(g - factor * delta)
            print(f"  [Attack] Client {client_id:02d} SIGN-FLIP | factor={self.scale_factor * self.attack_intensity:.2f}x")
            return poisoned

        if self.attack_type == "model_replacement":
            # Pushes a stronger crafted update relative to global model.
            poisoned = []
            for p, g in zip(params, global_params):
                delta = p - g
                factor = max(1.0, 8.0 * self.attack_intensity)
                poisoned.append(g + factor * delta)
            print(f"  [Attack] Client {client_id:02d} MODEL-REPLACEMENT | factor={8.0 * self.attack_intensity:.2f}x")
            return poisoned

        if self.attack_type == "sybil":
            if self._sybil_round_cache != self.current_round:
                self._sybil_round_cache = self.current_round
                self._sybil_delta_cache = None

            if self._sybil_delta_cache is None:
                self._sybil_delta_cache = [
                    (p - g) * max(1.0, self.scale_factor * self.attack_intensity)
                    for p, g in zip(params, global_params)
                ]

            poisoned = [g + d for g, d in zip(global_params, self._sybil_delta_cache)]
            print(f"  [Attack] Client {client_id:02d} SYBIL | coordinated malicious update")
            return poisoned

        return params

    def get_attack_summary(self) -> Dict:
        return {
            "attack_type":        self.attack_type,
            "malicious_clients":  sorted(self.malicious_clients),
            "flip_fraction":      self.flip_fraction,
            "scale_factor":       self.scale_factor,
            "attack_start_round": self.attack_start_round,
            "attack_end_round":   self.attack_end_round,
            "attack_intensity":   self.attack_intensity,
            "trigger_value":      self.trigger_value,
            "trigger_feature_count": self.trigger_feature_count,
            "backdoor_target_label": self.backdoor_target_label,
        }