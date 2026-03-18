"""
incremental_model.py — Policy-driven SGDClassifier wrapper for online learning.

v2: Upgraded from a naive partial_fit wrapper into a controlled trainer that:
  - uses a TrainingPolicy to govern batch construction and weighting
  - supports cursor-aware candidate selection (no double-training)
  - mixes fresh resolved samples with a replay slice from older history
  - applies class-balance weighting and ambiguity-aware sample weights
  - monitors for drift/collapse and emits structured diagnostics

The incumbent model family is SGDClassifier with modified_huber loss.
This may evolve, but the interface is stable.
"""

from __future__ import annotations

import sys
import math
import numpy as np
from collections import Counter
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from .policies import TrainingPolicy, DEFAULT_TRAINING_POLICY

CLASSES = np.array(["bearish", "bullish", "neutral"])

# Thresholds for model state determination
COLD_START_THRESHOLD = 10
WARMING_THRESHOLD = 50


class IncrementalCortexModel:
    """
    Controlled online learner wrapping SGDClassifier.

    Supports cold-start predictions with uniform probabilities,
    policy-driven batch construction, and diagnostic output.
    """

    def __init__(self):
        self.model = SGDClassifier(
            loss="modified_huber",  # gives probability estimates
            penalty="l2",
            alpha=1e-4,
            max_iter=1,
            tol=None,
            warm_start=True,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.trained_samples = 0
        self.model_version = "v1.0.0"
        self._fitted = False
        self._scaler_fitted = False

    @property
    def model_state(self) -> str:
        if self.trained_samples < COLD_START_THRESHOLD:
            return "cold_start"
        elif self.trained_samples < WARMING_THRESHOLD:
            return "warming"
        return "active"

    def predict(self, features: list[float]) -> dict:
        """
        Predict class probabilities from a feature vector.
        Returns uniform probabilities during cold start.
        """
        X = np.array([features])

        if not self._fitted:
            return {
                "predicted_class": "neutral",
                "probabilities": {"bullish": 0.333, "neutral": 0.334, "bearish": 0.333},
                "confidence": 0.0,
                "action_tendency": 0.0,
            }

        if self._scaler_fitted:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X

        try:
            proba = self.model.predict_proba(X_scaled)[0]
            class_prob = {}
            for cls, p in zip(self.model.classes_, proba):
                class_prob[cls] = float(p)

            for cls in ["bullish", "neutral", "bearish"]:
                if cls not in class_prob:
                    class_prob[cls] = 0.0

            predicted_class = max(class_prob, key=class_prob.get)
            confidence = class_prob[predicted_class]
            action_tendency = class_prob.get("bullish", 0.0) - class_prob.get("bearish", 0.0)
            action_tendency = max(-1.0, min(1.0, action_tendency))

        except Exception as ex:
            print(f"[MlCortex] Prediction error: {ex}", file=sys.stderr)
            return {
                "predicted_class": "neutral",
                "probabilities": {"bullish": 0.333, "neutral": 0.334, "bearish": 0.333},
                "confidence": 0.0,
                "action_tendency": 0.0,
            }

        return {
            "predicted_class": predicted_class,
            "probabilities": class_prob,
            "confidence": round(confidence, 4),
            "action_tendency": round(action_tendency, 4),
        }

    def class_distribution(self) -> dict:
        """Return class distribution from training history (approximate)."""
        if not self._fitted or not hasattr(self.model, "classes_"):
            return {}
        return {cls: 0 for cls in self.model.classes_}

    # ═══════════════════════════════════════════════════════════════
    # CONTROLLED TRAINING
    # ═══════════════════════════════════════════════════════════════

    def controlled_fit(
        self,
        fresh_samples: list[dict],
        replay_samples: list[dict],
        policy: TrainingPolicy | None = None,
    ) -> TrainResult:
        """
        Policy-driven incremental training cycle.

        Args:
          fresh_samples   — newly resolved records (not yet trained on)
          replay_samples  — older resolved records for replay mixing
          policy          — training policy governing this cycle

        Returns:
          TrainResult with diagnostics
        """
        tp = policy or DEFAULT_TRAINING_POLICY
        warnings: list[str] = []

        # ── Filter candidates ──
        fresh_valid = _filter_eligible(fresh_samples, tp, warnings)
        replay_valid = _filter_eligible(replay_samples, tp, warnings)

        # Cap fresh samples
        if len(fresh_valid) > tp.max_fresh_samples:
            fresh_valid = fresh_valid[:tp.max_fresh_samples]
            warnings.append(f"capped_fresh_at_{tp.max_fresh_samples}")

        # Build replay slice
        replay_slice = _build_replay_slice(replay_valid, tp)

        # Combine batch
        batch = fresh_valid + replay_slice

        if len(batch) < tp.min_samples_to_train:
            return TrainResult(
                samples_fitted=0,
                fresh_count=len(fresh_valid),
                replay_count=len(replay_slice),
                batch_class_distribution={},
                model_state=self.model_state,
                model_version=self.model_version,
                trained_samples_total=self.trained_samples,
                warnings=warnings + [f"batch_too_small:{len(batch)}<{tp.min_samples_to_train}"],
                drift_flags=[],
                policy_version=tp.version,
            )

        # ── Extract features, labels, weights ──
        X_list, y_list, w_list = [], [], []
        for sample in batch:
            features = sample.get("features", [])
            label = sample.get("actual_label") or sample.get("label", "")
            if not features or not label:
                continue
            X_list.append(features)
            y_list.append(label)
            w_list.append(_compute_sample_weight(sample, tp))

        if not X_list:
            return TrainResult(
                samples_fitted=0,
                fresh_count=0, replay_count=0,
                batch_class_distribution={},
                model_state=self.model_state,
                model_version=self.model_version,
                trained_samples_total=self.trained_samples,
                warnings=warnings + ["no_valid_features_in_batch"],
                drift_flags=[], policy_version=tp.version,
            )

        X = np.array(X_list)
        y = np.array(y_list)
        w = np.array(w_list)

        # ── Class balance weighting ──
        if tp.class_balance_mode == "weight":
            w = _apply_class_balance_weights(y, w)

        # ── Batch class distribution ──
        class_dist = dict(Counter(y_list))

        # ── Fit / update scaler ──
        if not self._scaler_fitted:
            self.scaler.fit(X)
            self._scaler_fitted = True
        else:
            self.scaler.partial_fit(X)

        X_scaled = self.scaler.transform(X)

        # ── Pre-fit snapshot for drift detection ──
        pre_fit_proba = None
        if self._fitted and len(X_scaled) > 0:
            try:
                pre_fit_proba = self.model.predict_proba(X_scaled[:min(10, len(X_scaled))])
            except Exception:
                pass

        # ── partial_fit ──
        self.model.partial_fit(X_scaled, y, classes=CLASSES, sample_weight=w)
        self._fitted = True
        self.trained_samples += len(y)

        # ── Drift detection ──
        drift_flags = []
        if pre_fit_proba is not None:
            try:
                post_fit_proba = self.model.predict_proba(X_scaled[:min(10, len(X_scaled))])
                drift = float(np.mean(np.abs(post_fit_proba - pre_fit_proba)))
                if drift > 0.3:
                    drift_flags.append(f"high_probability_shift:{drift:.3f}")
            except Exception:
                pass

        # Check for class collapse (model only predicting one class)
        if self._fitted:
            try:
                test_proba = self.model.predict_proba(X_scaled[:min(20, len(X_scaled))])
                pred_classes = set(self.model.classes_[np.argmax(test_proba, axis=1)])
                if len(pred_classes) == 1:
                    drift_flags.append(f"class_collapse:{list(pred_classes)[0]}")
            except Exception:
                pass

        return TrainResult(
            samples_fitted=len(y),
            fresh_count=len(fresh_valid),
            replay_count=len(replay_slice),
            batch_class_distribution=class_dist,
            model_state=self.model_state,
            model_version=self.model_version,
            trained_samples_total=self.trained_samples,
            warnings=warnings,
            drift_flags=drift_flags,
            policy_version=tp.version,
        )

    def partial_fit(self, features_batch: list[list[float]], labels: list[str]) -> int:
        """
        Legacy simple partial_fit for backward compatibility.
        Use controlled_fit for policy-driven training.
        """
        if not features_batch or not labels:
            return 0

        X = np.array(features_batch)
        y = np.array(labels)

        if not self._scaler_fitted:
            self.scaler.fit(X)
            self._scaler_fitted = True
        else:
            self.scaler.partial_fit(X)

        X_scaled = self.scaler.transform(X)
        self.model.partial_fit(X_scaled, y, classes=CLASSES)
        self._fitted = True
        self.trained_samples += len(labels)

        return len(labels)

    def get_state_dict(self) -> dict:
        """Return serializable state for persistence."""
        return {
            "trained_samples": self.trained_samples,
            "model_version": self.model_version,
            "model_state": self.model_state,
            "fitted": self._fitted,
            "scaler_fitted": self._scaler_fitted,
        }


# ═══════════════════════════════════════════════════════════════════
# TRAIN RESULT
# ═══════════════════════════════════════════════════════════════════

class TrainResult:
    """Structured output from a controlled training cycle."""

    def __init__(
        self,
        samples_fitted: int,
        fresh_count: int,
        replay_count: int,
        batch_class_distribution: dict,
        model_state: str,
        model_version: str,
        trained_samples_total: int,
        warnings: list[str],
        drift_flags: list[str],
        policy_version: str,
    ):
        self.samples_fitted = samples_fitted
        self.fresh_count = fresh_count
        self.replay_count = replay_count
        self.batch_class_distribution = batch_class_distribution
        self.model_state = model_state
        self.model_version = model_version
        self.trained_samples_total = trained_samples_total
        self.warnings = warnings
        self.drift_flags = drift_flags
        self.policy_version = policy_version

    def to_dict(self) -> dict:
        return {
            "samples_fitted": self.samples_fitted,
            "fresh_count": self.fresh_count,
            "replay_count": self.replay_count,
            "batch_class_distribution": self.batch_class_distribution,
            "model_state": self.model_state,
            "model_version": self.model_version,
            "trained_samples_total": self.trained_samples_total,
            "warnings": self.warnings,
            "drift_flags": self.drift_flags,
            "policy_version": self.policy_version,
        }


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def _filter_eligible(
    samples: list[dict],
    policy: TrainingPolicy,
    warnings: list[str],
) -> list[dict]:
    """Filter samples by training eligibility according to policy."""
    eligible = []
    blocked = 0
    for s in samples:
        if policy.require_temporal_safety and not s.get("point_in_time_safe", True):
            blocked += 1
            continue
        if not s.get("eligible_for_training", True):
            blocked += 1
            continue
        if policy.require_policy_compat:
            lpv = s.get("label_policy_version", "")
            if lpv and lpv not in policy.compatible_label_versions:
                blocked += 1
                continue
        eligible.append(s)

    if blocked:
        warnings.append(f"filtered_ineligible:{blocked}")
    return eligible


def _build_replay_slice(
    replay_pool: list[dict],
    policy: TrainingPolicy,
) -> list[dict]:
    """
    Select a replay slice from older resolved samples.
    Uses recency-weighted sampling: more recent samples get higher probability.
    """
    if not replay_pool or policy.replay_ratio <= 0:
        return []

    max_replay = min(policy.replay_max_samples, len(replay_pool))
    if max_replay <= 0:
        return []

    n = len(replay_pool)
    if policy.replay_recency_weight < 1.0 and n > 1:
        weights = np.array([
            policy.replay_recency_weight ** (n - 1 - i) for i in range(n)
        ])
        weights = weights / weights.sum()
    else:
        weights = np.ones(n) / n

    sample_size = min(max_replay, n)
    rng = np.random.default_rng(seed=42)
    indices = rng.choice(n, size=sample_size, replace=False, p=weights)
    return [replay_pool[i] for i in sorted(indices)]


def _compute_sample_weight(sample: dict, policy: TrainingPolicy) -> float:
    """Compute per-sample weight based on ambiguity and policy."""
    base_weight = 1.0
    ambiguity = sample.get("ambiguity_score", 0.0)

    if ambiguity > policy.max_ambiguity:
        base_weight *= policy.ambiguity_sample_weight

    return base_weight


def _apply_class_balance_weights(
    y: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Apply inverse-frequency class weighting to improve balance.
    Minority classes get higher weight; majority classes get lower weight.
    """
    counts = Counter(y)
    if not counts:
        return w

    total = len(y)
    n_classes = len(counts)
    w_out = w.copy()

    for i, label in enumerate(y):
        class_count = counts[label]
        class_weight = total / (n_classes * class_count) if class_count > 0 else 1.0
        w_out[i] *= class_weight

    return w_out
