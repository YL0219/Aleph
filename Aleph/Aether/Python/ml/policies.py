"""
policies.py — Versioned policy substrate for the Cortex learning loop.

Every tunable decision in the resolve/label/train pipeline is governed by
a serializable, versioned policy object.  This prevents threshold-burying,
enables challenger/incumbent comparison, and gives every resolved/trained
artifact full provenance.

Policy classes:
  LabelPolicy       — directional 3-class labeling rules
  ResolutionPolicy   — when/how to resolve pending predictions
  TrainingPolicy     — candidate selection, replay, weighting, fit control
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# LABEL POLICY
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LabelPolicy:
    """
    Governs how a realized return is mapped to a directional 3-class label.

    Fields:
      version             — policy identifier for provenance
      bullish_threshold_bps — minimum return (basis points) for 'bullish'
      bearish_threshold_bps — minimum negative return (bps) for 'bearish'
      price_basis         — which price column to anchor ('close', 'open', 'vwap')
      neutral_band_mode   — 'symmetric' | 'asymmetric' (future use)
      ambiguity_zone_bps  — returns within this band *around* thresholds get
                            an elevated ambiguity score (soft margin)
    """
    version: str = "lp_v1"
    bullish_threshold_bps: float = 100.0   # +1.00%
    bearish_threshold_bps: float = -100.0  # -1.00%
    price_basis: str = "close"
    neutral_band_mode: str = "symmetric"
    ambiguity_zone_bps: float = 30.0       # 0.30% band around thresholds

    def assign_label(self, return_bps: float) -> str:
        """Map realized return in basis points to 3-class label."""
        if return_bps > self.bullish_threshold_bps:
            return "bullish"
        if return_bps < self.bearish_threshold_bps:
            return "bearish"
        return "neutral"

    def compute_ambiguity(self, return_bps: float) -> float:
        """
        0.0 = clearly inside a class.
        1.0 = sitting right on a threshold boundary.
        """
        dist_bull = abs(return_bps - self.bullish_threshold_bps)
        dist_bear = abs(return_bps - self.bearish_threshold_bps)
        nearest = min(dist_bull, dist_bear)
        if self.ambiguity_zone_bps <= 0:
            return 0.0
        return max(0.0, 1.0 - nearest / self.ambiguity_zone_bps)

    def compute_move_strength(self, return_bps: float) -> float:
        """
        Normalized magnitude of the move. 0 = no move, 1+ = strong directional move.
        Reference scale is the bullish threshold.
        """
        ref = abs(self.bullish_threshold_bps) or 100.0
        return round(abs(return_bps) / ref, 4)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> LabelPolicy:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════
# RESOLUTION POLICY
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ResolutionPolicy:
    """
    Governs *when* and *how* a pending prediction can be resolved against
    the realized market truth from the parquet data lake.

    Fields:
      version                — policy identifier
      require_full_horizon   — if True, reject resolution when the target bar
                               doesn't exist yet in the parquet
      min_horizon_coverage   — fraction [0,1] of horizon bars that must be
                               present before we attempt resolution (future use)
      gap_tolerance_bars     — maximum consecutive missing bars before we flag
                               a data integrity problem
      max_pending_age_hours  — samples older than this are auto-expired
                               (never resolved, removed from queue)
      allowed_intervals      — intervals we know how to resolve
      price_column           — default column for truth anchoring
    """
    version: str = "rp_v1"
    require_full_horizon: bool = True
    min_horizon_coverage: float = 1.0
    gap_tolerance_bars: int = 3
    max_pending_age_hours: float = 168.0 * 4  # 4 weeks default
    allowed_intervals: tuple[str, ...] = ("1h", "4h", "1d")
    price_column: str = "close"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["allowed_intervals"] = list(d["allowed_intervals"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ResolutionPolicy:
        if "allowed_intervals" in d:
            d = dict(d)
            d["allowed_intervals"] = tuple(d["allowed_intervals"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════
# TRAINING POLICY
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TrainingPolicy:
    """
    Governs incremental training behavior.

    Fields:
      version                 — policy identifier
      max_fresh_samples       — max newly-resolved samples per training cycle
      replay_ratio            — fraction of batch that should be older replay data
      replay_max_samples      — hard cap on replay slice size
      replay_recency_weight   — exponential decay factor for replay sampling
                                (1.0 = uniform, <1.0 = favor recent)
      min_samples_to_train    — minimum batch size to bother training
      max_ambiguity           — samples with ambiguity > this are down-weighted
      ambiguity_sample_weight — weight applied to high-ambiguity samples [0,1]
      class_balance_mode      — 'none' | 'oversample_minority' | 'weight'
      require_temporal_safety — hard block on temporally unsafe samples
      require_policy_compat   — only train on samples resolved with compatible policies
      compatible_label_versions — set of label policy versions we accept for training
    """
    version: str = "tp_v1"
    max_fresh_samples: int = 200
    replay_ratio: float = 0.3
    replay_max_samples: int = 100
    replay_recency_weight: float = 0.95
    min_samples_to_train: int = 3
    max_ambiguity: float = 0.8
    ambiguity_sample_weight: float = 0.5
    class_balance_mode: str = "weight"
    require_temporal_safety: bool = True
    require_policy_compat: bool = False
    compatible_label_versions: tuple[str, ...] = ("lp_v1",)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["compatible_label_versions"] = list(d["compatible_label_versions"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> TrainingPolicy:
        if "compatible_label_versions" in d:
            d = dict(d)
            d["compatible_label_versions"] = tuple(d["compatible_label_versions"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ═══════════════════════════════════════════════════════════════════
# DEFAULT POLICY INSTANCES
# ═══════════════════════════════════════════════════════════════════

DEFAULT_LABEL_POLICY = LabelPolicy()
DEFAULT_RESOLUTION_POLICY = ResolutionPolicy()
DEFAULT_TRAINING_POLICY = TrainingPolicy()


def get_active_policies() -> dict[str, Any]:
    """Return the active policy set as a serializable dict."""
    return {
        "label_policy": DEFAULT_LABEL_POLICY.to_dict(),
        "resolution_policy": DEFAULT_RESOLUTION_POLICY.to_dict(),
        "training_policy": DEFAULT_TRAINING_POLICY.to_dict(),
    }
