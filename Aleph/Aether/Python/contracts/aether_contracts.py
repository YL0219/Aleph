"""
aether_contracts.py — Pydantic validation models for the C#↔Python IPC boundary.

These are the "DNA blueprints" of valid blood cells crossing the membrane.
Every JSON payload entering or leaving the Aether Python sandbox MUST pass
through one of these models. Malformed data is rejected with a structured
error — never silently accepted, never crashes the pipeline.

Design rules:
  - All fields use snake_case (Python convention). C# converts camelCase↔snake_case.
  - Optional fields default to safe fallbacks (0.0, empty list, "unknown").
  - Numeric fields are bounded via Pydantic validators — no NaN, no Infinity.
  - Output models define the exact shape C# expects to deserialize.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ═══════════════════════════════════════════════════════════════════════
# Shared Validators
# ═══════════════════════════════════════════════════════════════════════

def _sanitize_float(v: Any, field_name: str = "") -> float:
    """Reject NaN/Infinity — these poison JSON serialization and ML pipelines."""
    if v is None:
        return 0.0
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"Field '{field_name}' contains NaN or Infinity.")
    return f


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ═══════════════════════════════════════════════════════════════════════
# INPUT CONTRACTS — what C# sends TO Python
# ═══════════════════════════════════════════════════════════════════════


class TemporalEnvelopeInput(BaseModel):
    """Temporal provenance envelope from the MetabolicEvent."""
    bar_open_utc: Optional[str] = None
    bar_close_utc: Optional[str] = None
    observation_cutoff_utc: str = ""
    max_included_knowledge_utc: Optional[str] = None
    point_in_time_safe: bool = True
    temporal_policy_version: str = "tp_v1"
    historical_replay_mode: bool = False
    exclusion_reasons: List[str] = Field(default_factory=list)


class MetaInput(BaseModel):
    """Metadata section of the cortex predict payload."""
    symbol: str
    interval: str = "1h"
    asof_utc: str = ""
    source_event_id: str = ""
    metabolic_version: str = ""
    model_key: str = "cortex_sgd_1h_24bar"
    feature_version: str = "v2.0.0"
    active_horizon: str = "1d"
    horizon_bars: int = Field(default=24, ge=1, le=1000)


class HomeostasisInput(BaseModel):
    """Homeostasis state from the autonomic system."""
    stress_level: float = 0.0
    fatigue_level: float = 0.0
    is_overloaded: bool = False
    is_breathless: bool = False

    @field_validator("stress_level", "fatigue_level", mode="before")
    @classmethod
    def sanitize_levels(cls, v: Any) -> float:
        return _clamp(_sanitize_float(v, "homeostasis_level"), 0.0, 1.0)


class GovernanceInput(BaseModel):
    """Governance flags controlling prediction and training eligibility."""
    eligible_for_prediction: bool = True
    eligible_for_training: bool = False
    learning_block_reasons: List[str] = Field(default_factory=list)


class CortexPredictInput(BaseModel):
    """
    Top-level input contract for cortex_predict.
    This is the --payload JSON that C# passes to Python via CLI argument.
    """
    meta: MetaInput
    temporal: TemporalEnvelopeInput = Field(default_factory=TemporalEnvelopeInput)
    technical: Dict[str, Any] = Field(default_factory=dict)
    macro: Dict[str, Any] = Field(default_factory=dict)
    events: Dict[str, Any] = Field(default_factory=dict)
    homeostasis: HomeostasisInput = Field(default_factory=HomeostasisInput)
    governance: GovernanceInput = Field(default_factory=GovernanceInput)

    @model_validator(mode="after")
    def enforce_temporal_governance(self) -> "CortexPredictInput":
        """If temporal safety failed, force training eligibility off."""
        if not self.temporal.point_in_time_safe:
            self.governance.eligible_for_training = False
        return self


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT CONTRACTS — what Python sends BACK to C#
# ═══════════════════════════════════════════════════════════════════════


class ProbabilitiesOutput(BaseModel):
    """3-class probability distribution. Must sum to ~1.0."""
    bullish: float = 0.33
    neutral: float = 0.34
    bearish: float = 0.33

    @field_validator("bullish", "neutral", "bearish", mode="before")
    @classmethod
    def sanitize_prob(cls, v: Any) -> float:
        return _clamp(_sanitize_float(v, "probability"), 0.0, 1.0)

    @model_validator(mode="after")
    def check_sum(self) -> "ProbabilitiesOutput":
        total = self.bullish + self.neutral + self.bearish
        if not (0.95 <= total <= 1.05):
            # Normalize instead of rejecting — cold start models may be imprecise
            self.bullish /= total
            self.neutral /= total
            self.bearish /= total
        return self


class RegimeProbabilitiesOutput(BaseModel):
    risk_on: float = 0.0
    risk_off: float = 0.0
    inflation_pressure: float = 0.0
    growth_scare: float = 0.0
    policy_shock: float = 0.0
    flight_to_safety: float = 0.0

    @field_validator("*", mode="before")
    @classmethod
    def sanitize(cls, v: Any) -> float:
        return _clamp(_sanitize_float(v, "regime_prob"), 0.0, 1.0)


class EventProbabilitiesOutput(BaseModel):
    materiality: float = 0.0
    follow_through: float = 0.0
    volatility_expansion: float = 0.0

    @field_validator("*", mode="before")
    @classmethod
    def sanitize(cls, v: Any) -> float:
        return _clamp(_sanitize_float(v, "event_prob"), 0.0, 1.0)


class CatalystRefOutput(BaseModel):
    event_type: str
    scheduled_for_utc: str
    importance_probability: float = 0.0

    @field_validator("importance_probability", mode="before")
    @classmethod
    def sanitize_importance(cls, v: Any) -> float:
        return _clamp(_sanitize_float(v, "importance"), 0.0, 1.0)


class CortexPredictOutput(BaseModel):
    """
    Output contract for cortex_predict.
    This is the JSON that Python writes to stdout for C# to consume.
    """
    ok: bool = True
    prediction_id: str = ""
    predicted_class: str = "neutral"
    confidence: float = 0.0
    action_tendency: float = 0.0
    model_state: str = "cold_start"
    model_version: str = "v1.0.0"
    model_key: str = "cortex_sgd_1h_24bar"
    feature_version: str = "v2.0.0"
    trained_samples: int = 0
    temporal_security_passed: bool = True
    eligible_for_training: bool = False
    pending_sample_stored: bool = False
    training_occurred: bool = False
    priority_score: Optional[float] = None

    probabilities: ProbabilitiesOutput = Field(default_factory=ProbabilitiesOutput)
    regime_probabilities: Optional[RegimeProbabilitiesOutput] = None
    event_probabilities: Optional[EventProbabilitiesOutput] = None

    top_drivers: List[str] = Field(default_factory=list)
    top_risks: List[str] = Field(default_factory=list)
    watched_catalysts: List[CatalystRefOutput] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    @field_validator("confidence", mode="before")
    @classmethod
    def sanitize_confidence(cls, v: Any) -> float:
        return _clamp(_sanitize_float(v, "confidence"), 0.0, 1.0)

    @field_validator("action_tendency", mode="before")
    @classmethod
    def sanitize_tendency(cls, v: Any) -> float:
        return _clamp(_sanitize_float(v, "action_tendency"), -1.0, 1.0)

    @field_validator("predicted_class", mode="before")
    @classmethod
    def validate_class(cls, v: Any) -> str:
        allowed = {"bullish", "neutral", "bearish"}
        s = str(v).lower().strip()
        if s not in allowed:
            return "neutral"
        return s


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT: cortex_train
# ═══════════════════════════════════════════════════════════════════════


class TrainingResultOutput(BaseModel):
    """Nested training result within cortex_train output."""
    samples_fitted: int = 0
    fresh_count: int = 0
    replay_count: int = 0
    model_state: str = "cold_start"
    model_version: str = ""
    trained_samples: int = 0
    gate: Optional[str] = None
    gate_reason: Optional[str] = None
    class_distribution: Optional[Dict[str, int]] = None
    warnings: List[str] = Field(default_factory=list)

    @field_validator("samples_fitted", "fresh_count", "replay_count", "trained_samples", mode="before")
    @classmethod
    def sanitize_counts(cls, v: Any) -> int:
        n = int(v) if v is not None else 0
        return max(n, 0)


class CortexTrainOutput(BaseModel):
    """Output contract for cortex_train."""
    ok: bool = True
    domain: str = "ml"
    action: str = "cortex_train"
    symbol: str = ""
    horizon: str = "1d"
    training: TrainingResultOutput = Field(default_factory=TrainingResultOutput)
    cursor_sequence: int = 0
    consumed_count: int = 0
    warnings: List[str] = Field(default_factory=list)

    @field_validator("cursor_sequence", "consumed_count", mode="before")
    @classmethod
    def sanitize_counts(cls, v: Any) -> int:
        n = int(v) if v is not None else 0
        return max(n, 0)


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT: cortex_resolve
# ═══════════════════════════════════════════════════════════════════════


class ResolutionSummaryOutput(BaseModel):
    """Resolution summary nested within cortex_resolve output."""
    total_pending: int = 0
    eligible: int = 0
    resolved: int = 0
    deferred: int = 0
    expired: int = 0
    errored: int = 0
    accuracy: float = 0.0
    mean_brier: float = 0.0
    class_distribution: Optional[Dict[str, int]] = None

    @field_validator("accuracy", "mean_brier", mode="before")
    @classmethod
    def sanitize_metrics(cls, v: Any) -> float:
        return _clamp(_sanitize_float(v, "resolution_metric"), 0.0, 1.0)


class CycleScorecardOutput(BaseModel):
    """Cycle-level scorecard from resolve."""
    accuracy: float = 0.0
    brier: float = 0.0
    grade: str = "F"
    samples: int = 0


class CortexResolveOutput(BaseModel):
    """Output contract for cortex_resolve."""
    ok: bool = True
    domain: str = "ml"
    action: str = "cortex_resolve"
    symbol: str = ""
    horizon: str = "1d"
    resolution: ResolutionSummaryOutput = Field(default_factory=ResolutionSummaryOutput)
    pending_rewrite: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    cycle_scorecard: Optional[CycleScorecardOutput] = None


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT: cortex_status
# ═══════════════════════════════════════════════════════════════════════


class RollingScorecardOutput(BaseModel):
    """Rolling scorecard summary in cortex_status output."""
    accuracy: float = 0.0
    brier: float = 0.0
    grade: str = "F"
    samples: int = 0
    last_updated_utc: Optional[str] = None


class CortexStatusOutput(BaseModel):
    """Output contract for cortex_status."""
    ok: bool = True
    domain: str = "ml"
    action: str = "cortex_status"
    symbol: str = ""
    horizon: str = "1d"
    model_key: str = ""
    feature_version: str = ""
    model_state: str = "cold_start"
    model_version: str = ""
    trained_samples: int = 0
    pending_count: int = 0
    resolved_count: int = 0
    pending_eligible_count: int = 0
    pending_blocked_count: int = 0
    temporal_policy_version: str = ""
    last_train_utc: Optional[str] = None
    class_distribution: Optional[Dict[str, int]] = None
    cursor_sequence: int = 0
    total_samples_ever_trained: int = 0
    active_policies: Optional[Dict[str, Any]] = None
    rolling_scorecard: Optional[RollingScorecardOutput] = None


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT: cortex_evaluate
# ═══════════════════════════════════════════════════════════════════════


class CortexEvaluateOutput(BaseModel):
    """Output contract for cortex_evaluate."""
    ok: bool = True
    domain: str = "ml"
    action: str = "cortex_evaluate"
    symbol: str = ""
    horizon: str = "1d"
    evaluation: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# OUTPUT: dream_step (Simulation)
# ═══════════════════════════════════════════════════════════════════════


class DreamStepOutput(BaseModel):
    """Output contract for dream_step simulation action."""
    ok: bool = True
    domain: str = "sim"
    action: str = "dream_step"
    dream_id: str = ""
    step: int = 0
    bar_utc: Optional[str] = None
    predicted_class: str = "neutral"
    confidence: float = 0.0
    probabilities: Optional[ProbabilitiesOutput] = None
    model_state: str = "cold_start"
    pending_stored: bool = False
    complete: bool = False
    warnings: List[str] = Field(default_factory=list)

    @field_validator("predicted_class", mode="before")
    @classmethod
    def validate_class(cls, v: Any) -> str:
        allowed = {"bullish", "neutral", "bearish"}
        s = str(v).lower().strip()
        return s if s in allowed else "neutral"

    @field_validator("confidence", mode="before")
    @classmethod
    def sanitize_confidence(cls, v: Any) -> float:
        return _clamp(_sanitize_float(v, "confidence"), 0.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# INPUT: cortex_train (validates --payload or args from C#)
# ═══════════════════════════════════════════════════════════════════════


class CortexTrainInput(BaseModel):
    """Input contract for cortex_train. Validated before execution."""
    symbol: str
    horizon: str = "1d"
    max_samples: int = Field(default=100, ge=1, le=10000)


class CortexResolveInput(BaseModel):
    """Input contract for cortex_resolve."""
    symbol: str
    horizon: str = "1d"
    interval: str = "1h"


class CortexEvaluateInput(BaseModel):
    """Input contract for cortex_evaluate."""
    symbol: str
    horizon: str = "1d"
    challengers_json: str = ""


# ═══════════════════════════════════════════════════════════════════════
# GENERIC AETHER RESPONSE ENVELOPE
# ═══════════════════════════════════════════════════════════════════════


class AetherResponse(BaseModel):
    """
    Universal response wrapper for all Aether Python outputs.
    Guarantees that every response has ok + error fields.
    C# checks ok first, then parses domain-specific fields.
    """
    ok: bool = False
    domain: str = ""
    action: str = ""
    error: Optional[str] = None

    @classmethod
    def fail(cls, domain: str, action: str, error: str) -> "AetherResponse":
        return cls(ok=False, domain=domain, action=action, error=error)


class AetherErrorResponse(BaseModel):
    """Structured error response for quarantine logging on the C# side."""
    ok: bool = False
    error: str = "unknown_error"
    error_type: str = "validation"
    quarantine: bool = True
    details: Optional[str] = None
