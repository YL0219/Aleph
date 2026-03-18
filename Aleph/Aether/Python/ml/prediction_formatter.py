"""
prediction_formatter.py — Normalized output contract for the Python brain.

v3: Expanded for Phase 9 learning loop — adds format_resolve and format_controlled_train
alongside the existing predict/status/train formatters.

All functions return a dict that gets serialized to JSON by the router.
Every output has {ok, domain, action} as the root envelope.
"""

from .feature_adapter import FEATURE_VERSION


# ═══════════════════════════════════════════════════════════════════
# PREDICTION OUTPUT
# ═══════════════════════════════════════════════════════════════════

def format_prediction(
    predicted_class: str,
    probabilities: dict,
    confidence: float,
    action_tendency: float,
    model_state: str,
    model_version: str,
    trained_samples: int,
    # v2 fields
    prediction_id: str = "",
    model_key: str = "",
    feature_version: str = "",
    temporal_security_passed: bool = True,
    eligible_for_training: bool = True,
    regime_probabilities: dict | None = None,
    event_probabilities: dict | None = None,
    priority_score: float = 0.0,
    top_drivers: list[str] | None = None,
    top_risks: list[str] | None = None,
    watched_catalysts: list[str] | None = None,
    learning_block_reasons: list[str] | None = None,
    pending_sample_stored: bool = False,
    training_occurred: bool = False,
    warnings: list[str] | None = None,
) -> dict:
    """Build the normalized prediction output dict."""
    regime = regime_probabilities or {}
    event = event_probabilities or {}

    return {
        "ok": True,
        "domain": "ml",
        "action": "cortex_predict",
        # core prediction
        "predicted_class": predicted_class,
        "probabilities": {
            "bullish": round(probabilities.get("bullish", 0.333), 4),
            "neutral": round(probabilities.get("neutral", 0.334), 4),
            "bearish": round(probabilities.get("bearish", 0.333), 4),
        },
        "confidence": round(confidence, 4),
        "action_tendency": round(action_tendency, 4),
        # identity / versioning
        "prediction_id": prediction_id,
        "model_key": model_key,
        "model_state": model_state,
        "model_version": model_version,
        "feature_version": feature_version or FEATURE_VERSION,
        "trained_samples": trained_samples,
        # temporal security
        "temporal_security_passed": temporal_security_passed,
        "eligible_for_training": eligible_for_training,
        "learning_block_reasons": learning_block_reasons or [],
        # regime / event probabilities
        "regime_probabilities": {
            "risk_on": round(regime.get("risk_on", 0.0), 4),
            "risk_off": round(regime.get("risk_off", 0.0), 4),
            "inflation_pressure": round(regime.get("inflation_pressure", 0.0), 4),
            "growth_scare": round(regime.get("growth_scare", 0.0), 4),
            "policy_shock": round(regime.get("policy_shock", 0.0), 4),
            "flight_to_safety": round(regime.get("flight_to_safety", 0.0), 4),
        },
        "event_probabilities": {
            "materiality": round(event.get("materiality", 0.0), 4),
            "shock": round(event.get("shock", 0.0), 4),
            "schedule_tension": round(event.get("schedule_tension", 0.0), 4),
        },
        # scoring / explainability
        "priority_score": round(priority_score, 4),
        "top_drivers": top_drivers or [],
        "top_risks": top_risks or [],
        "watched_catalysts": watched_catalysts or [],
        # lifecycle
        "pending_sample_stored": pending_sample_stored,
        "training_occurred": training_occurred,
        "warnings": warnings or [],
    }


# ═══════════════════════════════════════════════════════════════════
# STATUS OUTPUT
# ═══════════════════════════════════════════════════════════════════

def format_status(
    symbol: str,
    horizon: str,
    model_state: str,
    model_version: str,
    trained_samples: int,
    pending_count: int,
    resolved_count: int,
    # v2 fields
    model_key: str = "",
    feature_version: str = "",
    pending_eligible_count: int = 0,
    pending_blocked_count: int = 0,
    temporal_policy_version: str = "",
    last_train_utc: str | None = None,
    class_distribution: dict | None = None,
    # v3 fields
    cursor_sequence: int = 0,
    total_samples_ever_trained: int = 0,
    active_policies: dict | None = None,
) -> dict:
    """Build the normalized status output dict."""
    return {
        "ok": True,
        "domain": "ml",
        "action": "cortex_status",
        "symbol": symbol,
        "horizon": horizon,
        "model_key": model_key,
        "model_state": model_state,
        "model_version": model_version,
        "feature_version": feature_version or FEATURE_VERSION,
        "trained_samples": trained_samples,
        "pending_count": pending_count,
        "pending_eligible_count": pending_eligible_count,
        "pending_blocked_count": pending_blocked_count,
        "resolved_count": resolved_count,
        "temporal_policy_version": temporal_policy_version,
        "last_train_utc": last_train_utc,
        "class_distribution": class_distribution or {},
        "cursor_sequence": cursor_sequence,
        "total_samples_ever_trained": total_samples_ever_trained,
        "active_policies": active_policies or {},
    }


# ═══════════════════════════════════════════════════════════════════
# RESOLVE OUTPUT
# ═══════════════════════════════════════════════════════════════════

def format_resolve(
    symbol: str,
    horizon: str,
    resolution_summary: dict,
    pending_rewrite_result: dict | None = None,
    warnings: list[str] | None = None,
) -> dict:
    """Build the normalized resolve output dict."""
    return {
        "ok": True,
        "domain": "ml",
        "action": "cortex_resolve",
        "symbol": symbol,
        "horizon": horizon,
        "resolution": resolution_summary,
        "pending_rewrite": pending_rewrite_result or {},
        "warnings": warnings or [],
    }


# ═══════════════════════════════════════════════════════════════════
# CONTROLLED TRAIN OUTPUT
# ═══════════════════════════════════════════════════════════════════

def format_controlled_train(
    symbol: str,
    horizon: str,
    train_result: dict,
    cursor_sequence: int = 0,
    consumed_count: int = 0,
    warnings: list[str] | None = None,
) -> dict:
    """Build the normalized controlled-train output dict."""
    return {
        "ok": True,
        "domain": "ml",
        "action": "cortex_train",
        "symbol": symbol,
        "horizon": horizon,
        "training": train_result,
        "cursor_sequence": cursor_sequence,
        "consumed_count": consumed_count,
        "warnings": warnings or [],
    }


# ═══════════════════════════════════════════════════════════════════
# LEGACY TRAIN OUTPUT (backward compat)
# ═══════════════════════════════════════════════════════════════════

def format_train_result(
    symbol: str,
    horizon: str,
    samples_fitted: int,
    model_state: str,
    model_version: str,
    trained_samples: int,
) -> dict:
    """Build the legacy training result output dict."""
    return {
        "ok": True,
        "domain": "ml",
        "action": "cortex_train",
        "symbol": symbol,
        "horizon": horizon,
        "samples_fitted": samples_fitted,
        "model_state": model_state,
        "model_version": model_version,
        "trained_samples": trained_samples,
    }
