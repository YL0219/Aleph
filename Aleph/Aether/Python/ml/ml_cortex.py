"""
ml_cortex.py - Main entrypoint for the ML Cortex Python brain.

v2: Reads nested metabolic payload, enforces temporal security,
returns expanded prediction contract with regime/event probabilities.

Called by ml_manager.py for cortex_predict, cortex_train, cortex_status actions.
Routes to the appropriate brain subsystem and returns a normalized JSON dict.

All stdout output = exactly one JSON object.
All logs go to stderr.
"""

import sys
import uuid

from .feature_adapter import extract_features, has_meaningful_features, FEATURE_VERSION
from .brain_state import load_model, save_model
from .temporal_security import check_temporal_safety, compute_eligibility
from .pending_memory import (
    store_pending_sample,
    load_resolved_samples,
    pending_count as get_pending_count,
    pending_eligible_count as get_pending_eligible_count,
    pending_blocked_count as get_pending_blocked_count,
)
from .prediction_formatter import format_prediction, format_status, format_train_result


def cortex_predict(symbol: str, interval: str, horizon: str, asof_utc: str, payload: dict) -> dict:
    """
    Real-time prediction path. Always available, even cold start.

    1. Read meta/temporal/governance from nested payload
    2. Check temporal safety
    3. Load model (or cold-start)
    4. Extract features
    5. Predict
    6. Store pending sample (with eligibility metadata)
    7. Return expanded prediction contract
    """
    warnings: list[str] = []

    # ── Read nested payload sections ──
    meta = payload.get("meta", {})
    temporal = payload.get("temporal", {})
    governance = payload.get("governance", {})
    homeostasis = payload.get("homeostasis", {})

    model_key = meta.get("model_key", "")
    feature_version = meta.get("feature_version", FEATURE_VERSION)
    active_horizon = meta.get("active_horizon", horizon)
    horizon_bars = meta.get("horizon_bars", 24)
    source_event_id = meta.get("source_event_id")
    observation_cutoff_utc = temporal.get("observation_cutoff_utc", asof_utc)
    temporal_policy_version = temporal.get("temporal_policy_version", "")

    # ── Temporal security check ──
    ts_result = check_temporal_safety(payload)
    temporal_passed = ts_result["passed"]

    # Combine with governance signals for eligibility
    eligible, block_reasons = compute_eligibility(
        temporal_passed=temporal_passed,
        governance={
            "breathless": homeostasis.get("is_breathless", False),
            "overloaded": homeostasis.get("is_overloaded", False),
            "learning_paused": governance.get("learning_paused", False),
        },
    )

    # Also respect C# governance decision if it was stricter
    cs_eligible = governance.get("eligible_for_training", True)
    cs_block_reasons = governance.get("learning_block_reasons", [])
    if not cs_eligible:
        eligible = False
        for r in cs_block_reasons:
            if r not in block_reasons:
                block_reasons.append(r)

    if ts_result["violations"]:
        warnings.append(f"temporal_violations:{','.join(ts_result['violations'])}")

    # ── Load model ──
    model = load_model(symbol, horizon)
    print(f"[MlCortex] Predict {symbol}/{horizon} — state={model.model_state}, samples={model.trained_samples}", file=sys.stderr)

    # ── Check feature quality ──
    if not has_meaningful_features(payload):
        warnings.append("insufficient_features")

    # ── Extract features ──
    features = extract_features(payload)

    # ── Predict ──
    result = model.predict(features)

    if model.model_state == "cold_start":
        warnings.append("cold_start_prediction")

    # ── Build regime/event probabilities from payload ──
    macro = payload.get("macro", {})
    regime_hints = macro.get("regime_hints", {}) if isinstance(macro, dict) else {}
    events = payload.get("events", {})

    regime_probabilities = {
        "risk_on": _sf(regime_hints.get("risk_on")),
        "risk_off": _sf(regime_hints.get("risk_off")),
        "inflation_pressure": _sf(regime_hints.get("inflation_pressure")),
        "growth_scare": _sf(regime_hints.get("growth_scare")),
        "policy_shock": _sf(regime_hints.get("policy_shock")),
        "flight_to_safety": _sf(regime_hints.get("flight_to_safety")),
    }

    event_probabilities = {
        "materiality": _sf(events.get("materiality")) if isinstance(events, dict) else 0.0,
        "shock": _sf(events.get("shock")) if isinstance(events, dict) else 0.0,
        "schedule_tension": _sf(events.get("schedule_tension")) if isinstance(events, dict) else 0.0,
    }

    # ── Compute priority score ──
    priority_score = _compute_priority(result, regime_probabilities, event_probabilities)

    # ── Extract context tags ──
    macro_tags = macro.get("macro_tags", []) if isinstance(macro, dict) else []
    headline_tags = events.get("headline_tags", []) if isinstance(events, dict) else []
    scheduled_event_types = []
    if isinstance(events, dict):
        for cat in events.get("scheduled_catalysts", []):
            if isinstance(cat, dict) and cat.get("event_type"):
                scheduled_event_types.append(cat["event_type"])

    # ── Generate prediction ID ──
    prediction_id = uuid.uuid4().hex[:16]

    # ── Get entry price ──
    technical = payload.get("technical", {})
    entry_price = technical.get("price") if isinstance(technical, dict) else None

    # ── Store pending sample ──
    pending_stored = store_pending_sample(
        symbol=symbol,
        horizon=horizon,
        features=features,
        predicted_class=result["predicted_class"],
        asof_utc=asof_utc,
        prediction_id=prediction_id,
        model_key=model_key,
        interval=interval,
        active_horizon=active_horizon,
        horizon_bars=horizon_bars,
        observation_cutoff_utc=observation_cutoff_utc,
        point_in_time_safe=temporal_passed,
        temporal_policy_version=temporal_policy_version,
        feature_version=feature_version,
        predicted_probabilities=result["probabilities"],
        regime_probabilities=regime_probabilities,
        event_probabilities=event_probabilities,
        priority_score=priority_score,
        macro_tags=macro_tags if isinstance(macro_tags, list) else [],
        headline_tags=headline_tags if isinstance(headline_tags, list) else [],
        scheduled_event_types=scheduled_event_types,
        eligible_for_training=eligible,
        learning_block_reasons=block_reasons,
        entry_price=entry_price,
        price_basis="close",
        source_event_id=source_event_id,
    )

    # ── Build watched catalysts (pass through from events) ──
    watched_catalysts = []
    if isinstance(events, dict):
        for cat in events.get("scheduled_catalysts", []):
            if isinstance(cat, dict):
                watched_catalysts.append(cat.get("event_type", "unknown"))

    return format_prediction(
        predicted_class=result["predicted_class"],
        probabilities=result["probabilities"],
        confidence=result["confidence"],
        action_tendency=result["action_tendency"],
        model_state=model.model_state,
        model_version=model.model_version,
        trained_samples=model.trained_samples,
        prediction_id=prediction_id,
        model_key=model_key,
        feature_version=feature_version,
        temporal_security_passed=temporal_passed,
        eligible_for_training=eligible,
        regime_probabilities=regime_probabilities,
        event_probabilities=event_probabilities,
        priority_score=priority_score,
        top_drivers=_extract_top_drivers(result, regime_probabilities),
        top_risks=_extract_top_risks(regime_probabilities, event_probabilities),
        watched_catalysts=watched_catalysts,
        learning_block_reasons=block_reasons,
        pending_sample_stored=pending_stored,
        training_occurred=False,
        warnings=warnings,
    )


def cortex_train(symbol: str, horizon: str, max_samples: int = 100) -> dict:
    """
    Incremental training path. Uses resolved samples to partial_fit the model.
    Should only be called during Calm/DeepWork windows.
    """
    model = load_model(symbol, horizon)
    print(f"[MlCortex] Train {symbol}/{horizon} — loading resolved samples", file=sys.stderr)

    # Load resolved (labeled) samples
    resolved = load_resolved_samples(symbol, horizon, max_samples=max_samples)

    if not resolved:
        print(f"[MlCortex] No resolved samples for {symbol}/{horizon}", file=sys.stderr)
        return format_train_result(
            symbol=symbol,
            horizon=horizon,
            samples_fitted=0,
            model_state=model.model_state,
            model_version=model.model_version,
            trained_samples=model.trained_samples,
        )

    # Extract features and labels from resolved samples
    features_batch = [s["features"] for s in resolved if "features" in s and "label" in s]
    labels = [s["label"] for s in resolved if "features" in s and "label" in s]

    if not features_batch:
        return format_train_result(
            symbol=symbol,
            horizon=horizon,
            samples_fitted=0,
            model_state=model.model_state,
            model_version=model.model_version,
            trained_samples=model.trained_samples,
        )

    # Partial fit
    fitted = model.partial_fit(features_batch, labels)
    print(f"[MlCortex] Fitted {fitted} samples for {symbol}/{horizon}", file=sys.stderr)

    # Save updated model
    save_model(symbol, horizon, model)

    return format_train_result(
        symbol=symbol,
        horizon=horizon,
        samples_fitted=fitted,
        model_state=model.model_state,
        model_version=model.model_version,
        trained_samples=model.trained_samples,
    )


def cortex_status(symbol: str, horizon: str) -> dict:
    """Get Cortex model status (v2 expanded)."""
    model = load_model(symbol, horizon)
    pc = get_pending_count(symbol, horizon)
    pe = get_pending_eligible_count(symbol, horizon)
    pb = get_pending_blocked_count(symbol, horizon)

    # Resolved count
    from .pending_memory import _resolved_path
    rp = _resolved_path(symbol, horizon)
    rc = 0
    if rp.exists():
        try:
            with open(rp, "r") as f:
                rc = sum(1 for line in f if line.strip())
        except Exception:
            pass

    # Class distribution from model
    class_dist = {}
    if hasattr(model, "class_distribution"):
        class_dist = model.class_distribution()

    return format_status(
        symbol=symbol,
        horizon=horizon,
        model_state=model.model_state,
        model_version=model.model_version,
        trained_samples=model.trained_samples,
        pending_count=pc,
        resolved_count=rc,
        model_key="",
        feature_version=FEATURE_VERSION,
        pending_eligible_count=pe,
        pending_blocked_count=pb,
        temporal_policy_version="tp_v1",
        last_train_utc=None,
        class_distribution=class_dist,
    )


# ─── Internal helpers ──────────────────────────────────────────────


def _sf(val) -> float:
    """Safe float conversion."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _compute_priority(prediction: dict, regime: dict, events: dict) -> float:
    """Simple priority score: higher confidence + higher event/regime signals = higher priority."""
    conf = prediction.get("confidence", 0.0)
    tend = abs(prediction.get("action_tendency", 0.0))
    event_signal = max(events.get("materiality", 0.0), events.get("shock", 0.0))
    regime_signal = max(regime.get("risk_off", 0.0), regime.get("policy_shock", 0.0), regime.get("flight_to_safety", 0.0))
    return round(conf * 0.4 + tend * 0.2 + event_signal * 0.2 + regime_signal * 0.2, 4)


def _extract_top_drivers(prediction: dict, regime: dict) -> list[str]:
    """Extract top contributing factors for the prediction."""
    drivers = []
    probs = prediction.get("probabilities", {})
    pred_class = prediction.get("predicted_class", "neutral")

    if pred_class == "bullish" and probs.get("bullish", 0) > 0.5:
        drivers.append("strong_bullish_signal")
    elif pred_class == "bearish" and probs.get("bearish", 0) > 0.5:
        drivers.append("strong_bearish_signal")

    if regime.get("risk_on", 0) > 0.6:
        drivers.append("regime_risk_on")
    if regime.get("risk_off", 0) > 0.6:
        drivers.append("regime_risk_off")

    return drivers[:5]


def _extract_top_risks(regime: dict, events: dict) -> list[str]:
    """Extract top risk factors."""
    risks = []
    if regime.get("policy_shock", 0) > 0.4:
        risks.append("policy_shock_elevated")
    if regime.get("flight_to_safety", 0) > 0.5:
        risks.append("flight_to_safety")
    if regime.get("growth_scare", 0) > 0.4:
        risks.append("growth_scare")
    if events.get("shock", 0) > 0.5:
        risks.append("event_shock_elevated")
    if events.get("schedule_tension", 0) > 0.6:
        risks.append("high_schedule_tension")
    return risks[:5]
