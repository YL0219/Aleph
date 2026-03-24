"""
feature_adapter.py - Maps nested metabolic payload to a fixed-order feature vector.

v2: Reads from the nested payload structure (technical/macro/events sections).
The feature order is canonical and must be stable across model versions.
New features should be appended, never inserted, to maintain compatibility.

Phase 10.5 note:
  The macro section now carries raw perception data as named subsections:
    macro.proxies.data  — proxy latest closes (VIX, DXY, BTC, SPY, QQQ, TLT, GLD)
    macro.calendar.data — economic calendar events (FOMC, CPI, NFP, GDP)
    macro.headlines.data — macro news headlines with tags
  Each subsection has _status / _fetched_at_utc metadata.

  The OLD v2 macro feature paths (cross_asset, regime_hints, events.materiality etc.)
  are backwards-compatible: they resolve to None → 0.0 defaults until a future v3
  adapter computes features from the raw perception data.

  To add new macro features from perception data:
    1. Add a _extract_perception_features() helper below
    2. Append new feature names to FEATURE_NAMES (never insert)
    3. Update feature_count() and the model
"""

import math

# ═══════════════════════════════════════════════════════════════════
# Canonical feature order — append-only for backwards compatibility
# v1 features (0-17): technical indicators
# v2 features (18+): macro/event scores
# ═══════════════════════════════════════════════════════════════════

FEATURE_NAMES_V1 = [
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "dist_sma_20",
    "dist_sma_50",
    "dist_sma_200",
    "atr_pct",
    "volatility_20",
    "bb_bandwidth",
    "factor_trend",
    "factor_momentum",
    "factor_volatility",
    "factor_participation",
    "composite_bullish",
    "composite_bearish",
    "composite_neutral",
    "composite_confidence",
]

FEATURE_NAMES_V2_MACRO = [
    "macro_equities_risk",
    "macro_bonds_risk",
    "macro_gold_strength",
    "macro_dollar_pressure",
    "macro_volatility_pressure",
    "macro_crypto_risk",
    "macro_liquidity_stress",
    "macro_correlation_stress",
    "regime_risk_on",
    "regime_risk_off",
    "regime_inflation_pressure",
    "regime_growth_scare",
    "regime_policy_shock",
    "regime_flight_to_safety",
    "event_materiality",
    "event_shock",
    "event_schedule_tension",
    "crypto_risk",
    "crypto_volatility",
    "crypto_weekend_stress",
]

FEATURE_NAMES = FEATURE_NAMES_V1 + FEATURE_NAMES_V2_MACRO

FEATURE_VERSION = "v2.0.0"


def _safe_float(val) -> float:
    """Convert a value to float, returning 0.0 for None/NaN/Inf/invalid."""
    if val is None:
        return 0.0
    if isinstance(val, (int, float)):
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return float(val)
    try:
        f = float(val)
        return 0.0 if math.isnan(f) or math.isinf(f) else f
    except (ValueError, TypeError):
        return 0.0


def _flatten_nested_payload(payload: dict) -> dict:
    """
    Flatten the nested C# payload into a flat dict for feature extraction.
    Supports both the new nested format (meta/technical/macro/events)
    and the legacy flat format for backward compatibility.
    """
    flat = {}

    # Check if this is a nested payload (has 'technical' key)
    tech = payload.get("technical")
    if tech and isinstance(tech, dict):
        # New nested format
        for k, v in tech.items():
            if k == "factors" and isinstance(v, dict):
                flat["factor_trend"] = v.get("trend")
                flat["factor_momentum"] = v.get("momentum")
                flat["factor_volatility"] = v.get("volatility")
                flat["factor_participation"] = v.get("participation")
            elif k == "composite" and isinstance(v, dict):
                flat["composite_bullish"] = v.get("bullish")
                flat["composite_bearish"] = v.get("bearish")
                flat["composite_neutral"] = v.get("neutral")
                flat["composite_confidence"] = v.get("confidence")
            else:
                flat[k] = v

        # Macro section
        macro = payload.get("macro", {})
        if isinstance(macro, dict):
            ca = macro.get("cross_asset", {})
            if isinstance(ca, dict):
                flat["macro_equities_risk"] = ca.get("equities_risk")
                flat["macro_bonds_risk"] = ca.get("bonds_risk")
                flat["macro_gold_strength"] = ca.get("gold_strength")
                flat["macro_dollar_pressure"] = ca.get("dollar_pressure")
                flat["macro_volatility_pressure"] = ca.get("volatility_pressure")
                flat["macro_crypto_risk"] = ca.get("crypto_risk")
                flat["macro_liquidity_stress"] = ca.get("liquidity_stress")
                flat["macro_correlation_stress"] = ca.get("correlation_stress")

            rh = macro.get("regime_hints", {})
            if isinstance(rh, dict):
                flat["regime_risk_on"] = rh.get("risk_on")
                flat["regime_risk_off"] = rh.get("risk_off")
                flat["regime_inflation_pressure"] = rh.get("inflation_pressure")
                flat["regime_growth_scare"] = rh.get("growth_scare")
                flat["regime_policy_shock"] = rh.get("policy_shock")
                flat["regime_flight_to_safety"] = rh.get("flight_to_safety")

        # Events section
        events = payload.get("events", {})
        if isinstance(events, dict):
            flat["event_materiality"] = events.get("materiality")
            flat["event_shock"] = events.get("shock")
            flat["event_schedule_tension"] = events.get("schedule_tension")

            cs = events.get("crypto_stress", {})
            if isinstance(cs, dict):
                flat["crypto_risk"] = cs.get("risk")
                flat["crypto_volatility"] = cs.get("volatility")
                flat["crypto_weekend_stress"] = cs.get("weekend_stress")
    else:
        # Legacy flat format — pass through directly
        flat = dict(payload)

    return flat


def extract_features(payload: dict) -> list[float]:
    """
    Extract a fixed-order feature vector from a metabolic payload dict.
    Supports both nested (v2) and flat (v1) payload formats.
    Missing values are replaced with 0.0 (safe default for SGDClassifier).
    """
    flat = _flatten_nested_payload(payload)
    return [_safe_float(flat.get(name)) for name in FEATURE_NAMES]


def feature_count() -> int:
    return len(FEATURE_NAMES)


def has_meaningful_features(payload: dict) -> bool:
    """Check if the payload has at least some non-null features for prediction."""
    flat = _flatten_nested_payload(payload)
    meaningful = 0
    for name in FEATURE_NAMES_V1:  # check core technical features
        val = flat.get(name)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            meaningful += 1
    return meaningful >= 3
