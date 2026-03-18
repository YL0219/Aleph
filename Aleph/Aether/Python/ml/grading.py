"""
grading.py — Sidecar grading computations for resolved predictions.

The primary training target is the directional 3-class label.
These grading sidecars evaluate the *quality* of the broader probability
surface emitted at prediction time vs. what actually happened.

Sidecar grades are stored in the resolved record for:
  - evaluation / replay analysis
  - future multi-head learning
  - Arbiter-guided self-improvement

They do NOT influence the incumbent SGDClassifier fit target.
"""

from __future__ import annotations

import math


def grade_directional(
    predicted_class: str,
    actual_label: str,
    predicted_probabilities: dict,
) -> dict:
    """
    Grade the directional prediction.

    Returns:
      correct           — bool, did the predicted class match the actual label?
      brier_score       — Brier score for the 3-class prediction [0=perfect, 2=worst]
      calibration_gap   — |predicted_confidence - accuracy| (per-sample proxy)
      grade_bucket      — 'correct' | 'wrong_direction' | 'missed_neutral' | 'false_neutral'
    """
    correct = predicted_class == actual_label

    # Brier score: sum of (p_i - y_i)^2 over all classes
    classes = ["bullish", "neutral", "bearish"]
    brier = 0.0
    for cls in classes:
        p = predicted_probabilities.get(cls, 1.0 / 3.0)
        y = 1.0 if cls == actual_label else 0.0
        brier += (p - y) ** 2

    # Calibration gap (per-sample)
    predicted_conf = predicted_probabilities.get(predicted_class, 1.0 / 3.0)
    actual_hit = 1.0 if correct else 0.0
    calibration_gap = abs(predicted_conf - actual_hit)

    # Grade bucket
    if correct:
        grade_bucket = "correct"
    elif actual_label == "neutral":
        grade_bucket = "missed_neutral"  # predicted directional, got neutral
    elif predicted_class == "neutral":
        grade_bucket = "false_neutral"   # predicted neutral, got directional
    else:
        grade_bucket = "wrong_direction"  # predicted bullish, got bearish (or vice versa)

    return {
        "correct": correct,
        "brier_score": round(brier, 6),
        "calibration_gap": round(calibration_gap, 4),
        "grade_bucket": grade_bucket,
    }


def grade_regime(
    predicted_regime: dict,
    realized_context: dict | None = None,
) -> dict:
    """
    Grade regime probability predictions.

    For now: store the predicted regime as-is and compute entropy/concentration
    as a self-consistency metric.  True regime grading requires a ground-truth
    regime labeler (future work).

    Returns:
      entropy           — Shannon entropy of the regime distribution
      dominant_regime    — highest-probability regime
      dominant_prob      — probability of the dominant regime
      concentration     — 1 - entropy/max_entropy (0=uniform, 1=concentrated)
    """
    regimes = ["risk_on", "risk_off", "inflation_pressure",
               "growth_scare", "policy_shock", "flight_to_safety"]

    probs = [max(predicted_regime.get(r, 0.0), 1e-10) for r in regimes]
    total = sum(probs)
    if total <= 0:
        total = 1.0
    normed = [p / total for p in probs]

    entropy = -sum(p * math.log(p) for p in normed if p > 0)
    max_entropy = math.log(len(regimes))

    dominant_idx = max(range(len(regimes)), key=lambda i: normed[i])

    return {
        "entropy": round(entropy, 4),
        "dominant_regime": regimes[dominant_idx],
        "dominant_prob": round(normed[dominant_idx], 4),
        "concentration": round(1.0 - entropy / max_entropy if max_entropy > 0 else 0.0, 4),
    }


def grade_event_surface(
    predicted_events: dict,
    realized_volatility_bps: float | None = None,
) -> dict:
    """
    Grade event/materiality predictions.

    Uses realized volatility as a proxy for event impact:
    if the model predicted high materiality/shock and a big move happened,
    that's a hit.

    Returns:
      predicted_materiality  — what was predicted
      predicted_shock        — what was predicted
      realized_vol_bps       — what actually happened (if available)
      event_hit              — bool, if available
    """
    mat = predicted_events.get("materiality", 0.0)
    shock = predicted_events.get("shock", 0.0)

    result: dict = {
        "predicted_materiality": round(mat, 4),
        "predicted_shock": round(shock, 4),
    }

    if realized_volatility_bps is not None:
        result["realized_vol_bps"] = round(realized_volatility_bps, 2)
        # Simple hit: predicted high event signal AND a big move happened
        high_signal = max(mat, shock) > 0.4
        big_move = abs(realized_volatility_bps) > 150  # >1.5% move
        result["event_hit"] = high_signal == big_move
    else:
        result["realized_vol_bps"] = None
        result["event_hit"] = None

    return result


def grade_volatility_expansion(
    predicted_regime: dict,
    realized_range_bps: float | None = None,
    entry_atr_bps: float | None = None,
) -> dict:
    """
    Volatility expansion sidecar — did vol expand as implied by regime?

    Returns:
      vol_expansion_ratio — realized range / entry ATR (>1 = expansion)
      regime_implied_vol  — synthetic vol expectation from regime probs
      vol_surprise        — expansion ratio / regime implied (>1 = surprise)
    """
    regime_vol_weights = {
        "risk_off": 1.5,
        "policy_shock": 2.0,
        "flight_to_safety": 1.8,
        "growth_scare": 1.3,
        "risk_on": 0.8,
        "inflation_pressure": 1.0,
    }
    implied = sum(
        predicted_regime.get(r, 0.0) * w
        for r, w in regime_vol_weights.items()
    )
    implied = max(implied, 0.5)

    result: dict = {
        "regime_implied_vol": round(implied, 4),
    }

    if realized_range_bps is not None and entry_atr_bps is not None and entry_atr_bps > 0:
        expansion = realized_range_bps / entry_atr_bps
        result["vol_expansion_ratio"] = round(expansion, 4)
        result["vol_surprise"] = round(expansion / implied, 4)
    else:
        result["vol_expansion_ratio"] = None
        result["vol_surprise"] = None

    return result
