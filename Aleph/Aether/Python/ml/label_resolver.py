"""
label_resolver.py — Policy-driven delayed label resolution against parquet truth.

Takes pending predictions and resolves them against the realized OHLCV market
history from the parquet data lake.  Produces rich resolved records with:
  - primary directional 3-class label (the incumbent training target)
  - sidecar grades for regime/event/volatility (for evaluation only, not fit)
  - full provenance chain

Hard invariants:
  - no look-ahead bias (temporal safety enforced)
  - no resolution unless the full horizon is observable
  - no resolved sample enters training without valid provenance
  - append-only resolved truth archive

Depends on:
  - quant/parquet_loader.py for OHLCV market data
  - policies.py for LabelPolicy and ResolutionPolicy
  - grading.py for sidecar evaluations
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from typing import Any

import pandas as pd

from .policies import LabelPolicy, ResolutionPolicy, DEFAULT_LABEL_POLICY, DEFAULT_RESOLUTION_POLICY
from .grading import grade_directional, grade_regime, grade_event_surface, grade_volatility_expansion


# ═══════════════════════════════════════════════════════════════════
# MAIN ENTRY: resolve_pending_batch
# ═══════════════════════════════════════════════════════════════════

def resolve_pending_batch(
    pending_samples: list[dict],
    ohlcv_df: pd.DataFrame | None,
    label_policy: LabelPolicy | None = None,
    resolution_policy: ResolutionPolicy | None = None,
    data_interval: str | None = None,
) -> ResolutionResult:
    """
    Attempt to resolve a batch of pending predictions against market truth.

    Args:
      pending_samples   — list of pending sample dicts from pending_memory
      ohlcv_df          — DataFrame of OHLCV candles (time, open, high, low, close, volume)
                          sorted by time ascending, with timezone-aware datetime index
      label_policy      — labeling rules; defaults to DEFAULT_LABEL_POLICY
      resolution_policy — resolution rules; defaults to DEFAULT_RESOLUTION_POLICY
      data_interval     — actual interval of the loaded OHLCV data (e.g. "1d" even
                          when the sample's interval is "1h"). When the data is coarser
                          than the sample interval, tolerances and coverage are adjusted.

    Returns:
      ResolutionResult with resolved, deferred, expired, and errored samples plus stats.
    """
    lp = label_policy or DEFAULT_LABEL_POLICY
    rp = resolution_policy or DEFAULT_RESOLUTION_POLICY

    resolved: list[dict] = []
    deferred: list[dict] = []
    expired: list[dict] = []
    errored: list[dict] = []
    warnings: list[str] = []

    now_utc = datetime.now(timezone.utc)

    # ── Validate OHLCV data ──
    if ohlcv_df is None or ohlcv_df.empty:
        warnings.append("no_ohlcv_data_available")
        # All samples are deferred — we can't resolve without data
        return ResolutionResult(
            resolved=[], deferred=list(pending_samples),
            expired=[], errored=[],
            warnings=warnings, label_policy=lp, resolution_policy=rp,
        )

    # Ensure time column is datetime and sorted
    df = _prepare_ohlcv(ohlcv_df, warnings)
    if df is None:
        return ResolutionResult(
            resolved=[], deferred=list(pending_samples),
            expired=[], errored=[],
            warnings=warnings, label_policy=lp, resolution_policy=rp,
        )

    # Check for data gaps
    gap_issues = _check_data_gaps(df, rp)
    if gap_issues:
        warnings.extend(gap_issues)

    # ── Process each pending sample ──
    for sample in pending_samples:
        try:
            outcome = _resolve_single(sample, df, lp, rp, now_utc, data_interval=data_interval)
        except Exception as ex:
            errored.append({**sample, "_error": str(ex)})
            continue

        if outcome["status"] == "resolved":
            resolved.append(outcome["record"])
        elif outcome["status"] == "deferred":
            deferred.append(sample)
            if outcome.get("reason"):
                warnings.append(f"deferred:{sample.get('prediction_id','?')}:{outcome['reason']}")
        elif outcome["status"] == "expired":
            expired.append(sample)
        elif outcome["status"] == "error":
            errored.append({**sample, "_error": outcome.get("reason", "unknown")})

    return ResolutionResult(
        resolved=resolved, deferred=deferred,
        expired=expired, errored=errored,
        warnings=warnings, label_policy=lp, resolution_policy=rp,
    )


# ═══════════════════════════════════════════════════════════════════
# SINGLE SAMPLE RESOLUTION
# ═══════════════════════════════════════════════════════════════════

def _resolve_single(
    sample: dict,
    df: pd.DataFrame,
    lp: LabelPolicy,
    rp: ResolutionPolicy,
    now_utc: datetime,
    data_interval: str | None = None,
) -> dict:
    """
    Attempt to resolve one pending sample.

    Returns:
      {"status": "resolved"|"deferred"|"expired"|"error", "record": ..., "reason": ...}
    """
    prediction_id = sample.get("prediction_id", "")
    asof_str = sample.get("asof_utc", "")
    horizon_bars = sample.get("horizon_bars", 24)
    interval = sample.get("interval", "1h")
    entry_price = sample.get("entry_price")
    price_basis = sample.get("price_basis", "close")

    # ── Parse prediction timestamp ──
    asof_dt = _parse_utc(asof_str)
    if asof_dt is None:
        return {"status": "error", "reason": "invalid_asof_utc"}

    # ── Check expiration ──
    stored_str = sample.get("stored_utc", asof_str)
    stored_dt = _parse_utc(stored_str) or asof_dt
    age_hours = (now_utc - stored_dt).total_seconds() / 3600.0
    if age_hours > rp.max_pending_age_hours:
        return {"status": "expired", "reason": "max_age_exceeded"}

    # ── Temporal safety: reject if sample was not point-in-time safe ──
    if not sample.get("point_in_time_safe", True):
        # Still resolve for grading, but mark ineligible for training
        pass  # handled below in eligibility

    # ── Compute target bar timestamp ──
    bar_duration = _interval_to_timedelta(interval)
    if bar_duration is None:
        return {"status": "error", "reason": f"unsupported_interval:{interval}"}

    # Total horizon duration in real time (e.g. 24 × 1h = 24h)
    horizon_duration = bar_duration * horizon_bars
    target_bar_dt = asof_dt + horizon_duration

    # ── Determine effective data bar duration ──
    # When the loaded OHLCV data is coarser than the sample interval
    # (e.g. daily bars for hourly predictions), use the data interval
    # for tolerance and coverage calculations.
    effective_interval = data_interval or interval
    data_bar_duration = _interval_to_timedelta(effective_interval) or bar_duration

    # ── Find anchor and target bars in OHLCV ──
    # Use floor-based (backward-looking) matching: find the latest bar
    # whose timestamp is ≤ the target.  This correctly maps exact
    # heartbeat timestamps (e.g. 21:48:12) to their enclosing bar
    # (e.g. the daily bar at 00:00:00 for that day).
    anchor_idx = _find_floor_bar(df, asof_dt, tolerance=data_bar_duration)
    target_idx = _find_floor_bar(df, target_bar_dt, tolerance=data_bar_duration)

    if anchor_idx is None:
        return {"status": "deferred", "reason": "anchor_bar_not_found"}

    if target_idx is None:
        if rp.require_full_horizon:
            return {"status": "deferred", "reason": "target_bar_not_yet_available"}
        else:
            return {"status": "deferred", "reason": "target_bar_missing"}

    # ── Ensure target is *after* anchor (no look-ahead) ──
    if target_idx <= anchor_idx:
        return {"status": "error", "reason": "target_not_after_anchor"}

    # ── Check horizon coverage ──
    # Coverage is measured in *time* rather than bar count, so that
    # coarser data (e.g. 1 daily bar spanning 24h) correctly counts
    # as full coverage for a 24h horizon.
    anchor_time = df.iloc[anchor_idx]["time"]
    target_time = df.iloc[target_idx]["time"]
    time_covered = (target_time - anchor_time).total_seconds()
    horizon_seconds = horizon_duration.total_seconds()
    coverage = time_covered / max(horizon_seconds, 1.0)
    if coverage < rp.min_horizon_coverage:
        return {"status": "deferred", "reason": f"insufficient_coverage:{coverage:.2f}"}

    # ── Extract prices ──
    price_col = rp.price_column
    if price_col not in df.columns:
        price_col = "close"

    anchor_price = float(df.iloc[anchor_idx][price_col])
    target_price = float(df.iloc[target_idx][price_col])
    target_bar_utc = str(df.iloc[target_idx]["time"])

    # Fallback entry price from anchor if not provided
    if entry_price is None or entry_price <= 0:
        entry_price = anchor_price

    if entry_price <= 0:
        return {"status": "error", "reason": "invalid_entry_price"}

    # ── Compute realized return ──
    return_bps = ((target_price - entry_price) / entry_price) * 10_000.0

    # ── Assign label via policy ──
    label = lp.assign_label(return_bps)
    ambiguity = lp.compute_ambiguity(return_bps)
    move_strength = lp.compute_move_strength(return_bps)

    # ── Compute range for volatility sidecar ──
    horizon_slice = df.iloc[anchor_idx:target_idx + 1]
    realized_range_bps = None
    if len(horizon_slice) > 1 and "high" in df.columns and "low" in df.columns:
        horizon_high = float(horizon_slice["high"].max())
        horizon_low = float(horizon_slice["low"].min())
        if entry_price > 0:
            realized_range_bps = ((horizon_high - horizon_low) / entry_price) * 10_000.0

    # ── Sidecar grading ──
    directional_grade = grade_directional(
        predicted_class=sample.get("predicted_class", "neutral"),
        actual_label=label,
        predicted_probabilities=sample.get("predicted_probabilities", {}),
    )

    regime_grade = grade_regime(
        predicted_regime=sample.get("regime_probabilities", {}),
    )

    event_grade = grade_event_surface(
        predicted_events=sample.get("event_probabilities", {}),
        realized_volatility_bps=abs(return_bps) if return_bps is not None else None,
    )

    # ATR proxy from features if available (feature index 7 = atr_14 in v1)
    features = sample.get("features", [])
    entry_atr_bps = None
    if len(features) > 7 and entry_price > 0:
        atr_raw = features[7]  # atr_14
        if atr_raw and atr_raw > 0:
            entry_atr_bps = (atr_raw / entry_price) * 10_000.0

    vol_grade = grade_volatility_expansion(
        predicted_regime=sample.get("regime_probabilities", {}),
        realized_range_bps=realized_range_bps,
        entry_atr_bps=entry_atr_bps,
    )

    # ── Training eligibility ──
    eligible = sample.get("eligible_for_training", True)
    block_reasons = list(sample.get("learning_block_reasons", []))
    if not sample.get("point_in_time_safe", True):
        eligible = False
        if "temporal_safety_failed" not in block_reasons:
            block_reasons.append("temporal_safety_failed")

    # ── Build rich resolved record ──
    record: dict[str, Any] = {
        # A. Provenance
        "prediction_id": prediction_id,
        "model_key": sample.get("model_key", ""),
        "feature_version": sample.get("feature_version", ""),
        "temporal_policy_version": sample.get("temporal_policy_version", ""),
        "label_policy_version": lp.version,
        "resolution_policy_version": rp.version,
        "source_event_id": sample.get("source_event_id"),

        # B. Original belief
        "symbol": sample.get("symbol", ""),
        "horizon": sample.get("horizon", ""),
        "interval": sample.get("interval", ""),
        "asof_utc": asof_str,
        "predicted_class": sample.get("predicted_class", ""),
        "predicted_probabilities": sample.get("predicted_probabilities", {}),
        "regime_probabilities": sample.get("regime_probabilities", {}),
        "event_probabilities": sample.get("event_probabilities", {}),
        "priority_score": sample.get("priority_score", 0.0),
        "macro_tags": sample.get("macro_tags", []),
        "headline_tags": sample.get("headline_tags", []),
        "scheduled_event_types": sample.get("scheduled_event_types", []),
        "entry_price": entry_price,
        "price_basis": price_basis,
        "observation_cutoff_utc": sample.get("observation_cutoff_utc"),

        # C. Realized truth
        "anchor_bar_utc": str(df.iloc[anchor_idx]["time"]),
        "target_bar_utc": target_bar_utc,
        "target_price": round(target_price, 6),
        "realized_return_bps": round(return_bps, 2),
        "actual_label": label,
        "move_strength": move_strength,
        "ambiguity_score": round(ambiguity, 4),
        "horizon_bars_used": target_idx - anchor_idx,
        "realized_range_bps": round(realized_range_bps, 2) if realized_range_bps is not None else None,

        # D. Grading sidecars
        "directional_grade": directional_grade,
        "regime_grade": regime_grade,
        "event_grade": event_grade,
        "volatility_grade": vol_grade,

        # E. Learning eligibility
        "eligible_for_training": eligible,
        "learning_block_reasons": block_reasons,
        "replay_eligible": eligible,  # initially same as training eligible

        # F. Features (carried forward for training)
        "features": features,

        # G. Resolution metadata
        "resolution_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_without_lookahead": True,
    }

    return {"status": "resolved", "record": record}


# ═══════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════

def _prepare_ohlcv(df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame | None:
    """Validate and prepare OHLCV DataFrame for resolution."""
    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        warnings.append(f"missing_ohlcv_columns:{sorted(missing)}")
        return None

    df = df.copy()

    # Ensure time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        try:
            df["time"] = pd.to_datetime(df["time"], utc=True)
        except Exception:
            warnings.append("cannot_parse_time_column")
            return None

    # Ensure timezone-aware
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")

    df = df.sort_values("time").reset_index(drop=True)

    if df.empty:
        warnings.append("ohlcv_empty_after_prepare")
        return None

    return df


def _check_data_gaps(df: pd.DataFrame, rp: ResolutionPolicy) -> list[str]:
    """Check for significant gaps in the OHLCV time series."""
    issues = []
    if len(df) < 2:
        return issues

    times = df["time"]
    diffs = times.diff().dropna()

    if len(diffs) == 0:
        return issues

    median_gap = diffs.median()
    if median_gap.total_seconds() <= 0:
        return issues

    # Check for gaps larger than tolerance * median
    large_gaps = diffs[diffs > median_gap * (rp.gap_tolerance_bars + 1)]
    if len(large_gaps) > 0:
        issues.append(f"data_gaps_detected:{len(large_gaps)}")

    return issues


def _find_floor_bar(
    df: pd.DataFrame,
    target_dt: datetime,
    tolerance: timedelta | None = None,
) -> int | None:
    """
    Find the index of the latest bar whose timestamp is ≤ target_dt
    (backward-looking / floor match).

    This correctly maps exact prediction timestamps (e.g. 21:48:12 UTC)
    to their enclosing bar — for example, the daily bar at 00:00:00 for
    that same day, or the hourly bar at 21:00:00.

    Returns None if no bar is at or before target_dt within tolerance.
    """
    if df.empty:
        return None

    target_ts = pd.Timestamp(target_dt)
    if target_ts.tz is None:
        target_ts = target_ts.tz_localize("UTC")

    # Find all bars at or before the target timestamp
    mask = df["time"] <= target_ts
    if not mask.any():
        # No bar at or before target — check if the first bar is
        # within tolerance (data might start slightly after target)
        if tolerance is not None:
            first_diff = (df["time"].iloc[0] - target_ts)
            if first_diff <= tolerance:
                return 0
        return None

    # Latest bar ≤ target
    floor_idx = int(mask[::-1].idxmax())

    # Verify within tolerance if specified
    if tolerance is not None:
        gap = target_ts - df.iloc[floor_idx]["time"]
        if gap > tolerance:
            return None

    return floor_idx


def _find_nearest_bar(
    df: pd.DataFrame,
    target_dt: datetime,
    tolerance: timedelta | None = None,
) -> int | None:
    """
    Find the index of the bar nearest to target_dt.
    Returns None if no bar is within tolerance.

    Kept for backward compatibility — new resolution code uses _find_floor_bar.
    """
    if df.empty:
        return None

    target_ts = pd.Timestamp(target_dt)
    if target_ts.tz is None:
        target_ts = target_ts.tz_localize("UTC")

    time_diffs = (df["time"] - target_ts).abs()
    nearest_idx = int(time_diffs.idxmin())

    if tolerance is not None:
        if time_diffs.iloc[nearest_idx] > tolerance:
            return None

    return nearest_idx


def _interval_to_timedelta(interval: str) -> timedelta | None:
    """Convert interval string to timedelta."""
    mapping = {
        "1m": timedelta(minutes=1),
        "5m": timedelta(minutes=5),
        "15m": timedelta(minutes=15),
        "30m": timedelta(minutes=30),
        "1h": timedelta(hours=1),
        "2h": timedelta(hours=2),
        "4h": timedelta(hours=4),
        "6h": timedelta(hours=6),
        "8h": timedelta(hours=8),
        "12h": timedelta(hours=12),
        "1d": timedelta(days=1),
        "1w": timedelta(weeks=1),
    }
    return mapping.get(interval.lower())


def _parse_utc(s: str | None) -> datetime | None:
    """Parse an ISO-8601 UTC timestamp."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


# ═══════════════════════════════════════════════════════════════════
# RESOLUTION RESULT
# ═══════════════════════════════════════════════════════════════════

class ResolutionResult:
    """Container for the output of resolve_pending_batch."""

    def __init__(
        self,
        resolved: list[dict],
        deferred: list[dict],
        expired: list[dict],
        errored: list[dict],
        warnings: list[str],
        label_policy: LabelPolicy,
        resolution_policy: ResolutionPolicy,
    ):
        self.resolved = resolved
        self.deferred = deferred
        self.expired = expired
        self.errored = errored
        self.warnings = warnings
        self.label_policy = label_policy
        self.resolution_policy = resolution_policy

    @property
    def total_processed(self) -> int:
        return len(self.resolved) + len(self.deferred) + len(self.expired) + len(self.errored)

    def summary(self) -> dict:
        """Produce a summary dict suitable for JSON output."""
        # Class distribution of resolved samples
        class_dist: dict[str, int] = {}
        brier_scores: list[float] = []
        correct_count = 0
        for r in self.resolved:
            lbl = r.get("actual_label", "unknown")
            class_dist[lbl] = class_dist.get(lbl, 0) + 1
            dg = r.get("directional_grade", {})
            if dg.get("correct"):
                correct_count += 1
            bs = dg.get("brier_score")
            if bs is not None:
                brier_scores.append(bs)

        accuracy = correct_count / len(self.resolved) if self.resolved else 0.0
        mean_brier = sum(brier_scores) / len(brier_scores) if brier_scores else 0.0

        return {
            "total_processed": self.total_processed,
            "resolved_count": len(self.resolved),
            "deferred_count": len(self.deferred),
            "expired_count": len(self.expired),
            "errored_count": len(self.errored),
            "class_distribution": class_dist,
            "accuracy": round(accuracy, 4),
            "mean_brier_score": round(mean_brier, 6),
            "label_policy_version": self.label_policy.version,
            "resolution_policy_version": self.resolution_policy.version,
            "warnings": self.warnings,
        }
