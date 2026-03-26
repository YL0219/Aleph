"""
operational_status.py — Rich operational status for ML cortex observability.

Computes an honest operational snapshot that clearly distinguishes between:
  - Healthy waiting   (predictions pending maturity — NORMAL behavior)
  - Healthy idle       (no pending work, system is quiet)
  - Healthy progressing (recently resolved/trained)
  - Blocked            (schema mismatch, missing data)
  - Stalled            (mature predictions not processing)
  - Degraded / Broken  (partial or critical failures)

Core design principle: a deferred 1-day prediction should surface as
"waiting for maturity" — NOT "the pipeline is broken".  This is the
scoreboard honesty requirement.

Storage reads:
  data_lake/cortex/pending/{symbol}/{horizon}/pending.jsonl
  data_lake/cortex/resolved/{symbol}/{horizon}/resolved.jsonl
  data_lake/cortex/cursor/{symbol}/{horizon}/cursor.json
  data_lake/cortex/state/{symbol}/{horizon}/metadata.json
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import pending_memory, training_cursor, brain_state, scorecard, policies
from .feature_adapter import FEATURE_VERSION


# ═══════════════════════════════════════════════════════════════════
# PIPELINE STATE CONSTANTS
# ═══════════════════════════════════════════════════════════════════

class PipelineState:
    """Enum-like string constants for pipeline health states."""

    HEALTHY_WAITING = "healthy_waiting"
    HEALTHY_IDLE = "healthy_idle"
    HEALTHY_PROGRESSING = "healthy_progressing"
    BLOCKED_SCHEMA = "blocked_schema"
    BLOCKED_DATA = "blocked_data"
    STALLED = "stalled"
    DEGRADED = "degraded"
    BROKEN = "broken"


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _cortex_root() -> Path:
    ml_dir = Path(__file__).parent
    python_dir = ml_dir.parent
    aether_dir = python_dir.parent
    content_root = aether_dir.parent
    return content_root / "data_lake" / "cortex"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_utc(value: str | None) -> datetime | None:
    """Parse an ISO timestamp string to a timezone-aware UTC datetime."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _interval_to_hours(interval: str) -> float:
    """Convert an interval string to hours per bar."""
    mapping = {"1h": 1.0, "4h": 4.0, "1d": 24.0, "1w": 168.0}
    return mapping.get(interval, 1.0)


def _safe_round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    try:
        return round(value, digits)
    except (TypeError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════════
# MATURITY TIMELINE
# ═══════════════════════════════════════════════════════════════════

def compute_maturity_timeline(
    pending_samples: list[dict],
    interval: str,
    horizon_bars: int,
) -> dict[str, Any]:
    """
    For each pending sample, compute time until horizon maturity.

    A prediction made at asof_utc matures at:
        asof_utc + (horizon_bars * hours_per_bar)

    Returns:
        next_maturity_utc   — ISO string of soonest maturity
        mature_count        — predictions whose horizon has elapsed
        immature_count      — predictions still waiting
        avg_hours           — average hours to maturity for immature predictions
        oldest_pending_utc  — oldest asof_utc in the pending set
    """
    if not pending_samples:
        return {
            "next_maturity_utc": None,
            "mature_count": 0,
            "immature_count": 0,
            "avg_hours_to_maturity": None,
            "oldest_pending_utc": None,
        }

    now = _now_utc()
    hours_per_bar = _interval_to_hours(interval)
    maturity_delta = timedelta(hours=horizon_bars * hours_per_bar)

    mature_count = 0
    immature_count = 0
    immature_hours: list[float] = []
    next_maturity: datetime | None = None
    oldest_asof: datetime | None = None

    for sample in pending_samples:
        asof = _parse_utc(sample.get("asof_utc"))
        if asof is None:
            continue

        # Track oldest pending
        if oldest_asof is None or asof < oldest_asof:
            oldest_asof = asof

        maturity_time = asof + maturity_delta

        if maturity_time <= now:
            mature_count += 1
        else:
            immature_count += 1
            hours_left = (maturity_time - now).total_seconds() / 3600.0
            immature_hours.append(hours_left)

            if next_maturity is None or maturity_time < next_maturity:
                next_maturity = maturity_time

    avg_hours = None
    if immature_hours:
        avg_hours = sum(immature_hours) / len(immature_hours)

    return {
        "next_maturity_utc": next_maturity.isoformat() if next_maturity else None,
        "mature_count": mature_count,
        "immature_count": immature_count,
        "avg_hours_to_maturity": _safe_round(avg_hours, 2),
        "oldest_pending_utc": oldest_asof.isoformat() if oldest_asof else None,
    }


# ═══════════════════════════════════════════════════════════════════
# SCHEMA HEALTH
# ═══════════════════════════════════════════════════════════════════

def assess_schema_health(
    pending_samples: list[dict],
    resolved_samples: list[dict],
    model_feature_version: str,
) -> dict[str, Any]:
    """
    Scan feature_version tags on pending and resolved samples,
    compare to the current model feature version.

    Returns:
        current_version   — the model's active feature version
        versions_found    — dict mapping version -> count across all samples
        mismatch_count    — number of samples with a different version
        mismatch_details  — list of version strings that don't match
        all_compatible    — True if every sample matches or no samples exist
    """
    versions_found: dict[str, int] = {}

    for sample in pending_samples:
        v = sample.get("feature_version", "")
        if v:
            versions_found[v] = versions_found.get(v, 0) + 1

    for sample in resolved_samples:
        v = sample.get("feature_version", "")
        if v:
            versions_found[v] = versions_found.get(v, 0) + 1

    mismatch_versions = [
        v for v in versions_found if v != model_feature_version
    ]
    mismatch_count = sum(versions_found.get(v, 0) for v in mismatch_versions)

    all_compatible = len(mismatch_versions) == 0

    return {
        "current_version": model_feature_version,
        "versions_found": versions_found,
        "mismatch_count": mismatch_count,
        "mismatch_details": mismatch_versions,
        "all_compatible": all_compatible,
    }


# ═══════════════════════════════════════════════════════════════════
# OPERATIONAL SNAPSHOT
# ═══════════════════════════════════════════════════════════════════

def compute_operational_snapshot(
    symbol: str,
    horizon: str,
    interval: str,
) -> dict[str, Any]:
    """
    Compute the full operational snapshot for a symbol/horizon pair.

    Loads all relevant state (pending, resolved, cursor, model, scorecard)
    and determines the pipeline state using a clear priority ladder.

    Returns an OperationalSnapshot dict.
    """
    now = _now_utc()
    errors: list[str] = []

    # ── Load pending samples ──
    try:
        pending = pending_memory.load_pending_samples(symbol, horizon)
    except Exception as ex:
        errors.append(f"pending_load_failed: {ex}")
        pending = []

    # ── Load resolved samples ──
    try:
        resolved = pending_memory.load_resolved_samples(symbol, horizon)
    except Exception as ex:
        errors.append(f"resolved_load_failed: {ex}")
        resolved = []

    # ── Load training cursor ──
    try:
        cursor = training_cursor.load_cursor(symbol, horizon)
    except Exception as ex:
        errors.append(f"cursor_load_failed: {ex}")
        cursor = training_cursor.TrainingCursor(
            symbol=symbol.upper(), horizon=horizon,
        )

    # ── Load model state ──
    try:
        model = brain_state.load_model(symbol, horizon)
        model_meta = model.get_state_dict()
    except Exception as ex:
        errors.append(f"model_load_failed: {ex}")
        model_meta = {
            "trained_samples": 0,
            "model_version": "unknown",
            "model_state": "cold_start",
            "fitted": False,
            "scaler_fitted": False,
        }

    # ── Derive horizon_bars from samples or default ──
    horizon_bars = 24  # default for 1h/24-bar or 1d/1-bar
    if pending:
        first_hb = pending[0].get("horizon_bars")
        if isinstance(first_hb, int) and first_hb > 0:
            horizon_bars = first_hb

    # ── Maturity timeline ──
    maturity = compute_maturity_timeline(pending, interval, horizon_bars)

    # ── Schema health ──
    schema = assess_schema_health(pending, resolved, FEATURE_VERSION)

    # ── Pending summary ──
    total_pending = len(pending)
    eligible_pending = sum(
        1 for s in pending if s.get("eligible_for_training", True)
    )
    blocked_pending = total_pending - eligible_pending

    pending_summary = {
        "total": total_pending,
        "eligible": eligible_pending,
        "blocked": blocked_pending,
        "mature": maturity["mature_count"],
        "immature": maturity["immature_count"],
    }

    # ── Resolved summary ──
    total_resolved = len(resolved)
    recent_24h = 0
    cutoff_24h = now - timedelta(hours=24)
    for s in resolved:
        res_time = _parse_utc(s.get("resolution_utc"))
        if res_time and res_time >= cutoff_24h:
            recent_24h += 1

    # Accuracy from last 50 resolved
    last_50 = resolved[-50:] if len(resolved) > 50 else resolved
    accuracy_last_50 = None
    if last_50:
        correct = sum(
            1 for s in last_50
            if s.get("predicted_class") == s.get("actual_label", s.get("label"))
        )
        accuracy_last_50 = round(correct / len(last_50), 4)

    resolved_summary = {
        "total": total_resolved,
        "recent_resolved_24h": recent_24h,
        "accuracy_last_50": accuracy_last_50,
    }

    # ── Cursor summary ──
    hours_since_last_train = None
    if cursor.last_train_utc:
        last_train_dt = _parse_utc(cursor.last_train_utc)
        if last_train_dt:
            hours_since_last_train = round(
                (now - last_train_dt).total_seconds() / 3600.0, 2,
            )

    cursor_summary = {
        "sequence": cursor.sequence,
        "last_train_utc": cursor.last_train_utc,
        "hours_since_last_train": hours_since_last_train,
        "total_ever": cursor.total_samples_ever,
    }

    # ── Model identity ──
    model_identity = {
        "model_key": f"{symbol.upper()}_{horizon}",
        "feature_version": FEATURE_VERSION,
        "model_state": model_meta.get("model_state", "cold_start"),
        "trained_samples": model_meta.get("trained_samples", 0),
    }

    # ── Scorecard summary ──
    scorecard_summary: dict[str, Any] = {
        "brier": None,
        "accuracy": None,
        "grade": "insufficient_data",
        "drift_detected": False,
        "warnings": [],
    }
    if len(resolved) >= 10:
        try:
            sc = scorecard.compute_rolling_scorecard(resolved)
            scorecard_summary = {
                "brier": sc.get("mean_brier_score"),
                "accuracy": sc.get("accuracy"),
                "grade": _grade_from_accuracy(sc.get("accuracy")),
                "drift_detected": sc.get("drift", {}).get("detected", False),
                "warnings": sc.get("warnings", []),
            }
        except Exception as ex:
            errors.append(f"scorecard_failed: {ex}")

    # ── Training readiness ──
    fresh_ids = cursor.get_unconsumed(
        [s.get("prediction_id", "") for s in resolved if s.get("prediction_id")]
    )
    has_fresh = len(fresh_ids) > 0
    tp = policies.DEFAULT_TRAINING_POLICY

    block_reasons: list[str] = []
    if not has_fresh:
        block_reasons.append("no_fresh_resolved_samples")
    if len(fresh_ids) < tp.min_samples_to_train:
        block_reasons.append(
            f"below_min_samples({len(fresh_ids)}<{tp.min_samples_to_train})"
        )
    if not schema["all_compatible"]:
        block_reasons.append(
            f"schema_mismatch({schema['mismatch_count']} samples)"
        )

    gate_status = "open" if not block_reasons else "blocked"

    training_readiness = {
        "has_fresh_samples": has_fresh,
        "fresh_sample_count": len(fresh_ids),
        "gate_status": gate_status,
        "block_reasons": block_reasons,
    }

    # ── Determine pipeline state ──
    pipeline_state, state_reason = _determine_pipeline_state(
        errors=errors,
        schema=schema,
        pending_summary=pending_summary,
        maturity=maturity,
        resolved_summary=resolved_summary,
        cursor_summary=cursor_summary,
        hours_since_last_train=hours_since_last_train,
        interval=interval,
    )

    # ── State since (best effort) ──
    state_since_utc: str | None = None
    if pipeline_state == PipelineState.HEALTHY_PROGRESSING and cursor.last_train_utc:
        state_since_utc = cursor.last_train_utc

    return {
        "pipeline_state": pipeline_state,
        "state_reason": state_reason,
        "state_since_utc": state_since_utc,
        "snapshot_utc": now.isoformat(),
        "symbol": symbol.upper(),
        "horizon": horizon,
        "model_identity": model_identity,
        "pending_summary": pending_summary,
        "maturity_timeline": maturity,
        "resolved_summary": resolved_summary,
        "cursor_summary": cursor_summary,
        "schema_health": schema,
        "scorecard_summary": scorecard_summary,
        "training_readiness": training_readiness,
        "errors": errors,
    }


def _determine_pipeline_state(
    *,
    errors: list[str],
    schema: dict[str, Any],
    pending_summary: dict[str, Any],
    maturity: dict[str, Any],
    resolved_summary: dict[str, Any],
    cursor_summary: dict[str, Any],
    hours_since_last_train: float | None,
    interval: str,
) -> tuple[str, str]:
    """
    Priority ladder for pipeline state determination.

    Returns (state, reason) tuple.
    """
    # 1. Critical errors in loading state → BROKEN
    if errors:
        return (
            PipelineState.BROKEN,
            f"Critical errors loading state: {'; '.join(errors)}",
        )

    total_pending = pending_summary["total"]
    total_resolved = resolved_summary["total"]
    mature_count = maturity["mature_count"]
    immature_count = maturity["immature_count"]

    # 2. Schema mismatches block all training → BLOCKED_SCHEMA
    if not schema["all_compatible"] and schema["mismatch_count"] > 0:
        pct = schema["mismatch_count"]
        details = ", ".join(schema["mismatch_details"])
        return (
            PipelineState.BLOCKED_SCHEMA,
            f"Schema mismatch: {pct} samples have incompatible versions [{details}]. "
            f"Current model expects {schema['current_version']}.",
        )

    # 3. No pending and no resolved → HEALTHY_IDLE
    if total_pending == 0 and total_resolved == 0:
        return (
            PipelineState.HEALTHY_IDLE,
            "No pending predictions and no resolved history. System is idle.",
        )

    # 4. Pending exist but none are mature → HEALTHY_WAITING
    if total_pending > 0 and mature_count == 0 and immature_count > 0:
        avg_h = maturity.get("avg_hours_to_maturity")
        next_m = maturity.get("next_maturity_utc", "unknown")
        avg_str = f"~{avg_h:.0f}h" if avg_h is not None else "unknown"
        return (
            PipelineState.HEALTHY_WAITING,
            f"{immature_count} predictions waiting for horizon maturity. "
            f"Next maturity: {next_m}. Average wait: {avg_str}.",
        )

    # 5. Recently resolved or recently trained → HEALTHY_PROGRESSING
    recent_24h = resolved_summary.get("recent_resolved_24h", 0)
    recently_trained = (
        hours_since_last_train is not None and hours_since_last_train < 48.0
    )

    if recent_24h > 0 or recently_trained:
        parts: list[str] = []
        if recent_24h > 0:
            parts.append(f"resolved {recent_24h} in last 24h")
        if recently_trained:
            parts.append(
                f"last training {hours_since_last_train:.1f}h ago"
            )
        return (
            PipelineState.HEALTHY_PROGRESSING,
            f"Pipeline progressing: {', '.join(parts)}.",
        )

    # 6. Mature predictions sitting unresolved for extended period → STALLED
    # "Extended period" = more than 2 cycle intervals worth of hours
    hours_per_bar = _interval_to_hours(interval)
    stall_threshold_hours = hours_per_bar * 2
    if stall_threshold_hours < 4.0:
        stall_threshold_hours = 4.0

    if mature_count > 0:
        # If we have mature predictions but haven't resolved recently
        # and haven't trained recently, we're stalled
        if recent_24h == 0 and (
            hours_since_last_train is None
            or hours_since_last_train > stall_threshold_hours
        ):
            return (
                PipelineState.STALLED,
                f"{mature_count} mature predictions not resolving. "
                f"Last training: "
                f"{'never' if hours_since_last_train is None else f'{hours_since_last_train:.1f}h ago'}. "
                f"Check OHLCV data availability.",
            )

    # 7. Mixed state: some mature, some immature, not clearly progressing
    if total_pending > 0 and total_resolved == 0:
        return (
            PipelineState.HEALTHY_WAITING,
            f"{total_pending} predictions pending. "
            f"{mature_count} mature, {immature_count} immature. "
            f"Awaiting first resolution cycle.",
        )

    # 8. Otherwise → DEGRADED
    return (
        PipelineState.DEGRADED,
        f"Pipeline in mixed state: {total_pending} pending "
        f"({mature_count} mature), {total_resolved} resolved, "
        f"last train: "
        f"{'never' if hours_since_last_train is None else f'{hours_since_last_train:.1f}h ago'}.",
    )


def _grade_from_accuracy(accuracy: float | None) -> str:
    """Map accuracy to a letter grade for quick assessment."""
    if accuracy is None:
        return "insufficient_data"
    if accuracy >= 0.60:
        return "A"
    if accuracy >= 0.50:
        return "B"
    if accuracy >= 0.40:
        return "C"
    if accuracy >= 0.30:
        return "D"
    return "F"


# ═══════════════════════════════════════════════════════════════════
# FORMAT FOR OUTPUT
# ═══════════════════════════════════════════════════════════════════

def format_operational_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """
    Format an operational snapshot for JSON output.

    Rounds floats, converts datetimes to ISO strings, and adds
    a human_summary one-liner.
    """
    state = snapshot.get("pipeline_state", PipelineState.BROKEN)
    reason = snapshot.get("state_reason", "")

    human_summary = _build_human_summary(snapshot)

    formatted: dict[str, Any] = {
        "pipeline_state": state,
        "state_reason": reason,
        "state_since_utc": snapshot.get("state_since_utc"),
        "snapshot_utc": snapshot.get("snapshot_utc"),
        "human_summary": human_summary,
        "symbol": snapshot.get("symbol"),
        "horizon": snapshot.get("horizon"),
        "model_identity": snapshot.get("model_identity", {}),
        "pending_summary": snapshot.get("pending_summary", {}),
        "maturity_timeline": snapshot.get("maturity_timeline", {}),
        "resolved_summary": snapshot.get("resolved_summary", {}),
        "cursor_summary": snapshot.get("cursor_summary", {}),
        "schema_health": _format_schema_health(
            snapshot.get("schema_health", {}),
        ),
        "scorecard_summary": snapshot.get("scorecard_summary", {}),
        "training_readiness": snapshot.get("training_readiness", {}),
    }

    if snapshot.get("errors"):
        formatted["errors"] = snapshot["errors"]

    return formatted


def _format_schema_health(schema: dict[str, Any]) -> dict[str, Any]:
    """Simplify schema health for JSON output."""
    return {
        "current_version": schema.get("current_version", ""),
        "versions_found": schema.get("versions_found", {}),
        "mismatch_count": schema.get("mismatch_count", 0),
        "mismatch_details": schema.get("mismatch_details", []),
        "all_compatible": schema.get("all_compatible", True),
    }


def _build_human_summary(snapshot: dict[str, Any]) -> str:
    """Build a one-line plain English summary of the current pipeline state."""
    state = snapshot.get("pipeline_state", "")
    pending_s = snapshot.get("pending_summary", {})
    maturity = snapshot.get("maturity_timeline", {})
    resolved_s = snapshot.get("resolved_summary", {})
    cursor_s = snapshot.get("cursor_summary", {})
    model_id = snapshot.get("model_identity", {})
    training = snapshot.get("training_readiness", {})

    total_pending = pending_s.get("total", 0)
    mature = pending_s.get("mature", 0)
    immature = pending_s.get("immature", 0)
    horizon = snapshot.get("horizon", "?")
    model_state = model_id.get("model_state", "unknown")

    if state == PipelineState.HEALTHY_IDLE:
        return "Idle: no predictions pending or resolved. Waiting for market data."

    if state == PipelineState.HEALTHY_WAITING:
        avg_h = maturity.get("avg_hours_to_maturity")
        if model_state == "cold_start":
            eta = f" First maturity in ~{avg_h:.0f}h." if avg_h else ""
            return (
                f"Waiting: cold start model, {total_pending} predictions "
                f"pending.{eta}"
            )
        eta = f" Next resolution in ~{avg_h:.0f}h." if avg_h else ""
        return (
            f"Healthy: {total_pending} predictions waiting for "
            f"{horizon} horizon maturity.{eta}"
        )

    if state == PipelineState.HEALTHY_PROGRESSING:
        recent = resolved_s.get("recent_resolved_24h", 0)
        gate = training.get("gate_status", "unknown")
        parts: list[str] = []
        if recent > 0:
            parts.append(f"resolved {recent} predictions this cycle")
        if gate == "open":
            parts.append("training gate open")
        elif cursor_s.get("hours_since_last_train") is not None:
            h = cursor_s["hours_since_last_train"]
            parts.append(f"trained {h:.0f}h ago")
        return f"Progressing: {', '.join(parts)}." if parts else "Progressing."

    if state == PipelineState.BLOCKED_SCHEMA:
        schema = snapshot.get("schema_health", {})
        mc = schema.get("mismatch_count", 0)
        return f"Blocked: schema mismatch on {mc} samples. Training halted."

    if state == PipelineState.BLOCKED_DATA:
        return "Blocked: OHLCV data missing for resolution. Check data ingestion."

    if state == PipelineState.STALLED:
        return (
            f"Stalled: {mature} mature predictions not resolving. "
            f"Check OHLCV data availability."
        )

    if state == PipelineState.BROKEN:
        errs = snapshot.get("errors", [])
        first_err = errs[0] if errs else "unknown error"
        return f"Broken: {first_err}"

    # DEGRADED or unknown
    return (
        f"Degraded: {total_pending} pending ({mature} mature), "
        f"{resolved_s.get('total', 0)} resolved."
    )
