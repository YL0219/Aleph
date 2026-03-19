"""
cortex_dashboard.py — Compact CLI diagnostic for the Aleph ML Cortex.

A single command that renders the full operational picture:
  - Pending memory status (count, age, eligibility)
  - Resolved archive status (count, class distribution)
  - Training cursor state (sequence, consumed, last train)
  - Model state (cold/warming/active, samples)
  - Rolling scorecard summary (Brier, accuracy, drift)
  - Last challenger evaluation summary (if available)
  - Organism health assessment

Usage:
  python cortex_dashboard.py --symbol BTCUSDT --horizon 1d
  python cortex_dashboard.py --symbol BTCUSDT --horizon 1d --full
  python cortex_dashboard.py --symbol BTCUSDT --horizon 1d --evaluate

Flags:
  --full       Show calibration curve and full scorecard detail
  --evaluate   Run challenger evaluation and show results
  --json       Output raw JSON instead of formatted text
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure the ml package is importable
_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

from ml.pending_memory import (
    load_pending_samples, load_resolved_samples,
    pending_count, pending_eligible_count, pending_blocked_count,
    resolved_count,
)
from ml.training_cursor import load_cursor
from ml.brain_state import load_model
from ml.scorecard import compute_rolling_scorecard, DEFAULT_SCORECARD_POLICY
from ml.policies import get_active_policies


# ═══════════════════════════════════════════════════════════════════
# TERMINAL FORMATTING
# ═══════════════════════════════════════════════════════════════════

_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_MAGENTA = "\033[35m"
_WHITE = "\033[37m"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}"


def _header(title: str) -> str:
    line = "─" * 60
    return f"\n{_color(line, _DIM)}\n{_color(f'  {title}', _BOLD + _CYAN)}\n{_color(line, _DIM)}"


def _kv(key: str, value, warn: bool = False, good: bool = False) -> str:
    color = _RED if warn else (_GREEN if good else _WHITE)
    return f"  {_color(key + ':', _DIM)}  {_color(str(value), color)}"


def _bar(value: float, width: int = 20, label: str = "") -> str:
    """Render a simple horizontal bar chart."""
    filled = int(value * width)
    bar = "█" * filled + "░" * (width - filled)
    pct = f"{value:.0%}"
    return f"  {bar} {pct} {_color(label, _DIM)}"


def _health_indicator(level: str) -> str:
    if level == "healthy":
        return _color("● HEALTHY", _GREEN + _BOLD)
    elif level == "degraded":
        return _color("● DEGRADED", _YELLOW + _BOLD)
    elif level == "critical":
        return _color("● CRITICAL", _RED + _BOLD)
    return _color("● UNKNOWN", _DIM)


# ═══════════════════════════════════════════════════════════════════
# DATA COLLECTORS
# ═══════════════════════════════════════════════════════════════════

def collect_pending_status(symbol: str, horizon: str) -> dict:
    """Gather pending memory statistics."""
    samples = load_pending_samples(symbol, horizon, max_samples=50000)
    total = len(samples)
    eligible = sum(1 for s in samples if s.get("eligible_for_training", False))
    blocked = total - eligible

    # Age analysis
    now = datetime.now(timezone.utc)
    ages_hours = []
    for s in samples:
        asof = s.get("asof_utc", "")
        try:
            dt = datetime.fromisoformat(asof.replace("Z", "+00:00"))
            age = (now - dt).total_seconds() / 3600.0
            ages_hours.append(age)
        except (ValueError, TypeError):
            pass

    oldest_hours = max(ages_hours) if ages_hours else 0
    newest_hours = min(ages_hours) if ages_hours else 0
    median_hours = sorted(ages_hours)[len(ages_hours) // 2] if ages_hours else 0

    return {
        "total": total,
        "eligible": eligible,
        "blocked": blocked,
        "oldest_hours": round(oldest_hours, 1),
        "newest_hours": round(newest_hours, 1),
        "median_hours": round(median_hours, 1),
    }


def collect_resolved_status(symbol: str, horizon: str) -> dict:
    """Gather resolved archive statistics."""
    total = resolved_count(symbol, horizon)
    samples = load_resolved_samples(symbol, horizon) if total > 0 else []

    from collections import Counter
    label_dist = dict(Counter(s.get("actual_label", "?") for s in samples))
    pred_dist = dict(Counter(s.get("predicted_class", "?") for s in samples))

    # Accuracy from directional grades
    correct = sum(
        1 for s in samples
        if s.get("directional_grade", {}).get("correct", False)
    )
    accuracy = correct / len(samples) if samples else 0

    return {
        "total": total,
        "label_distribution": label_dist,
        "predicted_distribution": pred_dist,
        "archive_accuracy": round(accuracy, 4),
    }


def collect_cursor_status(symbol: str, horizon: str) -> dict:
    """Gather training cursor state."""
    cursor = load_cursor(symbol, horizon)
    return {
        "sequence": cursor.sequence,
        "consumed_count": len(cursor.consumed_ids),
        "last_train_utc": cursor.last_train_utc or "never",
        "total_samples_ever": cursor.total_samples_ever,
        "last_policy": cursor.last_train_policy or "none",
    }


def collect_model_status(symbol: str, horizon: str) -> dict:
    """Gather model state."""
    model = load_model(symbol, horizon)
    class_dist = model.class_distribution() if hasattr(model, "class_distribution") else {}
    return {
        "state": model.model_state,
        "version": model.model_version,
        "trained_samples": model.trained_samples,
        "class_distribution": class_dist,
    }


def collect_scorecard(symbol: str, horizon: str) -> dict | None:
    """Compute rolling scorecard if enough resolved data."""
    samples = load_resolved_samples(symbol, horizon)
    if not samples:
        return None
    return compute_rolling_scorecard(samples, DEFAULT_SCORECARD_POLICY)


def assess_health(
    pending: dict,
    resolved: dict,
    cursor: dict,
    model: dict,
    scorecard: dict | None,
) -> dict:
    """Produce a high-level health assessment."""
    issues = []
    level = "healthy"

    # Model state checks
    if model["state"] == "cold_start":
        issues.append("model_cold_start")
        level = "degraded"

    if model["trained_samples"] == 0:
        issues.append("no_training_ever")

    # Pending backlog
    if pending["total"] > 100:
        issues.append(f"large_pending_backlog:{pending['total']}")
        level = "degraded"

    if pending["oldest_hours"] > 168:  # 7 days
        issues.append(f"stale_pending:{pending['oldest_hours']:.0f}h")
        level = "degraded"

    # Cursor stall
    if cursor["sequence"] == 0 and resolved["total"] > 10:
        issues.append("cursor_never_advanced")
        level = "degraded"

    # Scorecard warnings
    if scorecard and scorecard.get("status") == "ok":
        brier = scorecard.get("mean_brier_score", 0)
        acc = scorecard.get("accuracy", 0)
        if brier > 0.4:
            issues.append(f"high_brier:{brier:.3f}")
            level = "degraded"
        if acc < 0.25:
            issues.append(f"very_low_accuracy:{acc:.3f}")
            level = "critical"
        if scorecard.get("drift", {}).get("detected"):
            issues.append("drift_detected")
            level = "degraded"
        warnings = scorecard.get("warnings", [])
        if any("class_collapse" in w or "never_predicted" in w for w in warnings):
            issues.append("class_stability_concern")
            level = "degraded"

    # Resolved archive empty
    if resolved["total"] == 0:
        issues.append("no_resolved_samples")
        if model["trained_samples"] == 0:
            level = "degraded"

    return {"level": level, "issues": issues}


# ═══════════════════════════════════════════════════════════════════
# RENDERERS
# ═══════════════════════════════════════════════════════════════════

def render_dashboard(
    symbol: str,
    horizon: str,
    pending: dict,
    resolved: dict,
    cursor: dict,
    model: dict,
    scorecard: dict | None,
    health: dict,
    full: bool = False,
) -> str:
    """Render the full dashboard as a formatted string."""
    lines = []
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines.append("")
    lines.append(_color(f"  ALEPH CORTEX DASHBOARD — {symbol}/{horizon}", _BOLD + _MAGENTA))
    lines.append(_color(f"  {now_str}", _DIM))
    lines.append(f"  Health: {_health_indicator(health['level'])}")
    if health["issues"]:
        for issue in health["issues"]:
            lines.append(f"    {_color('!', _YELLOW)} {issue}")

    # ── Model ──
    lines.append(_header("Model"))
    state_color = _GREEN if model["state"] == "active" else (_YELLOW if model["state"] == "warming" else _RED)
    lines.append(f"  State:    {_color(model['state'], state_color)}")
    lines.append(_kv("Version", model["version"]))
    lines.append(_kv("Trained", f"{model['trained_samples']} samples"))
    if model["class_distribution"]:
        dist = model["class_distribution"]
        lines.append(f"  Classes:  {_color(json.dumps(dist), _DIM)}")

    # ── Pending Memory ──
    lines.append(_header("Pending Memory"))
    lines.append(_kv("Total", pending["total"],
                      warn=pending["total"] > 100))
    lines.append(_kv("Eligible", pending["eligible"],
                      good=pending["eligible"] > 0))
    lines.append(_kv("Blocked", pending["blocked"],
                      warn=pending["blocked"] > pending["eligible"]))
    if pending["total"] > 0:
        lines.append(_kv("Age (oldest)", f"{pending['oldest_hours']}h",
                          warn=pending["oldest_hours"] > 168))
        lines.append(_kv("Age (newest)", f"{pending['newest_hours']}h"))
        lines.append(_kv("Age (median)", f"{pending['median_hours']}h"))

    # ── Resolved Archive ──
    lines.append(_header("Resolved Archive"))
    lines.append(_kv("Total", resolved["total"],
                      good=resolved["total"] > 30))
    if resolved["total"] > 0:
        lines.append(_kv("Archive accuracy", f"{resolved['archive_accuracy']:.1%}",
                          warn=resolved["archive_accuracy"] < 0.3,
                          good=resolved["archive_accuracy"] >= 0.5))
        lines.append(f"  Labels:      {_color(json.dumps(resolved['label_distribution']), _DIM)}")
        lines.append(f"  Predictions: {_color(json.dumps(resolved['predicted_distribution']), _DIM)}")

    # ── Training Cursor ──
    lines.append(_header("Training Cursor"))
    lines.append(_kv("Sequence", cursor["sequence"],
                      good=cursor["sequence"] > 0))
    lines.append(_kv("Consumed IDs", cursor["consumed_count"]))
    lines.append(_kv("Total ever trained", cursor["total_samples_ever"]))
    lines.append(_kv("Last train", cursor["last_train_utc"]))
    lines.append(_kv("Last policy", cursor["last_policy"]))

    # ── Rolling Scorecard ──
    lines.append(_header("Rolling Scorecard"))
    if scorecard is None or scorecard.get("status") == "insufficient_data":
        lines.append(_kv("Status", "insufficient data", warn=True))
        min_req = scorecard.get("min_required", "?") if scorecard else "?"
        lines.append(_kv("Minimum required", min_req))
    else:
        brier = scorecard["mean_brier_score"]
        acc = scorecard["accuracy"]
        lines.append(_kv("Window", f"{scorecard.get('window_actual', '?')}/{scorecard.get('window_size', '?')} samples"))
        lines.append(_kv("Brier score", f"{brier:.4f}",
                          warn=brier > 0.35, good=brier < 0.25))
        lines.append(_kv("Accuracy", f"{acc:.1%}",
                          warn=acc < 0.3, good=acc >= 0.5))
        lines.append(_kv("Calibration gap", f"{scorecard.get('mean_calibration_gap', 0):.4f}"))

        # Grade buckets
        buckets = scorecard.get("grade_buckets", {})
        if buckets:
            total_b = sum(buckets.values())
            for bucket, count in buckets.items():
                pct = count / total_b if total_b > 0 else 0
                lines.append(f"    {bucket:20s} {count:4d}  ({pct:.0%})")

        # Streak
        streak = scorecard.get("current_streak", {})
        if streak.get("length", 0) > 0:
            streak_color = _GREEN if streak["type"] == "correct" else _RED
            lines.append(f"  Streak:  {_color(f'{streak[\"type\"]} x{streak[\"length\"]}', streak_color)}")

        # Drift
        drift = scorecard.get("drift", {})
        if drift.get("detected"):
            lines.append(f"  {_color('DRIFT DETECTED', _RED + _BOLD)}: {', '.join(drift.get('flags', []))}")
            lines.append(f"    Brier shift:    {drift.get('brier_shift', 0):+.4f}")
            lines.append(f"    Accuracy shift: {drift.get('accuracy_shift', 0):+.4f}")

        # Warnings
        warnings = scorecard.get("warnings", [])
        if warnings:
            lines.append(f"  Warnings ({len(warnings)}):")
            for w in warnings:
                lines.append(f"    {_color('!', _YELLOW)} {w}")

        # Full mode: calibration curve
        if full and scorecard.get("calibration"):
            lines.append("")
            lines.append(_color("  Calibration Curve:", _BOLD))
            lines.append(f"  {'Bin':>12s}  {'Count':>5s}  {'Predicted':>9s}  {'Actual':>7s}  {'Gap':>5s}")
            for b in scorecard["calibration"]:
                bin_label = f"{b['bin_lower']:.2f}-{b['bin_upper']:.2f}"
                lines.append(
                    f"  {bin_label:>12s}  {b['count']:5d}  "
                    f"{b['mean_predicted_prob']:9.4f}  {b['actual_hit_rate']:7.4f}  {b['gap']:5.4f}"
                )

    # ── Active Policies ──
    lines.append(_header("Active Policies"))
    policies = get_active_policies()
    for name, pol in policies.items():
        version = pol.get("version", "?") if isinstance(pol, dict) else "?"
        lines.append(_kv(name, version))

    lines.append("")
    lines.append(_color("─" * 60, _DIM))
    return "\n".join(lines)


def render_evaluation_summary(eval_result: dict) -> str:
    """Render challenger evaluation results."""
    lines = []
    lines.append(_header("Challenger Evaluation"))

    if not eval_result.get("ok", False) and "error" in eval_result:
        lines.append(_kv("Error", eval_result["error"], warn=True))
        return "\n".join(lines)

    evaluation = eval_result.get("evaluation", eval_result)

    lines.append(_kv("Samples", evaluation.get("sample_count", "?")))
    lines.append(_kv("Challengers", evaluation.get("challengers_evaluated", "?")))

    summary = evaluation.get("summary", {})
    lines.append(_kv("Promote", summary.get("promote", 0),
                      good=summary.get("promote", 0) > 0))
    lines.append(_kv("Reject", summary.get("reject", 0)))
    lines.append(_kv("Inconclusive", summary.get("inconclusive", 0)))

    best = summary.get("best_challenger")
    if best:
        lines.append(_kv("Best challenger", best, good=True))

    # Per-challenger results
    comparisons = evaluation.get("comparisons", [])
    for comp in comparisons:
        name = comp.get("challenger_name", "?")
        decision = comp.get("promotion_decision", {})
        dec = decision.get("decision", "?")
        dec_color = _GREEN if dec == "promote" else (_RED if dec == "reject" else _YELLOW)

        lines.append("")
        lines.append(f"  {_color(name, _BOLD)}  →  {_color(dec.upper(), dec_color)}")

        delta = comp.get("delta", {})
        brier_diff = delta.get("brier_score_diff")
        acc_diff = delta.get("accuracy_diff")
        if brier_diff is not None:
            improvement = -brier_diff
            lines.append(f"    Brier improvement: {_color(f'{improvement:+.4f}', _GREEN if improvement > 0 else _RED)}")
        if acc_diff is not None:
            lines.append(f"    Accuracy change:   {_color(f'{acc_diff:+.4f}', _GREEN if acc_diff > 0 else _RED)}")

        reasons = decision.get("reasons", [])
        if reasons:
            lines.append(f"    Reasons: {', '.join(reasons[:3])}")

        vetoes = decision.get("vetoes", [])
        if vetoes:
            for v in vetoes:
                lines.append(f"    {_color('VETO', _RED)}: {v}")

    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# JSON OUTPUT MODE
# ═══════════════════════════════════════════════════════════════════

def collect_all_as_json(symbol: str, horizon: str, run_eval: bool = False) -> dict:
    """Collect all diagnostics as a single JSON-serializable dict."""
    pending = collect_pending_status(symbol, horizon)
    resolved = collect_resolved_status(symbol, horizon)
    cursor = collect_cursor_status(symbol, horizon)
    model = collect_model_status(symbol, horizon)
    scorecard = collect_scorecard(symbol, horizon)
    health = assess_health(pending, resolved, cursor, model, scorecard)

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "horizon": horizon,
        "health": health,
        "model": model,
        "pending": pending,
        "resolved": resolved,
        "cursor": cursor,
        "scorecard": scorecard,
        "policies": get_active_policies(),
    }

    if run_eval:
        from ml.challenger_runner import run_challenger_comparison, build_default_challengers
        from ml.promotion import DEFAULT_PROMOTION_POLICY
        resolved_samples = load_resolved_samples(symbol, horizon)
        if resolved_samples:
            eval_result = run_challenger_comparison(
                resolved_samples=resolved_samples,
                challengers=build_default_challengers(),
                scorecard_policy=DEFAULT_SCORECARD_POLICY,
                promotion_policy=DEFAULT_PROMOTION_POLICY,
            )
            result["evaluation"] = eval_result

    return result


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Aleph Cortex Dashboard — operational diagnostics CLI",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--horizon", default="1d", help="Prediction horizon")
    parser.add_argument("--full", action="store_true", help="Show full detail (calibration curve, etc.)")
    parser.add_argument("--evaluate", action="store_true", help="Run challenger evaluation")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    symbol = args.symbol.strip().upper()
    horizon = args.horizon.strip().lower()

    if args.json:
        result = collect_all_as_json(symbol, horizon, run_eval=args.evaluate)
        print(json.dumps(result, indent=2, default=str))
        return

    # Collect data
    pending = collect_pending_status(symbol, horizon)
    resolved = collect_resolved_status(symbol, horizon)
    cursor = collect_cursor_status(symbol, horizon)
    model = collect_model_status(symbol, horizon)
    scorecard = collect_scorecard(symbol, horizon)
    health = assess_health(pending, resolved, cursor, model, scorecard)

    # Render dashboard
    output = render_dashboard(
        symbol, horizon, pending, resolved, cursor, model, scorecard, health,
        full=args.full,
    )
    print(output)

    # Optional evaluation
    if args.evaluate:
        from ml.challenger_runner import run_challenger_comparison, build_default_challengers
        from ml.promotion import DEFAULT_PROMOTION_POLICY
        resolved_samples = load_resolved_samples(symbol, horizon)
        if resolved_samples:
            eval_result = run_challenger_comparison(
                resolved_samples=resolved_samples,
                challengers=build_default_challengers(),
                scorecard_policy=DEFAULT_SCORECARD_POLICY,
                promotion_policy=DEFAULT_PROMOTION_POLICY,
            )
            print(render_evaluation_summary({"ok": True, "evaluation": eval_result}))
        else:
            print(_kv("Evaluation", "No resolved samples available", warn=True))


if __name__ == "__main__":
    main()
