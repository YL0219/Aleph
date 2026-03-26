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

Compatible with Python 3.10+.
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
from ml.operational_status import compute_operational_snapshot, format_operational_snapshot


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
    return "{}{}{}".format(code, text, _RESET)


def _header(title: str) -> str:
    line = "\u2500" * 60
    dim_line = _color(line, _DIM)
    bold_title = _color("  " + title, _BOLD + _CYAN)
    return "\n{}\n{}\n{}".format(dim_line, bold_title, dim_line)


def _kv(key: str, value, warn: bool = False, good: bool = False) -> str:
    color = _RED if warn else (_GREEN if good else _WHITE)
    label = _color(key + ":", _DIM)
    val = _color(str(value), color)
    return "  {}  {}".format(label, val)


def _bar(value: float, width: int = 20, label: str = "") -> str:
    """Render a simple horizontal bar chart."""
    filled = int(value * width)
    bar = "\u2588" * filled + "\u2591" * (width - filled)
    pct = "{:.0%}".format(value)
    dim_label = _color(label, _DIM)
    return "  {} {} {}".format(bar, pct, dim_label)


def _health_indicator(level: str) -> str:
    if level == "healthy":
        return _color("\u25cf HEALTHY", _GREEN + _BOLD)
    elif level == "degraded":
        return _color("\u25cf DEGRADED", _YELLOW + _BOLD)
    elif level == "critical":
        return _color("\u25cf CRITICAL", _RED + _BOLD)
    return _color("\u25cf UNKNOWN", _DIM)


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


def collect_scorecard(symbol: str, horizon: str):
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
    scorecard,
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
        issues.append("large_pending_backlog:{}".format(pending["total"]))
        level = "degraded"

    if pending["oldest_hours"] > 168:  # 7 days
        issues.append("stale_pending:{:.0f}h".format(pending["oldest_hours"]))
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
            issues.append("high_brier:{:.3f}".format(brier))
            level = "degraded"
        if acc < 0.25:
            issues.append("very_low_accuracy:{:.3f}".format(acc))
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
    scorecard,
    health: dict,
    full: bool = False,
) -> str:
    """Render the full dashboard as a formatted string."""
    lines = []
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    title = "  ALEPH CORTEX DASHBOARD \u2014 {}/{}".format(symbol, horizon)
    lines.append("")
    lines.append(_color(title, _BOLD + _MAGENTA))
    lines.append(_color("  " + now_str, _DIM))
    health_level = health["level"]
    lines.append("  Health: {}".format(_health_indicator(health_level)))
    if health["issues"]:
        for issue in health["issues"]:
            bang = _color("!", _YELLOW)
            lines.append("    {} {}".format(bang, issue))

    # ── Operational Status ──
    lines.append(_header("Pipeline Status"))
    try:
        op_snap = compute_operational_snapshot(symbol, horizon, "1h")
        op_formatted = format_operational_snapshot(op_snap)

        # Pipeline state with color
        state = op_formatted.get("pipeline_state", "unknown")
        state_colors = {
            "healthy_waiting": _GREEN,
            "healthy_idle": _GREEN,
            "healthy_progressing": _GREEN + _BOLD,
            "blocked_schema": _RED,
            "blocked_data": _RED,
            "stalled": _YELLOW + _BOLD,
            "degraded": _YELLOW,
            "broken": _RED + _BOLD,
        }
        state_color = state_colors.get(state, _DIM)
        lines.append("  State:  {}".format(_color(state.upper().replace("_", " "), state_color)))

        # Human summary
        human = op_formatted.get("human_summary", "")
        if human:
            lines.append("  {}".format(_color(human, _DIM)))

        # Maturity timeline
        maturity = op_formatted.get("maturity_timeline", {})
        mature_count = maturity.get("mature_count", 0)
        immature_count = maturity.get("immature_count", 0)
        if mature_count > 0 or immature_count > 0:
            lines.append(_kv("Mature predictions", mature_count, good=mature_count > 0))
            lines.append(_kv("Awaiting maturity", immature_count))
            next_mat = maturity.get("next_maturity_utc")
            if next_mat:
                lines.append(_kv("Next maturity", next_mat))
            avg_h = maturity.get("avg_hours_to_maturity", 0)
            if avg_h > 0:
                lines.append(_kv("Avg time to maturity", "{:.1f}h".format(avg_h)))

        # Schema health
        schema = op_formatted.get("schema_health", {})
        mismatches = schema.get("mismatch_count", 0)
        if mismatches > 0:
            lines.append(_kv("Schema mismatches", mismatches, warn=True))

        # Training readiness
        readiness = op_formatted.get("training_readiness", {})
        gate = readiness.get("gate_status", "unknown")
        if gate == "ready":
            lines.append(_kv("Training gate", "OPEN", good=True))
        elif gate == "blocked":
            reasons = readiness.get("block_reasons", [])
            lines.append(_kv("Training gate", "BLOCKED: {}".format(", ".join(reasons[:3])), warn=True))
    except Exception as ex:
        lines.append(_kv("Status", "error: {}".format(ex), warn=True))

    # ── Model ──
    lines.append(_header("Model"))
    model_state = model["state"]
    if model_state == "active":
        state_color = _GREEN
    elif model_state == "warming":
        state_color = _YELLOW
    else:
        state_color = _RED
    lines.append("  State:    {}".format(_color(model_state, state_color)))
    lines.append(_kv("Version", model["version"]))
    lines.append(_kv("Trained", "{} samples".format(model["trained_samples"])))
    if model["class_distribution"]:
        dist_str = json.dumps(model["class_distribution"])
        lines.append("  Classes:  {}".format(_color(dist_str, _DIM)))

    # ── Pending Memory ──
    lines.append(_header("Pending Memory"))
    lines.append(_kv("Total", pending["total"],
                      warn=pending["total"] > 100))
    lines.append(_kv("Eligible", pending["eligible"],
                      good=pending["eligible"] > 0))
    lines.append(_kv("Blocked", pending["blocked"],
                      warn=pending["blocked"] > pending["eligible"]))
    if pending["total"] > 0:
        lines.append(_kv("Age (oldest)", "{}h".format(pending["oldest_hours"]),
                          warn=pending["oldest_hours"] > 168))
        lines.append(_kv("Age (newest)", "{}h".format(pending["newest_hours"])))
        lines.append(_kv("Age (median)", "{}h".format(pending["median_hours"])))

    # ── Resolved Archive ──
    lines.append(_header("Resolved Archive"))
    lines.append(_kv("Total", resolved["total"],
                      good=resolved["total"] > 30))
    if resolved["total"] > 0:
        acc = resolved["archive_accuracy"]
        acc_str = "{:.1%}".format(acc)
        lines.append(_kv("Archive accuracy", acc_str,
                          warn=acc < 0.3,
                          good=acc >= 0.5))
        label_str = json.dumps(resolved["label_distribution"])
        pred_str = json.dumps(resolved["predicted_distribution"])
        lines.append("  Labels:      {}".format(_color(label_str, _DIM)))
        lines.append("  Predictions: {}".format(_color(pred_str, _DIM)))

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
        window_actual = scorecard.get("window_actual", "?")
        window_size = scorecard.get("window_size", "?")
        window_str = "{}/{} samples".format(window_actual, window_size)
        lines.append(_kv("Window", window_str))
        brier_str = "{:.4f}".format(brier)
        lines.append(_kv("Brier score", brier_str,
                          warn=brier > 0.35, good=brier < 0.25))
        acc_str = "{:.1%}".format(acc)
        lines.append(_kv("Accuracy", acc_str,
                          warn=acc < 0.3, good=acc >= 0.5))
        cal_gap = scorecard.get("mean_calibration_gap", 0)
        lines.append(_kv("Calibration gap", "{:.4f}".format(cal_gap)))

        # Grade buckets
        buckets = scorecard.get("grade_buckets", {})
        if buckets:
            total_b = sum(buckets.values())
            for bucket, count in buckets.items():
                pct = count / total_b if total_b > 0 else 0
                lines.append("    {:20s} {:4d}  ({:.0%})".format(bucket, count, pct))

        # Streak
        streak = scorecard.get("current_streak", {})
        streak_len = streak.get("length", 0)
        if streak_len > 0:
            streak_type = streak.get("type", "?")
            streak_color = _GREEN if streak_type == "correct" else _RED
            streak_text = "{} x{}".format(streak_type, streak_len)
            lines.append("  Streak:  {}".format(_color(streak_text, streak_color)))

        # Drift
        drift = scorecard.get("drift", {})
        if drift.get("detected"):
            drift_flags = ", ".join(drift.get("flags", []))
            drift_label = _color("DRIFT DETECTED", _RED + _BOLD)
            lines.append("  {}: {}".format(drift_label, drift_flags))
            brier_shift = drift.get("brier_shift", 0)
            acc_shift = drift.get("accuracy_shift", 0)
            lines.append("    Brier shift:    {:+.4f}".format(brier_shift))
            lines.append("    Accuracy shift: {:+.4f}".format(acc_shift))

        # Warnings
        warnings = scorecard.get("warnings", [])
        if warnings:
            lines.append("  Warnings ({}):".format(len(warnings)))
            for w in warnings:
                bang = _color("!", _YELLOW)
                lines.append("    {} {}".format(bang, w))

        # Full mode: calibration curve
        if full and scorecard.get("calibration"):
            lines.append("")
            lines.append(_color("  Calibration Curve:", _BOLD))
            lines.append("  {:>12s}  {:>5s}  {:>9s}  {:>7s}  {:>5s}".format(
                "Bin", "Count", "Predicted", "Actual", "Gap"))
            for b in scorecard["calibration"]:
                bin_label = "{:.2f}-{:.2f}".format(b["bin_lower"], b["bin_upper"])
                b_count = b["count"]
                b_pred = b["mean_predicted_prob"]
                b_actual = b["actual_hit_rate"]
                b_gap = b["gap"]
                lines.append(
                    "  {:>12s}  {:5d}  {:9.4f}  {:7.4f}  {:5.4f}".format(
                        bin_label, b_count, b_pred, b_actual, b_gap
                    )
                )

    # ── Active Policies ──
    lines.append(_header("Active Policies"))
    policies = get_active_policies()
    for name, pol in policies.items():
        version = pol.get("version", "?") if isinstance(pol, dict) else "?"
        lines.append(_kv(name, version))

    lines.append("")
    lines.append(_color("\u2500" * 60, _DIM))
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
    promote_count = summary.get("promote", 0)
    lines.append(_kv("Promote", promote_count,
                      good=promote_count > 0))
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
        if dec == "promote":
            dec_color = _GREEN
        elif dec == "reject":
            dec_color = _RED
        else:
            dec_color = _YELLOW

        lines.append("")
        name_colored = _color(name, _BOLD)
        dec_colored = _color(dec.upper(), dec_color)
        lines.append("  {}  \u2192  {}".format(name_colored, dec_colored))

        delta = comp.get("delta", {})
        brier_diff = delta.get("brier_score_diff")
        acc_diff = delta.get("accuracy_diff")
        if brier_diff is not None:
            improvement = -brier_diff
            imp_color = _GREEN if improvement > 0 else _RED
            imp_str = _color("{:+.4f}".format(improvement), imp_color)
            lines.append("    Brier improvement: {}".format(imp_str))
        if acc_diff is not None:
            acc_color = _GREEN if acc_diff > 0 else _RED
            acc_str = _color("{:+.4f}".format(acc_diff), acc_color)
            lines.append("    Accuracy change:   {}".format(acc_str))

        reasons = decision.get("reasons", [])
        if reasons:
            reasons_str = ", ".join(reasons[:3])
            lines.append("    Reasons: {}".format(reasons_str))

        vetoes = decision.get("vetoes", [])
        if vetoes:
            for v in vetoes:
                veto_label = _color("VETO", _RED)
                lines.append("    {}: {}".format(veto_label, v))

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

    try:
        op_snap = compute_operational_snapshot(symbol, horizon, "1h")
        result["operational_status"] = format_operational_snapshot(op_snap)
    except Exception:
        pass

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
        description="Aleph Cortex Dashboard \u2014 operational diagnostics CLI",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--horizon", default="1d", help="Prediction horizon")
    parser.add_argument("--full", action="store_true", help="Show full detail (calibration curve, etc.)")
    parser.add_argument("--evaluate", action="store_true", help="Run challenger evaluation")
    parser.add_argument("--dreams", action="store_true", help="Show dream state summary")
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

    # Optional dream state summary
    if args.dreams:
        try:
            from ml.dream_state import list_dreams
            dreams = list_dreams()
            if dreams:
                print(_header("Dream States"))
                for d in dreams:
                    status = d.get("status", "?")
                    dream_id = d.get("dream_id", "?")
                    progress = d.get("progress_pct", 0)
                    status_color = _GREEN if status == "completed" else (_YELLOW if status == "running" else _DIM)
                    print("  {}  {}  {:.0f}%".format(
                        _color(dream_id, _BOLD),
                        _color(status, status_color),
                        progress
                    ))
            else:
                print(_kv("Dreams", "No dream states found", warn=False))
        except Exception as ex:
            print(_kv("Dreams", "error: {}".format(ex), warn=True))


if __name__ == "__main__":
    main()
