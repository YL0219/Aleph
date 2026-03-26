"""
ml_cortex.py — Main orchestrator for the ML Cortex Python brain.

v4: Phase 9.1 — adds scorecard integration and challenger evaluation:
  - cortex_predict   — real-time inference (always available, even cold start)
  - cortex_resolve   — resolve pending + cycle-level scorecard
  - cortex_train     — cursor-aware policy-driven incremental training
  - cortex_status    — diagnostics + rolling scorecard summary
  - cortex_evaluate  — offline challenger-vs-incumbent comparison

All stdout output = exactly one JSON object.
All logs go to stderr.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

from .feature_adapter import extract_features, has_meaningful_features, FEATURE_VERSION
from .brain_state import load_model, save_model
from .temporal_security import check_temporal_safety, compute_eligibility
from .policies import (
    DEFAULT_LABEL_POLICY, DEFAULT_RESOLUTION_POLICY, DEFAULT_TRAINING_POLICY,
    get_active_policies,
)
from .training_cursor import load_cursor, save_cursor
from .pending_memory import (
    store_pending_sample,
    load_pending_samples,
    load_resolved_samples,
    load_resolved_since_cursor,
    append_resolved_samples,
    rewrite_pending_after_resolve,
    pending_count as get_pending_count,
    pending_eligible_count as get_pending_eligible_count,
    pending_blocked_count as get_pending_blocked_count,
    resolved_count as get_resolved_count,
)
from .label_resolver import resolve_pending_batch
from .prediction_formatter import (
    format_prediction,
    format_status,
    format_resolve,
    format_controlled_train,
    format_train_result,
    format_evaluation,
)
from .scorecard import compute_scorecard, compute_rolling_scorecard, DEFAULT_SCORECARD_POLICY
from .challenger_runner import run_challenger_comparison, build_default_challengers, ChallengerSpec
from .promotion import DEFAULT_PROMOTION_POLICY
from .schema_gate import filter_compatible_memories, check_memory_compatibility
from .model_registry import ModelRoster


# ═══════════════════════════════════════════════════════════════════
# VERB 1: PREDICT (real-time inference)
# ═══════════════════════════════════════════════════════════════════

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

    # ── Touch model registry (best-effort) ──
    try:
        roster = ModelRoster.load()
        if not roster.get_model(model_key) and model_key:
            from .model_registry import ModelIdentity
            roster.register_model(ModelIdentity(
                model_key=model_key,
                feature_version=feature_version,
                label_policy_version=DEFAULT_LABEL_POLICY.version,
                description=f"Auto-registered from predict for {symbol}/{horizon}",
            ))
            roster.save()
    except Exception:
        pass  # non-blocking

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

    # ── Build watched catalysts ──
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


# ═══════════════════════════════════════════════════════════════════
# VERB 2: RESOLVE (delayed label resolution against parquet truth)
# ═══════════════════════════════════════════════════════════════════

def cortex_resolve(symbol: str, horizon: str, interval: str = "1h") -> dict:
    """
    Resolve mature pending predictions against realized market truth.

    Steps:
      1. Load pending samples
      2. Load OHLCV parquet truth for this symbol/interval
      3. Run policy-driven batch resolution (label_resolver)
      4. Append resolved records to truth archive
      5. Atomically rewrite pending queue (keep unresolved, remove resolved/expired)
      6. Return summary
    """
    warnings: list[str] = []

    # ── Load pending ──
    pending = load_pending_samples(symbol, horizon)
    if not pending:
        return format_resolve(
            symbol=symbol, horizon=horizon,
            resolution_summary={
                "total_processed": 0, "resolved_count": 0,
                "deferred_count": 0, "expired_count": 0, "errored_count": 0,
                "class_distribution": {}, "accuracy": 0.0, "mean_brier_score": 0.0,
                "label_policy_version": DEFAULT_LABEL_POLICY.version,
                "resolution_policy_version": DEFAULT_RESOLUTION_POLICY.version,
                "warnings": ["no_pending_samples"],
            },
            warnings=["no_pending_samples"],
        )

    print(f"[MlCortex] Resolve {symbol}/{horizon} — {len(pending)} pending samples", file=sys.stderr)

    # ── Load OHLCV truth from parquet ──
    ohlcv_df = _load_ohlcv_truth(symbol, interval, warnings)

    # ── Run resolution ──
    result = resolve_pending_batch(
        pending_samples=pending,
        ohlcv_df=ohlcv_df,
        label_policy=DEFAULT_LABEL_POLICY,
        resolution_policy=DEFAULT_RESOLUTION_POLICY,
    )

    summary = result.summary()
    print(
        f"[MlCortex] Resolution complete: resolved={len(result.resolved)}, "
        f"deferred={len(result.deferred)}, expired={len(result.expired)}, "
        f"errored={len(result.errored)}",
        file=sys.stderr,
    )

    # ── Append resolved to truth archive ──
    if result.resolved:
        written = append_resolved_samples(symbol, horizon, result.resolved)
        print(f"[MlCortex] Wrote {written} resolved records to truth archive", file=sys.stderr)
        if written < len(result.resolved):
            warnings.append(f"partial_archive_write:{written}/{len(result.resolved)}")

    # ── Rewrite pending queue ──
    resolved_ids = {r["prediction_id"] for r in result.resolved if r.get("prediction_id")}
    expired_ids = {s.get("prediction_id", "") for s in result.expired if s.get("prediction_id")}

    rewrite_result = {}
    if resolved_ids or expired_ids:
        rewrite_result = rewrite_pending_after_resolve(symbol, horizon, resolved_ids, expired_ids)
        print(f"[MlCortex] Pending rewrite: {rewrite_result}", file=sys.stderr)

    warnings.extend(result.warnings)

    # ── Compute cycle-level scorecard over newly resolved batch ──
    cycle_scorecard = None
    if result.resolved:
        try:
            cycle_scorecard = compute_scorecard(result.resolved, DEFAULT_SCORECARD_POLICY)
            print(
                f"[MlCortex] Cycle scorecard: brier={cycle_scorecard.get('mean_brier_score')}, "
                f"acc={cycle_scorecard.get('accuracy')}, warnings={cycle_scorecard.get('warning_count', 0)}",
                file=sys.stderr,
            )
        except Exception as ex:
            warnings.append(f"scorecard_error:{ex}")

    return format_resolve(
        symbol=symbol,
        horizon=horizon,
        resolution_summary=summary,
        pending_rewrite_result=rewrite_result,
        warnings=warnings,
        cycle_scorecard=cycle_scorecard,
    )


# ═══════════════════════════════════════════════════════════════════
# VERB 3: TRAIN (cursor-aware policy-driven incremental training)
# ═══════════════════════════════════════════════════════════════════

def cortex_train(symbol: str, horizon: str, max_samples: int = 200) -> dict:
    """
    Incremental training path.

    Steps:
      1. Load training cursor (what's already been consumed)
      2. Load resolved samples, split into fresh vs replay
      3. Load model
      4. Run controlled_fit with policy-driven batch construction
      5. Update cursor
      6. Save model
      7. Return summary
    """
    warnings: list[str] = []

    # ── Load cursor ──
    cursor = load_cursor(symbol, horizon)
    print(
        f"[MlCortex] Train {symbol}/{horizon} — cursor seq={cursor.sequence}, "
        f"consumed={len(cursor.consumed_ids)}",
        file=sys.stderr,
    )

    # ── Load resolved samples split by cursor ──
    fresh, replay_pool = load_resolved_since_cursor(
        symbol, horizon,
        consumed_ids=cursor.consumed_ids,
        max_samples=max_samples * 3,  # load more for replay pool
    )

    # ── Schema compatibility gate ──
    if fresh:
        compatible, incompatible, adapted = filter_compatible_memories(
            fresh, FEATURE_VERSION
        )
        if incompatible:
            warnings.append(f"schema_filtered:{len(incompatible)}_incompatible_memories")
            print(
                f"[MlCortex] Schema gate: {len(compatible)} compatible, "
                f"{len(adapted)} adapted, {len(incompatible)} incompatible",
                file=sys.stderr,
            )
        fresh = compatible + adapted

    if replay_pool:
        replay_compat, replay_incompat, replay_adapted = filter_compatible_memories(
            replay_pool, FEATURE_VERSION
        )
        if replay_incompat:
            warnings.append(f"schema_filtered_replay:{len(replay_incompat)}_incompatible")
        replay_pool = replay_compat + replay_adapted

    if not fresh:
        print(f"[MlCortex] No fresh resolved samples for {symbol}/{horizon}", file=sys.stderr)
        return format_controlled_train(
            symbol=symbol, horizon=horizon,
            train_result={
                "samples_fitted": 0, "fresh_count": 0, "replay_count": 0,
                "batch_class_distribution": {},
                "model_state": "unknown", "model_version": "unknown",
                "trained_samples_total": 0,
                "warnings": ["no_fresh_samples"],
                "drift_flags": [],
                "policy_version": DEFAULT_TRAINING_POLICY.version,
            },
            cursor_sequence=cursor.sequence,
            consumed_count=len(cursor.consumed_ids),
            warnings=["no_fresh_resolved_samples"],
        )

    # ── Load model ──
    model = load_model(symbol, horizon)
    print(
        f"[MlCortex] Model loaded: state={model.model_state}, "
        f"samples={model.trained_samples}",
        file=sys.stderr,
    )

    # ── Controlled fit ──
    train_result = model.controlled_fit(
        fresh_samples=fresh,
        replay_samples=replay_pool,
        policy=DEFAULT_TRAINING_POLICY,
    )

    print(
        f"[MlCortex] Training complete: fitted={train_result.samples_fitted}, "
        f"fresh={train_result.fresh_count}, replay={train_result.replay_count}",
        file=sys.stderr,
    )

    # ── Update cursor if training occurred ──
    if train_result.samples_fitted > 0:
        # Mark all fresh sample prediction_ids as consumed
        fresh_ids = [s.get("prediction_id", "") for s in fresh if s.get("prediction_id")]
        cursor.mark_consumed(fresh_ids, DEFAULT_TRAINING_POLICY.version)
        cursor.prune_old_ids()
        save_cursor(cursor)

        # Save updated model
        save_model(symbol, horizon, model)
        print(f"[MlCortex] Model and cursor saved. Seq={cursor.sequence}", file=sys.stderr)

    if train_result.drift_flags:
        warnings.extend([f"drift:{f}" for f in train_result.drift_flags])

    return format_controlled_train(
        symbol=symbol,
        horizon=horizon,
        train_result=train_result.to_dict(),
        cursor_sequence=cursor.sequence,
        consumed_count=len(cursor.consumed_ids),
        warnings=warnings + train_result.warnings,
    )


# ═══════════════════════════════════════════════════════════════════
# VERB 4: STATUS (expanded diagnostics)
# ═══════════════════════════════════════════════════════════════════

def cortex_status(symbol: str, horizon: str) -> dict:
    """Get Cortex model/memory/cursor status with rolling scorecard."""
    model = load_model(symbol, horizon)
    cursor = load_cursor(symbol, horizon)

    pc = get_pending_count(symbol, horizon)
    pe = get_pending_eligible_count(symbol, horizon)
    pb = get_pending_blocked_count(symbol, horizon)
    rc = get_resolved_count(symbol, horizon)

    class_dist = {}
    if hasattr(model, "class_distribution"):
        class_dist = model.class_distribution()

    # ── Compute rolling scorecard over resolved history ──
    rolling_sc = None
    if rc > 0:
        try:
            resolved = load_resolved_samples(symbol, horizon)
            if resolved:
                rolling_sc = compute_rolling_scorecard(resolved, DEFAULT_SCORECARD_POLICY)
                print(
                    f"[MlCortex] Rolling scorecard ({rolling_sc.get('window_actual', 0)} samples): "
                    f"brier={rolling_sc.get('mean_brier_score')}, acc={rolling_sc.get('accuracy')}",
                    file=sys.stderr,
                )
        except Exception as ex:
            print(f"[MlCortex] Rolling scorecard error: {ex}", file=sys.stderr)

    # ── Operational status assessment ──
    operational = None
    try:
        from .operational_status import compute_operational_snapshot, format_operational_snapshot
        op_snap = compute_operational_snapshot(symbol, horizon, "1h")
        operational = format_operational_snapshot(op_snap)
    except Exception as ex:
        print(f"[MlCortex] Operational status error: {ex}", file=sys.stderr)

    result = format_status(
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
        last_train_utc=cursor.last_train_utc,
        class_distribution=class_dist,
        cursor_sequence=cursor.sequence,
        total_samples_ever_trained=cursor.total_samples_ever,
        active_policies=get_active_policies(),
        rolling_scorecard=rolling_sc,
    )
    if operational is not None:
        result["operational_status"] = operational
    return result


# ═══════════════════════════════════════════════════════════════════
# VERB 5: EVALUATE (offline challenger comparison) — Phase 9.1
# ═══════════════════════════════════════════════════════════════════

def cortex_evaluate(symbol: str, horizon: str, challengers_json: str = "") -> dict:
    """
    Run offline challenger-vs-incumbent evaluation against resolved history.

    If no challengers_json is provided, uses the default challenger set
    covering label threshold and replay ratio variations.
    """
    warnings: list[str] = []

    # ── Load resolved history ──
    resolved = load_resolved_samples(symbol, horizon)
    if not resolved:
        return format_evaluation(
            symbol=symbol, horizon=horizon,
            evaluation_result={
                "ok": False,
                "error": "no_resolved_history",
                "sample_count": 0,
            },
            warnings=["no_resolved_history"],
        )

    print(
        f"[MlCortex] Evaluate {symbol}/{horizon} — {len(resolved)} resolved samples",
        file=sys.stderr,
    )

    # ── Parse challenger specs or use defaults ──
    challengers = []
    if challengers_json:
        try:
            import json
            specs = json.loads(challengers_json)
            if isinstance(specs, list):
                for spec in specs:
                    lp = None
                    tp = None
                    if "label_policy" in spec:
                        from .policies import LabelPolicy
                        lp = LabelPolicy.from_dict(spec["label_policy"])
                    if "training_policy" in spec:
                        from .policies import TrainingPolicy
                        tp = TrainingPolicy.from_dict(spec["training_policy"])
                    challengers.append(ChallengerSpec(
                        name=spec.get("name", "unnamed"),
                        label_policy=lp,
                        training_policy=tp,
                        description=spec.get("description", ""),
                    ))
        except Exception as ex:
            warnings.append(f"challengers_parse_error:{ex}")

    if not challengers:
        challengers = build_default_challengers()
        print(
            f"[MlCortex] Using {len(challengers)} default challengers",
            file=sys.stderr,
        )

    # ── Run comparison ──
    try:
        result = run_challenger_comparison(
            resolved_samples=resolved,
            challengers=challengers,
            incumbent_label_policy=DEFAULT_LABEL_POLICY,
            incumbent_training_policy=DEFAULT_TRAINING_POLICY,
            scorecard_policy=DEFAULT_SCORECARD_POLICY,
            promotion_policy=DEFAULT_PROMOTION_POLICY,
        )
    except Exception as ex:
        return format_evaluation(
            symbol=symbol, horizon=horizon,
            evaluation_result={"ok": False, "error": str(ex)},
            warnings=[f"evaluation_error:{ex}"],
        )

    summary = result.get("summary", {})
    print(
        f"[MlCortex] Evaluation complete: "
        f"promote={summary.get('promote', 0)}, "
        f"reject={summary.get('reject', 0)}, "
        f"inconclusive={summary.get('inconclusive', 0)}, "
        f"best={summary.get('best_challenger')}",
        file=sys.stderr,
    )

    return format_evaluation(
        symbol=symbol, horizon=horizon,
        evaluation_result=result,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════
# OHLCV TRUTH LOADER
# ═══════════════════════════════════════════════════════════════════

def _load_ohlcv_truth(symbol: str, interval: str, warnings: list[str]):
    """
    Load OHLCV data from the parquet data lake.
    Uses the quant/parquet_loader module.
    """
    try:
        # Import parquet_loader from the quant sibling package
        parent = Path(__file__).resolve().parent.parent
        quant_path = str(parent / "quant")
        if quant_path not in sys.path:
            sys.path.insert(0, str(parent))

        from quant.parquet_loader import load_ohlcv
        df, load_warnings = load_ohlcv(symbol, timeframe=interval, days=0)
        if load_warnings:
            warnings.extend(load_warnings)
        return df
    except ImportError as ex:
        warnings.append(f"parquet_loader_import_error:{ex}")
        return None
    except Exception as ex:
        warnings.append(f"ohlcv_load_error:{ex}")
        return None


# ═══════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════

def _sf(val) -> float:
    """Safe float conversion."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _compute_priority(prediction: dict, regime: dict, events: dict) -> float:
    """Higher confidence + higher event/regime signals = higher priority."""
    conf = prediction.get("confidence", 0.0)
    tend = abs(prediction.get("action_tendency", 0.0))
    event_signal = max(events.get("materiality", 0.0), events.get("shock", 0.0))
    regime_signal = max(regime.get("risk_off", 0.0), regime.get("policy_shock", 0.0),
                        regime.get("flight_to_safety", 0.0))
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
