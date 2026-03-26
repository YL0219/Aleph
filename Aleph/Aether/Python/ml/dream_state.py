"""
dream_state.py - Dream State / Simulation engine for the ML Cortex.

Replays historical data through the same ML pipeline without touching live
state.  Think of it as the organism sleeping and dreaming about historical
scenarios to learn.

The dream operates in an **isolated namespace**: every storage path is
redirected from the live ``data_lake/cortex/`` to
``data_lake/cortex/dreams/{dream_id}/``.  Multiple dreams can coexist and
live state is never mutated.

Storage layout per dream:
  data_lake/cortex/dreams/{dream_id}/manifest.json
  data_lake/cortex/dreams/{dream_id}/models/{symbol}/{horizon}/
  data_lake/cortex/dreams/{dream_id}/pending/{symbol}/{horizon}/
  data_lake/cortex/dreams/{dream_id}/resolved/{symbol}/{horizon}/
  data_lake/cortex/dreams/{dream_id}/cursor/{symbol}/{horizon}/

Key verbs (called from aether_router / C#):
  create_dream   -> initialise namespace + optional warm-start model clone
  run_dream_step -> predict at virtual time T, advance clock
  resolve_dream  -> resolve all pending predictions against parquet truth
  train_dream    -> incremental training on resolved dream data
  evaluate_dream -> scorecard + live-vs-dream comparison
  get_dream_status / list_dreams / abort_dream  -> lifecycle management
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .brain_state import load_model as live_load_model, _cortex_root, _model_dir as live_model_dir, _state_dir as live_state_dir
from .incremental_model import IncrementalCortexModel
from .pending_memory import (
    load_pending_samples as _live_load_pending,
    load_resolved_samples as _live_load_resolved,
)
from .training_cursor import TrainingCursor
from .feature_adapter import extract_features, FEATURE_VERSION
from .label_resolver import resolve_pending_batch
from .policies import (
    DEFAULT_LABEL_POLICY,
    DEFAULT_RESOLUTION_POLICY,
    DEFAULT_TRAINING_POLICY,
    LabelPolicy,
    ResolutionPolicy,
    TrainingPolicy,
)
from .scorecard import compute_scorecard, compute_rolling_scorecard, DEFAULT_SCORECARD_POLICY


# ═══════════════════════════════════════════════════════════════════════════
# PATH HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _dream_root(dream_id: str) -> Path:
    return _cortex_root() / "dreams" / dream_id


# ═══════════════════════════════════════════════════════════════════════════
# DREAM NAMESPACE
# ═══════════════════════════════════════════════════════════════════════════

class DreamNamespace:
    """
    Provides path resolution for a dream that mirrors the live cortex layout
    but rooted under ``data_lake/cortex/dreams/{dream_id}/``.
    """

    def __init__(self, dream_id: str) -> None:
        self.dream_id = dream_id
        self.root = _dream_root(dream_id)

    # ── directory helpers ──

    def model_dir(self, symbol: str, horizon: str) -> Path:
        return self.root / "models" / symbol.upper() / horizon

    def pending_path(self, symbol: str, horizon: str) -> Path:
        return self.root / "pending" / symbol.upper() / horizon / "pending.jsonl"

    def resolved_path(self, symbol: str, horizon: str) -> Path:
        return self.root / "resolved" / symbol.upper() / horizon / "resolved.jsonl"

    def cursor_path(self, symbol: str, horizon: str) -> Path:
        return self.root / "cursor" / symbol.upper() / horizon / "cursor.json"

    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    # ── ensure dirs ──

    def ensure_dirs(self, symbol: str, horizon: str) -> None:
        self.model_dir(symbol, horizon).mkdir(parents=True, exist_ok=True)
        self.pending_path(symbol, horizon).parent.mkdir(parents=True, exist_ok=True)
        self.resolved_path(symbol, horizon).parent.mkdir(parents=True, exist_ok=True)
        self.cursor_path(symbol, horizon).parent.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# DREAM MANIFEST
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DreamManifest:
    dream_id: str
    created_utc: str
    symbol: str
    horizon: str
    interval: str
    replay_start_utc: str
    replay_end_utc: str
    model_key: str
    feature_version: str
    status: str = "created"          # created | running | completed | failed | aborted
    current_step: int = 0
    total_steps: int = 0
    progress_pct: float = 0.0
    metrics: dict = field(default_factory=lambda: {
        "total_predictions": 0,
        "total_resolved": 0,
        "total_trained": 0,
        "dream_accuracy": 0.0,
        "dream_brier": 0.0,
    })
    config: dict = field(default_factory=dict)

    # ── serialisation ──

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DreamManifest:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})

    # ── persistence ──

    def save(self) -> None:
        ns = DreamNamespace(self.dream_id)
        path = ns.manifest_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        if os.name == "nt" and path.exists():
            os.remove(path)
        os.rename(tmp, path)

    @classmethod
    def load(cls, dream_id: str) -> DreamManifest | None:
        path = DreamNamespace(dream_id).manifest_path()
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                return cls.from_dict(json.load(f))
        except Exception as ex:
            print(f"[DreamState] Failed to load manifest {dream_id}: {ex}", file=sys.stderr)
            return None


# ═══════════════════════════════════════════════════════════════════════════
# DREAM CLOCK  (virtual time driver)
# ═══════════════════════════════════════════════════════════════════════════

_INTERVAL_MAP: dict[str, timedelta] = {
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


class DreamClock:
    """Virtual clock for deterministic time advancement within a dream."""

    def __init__(self, start_utc: datetime) -> None:
        self.current_utc = start_utc

    def advance(self, bars: int, interval: str) -> None:
        delta = _INTERVAL_MAP.get(interval.lower())
        if delta is None:
            raise ValueError(f"Unsupported interval: {interval}")
        self.current_utc += delta * bars

    def is_past(self, timestamp: str) -> bool:
        try:
            dt = datetime.fromisoformat(timestamp)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return self.current_utc >= dt
        except (ValueError, TypeError):
            return False


# ═══════════════════════════════════════════════════════════════════════════
# DREAM MODEL I/O  (namespace-redirected brain_state operations)
# ═══════════════════════════════════════════════════════════════════════════

def _dream_load_model(ns: DreamNamespace, symbol: str, horizon: str) -> IncrementalCortexModel:
    """Load a model from the dream namespace (mirrors brain_state.load_model)."""
    model = IncrementalCortexModel()

    model_path = ns.model_dir(symbol, horizon) / "model.pkl"
    scaler_path = ns.model_dir(symbol, horizon) / "scaler.pkl"
    meta_path = ns.model_dir(symbol, horizon) / "metadata.json"

    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                model.model = pickle.load(f)
            model._fitted = True
        except Exception as ex:
            print(f"[DreamState] Failed to load dream model: {ex}", file=sys.stderr)

    if scaler_path.exists():
        try:
            with open(scaler_path, "rb") as f:
                model.scaler = pickle.load(f)
            model._scaler_fitted = True
        except Exception as ex:
            print(f"[DreamState] Failed to load dream scaler: {ex}", file=sys.stderr)

    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            model.trained_samples = meta.get("trained_samples", 0)
            model.model_version = meta.get("model_version", "v1.0.0")
        except Exception as ex:
            print(f"[DreamState] Failed to load dream metadata: {ex}", file=sys.stderr)

    return model


def _dream_save_model(ns: DreamNamespace, symbol: str, horizon: str, model: IncrementalCortexModel) -> None:
    """Save a model into the dream namespace (mirrors brain_state.save_model)."""
    mdir = ns.model_dir(symbol, horizon)
    mdir.mkdir(parents=True, exist_ok=True)

    if model._fitted:
        try:
            with open(mdir / "model.pkl", "wb") as f:
                pickle.dump(model.model, f)
        except Exception as ex:
            print(f"[DreamState] Failed to save dream model: {ex}", file=sys.stderr)

    if model._scaler_fitted:
        try:
            with open(mdir / "scaler.pkl", "wb") as f:
                pickle.dump(model.scaler, f)
        except Exception as ex:
            print(f"[DreamState] Failed to save dream scaler: {ex}", file=sys.stderr)

    try:
        meta = model.get_state_dict()
        with open(mdir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as ex:
        print(f"[DreamState] Failed to save dream metadata: {ex}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════════════════════
# DREAM PENDING MEMORY  (namespace-redirected pending_memory operations)
# ═══════════════════════════════════════════════════════════════════════════

def _dream_store_pending(ns: DreamNamespace, symbol: str, horizon: str, sample: dict) -> bool:
    path = ns.pending_path(symbol, horizon)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "a") as f:
            f.write(json.dumps(sample, separators=(",", ":")) + "\n")
        return True
    except Exception as ex:
        print(f"[DreamState] Failed to store dream pending sample: {ex}", file=sys.stderr)
        return False


def _dream_load_pending(ns: DreamNamespace, symbol: str, horizon: str) -> list[dict]:
    path = ns.pending_path(symbol, horizon)
    if not path.exists():
        return []
    samples: list[dict] = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                    if not s.get("resolved", False):
                        samples.append(s)
                except json.JSONDecodeError:
                    continue
    except Exception as ex:
        print(f"[DreamState] Failed to load dream pending: {ex}", file=sys.stderr)
    return samples


def _dream_append_resolved(ns: DreamNamespace, symbol: str, horizon: str, records: list[dict]) -> int:
    if not records:
        return 0
    path = ns.resolved_path(symbol, horizon)
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with open(path, "a") as f:
            for record in records:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
                written += 1
    except Exception as ex:
        print(f"[DreamState] Failed to append dream resolved: {ex}", file=sys.stderr)
    return written


def _dream_load_resolved(ns: DreamNamespace, symbol: str, horizon: str) -> list[dict]:
    path = ns.resolved_path(symbol, horizon)
    if not path.exists():
        return []
    samples: list[dict] = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as ex:
        print(f"[DreamState] Failed to load dream resolved: {ex}", file=sys.stderr)
    return samples


def _dream_rewrite_pending(ns: DreamNamespace, symbol: str, horizon: str, resolved_ids: set[str]) -> dict:
    """Remove resolved predictions from the dream pending queue."""
    path = ns.pending_path(symbol, horizon)
    if not path.exists():
        return {"kept": 0, "removed": 0}
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except Exception:
        return {"kept": 0, "removed": 0}

    kept_lines: list[str] = []
    removed = 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
            pid = sample.get("prediction_id", "")
            if pid in resolved_ids:
                removed += 1
                continue
            kept_lines.append(line)
        except json.JSONDecodeError:
            continue

    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            for line in kept_lines:
                f.write(line + "\n")
        if os.name == "nt" and path.exists():
            os.remove(path)
        os.rename(tmp, path)
    except Exception as ex:
        print(f"[DreamState] Pending rewrite failed: {ex}", file=sys.stderr)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return {"kept": len(kept_lines), "removed": 0}

    return {"kept": len(kept_lines), "removed": removed}


# ═══════════════════════════════════════════════════════════════════════════
# DREAM CURSOR  (namespace-redirected training_cursor operations)
# ═══════════════════════════════════════════════════════════════════════════

def _dream_load_cursor(ns: DreamNamespace, symbol: str, horizon: str) -> TrainingCursor:
    path = ns.cursor_path(symbol, horizon)
    if not path.exists():
        return TrainingCursor(symbol=symbol.upper(), horizon=horizon)
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return TrainingCursor.from_dict(data)
    except Exception:
        return TrainingCursor(symbol=symbol.upper(), horizon=horizon)


def _dream_save_cursor(ns: DreamNamespace, cursor: TrainingCursor) -> bool:
    path = ns.cursor_path(cursor.symbol, cursor.horizon)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(cursor.to_dict(), f, indent=2)
        if os.name == "nt" and path.exists():
            os.remove(path)
        os.rename(tmp, path)
        return True
    except Exception as ex:
        print(f"[DreamState] Failed to save dream cursor: {ex}", file=sys.stderr)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False


# ═══════════════════════════════════════════════════════════════════════════
# OHLCV TRUTH LOADER  (shared read-only from the live data lake)
# ═══════════════════════════════════════════════════════════════════════════

def _load_ohlcv_truth(symbol: str, interval: str, warnings: list[str]):
    """Load OHLCV data from the shared parquet data lake (read-only)."""
    try:
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


def _slice_ohlcv(df, start_utc: datetime, end_utc: datetime):
    """Slice an OHLCV DataFrame to the [start, end] window."""
    import pandas as pd

    if df is None or df.empty:
        return df

    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], utc=True)

    if df["time"].dt.tz is None:
        df = df.copy()
        df["time"] = df["time"].dt.tz_localize("UTC")

    start_ts = pd.Timestamp(start_utc)
    end_ts = pd.Timestamp(end_utc)
    if start_ts.tz is None:
        start_ts = start_ts.tz_localize("UTC")
    if end_ts.tz is None:
        end_ts = end_ts.tz_localize("UTC")

    mask = (df["time"] >= start_ts) & (df["time"] <= end_ts)
    return df.loc[mask].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# CLONE LIVE MODEL INTO DREAM  (optional warm-start)
# ═══════════════════════════════════════════════════════════════════════════

def _clone_live_model(ns: DreamNamespace, symbol: str, horizon: str) -> bool:
    """Copy the live model/scaler/metadata into the dream namespace."""
    src_model = live_model_dir(symbol, horizon)
    src_state = live_state_dir(symbol, horizon)
    dst = ns.model_dir(symbol, horizon)
    dst.mkdir(parents=True, exist_ok=True)

    copied = False
    for filename in ("model.pkl", "scaler.pkl"):
        src = src_model / filename
        if src.exists():
            try:
                shutil.copy2(str(src), str(dst / filename))
                copied = True
            except Exception as ex:
                print(f"[DreamState] Clone {filename} failed: {ex}", file=sys.stderr)

    meta_src = src_state / "metadata.json"
    if meta_src.exists():
        try:
            shutil.copy2(str(meta_src), str(dst / "metadata.json"))
        except Exception as ex:
            print(f"[DreamState] Clone metadata.json failed: {ex}", file=sys.stderr)

    return copied


# ═══════════════════════════════════════════════════════════════════════════
# VERB: CREATE DREAM
# ═══════════════════════════════════════════════════════════════════════════

def create_dream(
    symbol: str,
    horizon: str,
    interval: str,
    replay_start: str,
    replay_end: str,
    model_key: str = "",
    feature_version: str = "",
    config: dict | None = None,
) -> dict:
    """
    Create a new dream namespace and return its manifest.

    Config keys:
      warm_start: bool (default True) - clone live model into dream
      label_policy: dict - override label policy for this dream
      resolution_policy: dict - override resolution policy
      training_policy: dict - override training policy
      horizon_bars: int - bars per horizon (default 24)
    """
    cfg = config or {}
    fv = feature_version or FEATURE_VERSION
    dream_id = f"dream_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{symbol.lower()}_{horizon}"

    ns = DreamNamespace(dream_id)
    ns.ensure_dirs(symbol, horizon)

    # ── Compute total steps from replay window ──
    start_dt = _parse_utc(replay_start)
    end_dt = _parse_utc(replay_end)
    total_steps = 0
    if start_dt and end_dt:
        bar_delta = _INTERVAL_MAP.get(interval.lower())
        if bar_delta and bar_delta.total_seconds() > 0:
            total_seconds = (end_dt - start_dt).total_seconds()
            total_steps = max(0, int(total_seconds / bar_delta.total_seconds()))

    manifest = DreamManifest(
        dream_id=dream_id,
        created_utc=datetime.now(timezone.utc).isoformat(),
        symbol=symbol.upper(),
        horizon=horizon,
        interval=interval,
        replay_start_utc=replay_start,
        replay_end_utc=replay_end,
        model_key=model_key,
        feature_version=fv,
        status="created",
        total_steps=total_steps,
        config=cfg,
    )
    manifest.save()

    # ── Optional warm-start: clone live model ──
    warm = cfg.get("warm_start", True)
    cloned = False
    if warm:
        cloned = _clone_live_model(ns, symbol, horizon)
        if cloned:
            print(f"[DreamState] Cloned live model into dream {dream_id}", file=sys.stderr)

    return {
        "ok": True,
        "dream_id": dream_id,
        "manifest": manifest.to_dict(),
        "model_cloned": cloned,
        "total_steps": total_steps,
    }


# ═══════════════════════════════════════════════════════════════════════════
# VERB: RUN DREAM STEP  (predict at virtual time T, advance clock)
# ═══════════════════════════════════════════════════════════════════════════

def run_dream_step(dream_id: str, step_payload: dict) -> dict:
    """
    Execute one simulation step inside a dream.

    step_payload:
      virtual_utc: str       - current virtual timestamp
      features_payload: dict  - metabolic payload (same schema as live)
      step_number: int        - ordinal step index (optional)

    Returns step result with prediction, virtual time, step number.
    """
    manifest = DreamManifest.load(dream_id)
    if manifest is None:
        return {"ok": False, "error": "dream_not_found", "dream_id": dream_id}

    if manifest.status == "aborted":
        return {"ok": False, "error": "dream_aborted", "dream_id": dream_id}

    if manifest.status == "completed":
        return {"ok": False, "error": "dream_already_completed", "dream_id": dream_id}

    ns = DreamNamespace(dream_id)
    symbol = manifest.symbol
    horizon = manifest.horizon
    interval = manifest.interval

    virtual_utc = step_payload.get("virtual_utc", "")
    features_payload = step_payload.get("features_payload", {})
    step_number = step_payload.get("step_number", manifest.current_step + 1)

    # ── Update status to running ──
    if manifest.status == "created":
        manifest.status = "running"

    # ── Load dream model ──
    model = _dream_load_model(ns, symbol, horizon)

    # ── Extract features ──
    features = extract_features(features_payload)

    # ── Predict ──
    result = model.predict(features)

    # ── Generate prediction ID ──
    prediction_id = uuid.uuid4().hex[:16]

    # ── Store pending in dream namespace ──
    horizon_bars = manifest.config.get("horizon_bars", 24)
    sample = {
        "prediction_id": prediction_id,
        "asof_utc": virtual_utc,
        "stored_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "horizon": horizon,
        "interval": interval,
        "active_horizon": horizon,
        "horizon_bars": horizon_bars,
        "model_key": manifest.model_key,
        "feature_version": manifest.feature_version,
        "features": features,
        "predicted_class": result["predicted_class"],
        "predicted_probabilities": result["probabilities"],
        "regime_probabilities": {},
        "event_probabilities": {},
        "priority_score": 0.0,
        "observation_cutoff_utc": virtual_utc,
        "point_in_time_safe": True,
        "temporal_policy_version": "",
        "macro_tags": [],
        "headline_tags": [],
        "scheduled_event_types": [],
        "eligible_for_training": True,
        "learning_block_reasons": [],
        "entry_price": features_payload.get("technical", {}).get("price") if isinstance(features_payload.get("technical"), dict) else None,
        "price_basis": "close",
        "source_event_id": None,
        "resolved": False,
        "dream_id": dream_id,
    }
    _dream_store_pending(ns, symbol, horizon, sample)

    # ── Update manifest progress ──
    manifest.current_step = step_number
    manifest.metrics["total_predictions"] = manifest.metrics.get("total_predictions", 0) + 1
    if manifest.total_steps > 0:
        manifest.progress_pct = round(step_number / manifest.total_steps * 100.0, 2)
    manifest.save()

    return {
        "ok": True,
        "dream_id": dream_id,
        "step_number": step_number,
        "virtual_utc": virtual_utc,
        "prediction_id": prediction_id,
        "predicted_class": result["predicted_class"],
        "probabilities": result["probabilities"],
        "confidence": result["confidence"],
        "model_state": model.model_state,
        "progress_pct": manifest.progress_pct,
    }


# ═══════════════════════════════════════════════════════════════════════════
# VERB: RESOLVE DREAM  (resolve pending predictions against parquet truth)
# ═══════════════════════════════════════════════════════════════════════════

def resolve_dream(dream_id: str) -> dict:
    """
    After replay is complete, resolve all pending dream predictions against
    the historical parquet truth.  Uses the shared OHLCV data lake (read-only).
    """
    manifest = DreamManifest.load(dream_id)
    if manifest is None:
        return {"ok": False, "error": "dream_not_found", "dream_id": dream_id}

    ns = DreamNamespace(dream_id)
    symbol = manifest.symbol
    horizon = manifest.horizon
    interval = manifest.interval
    warnings: list[str] = []

    # ── Load dream pending samples ──
    pending = _dream_load_pending(ns, symbol, horizon)
    if not pending:
        return {
            "ok": True,
            "dream_id": dream_id,
            "resolved_count": 0,
            "deferred_count": 0,
            "expired_count": 0,
            "warnings": ["no_pending_samples"],
        }

    print(f"[DreamState] Resolving dream {dream_id}: {len(pending)} pending", file=sys.stderr)

    # ── Load OHLCV truth ──
    ohlcv_df = _load_ohlcv_truth(symbol, interval, warnings)

    # ── Parse policy overrides from config ──
    label_policy = DEFAULT_LABEL_POLICY
    resolution_policy = DEFAULT_RESOLUTION_POLICY
    if manifest.config.get("label_policy"):
        try:
            label_policy = LabelPolicy.from_dict(manifest.config["label_policy"])
        except Exception:
            pass
    if manifest.config.get("resolution_policy"):
        try:
            resolution_policy = ResolutionPolicy.from_dict(manifest.config["resolution_policy"])
        except Exception:
            pass

    # ── Run batch resolution ──
    result = resolve_pending_batch(
        pending_samples=pending,
        ohlcv_df=ohlcv_df,
        label_policy=label_policy,
        resolution_policy=resolution_policy,
    )

    summary = result.summary()

    # ── Append resolved to dream truth archive ──
    if result.resolved:
        written = _dream_append_resolved(ns, symbol, horizon, result.resolved)
        print(f"[DreamState] Wrote {written} dream resolved records", file=sys.stderr)

    # ── Rewrite dream pending ──
    resolved_ids = {r["prediction_id"] for r in result.resolved if r.get("prediction_id")}
    expired_ids = {s.get("prediction_id", "") for s in result.expired if s.get("prediction_id")}
    if resolved_ids or expired_ids:
        _dream_rewrite_pending(ns, symbol, horizon, resolved_ids | expired_ids)

    # ── Update manifest metrics ──
    manifest.metrics["total_resolved"] = manifest.metrics.get("total_resolved", 0) + len(result.resolved)
    manifest.metrics["dream_accuracy"] = summary.get("accuracy", 0.0)
    manifest.metrics["dream_brier"] = summary.get("mean_brier_score", 0.0)
    manifest.save()

    warnings.extend(result.warnings)

    return {
        "ok": True,
        "dream_id": dream_id,
        "resolution_summary": summary,
        "resolved_count": len(result.resolved),
        "deferred_count": len(result.deferred),
        "expired_count": len(result.expired),
        "errored_count": len(result.errored),
        "warnings": warnings,
    }


# ═══════════════════════════════════════════════════════════════════════════
# VERB: TRAIN DREAM  (incremental training on resolved dream data)
# ═══════════════════════════════════════════════════════════════════════════

def train_dream(dream_id: str) -> dict:
    """
    Train the dream model on its resolved data using the same
    cursor-aware controlled_fit pipeline as live training.
    """
    manifest = DreamManifest.load(dream_id)
    if manifest is None:
        return {"ok": False, "error": "dream_not_found", "dream_id": dream_id}

    ns = DreamNamespace(dream_id)
    symbol = manifest.symbol
    horizon = manifest.horizon
    warnings: list[str] = []

    # ── Parse training policy override ──
    training_policy = DEFAULT_TRAINING_POLICY
    if manifest.config.get("training_policy"):
        try:
            training_policy = TrainingPolicy.from_dict(manifest.config["training_policy"])
        except Exception:
            pass

    # ── Load dream cursor ──
    cursor = _dream_load_cursor(ns, symbol, horizon)

    # ── Load dream resolved samples, split by cursor ──
    all_resolved = _dream_load_resolved(ns, symbol, horizon)
    fresh: list[dict] = []
    replay_pool: list[dict] = []
    for s in all_resolved:
        pid = s.get("prediction_id", "")
        if pid and pid in cursor.consumed_ids:
            replay_pool.append(s)
        else:
            fresh.append(s)

    if not fresh:
        return {
            "ok": True,
            "dream_id": dream_id,
            "samples_fitted": 0,
            "warnings": ["no_fresh_resolved_samples"],
        }

    print(
        f"[DreamState] Training dream {dream_id}: "
        f"fresh={len(fresh)}, replay_pool={len(replay_pool)}",
        file=sys.stderr,
    )

    # ── Load dream model ──
    model = _dream_load_model(ns, symbol, horizon)

    # ── Controlled fit ──
    train_result = model.controlled_fit(
        fresh_samples=fresh,
        replay_samples=replay_pool,
        policy=training_policy,
    )

    # ── Update cursor ──
    if train_result.samples_fitted > 0:
        fresh_ids = [s.get("prediction_id", "") for s in fresh if s.get("prediction_id")]
        cursor.mark_consumed(fresh_ids, training_policy.version)
        cursor.prune_old_ids()
        _dream_save_cursor(ns, cursor)
        _dream_save_model(ns, symbol, horizon, model)
        print(
            f"[DreamState] Dream training complete: fitted={train_result.samples_fitted}",
            file=sys.stderr,
        )

    if train_result.drift_flags:
        warnings.extend([f"drift:{f}" for f in train_result.drift_flags])

    # ── Update manifest metrics ──
    manifest.metrics["total_trained"] = manifest.metrics.get("total_trained", 0) + train_result.samples_fitted
    manifest.save()

    return {
        "ok": True,
        "dream_id": dream_id,
        "train_result": train_result.to_dict(),
        "cursor_sequence": cursor.sequence,
        "consumed_count": len(cursor.consumed_ids),
        "warnings": warnings + train_result.warnings,
    }


# ═══════════════════════════════════════════════════════════════════════════
# VERB: EVALUATE DREAM  (scorecard + live comparison)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_dream(dream_id: str) -> dict:
    """
    Run full evaluation on the dream's resolved history and compare
    dream model performance vs the live model.
    """
    manifest = DreamManifest.load(dream_id)
    if manifest is None:
        return {"ok": False, "error": "dream_not_found", "dream_id": dream_id}

    ns = DreamNamespace(dream_id)
    symbol = manifest.symbol
    horizon = manifest.horizon
    warnings: list[str] = []

    # ── Load dream resolved samples ──
    dream_resolved = _dream_load_resolved(ns, symbol, horizon)
    if not dream_resolved:
        return {
            "ok": True,
            "dream_id": dream_id,
            "dream_scorecard": None,
            "live_scorecard": None,
            "comparison": None,
            "warnings": ["no_dream_resolved_history"],
        }

    # ── Compute dream scorecard ──
    dream_sc = compute_scorecard(dream_resolved, DEFAULT_SCORECARD_POLICY)

    # ── Compute live scorecard over the same resolved samples (if available) ──
    live_sc = None
    try:
        live_resolved = _live_load_resolved(symbol, horizon)
        if live_resolved:
            live_sc = compute_scorecard(live_resolved, DEFAULT_SCORECARD_POLICY)
    except Exception as ex:
        warnings.append(f"live_scorecard_error:{ex}")

    # ── Comparison ──
    comparison = None
    if live_sc is not None and dream_sc.get("status") == "ok" and live_sc.get("status") == "ok":
        comparison = {
            "brier_diff": _safe_diff(dream_sc.get("mean_brier_score"), live_sc.get("mean_brier_score")),
            "accuracy_diff": _safe_diff(dream_sc.get("accuracy"), live_sc.get("accuracy")),
            "calibration_diff": _safe_diff(dream_sc.get("mean_calibration_gap"), live_sc.get("mean_calibration_gap")),
            "dream_sample_count": dream_sc.get("sample_count", 0),
            "live_sample_count": live_sc.get("sample_count", 0),
            "interpretation": {
                "brier": "negative = dream better (lower Brier)",
                "accuracy": "positive = dream better (higher accuracy)",
                "calibration": "negative = dream better (lower gap)",
            },
        }

    # ── Update manifest status ──
    manifest.status = "completed"
    manifest.metrics["dream_accuracy"] = dream_sc.get("accuracy", 0.0)
    manifest.metrics["dream_brier"] = dream_sc.get("mean_brier_score", 0.0)
    manifest.save()

    return {
        "ok": True,
        "dream_id": dream_id,
        "dream_scorecard": dream_sc,
        "live_scorecard": live_sc,
        "comparison": comparison,
        "manifest": manifest.to_dict(),
        "warnings": warnings,
    }


# ═══════════════════════════════════════════════════════════════════════════
# VERB: GET DREAM STATUS
# ═══════════════════════════════════════════════════════════════════════════

def get_dream_status(dream_id: str) -> dict:
    """Return current dream state: manifest, progress, metrics, counts."""
    manifest = DreamManifest.load(dream_id)
    if manifest is None:
        return {"ok": False, "error": "dream_not_found", "dream_id": dream_id}

    ns = DreamNamespace(dream_id)
    symbol = manifest.symbol
    horizon = manifest.horizon

    # Count pending/resolved
    pending = _dream_load_pending(ns, symbol, horizon)
    resolved = _dream_load_resolved(ns, symbol, horizon)

    return {
        "ok": True,
        "dream_id": dream_id,
        "manifest": manifest.to_dict(),
        "pending_count": len(pending),
        "resolved_count": len(resolved),
    }


# ═══════════════════════════════════════════════════════════════════════════
# VERB: LIST DREAMS
# ═══════════════════════════════════════════════════════════════════════════

def list_dreams() -> list[dict]:
    """List all dream manifests in the data lake."""
    dreams_dir = _cortex_root() / "dreams"
    if not dreams_dir.exists():
        return []

    manifests: list[dict] = []
    try:
        for entry in sorted(dreams_dir.iterdir()):
            if not entry.is_dir():
                continue
            manifest_path = entry / "manifest.json"
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r") as f:
                        manifests.append(json.load(f))
                except Exception:
                    continue
    except Exception as ex:
        print(f"[DreamState] Failed to list dreams: {ex}", file=sys.stderr)

    return manifests


# ═══════════════════════════════════════════════════════════════════════════
# VERB: ABORT DREAM
# ═══════════════════════════════════════════════════════════════════════════

def abort_dream(dream_id: str) -> dict:
    """Mark a dream as aborted.  Does not delete files."""
    manifest = DreamManifest.load(dream_id)
    if manifest is None:
        return {"ok": False, "error": "dream_not_found", "dream_id": dream_id}

    manifest.status = "aborted"
    manifest.save()

    print(f"[DreamState] Dream {dream_id} aborted", file=sys.stderr)

    return {
        "ok": True,
        "dream_id": dream_id,
        "status": "aborted",
        "manifest": manifest.to_dict(),
    }


# ═══════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _parse_utc(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _safe_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    try:
        return round(float(a) - float(b), 5)
    except (TypeError, ValueError):
        return None
