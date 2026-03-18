"""
pending_memory.py - Store/load pending unresolved samples for delayed labeling.

v2: Expanded schema for Phase 9.5 — prediction_id, model_key, temporal
security metadata, regime/event probabilities, eligibility tracking.

Storage layout:
  data_lake/cortex/pending/{symbol}/{horizon}/pending.jsonl
  data_lake/cortex/resolved/{symbol}/{horizon}/resolved.jsonl
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _cortex_root() -> Path:
    ml_dir = Path(__file__).parent
    python_dir = ml_dir.parent
    aether_dir = python_dir.parent
    content_root = aether_dir.parent
    return content_root / "data_lake" / "cortex"


def _pending_path(symbol: str, horizon: str) -> Path:
    return _cortex_root() / "pending" / symbol.upper() / horizon / "pending.jsonl"


def _resolved_path(symbol: str, horizon: str) -> Path:
    return _cortex_root() / "resolved" / symbol.upper() / horizon / "resolved.jsonl"


def store_pending_sample(
    symbol: str,
    horizon: str,
    features: list[float],
    predicted_class: str,
    asof_utc: str,
    # v2 expanded fields
    prediction_id: str = "",
    model_key: str = "",
    interval: str = "1h",
    active_horizon: str = "",
    horizon_bars: int = 24,
    observation_cutoff_utc: str | None = None,
    point_in_time_safe: bool = True,
    temporal_policy_version: str = "",
    feature_version: str = "",
    predicted_probabilities: dict | None = None,
    regime_probabilities: dict | None = None,
    event_probabilities: dict | None = None,
    priority_score: float = 0.0,
    macro_tags: list[str] | None = None,
    headline_tags: list[str] | None = None,
    scheduled_event_types: list[str] | None = None,
    eligible_for_training: bool = True,
    learning_block_reasons: list[str] | None = None,
    entry_price: float | None = None,
    price_basis: str = "close",
    source_event_id: str | None = None,
) -> bool:
    """
    Store a pending sample for future label resolution.
    Returns True if stored successfully.
    """
    path = _pending_path(symbol, horizon)
    path.parent.mkdir(parents=True, exist_ok=True)

    sample = {
        "prediction_id": prediction_id,
        "asof_utc": asof_utc,
        "stored_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol.upper(),
        "horizon": horizon,
        "interval": interval,
        "active_horizon": active_horizon or horizon,
        "horizon_bars": horizon_bars,
        "model_key": model_key,
        "feature_version": feature_version,
        "features": features,
        "predicted_class": predicted_class,
        "predicted_probabilities": predicted_probabilities or {},
        "regime_probabilities": regime_probabilities or {},
        "event_probabilities": event_probabilities or {},
        "priority_score": round(priority_score, 4),
        # temporal security
        "observation_cutoff_utc": observation_cutoff_utc,
        "point_in_time_safe": point_in_time_safe,
        "temporal_policy_version": temporal_policy_version,
        # context tags
        "macro_tags": macro_tags or [],
        "headline_tags": headline_tags or [],
        "scheduled_event_types": scheduled_event_types or [],
        # eligibility
        "eligible_for_training": eligible_for_training,
        "learning_block_reasons": learning_block_reasons or [],
        # price
        "entry_price": entry_price,
        "price_basis": price_basis,
        "source_event_id": source_event_id,
        "resolved": False,
    }

    try:
        with open(path, "a") as f:
            f.write(json.dumps(sample, separators=(",", ":")) + "\n")
        return True
    except Exception as ex:
        print(f"[MlCortex] Failed to store pending sample: {ex}", file=sys.stderr)
        return False


def load_pending_samples(symbol: str, horizon: str, max_samples: int = 1000) -> list[dict]:
    """Load pending (unresolved) samples."""
    path = _pending_path(symbol, horizon)
    if not path.exists():
        return []

    samples = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    if not sample.get("resolved", False):
                        samples.append(sample)
                except json.JSONDecodeError:
                    continue
                if len(samples) >= max_samples:
                    break
    except Exception as ex:
        print(f"[MlCortex] Failed to load pending samples: {ex}", file=sys.stderr)

    return samples


def load_eligible_pending_samples(symbol: str, horizon: str, max_samples: int = 1000) -> list[dict]:
    """Load only training-eligible pending samples."""
    all_pending = load_pending_samples(symbol, horizon, max_samples=max_samples * 2)
    return [s for s in all_pending if s.get("eligible_for_training", True)][:max_samples]


def store_resolved_sample(
    symbol: str,
    horizon: str,
    features: list[float],
    label: str,
    asof_utc: str,
    resolution_utc: str | None = None,
    # v2 expanded fields
    prediction_id: str = "",
    model_key: str = "",
    feature_version: str = "",
    target_bar_utc: str | None = None,
    target_price: float | None = None,
    future_return_bps: float | None = None,
    label_policy_version: str = "",
    resolved_without_lookahead: bool = True,
) -> bool:
    """Store a resolved (labeled) sample for training."""
    path = _resolved_path(symbol, horizon)
    path.parent.mkdir(parents=True, exist_ok=True)

    sample = {
        "prediction_id": prediction_id,
        "asof_utc": asof_utc,
        "resolution_utc": resolution_utc or datetime.now(timezone.utc).isoformat(),
        "symbol": symbol.upper(),
        "horizon": horizon,
        "model_key": model_key,
        "feature_version": feature_version,
        "features": features,
        "label": label,
        "target_bar_utc": target_bar_utc,
        "target_price": target_price,
        "future_return_bps": future_return_bps,
        "label_policy_version": label_policy_version,
        "resolved_without_lookahead": resolved_without_lookahead,
    }

    try:
        with open(path, "a") as f:
            f.write(json.dumps(sample, separators=(",", ":")) + "\n")
        return True
    except Exception as ex:
        print(f"[MlCortex] Failed to store resolved sample: {ex}", file=sys.stderr)
        return False


def load_resolved_samples(symbol: str, horizon: str, max_samples: int = 10000) -> list[dict]:
    """Load resolved (labeled) samples for training."""
    path = _resolved_path(symbol, horizon)
    if not path.exists():
        return []

    samples = []
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
                if len(samples) >= max_samples:
                    break
    except Exception as ex:
        print(f"[MlCortex] Failed to load resolved samples: {ex}", file=sys.stderr)

    return samples


def pending_count(symbol: str, horizon: str) -> int:
    """Count pending samples without loading them all."""
    path = _pending_path(symbol, horizon)
    if not path.exists():
        return 0
    try:
        with open(path, "r") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0


def pending_eligible_count(symbol: str, horizon: str) -> int:
    """Count training-eligible pending samples."""
    path = _pending_path(symbol, horizon)
    if not path.exists():
        return 0
    eligible = 0
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                    if not s.get("resolved", False) and s.get("eligible_for_training", True):
                        eligible += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return eligible


def pending_blocked_count(symbol: str, horizon: str) -> int:
    """Count training-blocked pending samples."""
    path = _pending_path(symbol, horizon)
    if not path.exists():
        return 0
    blocked = 0
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                    if not s.get("resolved", False) and not s.get("eligible_for_training", True):
                        blocked += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return blocked
