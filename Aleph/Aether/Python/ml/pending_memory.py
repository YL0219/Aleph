"""
pending_memory.py — Complete lifecycle for pending/resolved prediction memory.

v3: Full lifecycle with:
  - pending.jsonl   — active unresolved queue
  - resolved.jsonl  — append-only truth archive
  - atomic pending rewrite after resolve (unresolved stay, resolved move out)
  - rich resolved record schema with full provenance and grading sidecars

Storage layout:
  data_lake/cortex/pending/{symbol}/{horizon}/pending.jsonl
  data_lake/cortex/resolved/{symbol}/{horizon}/resolved.jsonl
"""

from __future__ import annotations

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


# ═══════════════════════════════════════════════════════════════════
# PENDING STORAGE
# ═══════════════════════════════════════════════════════════════════

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


def load_pending_samples(symbol: str, horizon: str, max_samples: int = 10000) -> list[dict]:
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


# ═══════════════════════════════════════════════════════════════════
# RESOLVED STORAGE (APPEND-ONLY TRUTH ARCHIVE)
# ═══════════════════════════════════════════════════════════════════

def append_resolved_samples(
    symbol: str,
    horizon: str,
    resolved_records: list[dict],
) -> int:
    """
    Append resolved records to the truth archive.
    Returns count of successfully written records.
    """
    if not resolved_records:
        return 0

    path = _resolved_path(symbol, horizon)
    path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    try:
        with open(path, "a") as f:
            for record in resolved_records:
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
                written += 1
    except Exception as ex:
        print(f"[MlCortex] Failed to append resolved samples: {ex}", file=sys.stderr)

    return written


def store_resolved_sample(
    symbol: str,
    horizon: str,
    features: list[float],
    label: str,
    asof_utc: str,
    resolution_utc: str | None = None,
    prediction_id: str = "",
    model_key: str = "",
    feature_version: str = "",
    target_bar_utc: str | None = None,
    target_price: float | None = None,
    future_return_bps: float | None = None,
    label_policy_version: str = "",
    resolved_without_lookahead: bool = True,
) -> bool:
    """Legacy single-sample resolved store (backward compat). Prefer append_resolved_samples."""
    record = {
        "prediction_id": prediction_id,
        "asof_utc": asof_utc,
        "resolution_utc": resolution_utc or datetime.now(timezone.utc).isoformat(),
        "symbol": symbol.upper(),
        "horizon": horizon,
        "model_key": model_key,
        "feature_version": feature_version,
        "features": features,
        "label": label,
        "actual_label": label,
        "target_bar_utc": target_bar_utc,
        "target_price": target_price,
        "future_return_bps": future_return_bps,
        "realized_return_bps": future_return_bps,
        "label_policy_version": label_policy_version,
        "resolved_without_lookahead": resolved_without_lookahead,
        "eligible_for_training": True,
        "learning_block_reasons": [],
    }
    return append_resolved_samples(symbol, horizon, [record]) == 1


def load_resolved_samples(symbol: str, horizon: str, max_samples: int = 50000) -> list[dict]:
    """Load resolved (labeled) samples from the truth archive."""
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


def load_resolved_since_cursor(
    symbol: str,
    horizon: str,
    consumed_ids: set[str],
    max_samples: int = 10000,
) -> tuple[list[dict], list[dict]]:
    """
    Load resolved samples, splitting into:
      - fresh: prediction_id NOT in consumed_ids
      - replay_pool: prediction_id IS in consumed_ids (already trained)

    Returns (fresh, replay_pool).
    """
    all_resolved = load_resolved_samples(symbol, horizon, max_samples=max_samples)
    fresh = []
    replay = []
    for s in all_resolved:
        pid = s.get("prediction_id", "")
        if pid and pid in consumed_ids:
            replay.append(s)
        else:
            fresh.append(s)
    return fresh, replay


# ═══════════════════════════════════════════════════════════════════
# ATOMIC PENDING REWRITE
# ═══════════════════════════════════════════════════════════════════

def rewrite_pending_after_resolve(
    symbol: str,
    horizon: str,
    resolved_ids: set[str],
    expired_ids: set[str] | None = None,
) -> dict:
    """
    Atomically rewrite pending.jsonl:
      - Remove samples whose prediction_id is in resolved_ids or expired_ids
      - Keep all other unresolved samples

    Uses write-to-tmp-then-rename for safety.

    Returns:
      {"kept": int, "removed": int, "expired": int}
    """
    path = _pending_path(symbol, horizon)
    expired_ids = expired_ids or set()
    all_remove = resolved_ids | expired_ids

    if not path.exists():
        return {"kept": 0, "removed": 0, "expired": 0}

    # Read all lines
    try:
        with open(path, "r") as f:
            lines = f.readlines()
    except Exception as ex:
        print(f"[MlCortex] Failed to read pending for rewrite: {ex}", file=sys.stderr)
        return {"kept": 0, "removed": 0, "expired": 0}

    kept_lines = []
    removed = 0
    expired_count = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            sample = json.loads(line)
            pid = sample.get("prediction_id", "")
            if pid in all_remove:
                if pid in expired_ids:
                    expired_count += 1
                else:
                    removed += 1
                continue
            kept_lines.append(line)
        except json.JSONDecodeError:
            # Drop malformed lines
            continue

    # Write atomically
    tmp_path = path.with_suffix(".tmp")
    try:
        with open(tmp_path, "w") as f:
            for line in kept_lines:
                f.write(line + "\n")
        if os.name == "nt" and path.exists():
            os.remove(path)
        os.rename(tmp_path, path)
    except Exception as ex:
        print(f"[MlCortex] Failed to rewrite pending: {ex}", file=sys.stderr)
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return {"kept": len(kept_lines), "removed": 0, "expired": 0}

    return {"kept": len(kept_lines), "removed": removed, "expired": expired_count}


# ═══════════════════════════════════════════════════════════════════
# COUNTING HELPERS
# ═══════════════════════════════════════════════════════════════════

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


def resolved_count(symbol: str, horizon: str) -> int:
    """Count resolved samples."""
    path = _resolved_path(symbol, horizon)
    if not path.exists():
        return 0
    try:
        with open(path, "r") as f:
            return sum(1 for line in f if line.strip())
    except Exception:
        return 0
