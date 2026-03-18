"""
training_cursor.py — Tracks which resolved samples have been consumed by training.

Prevents accidental double-training of the same samples across sleep cycles.
The cursor stores the set of prediction_ids that have already been used for
incremental fitting, plus a monotonic sequence number.

Storage:
  data_lake/cortex/cursor/{symbol}/{horizon}/cursor.json

The cursor is small, atomic-written, and self-describing.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _cortex_root() -> Path:
    ml_dir = Path(__file__).parent
    python_dir = ml_dir.parent
    aether_dir = python_dir.parent
    content_root = aether_dir.parent
    return content_root / "data_lake" / "cortex"


def _cursor_path(symbol: str, horizon: str) -> Path:
    return _cortex_root() / "cursor" / symbol.upper() / horizon / "cursor.json"


def load_cursor(symbol: str, horizon: str) -> TrainingCursor:
    """Load cursor from disk, or return a fresh empty cursor."""
    path = _cursor_path(symbol, horizon)
    if not path.exists():
        return TrainingCursor(symbol=symbol.upper(), horizon=horizon)

    try:
        with open(path, "r") as f:
            data = json.load(f)
        return TrainingCursor.from_dict(data)
    except Exception as ex:
        print(f"[TrainingCursor] Failed to load cursor: {ex}", file=sys.stderr)
        return TrainingCursor(symbol=symbol.upper(), horizon=horizon)


def save_cursor(cursor: TrainingCursor) -> bool:
    """Atomically write cursor to disk (write-tmp then rename)."""
    path = _cursor_path(cursor.symbol, cursor.horizon)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")

    try:
        with open(tmp_path, "w") as f:
            json.dump(cursor.to_dict(), f, indent=2)
        # Atomic rename (POSIX) / best-effort on Windows
        if os.name == "nt":
            if path.exists():
                os.remove(path)
        os.rename(tmp_path, path)
        return True
    except Exception as ex:
        print(f"[TrainingCursor] Failed to save cursor: {ex}", file=sys.stderr)
        # Clean up tmp if it lingers
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return False


class TrainingCursor:
    """
    Tracks training consumption state for a single symbol/horizon.

    Attributes:
      symbol              — asset symbol
      horizon             — prediction horizon key
      sequence            — monotonically increasing training-cycle counter
      consumed_ids        — set of prediction_ids already used for training
      last_train_utc      — ISO timestamp of last training cycle
      last_train_policy   — training policy version used in last cycle
      total_samples_ever  — cumulative count of all samples ever consumed
    """

    def __init__(
        self,
        symbol: str = "",
        horizon: str = "",
        sequence: int = 0,
        consumed_ids: set[str] | None = None,
        last_train_utc: str | None = None,
        last_train_policy: str = "",
        total_samples_ever: int = 0,
    ):
        self.symbol = symbol
        self.horizon = horizon
        self.sequence = sequence
        self.consumed_ids: set[str] = consumed_ids or set()
        self.last_train_utc = last_train_utc
        self.last_train_policy = last_train_policy
        self.total_samples_ever = total_samples_ever

    def is_consumed(self, prediction_id: str) -> bool:
        """Check if a prediction_id has already been consumed by training."""
        return prediction_id in self.consumed_ids

    def mark_consumed(self, prediction_ids: list[str], policy_version: str) -> None:
        """Mark a batch of prediction_ids as consumed and advance the cursor."""
        self.consumed_ids.update(prediction_ids)
        self.sequence += 1
        self.last_train_utc = datetime.now(timezone.utc).isoformat()
        self.last_train_policy = policy_version
        self.total_samples_ever += len(prediction_ids)

    def get_unconsumed(self, candidate_ids: list[str]) -> list[str]:
        """Filter to only prediction_ids not yet consumed."""
        return [pid for pid in candidate_ids if pid not in self.consumed_ids]

    def prune_old_ids(self, max_ids: int = 50_000) -> int:
        """
        Prune the oldest consumed IDs if the set grows too large.
        We can't perfectly age-order hex IDs, but we can cap the set size.
        Returns count of pruned IDs.
        """
        if len(self.consumed_ids) <= max_ids:
            return 0
        excess = len(self.consumed_ids) - max_ids
        # Remove arbitrary excess (set has no order, but that's acceptable
        # since very old IDs will never reappear in resolved.jsonl anyway)
        pruned_ids = list(self.consumed_ids)[:excess]
        for pid in pruned_ids:
            self.consumed_ids.discard(pid)
        return excess

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "horizon": self.horizon,
            "sequence": self.sequence,
            "consumed_ids": sorted(self.consumed_ids),
            "last_train_utc": self.last_train_utc,
            "last_train_policy": self.last_train_policy,
            "total_samples_ever": self.total_samples_ever,
            "cursor_version": "cursor_v1",
        }

    @classmethod
    def from_dict(cls, d: dict) -> TrainingCursor:
        return cls(
            symbol=d.get("symbol", ""),
            horizon=d.get("horizon", ""),
            sequence=d.get("sequence", 0),
            consumed_ids=set(d.get("consumed_ids", [])),
            last_train_utc=d.get("last_train_utc"),
            last_train_policy=d.get("last_train_policy", ""),
            total_samples_ever=d.get("total_samples_ever", 0),
        )
