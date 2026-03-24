"""
perception_snapshot.py — Read-side helper for the Aleph perception data lake.

Reads the latest perception artifacts (manifest, proxies, calendar, headlines)
and returns a consolidated snapshot for downstream consumers (Liver, ML Cortex).

This module never fetches from external sources — it only reads local files.
If artifacts are stale or missing, the snapshot reports this via freshness flags
so the consumer can decide whether to proceed with degraded context.

Output contract:
  - stdout: single JSON object with the consolidated perception snapshot
  - stderr: debug logs
  - exit code 0: snapshot produced (sections may be missing/stale)
  - exit code 1: fatal error

Compatible with Python 3.10+.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Any


try:
    import pandas as pd
except ImportError:
    pd = None


def _sanitize_for_json(value: Any) -> Any:
    """Recursively map NaN/NA-like values to None for strict JSON output."""
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(v) for v in value]

    if value is None:
        return None

    if isinstance(value, float) and math.isnan(value):
        return None

    if pd is not None:
        try:
            if bool(pd.isna(value)):
                return None
        except Exception:
            pass

    return value


# ── Constants ────────────────────────────────────────────────────────

MANIFEST_PATH = "data_lake/perception/manifest.json"
CALENDAR_PATH = "data_lake/macro/calendar/latest.json"
HEADLINES_PATH = "data_lake/macro/headlines/latest.json"
PROXY_ROOT = "data_lake/macro/proxies"

# Staleness thresholds (hours)
PROXY_STALE_HOURS = 26       # Proxies are daily — 26h gives buffer
CALENDAR_STALE_HOURS = 72    # Calendar events move slowly
HEADLINES_STALE_HOURS = 6    # Headlines go stale faster

DEFAULT_PROXY_NAMES = ["vix", "dxy", "btc", "spy", "qqq", "tlt", "gld"]


def _log(msg: str) -> None:
    print("[perception-snapshot] {}".format(msg), file=sys.stderr)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _is_stale(fetched_at: str | None, threshold_hours: float) -> bool:
    """Check if a timestamp is older than threshold_hours from now."""
    dt = _parse_iso(fetched_at)
    if dt is None:
        return True
    age = _now_utc() - dt
    return age > timedelta(hours=threshold_hours)


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: READ MANIFEST
# ═══════════════════════════════════════════════════════════════════

def read_manifest() -> dict[str, Any] | None:
    """Read the perception manifest. Returns None if missing."""
    if not os.path.exists(MANIFEST_PATH):
        _log("Manifest not found at {}".format(MANIFEST_PATH))
        return None

    try:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        _log("Failed to read manifest: {}".format(e))
        return None


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: READ PROXY SUMMARIES
# ═══════════════════════════════════════════════════════════════════

def read_proxy_summaries(proxy_names: list[str] | None = None) -> dict[str, Any]:
    """
    Read proxy parquet files and return latest close + basic stats for each.
    Does NOT load full DataFrames into the snapshot — just summary metrics.
    """
    names = proxy_names or DEFAULT_PROXY_NAMES
    summaries = {}

    if pd is None:
        _log("pandas not available — skipping proxy summaries")
        return {"available": False, "error": "pandas not installed", "proxies": {}}

    for name in names:
        parquet_path = os.path.join(PROXY_ROOT, name, "latest.parquet")
        entry = {
            "available": False,
            "parquetPath": parquet_path.replace("\\", "/"),
            "latestClose": None,
            "latestTime": None,
            "rows": 0,
            "dataStartUtc": None,
            "dataEndUtc": None,
        }

        if not os.path.exists(parquet_path):
            summaries[name] = entry
            continue

        try:
            df = pd.read_parquet(parquet_path, engine="pyarrow")
            if df.empty:
                summaries[name] = entry
                continue

            entry["available"] = True
            entry["rows"] = len(df)

            if "time" in df.columns:
                entry["dataStartUtc"] = str(df["time"].min()) + "Z"
                entry["dataEndUtc"] = str(df["time"].max()) + "Z"
                entry["latestTime"] = str(df["time"].max()) + "Z"

            if "close" in df.columns:
                entry["latestClose"] = float(df["close"].iloc[-1])

        except Exception as e:
            entry["error"] = str(e)
            _log("Failed to read proxy {}: {}".format(name, e))

        summaries[name] = entry

    available_count = sum(1 for v in summaries.values() if v.get("available"))
    return {
        "available": available_count > 0,
        "count": available_count,
        "total": len(names),
        "proxies": summaries,
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: READ CALENDAR
# ═══════════════════════════════════════════════════════════════════

def read_calendar() -> dict[str, Any]:
    """Read the economic calendar and filter to upcoming events."""
    if not os.path.exists(CALENDAR_PATH):
        return {"available": False, "error": "Calendar file not found", "events": []}

    try:
        with open(CALENDAR_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"available": False, "error": str(e), "events": []}

    events = data.get("events", [])
    now_iso = _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ")

    # Filter to upcoming events only
    upcoming = [e for e in events if e.get("scheduledUtc", "") >= now_iso]
    # Take next 10 upcoming
    upcoming = upcoming[:10]

    fetched_at = data.get("fetchedAtUtc")
    stale = _is_stale(fetched_at, CALENDAR_STALE_HOURS)

    return {
        "available": True,
        "stale": stale,
        "fetchedAtUtc": fetched_at,
        "upcomingCount": len(upcoming),
        "totalCount": len(events),
        "events": upcoming,
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: READ HEADLINES
# ═══════════════════════════════════════════════════════════════════

def read_headlines(limit: int = 10) -> dict[str, Any]:
    """Read the latest macro headlines."""
    if not os.path.exists(HEADLINES_PATH):
        return {"available": False, "error": "Headlines file not found", "headlines": []}

    try:
        with open(HEADLINES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {"available": False, "error": str(e), "headlines": []}

    headlines = data.get("headlines", [])[:limit]
    fetched_at = data.get("fetchedAtUtc")
    stale = _is_stale(fetched_at, HEADLINES_STALE_HOURS)

    return {
        "available": len(headlines) > 0,
        "stale": stale,
        "fetchedAtUtc": fetched_at,
        "headlineCount": len(headlines),
        "headlines": headlines,
    }


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: CONSOLIDATED SNAPSHOT
# ═══════════════════════════════════════════════════════════════════

def build_snapshot(headline_limit: int = 10) -> dict[str, Any]:
    """Build a full perception snapshot from local artifacts."""
    manifest = read_manifest()
    proxies = read_proxy_summaries()
    calendar = read_calendar()
    headlines = read_headlines(headline_limit)

    # Determine overall freshness
    manifest_fetched = manifest.get("fetchedAtUtc") if manifest else None
    manifest_stale = _is_stale(manifest_fetched, PROXY_STALE_HOURS)

    sections_available = sum([
        proxies.get("available", False),
        calendar.get("available", False),
        headlines.get("available", False),
    ])

    any_stale = any([
        manifest_stale,
        calendar.get("stale", True),
        headlines.get("stale", True),
    ])

    return {
        "ok": True,
        "snapshotAtUtc": _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "manifestPresent": manifest is not None,
        "manifestFetchedAtUtc": manifest_fetched,
        "manifestStale": manifest_stale,
        "sectionsAvailable": sections_available,
        "anyStale": any_stale,
        "proxies": proxies,
        "calendar": calendar,
        "headlines": headlines,
    }


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main(argv=None):
    parser = argparse.ArgumentParser(description="Aleph Perception Snapshot Reader")
    parser.add_argument("--headlineLimit", type=int, default=10,
                        help="Max headlines to include (default: 10)")
    args = parser.parse_args(argv)

    try:
        snapshot = build_snapshot(args.headlineLimit)
        print(json.dumps(_sanitize_for_json(snapshot), default=str, allow_nan=False))
    except Exception as e:
        _log("Fatal error: {}".format(e))
        print(json.dumps(_sanitize_for_json({"ok": False, "error": str(e)}), allow_nan=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
