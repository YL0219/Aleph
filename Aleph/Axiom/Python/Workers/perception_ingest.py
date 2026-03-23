"""
perception_ingest.py — Macro perception worker for Aleph Axiom.

Fetches, normalizes, and persists three families of perception data:
  1. Macro proxy OHLCV — VIX, DXY, BTC, plus macro basket (SPY, QQQ, TLT, GLD)
  2. Economic calendar — upcoming scheduled macro events (FOMC, CPI, NFP, etc.)
  3. Macro headlines — world/market news context

All artifacts are persisted locally in the data lake with provenance metadata.
Downstream consumers (Liver, ML Cortex) read from the local artifacts only.

Design:
  - Each section is independent and fault-tolerant: one failure does not block others.
  - yfinance is the primary provider for proxy OHLCV (these are well-known tickers).
  - Economic calendar uses a built-in schedule of known events + optional OpenBB.
  - Headlines reuse the existing news_headlines.py normalization logic.
  - A perception manifest records what was fetched, when, and whether it is fresh.

Output contract:
  - stdout: single JSON object with multi-section results
  - stderr: progress/debug logs
  - exit code 0: report printed (individual sections may have failures)
  - exit code 1: fatal error before any processing

Data lake layout:
  data_lake/macro/proxies/{name}/latest.parquet  (OHLCV)
  data_lake/macro/calendar/latest.json           (economic events)
  data_lake/macro/headlines/latest.json           (news items)
  data_lake/perception/manifest.json              (freshness/provenance)

Compatible with Python 3.10+.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import shutil
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any


# ── Dependency checks ────────────────────────────────────────────────

try:
    import yfinance as yf
except ImportError:
    print(json.dumps({"ok": False, "error": "yfinance is not installed"}))
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print(json.dumps({"ok": False, "error": "pandas is not installed"}))
    sys.exit(1)

try:
    import pyarrow  # noqa: F401
except ImportError:
    print(json.dumps({"ok": False, "error": "pyarrow is not installed"}))
    sys.exit(1)


# ── Constants ────────────────────────────────────────────────────────

# Default macro proxy tickers: yfinance ticker → lake name
DEFAULT_PROXY_MAP = {
    "^VIX":     "vix",
    "DX-Y.NYB": "dxy",
    "BTC-USD":  "btc",
    "SPY":      "spy",
    "QQQ":      "qqq",
    "TLT":      "tlt",
    "GLD":      "gld",
}

PROXY_OUT_ROOT = "data_lake/macro/proxies"
CALENDAR_OUT_PATH = "data_lake/macro/calendar/latest.json"
HEADLINES_OUT_PATH = "data_lake/macro/headlines/latest.json"
MANIFEST_OUT_PATH = "data_lake/perception/manifest.json"

DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_HEADLINE_LIMIT = 15


def _log(msg: str) -> None:
    print("[perception] {}".format(msg), file=sys.stderr)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: MACRO PROXY OHLCV
# ═══════════════════════════════════════════════════════════════════

def _write_parquet_atomic(df: pd.DataFrame, path: str) -> str:
    """Atomic Parquet write (tmp file -> rename). Returns final path."""
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(suffix=".parquet", dir=dir_path)
    os.close(fd)
    try:
        df.to_parquet(tmp_path, engine="pyarrow", index=False)
        shutil.move(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path


def _write_json_atomic(data: Any, path: str) -> str:
    """Atomic JSON write (tmp file -> rename). Returns final path."""
    dir_path = os.path.dirname(path)
    os.makedirs(dir_path, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=dir_path)
    os.close(fd)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        shutil.move(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    return path


def fetch_macro_proxies(
    proxy_map: dict[str, str],
    lookback_days: int,
    out_root: str,
) -> list[dict[str, Any]]:
    """
    Fetch OHLCV for each macro proxy ticker via yfinance.
    Stores each as data_lake/macro/proxies/{lake_name}/latest.parquet.
    Returns a list of per-proxy result dicts.
    """
    results = []
    end_date = _now_utc().strftime("%Y-%m-%d")
    start_date = (_now_utc() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

    for ticker, lake_name in proxy_map.items():
        result = {
            "ticker": ticker,
            "lakeName": lake_name,
            "provider": "yfinance",
            "isSuccess": False,
            "rowsWritten": 0,
            "parquetPath": "",
            "dataStartUtc": None,
            "dataEndUtc": None,
            "error": None,
        }

        try:
            _log("Fetching proxy {} ({})...".format(ticker, lake_name))
            t = yf.Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, interval="1d")

            if hist.empty:
                result["error"] = "No data returned for {}".format(ticker)
                _log("  {} — no data".format(ticker))
                results.append(result)
                continue

            # Normalize to stable OHLCV schema
            idx = hist.index
            if idx.tz is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)

            df = pd.DataFrame({
                "time":   idx,
                "open":   hist["Open"].values,
                "high":   hist["High"].values,
                "low":    hist["Low"].values,
                "close":  hist["Close"].values,
                "volume": hist["Volume"].values.astype("int64"),
            })

            parquet_path = os.path.join(out_root, lake_name, "latest.parquet")
            _write_parquet_atomic(df, parquet_path)

            data_start = df["time"].min().isoformat() + "Z"
            data_end = df["time"].max().isoformat() + "Z"

            result["isSuccess"] = True
            result["rowsWritten"] = len(df)
            result["parquetPath"] = parquet_path.replace("\\", "/")
            result["dataStartUtc"] = data_start
            result["dataEndUtc"] = data_end

            _log("  {} — {} rows -> {}".format(ticker, len(df), parquet_path))

        except Exception as e:
            result["error"] = str(e)
            _log("  {} — FAILED: {}".format(ticker, e))

        results.append(result)

    return results


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: ECONOMIC CALENDAR
# ═══════════════════════════════════════════════════════════════════

def fetch_economic_calendar(horizon_days: int = 30) -> dict[str, Any]:
    """
    Fetch upcoming economic calendar events.
    Strategy: built-in known schedule + optional OpenBB enrichment.
    Returns a dict with events list and provenance.
    """
    now = _now_utc()
    events = []
    provider = "builtin"

    # ── Built-in known recurring events ──
    # These are the major market-moving events with known schedules.
    # Updated annually. Dates are approximate — actual dates may shift.
    builtin_events = _get_builtin_calendar(now, horizon_days)
    events.extend(builtin_events)

    # ── Try OpenBB enrichment ──
    openbb_events = _try_openbb_calendar(horizon_days)
    if openbb_events:
        provider = "openbb+builtin"
        # Merge: deduplicate by event type + date proximity
        events = _merge_calendar_events(events, openbb_events)

    # Sort by date
    events.sort(key=lambda e: e.get("scheduledUtc", ""))

    return {
        "ok": True,
        "provider": provider,
        "fetchedAtUtc": _iso(now),
        "horizonDays": horizon_days,
        "eventCount": len(events),
        "events": events,
    }


def _get_builtin_calendar(now: datetime, horizon_days: int) -> list[dict[str, Any]]:
    """
    Generate a list of known macro events within the horizon window.
    Uses approximate schedules for major recurring events.
    """
    events = []
    start = now
    end = now + timedelta(days=horizon_days)

    # Known 2025-2026 FOMC meeting dates (2-day meetings, end date listed)
    fomc_dates = [
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-17",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-16",
    ]
    for d in fomc_dates:
        dt = datetime.fromisoformat(d + "T18:00:00+00:00")
        if start <= dt <= end:
            events.append({
                "eventType": "FOMC",
                "name": "FOMC Rate Decision",
                "scheduledUtc": _iso(dt),
                "importance": "high",
                "source": "builtin",
                "description": "Federal Open Market Committee interest rate decision",
            })

    # CPI: typically 2nd week of month, ~8:30 AM ET (13:30 UTC)
    # NFP: typically 1st Friday of month, ~8:30 AM ET (13:30 UTC)
    # GDP: typically last week of month (quarterly)
    for month_offset in range(horizon_days // 28 + 2):
        year = now.year + (now.month + month_offset - 1) // 12
        month = (now.month + month_offset - 1) % 12 + 1

        # CPI — approximate: 12th of month
        cpi_dt = datetime(year, month, 12, 13, 30, tzinfo=timezone.utc)
        if start <= cpi_dt <= end:
            events.append({
                "eventType": "CPI",
                "name": "Consumer Price Index",
                "scheduledUtc": _iso(cpi_dt),
                "importance": "high",
                "source": "builtin",
                "description": "Bureau of Labor Statistics CPI release",
            })

        # NFP — approximate: 1st Friday
        first_day = datetime(year, month, 1, 13, 30, tzinfo=timezone.utc)
        days_until_friday = (4 - first_day.weekday()) % 7
        nfp_dt = first_day + timedelta(days=days_until_friday)
        if start <= nfp_dt <= end:
            events.append({
                "eventType": "NFP",
                "name": "Non-Farm Payrolls",
                "scheduledUtc": _iso(nfp_dt),
                "importance": "high",
                "source": "builtin",
                "description": "Bureau of Labor Statistics employment report",
            })

        # GDP — approximate: last Thursday of month (only Jan, Apr, Jul, Oct for quarterly)
        if month in (1, 4, 7, 10):
            # Find last Thursday
            if month == 12:
                next_month_start = datetime(year + 1, 1, 1, 13, 30, tzinfo=timezone.utc)
            else:
                next_month_start = datetime(year, month + 1, 1, 13, 30, tzinfo=timezone.utc)
            last_day = next_month_start - timedelta(days=1)
            days_back = (last_day.weekday() - 3) % 7
            gdp_dt = last_day - timedelta(days=days_back)
            gdp_dt = gdp_dt.replace(hour=13, minute=30)
            if start <= gdp_dt <= end:
                events.append({
                    "eventType": "GDP",
                    "name": "GDP Report",
                    "scheduledUtc": _iso(gdp_dt),
                    "importance": "high",
                    "source": "builtin",
                    "description": "Bureau of Economic Analysis GDP estimate",
                })

    return events


def _try_openbb_calendar(horizon_days: int) -> list[dict[str, Any]]:
    """Attempt to fetch economic calendar from OpenBB. Returns empty list on failure."""
    try:
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
            from openbb import obb
    except Exception as e:
        _log("OpenBB import failed for calendar: {}".format(e))
        return []

    try:
        start_date = _now_utc().strftime("%Y-%m-%d")
        end_date = (_now_utc() + timedelta(days=horizon_days)).strftime("%Y-%m-%d")

        with __import__("contextlib").redirect_stdout(sys.stderr):
            # Try various OpenBB calendar endpoints
            for method_name in ["calendar", "economic_calendar"]:
                fn = getattr(obb.economy, method_name, None)
                if fn is None:
                    continue
                try:
                    result = fn(start_date=start_date, end_date=end_date)
                    if hasattr(result, "to_dataframe"):
                        df = result.to_dataframe()
                    elif hasattr(result, "to_df"):
                        df = result.to_df()
                    else:
                        continue

                    if df.empty:
                        continue

                    events = []
                    for _, row in df.iterrows():
                        event = _normalize_openbb_calendar_row(row)
                        if event:
                            events.append(event)

                    if events:
                        _log("OpenBB calendar: {} events".format(len(events)))
                        return events
                except Exception as e:
                    _log("OpenBB calendar method {} failed: {}".format(method_name, e))
                    continue

    except Exception as e:
        _log("OpenBB calendar fetch failed: {}".format(e))

    return []


def _normalize_openbb_calendar_row(row) -> dict[str, Any] | None:
    """Normalize an OpenBB calendar row to our schema."""
    name = None
    for key in ["event", "name", "title", "description"]:
        val = getattr(row, key, None) if hasattr(row, key) else row.get(key, None) if isinstance(row, dict) else None
        if val and str(val).strip():
            name = str(val).strip()
            break

    if not name:
        return None

    scheduled = None
    for key in ["date", "datetime", "scheduled", "time"]:
        val = getattr(row, key, None) if hasattr(row, key) else row.get(key, None) if isinstance(row, dict) else None
        if val is not None:
            try:
                if isinstance(val, str):
                    dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
                elif hasattr(val, "isoformat"):
                    dt = val
                else:
                    continue
                if hasattr(dt, "tzinfo") and dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                scheduled = _iso(dt)
                break
            except Exception:
                continue

    if not scheduled:
        return None

    # Classify importance
    importance = "medium"
    name_lower = name.lower()
    if any(kw in name_lower for kw in ["fomc", "interest rate", "fed funds", "federal reserve"]):
        importance = "high"
    elif any(kw in name_lower for kw in ["cpi", "consumer price", "inflation"]):
        importance = "high"
    elif any(kw in name_lower for kw in ["nonfarm", "non-farm", "payroll", "employment"]):
        importance = "high"
    elif any(kw in name_lower for kw in ["gdp", "gross domestic"]):
        importance = "high"

    # Classify event type
    event_type = "OTHER"
    if any(kw in name_lower for kw in ["fomc", "interest rate", "fed funds"]):
        event_type = "FOMC"
    elif any(kw in name_lower for kw in ["cpi", "consumer price"]):
        event_type = "CPI"
    elif any(kw in name_lower for kw in ["nonfarm", "non-farm", "payroll"]):
        event_type = "NFP"
    elif any(kw in name_lower for kw in ["gdp", "gross domestic"]):
        event_type = "GDP"

    return {
        "eventType": event_type,
        "name": name,
        "scheduledUtc": scheduled,
        "importance": importance,
        "source": "openbb",
        "description": "",
    }


def _merge_calendar_events(
    builtin: list[dict[str, Any]],
    openbb: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge builtin and OpenBB events, preferring OpenBB for known types."""
    # For known high-importance events (FOMC, CPI, NFP, GDP), prefer OpenBB dates
    # if they fall within 5 days of our builtin estimate.
    known_types = {"FOMC", "CPI", "NFP", "GDP"}
    merged = list(openbb)  # start with all OpenBB events

    openbb_known = {}
    for e in openbb:
        etype = e.get("eventType", "")
        if etype in known_types:
            sched = e.get("scheduledUtc", "")
            openbb_known.setdefault(etype, []).append(sched)

    for e in builtin:
        etype = e.get("eventType", "")
        sched = e.get("scheduledUtc", "")
        if etype in known_types and etype in openbb_known:
            # Skip builtin if OpenBB has same event type within 5 days
            dominated = False
            for ob_sched in openbb_known[etype]:
                try:
                    ob_dt = datetime.fromisoformat(ob_sched.replace("Z", "+00:00"))
                    bi_dt = datetime.fromisoformat(sched.replace("Z", "+00:00"))
                    if abs((ob_dt - bi_dt).total_seconds()) < 5 * 86400:
                        dominated = True
                        break
                except Exception:
                    pass
            if dominated:
                continue

        merged.append(e)

    return merged


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: MACRO HEADLINES
# ═══════════════════════════════════════════════════════════════════

def fetch_macro_headlines(limit: int = DEFAULT_HEADLINE_LIMIT) -> dict[str, Any]:
    """
    Fetch world/macro headlines using the existing news_headlines logic.
    Calls without a symbol to get broad macro/world news.
    """
    now = _now_utc()

    # Import normalization utilities from the existing news_headlines worker
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        from news_headlines import (
            _fetch_openbb, _fetch_rss, _normalize_item, _dedupe,
        )
    except ImportError as e:
        _log("Could not import news_headlines utilities: {}".format(e))
        return {
            "ok": False,
            "provider": None,
            "fetchedAtUtc": _iso(now),
            "headlineCount": 0,
            "headlines": [],
            "error": "news_headlines import failed: {}".format(e),
        }

    headlines = []
    provider = None

    # Try OpenBB for world/macro news (no symbol = broad)
    openbb_items, openbb_err = _fetch_openbb(None, limit)
    if openbb_items:
        headlines = openbb_items
        provider = "openbb"
    else:
        # Fallback to RSS
        rss_items, rss_err = _fetch_rss(None, limit)
        if rss_items:
            headlines = rss_items
            provider = "rss"

    # Tag headlines with macro topics
    tagged = []
    for h in headlines:
        tags = _extract_macro_tags(h.get("title", ""), h.get("summary", ""))
        tagged.append({**h, "macroTags": tags})

    return {
        "ok": len(tagged) > 0,
        "provider": provider,
        "fetchedAtUtc": _iso(now),
        "headlineCount": len(tagged),
        "headlines": tagged,
        "error": None if tagged else "No headlines available",
    }


def _extract_macro_tags(title: str, summary: str) -> list[str]:
    """Extract simple topic tags from headline text."""
    text = "{} {}".format(title, summary).lower()
    tags = []

    tag_keywords = {
        "fed": ["fed ", "federal reserve", "fomc", "powell"],
        "rates": ["interest rate", "rate hike", "rate cut", "basis point", "bps"],
        "inflation": ["inflation", "cpi", "consumer price", "pce"],
        "jobs": ["jobs", "employment", "payroll", "unemployment", "labor"],
        "gdp": ["gdp", "gross domestic", "economic growth"],
        "earnings": ["earnings", "quarterly results", "profit"],
        "oil": ["oil", "crude", "opec", "brent", "wti"],
        "crypto": ["bitcoin", "crypto", "btc", "ethereum"],
        "china": ["china", "chinese", "beijing", "pbo"],
        "geopolitical": ["war", "sanctions", "tariff", "trade war", "conflict"],
        "banking": ["bank", "credit", "lending", "deposit"],
        "housing": ["housing", "mortgage", "home sales", "real estate"],
        "tech": ["tech", "semiconductor", "ai ", "artificial intelligence", "nvidia", "apple"],
        "treasury": ["treasury", "bond", "yield", "10-year", "t-bill"],
    }

    for tag, keywords in tag_keywords.items():
        if any(kw in text for kw in keywords):
            tags.append(tag)

    return tags


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: PERCEPTION MANIFEST
# ═══════════════════════════════════════════════════════════════════

def build_manifest(
    proxy_results: list[dict[str, Any]],
    calendar_result: dict[str, Any],
    headline_result: dict[str, Any],
    started_utc: datetime,
    finished_utc: datetime,
) -> dict[str, Any]:
    """Build a perception manifest summarizing freshness and provenance."""
    proxy_ok = sum(1 for r in proxy_results if r.get("isSuccess"))
    proxy_total = len(proxy_results)

    return {
        "schemaVersion": 1,
        "fetchedAtUtc": _iso(finished_utc),
        "durationMs": int((finished_utc - started_utc).total_seconds() * 1000),
        "sections": {
            "macroProxies": {
                "ok": proxy_ok > 0,
                "succeeded": proxy_ok,
                "total": proxy_total,
                "tickers": {
                    r["lakeName"]: {
                        "ticker": r["ticker"],
                        "ok": r["isSuccess"],
                        "rows": r["rowsWritten"],
                        "dataEndUtc": r.get("dataEndUtc"),
                        "parquetPath": r.get("parquetPath", ""),
                    }
                    for r in proxy_results
                },
            },
            "calendar": {
                "ok": calendar_result.get("ok", False),
                "provider": calendar_result.get("provider"),
                "eventCount": calendar_result.get("eventCount", 0),
                "path": CALENDAR_OUT_PATH,
            },
            "headlines": {
                "ok": headline_result.get("ok", False),
                "provider": headline_result.get("provider"),
                "headlineCount": headline_result.get("headlineCount", 0),
                "path": HEADLINES_OUT_PATH,
            },
        },
        "overallFresh": proxy_ok > 0,
    }


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main(argv=None):
    parser = argparse.ArgumentParser(description="Aleph Perception Ingestion Worker")
    parser.add_argument("--lookbackDays", type=int, default=DEFAULT_LOOKBACK_DAYS,
                        help="Days of proxy OHLCV history (default: 365)")
    parser.add_argument("--headlineLimit", type=int, default=DEFAULT_HEADLINE_LIMIT,
                        help="Max headlines to fetch (default: 15)")
    parser.add_argument("--calendarHorizonDays", type=int, default=30,
                        help="Days ahead for calendar events (default: 30)")
    parser.add_argument("--proxyOutRoot", default=PROXY_OUT_ROOT,
                        help="Output root for proxy parquet files")
    parser.add_argument("--skipProxies", action="store_true",
                        help="Skip macro proxy OHLCV fetch")
    parser.add_argument("--skipCalendar", action="store_true",
                        help="Skip economic calendar fetch")
    parser.add_argument("--skipHeadlines", action="store_true",
                        help="Skip headline fetch")
    args = parser.parse_args(argv)

    started = _now_utc()
    job_id = str(uuid.uuid4())

    _log("Perception ingest starting. job={}".format(job_id))

    # ── Section 1: Macro proxies ──
    proxy_results = []
    if not args.skipProxies:
        _log("=== Fetching macro proxies ===")
        try:
            proxy_results = fetch_macro_proxies(
                DEFAULT_PROXY_MAP, args.lookbackDays, args.proxyOutRoot)
        except Exception as e:
            _log("Proxy section failed: {}".format(e))
            proxy_results = [{
                "ticker": "ALL", "lakeName": "all", "provider": "yfinance",
                "isSuccess": False, "rowsWritten": 0, "parquetPath": "",
                "error": str(e),
            }]
    else:
        _log("Skipping macro proxies (--skipProxies)")

    # ── Section 2: Economic calendar ──
    calendar_result = {"ok": False, "provider": None, "eventCount": 0, "events": []}
    if not args.skipCalendar:
        _log("=== Fetching economic calendar ===")
        try:
            calendar_result = fetch_economic_calendar(args.calendarHorizonDays)
            _write_json_atomic(calendar_result, CALENDAR_OUT_PATH)
            _log("Calendar: {} events -> {}".format(
                calendar_result["eventCount"], CALENDAR_OUT_PATH))
        except Exception as e:
            _log("Calendar section failed: {}".format(e))
            calendar_result["error"] = str(e)
    else:
        _log("Skipping calendar (--skipCalendar)")

    # ── Section 3: Headlines ──
    headline_result = {"ok": False, "provider": None, "headlineCount": 0, "headlines": []}
    if not args.skipHeadlines:
        _log("=== Fetching macro headlines ===")
        try:
            headline_result = fetch_macro_headlines(args.headlineLimit)
            _write_json_atomic(headline_result, HEADLINES_OUT_PATH)
            _log("Headlines: {} items -> {}".format(
                headline_result["headlineCount"], HEADLINES_OUT_PATH))
        except Exception as e:
            _log("Headlines section failed: {}".format(e))
            headline_result["error"] = str(e)
    else:
        _log("Skipping headlines (--skipHeadlines)")

    # ── Build and persist manifest ──
    finished = _now_utc()
    manifest = build_manifest(
        proxy_results, calendar_result, headline_result, started, finished)
    try:
        _write_json_atomic(manifest, MANIFEST_OUT_PATH)
        _log("Manifest -> {}".format(MANIFEST_OUT_PATH))
    except Exception as e:
        _log("Failed to write manifest: {}".format(e))

    # ── Build stdout report ──
    duration_ms = int((finished - started).total_seconds() * 1000)
    proxy_ok = sum(1 for r in proxy_results if r.get("isSuccess"))

    report = {
        "ok": True,
        "schemaVersion": 1,
        "jobId": job_id,
        "startedAtUtc": _iso(started),
        "finishedAtUtc": _iso(finished),
        "durationMs": duration_ms,
        "proxies": {
            "succeeded": proxy_ok,
            "total": len(proxy_results),
            "results": proxy_results,
        },
        "calendar": {
            "ok": calendar_result.get("ok", False),
            "provider": calendar_result.get("provider"),
            "eventCount": calendar_result.get("eventCount", 0),
        },
        "headlines": {
            "ok": headline_result.get("ok", False),
            "provider": headline_result.get("provider"),
            "headlineCount": headline_result.get("headlineCount", 0),
        },
        "manifestPath": MANIFEST_OUT_PATH,
    }

    _log("Perception ingest complete in {}ms".format(duration_ms))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
