"""
news_headlines.py - Atomic headlines worker for Deep Blue.

Priority:
1) OpenBB
2) yfinance (symbol mode)
3) RSS via feedparser

Stdout contract:
  Exactly ONE JSON object per invocation.
Stderr:
  Logs only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import urllib.parse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any


MIN_LIMIT = 1
MAX_LIMIT = 30
DEFAULT_LIMIT = 10


def _log(msg: str) -> None:
    print(f"[news_headlines] {msg}", file=sys.stderr)


def _format_error(symbol: str | None, message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "providerUsed": None,
        "symbol": symbol,
        "items": [],
        "error": message,
    }


def _parse_iso_utc(value: Any) -> str | None:
    if value is None:
        return None

    try:
        if isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None

            if raw.isdigit():
                dt = datetime.fromtimestamp(float(raw), tz=timezone.utc)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

            try:
                dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                pass

            try:
                dt = parsedate_to_datetime(raw)
                if dt is None:
                    return None
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except Exception:
                return None
    except Exception:
        return None

    return None


def _canonicalize_url(url: Any) -> str | None:
    if not isinstance(url, str):
        return None

    raw = url.strip()
    if not raw:
        return None

    try:
        parsed = urllib.parse.urlsplit(raw)
    except Exception:
        return None

    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        return None

    host = (parsed.hostname or "").lower()
    if not host:
        return None

    host_display = f"[{host}]" if ":" in host and not host.startswith("[") else host
    try:
        port = parsed.port
    except ValueError:
        return None
    if port is not None:
        default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
        if not default_port:
            host_display = f"{host_display}:{port}"

    path = parsed.path or "/"
    return urllib.parse.urlunsplit((scheme, host_display, path, parsed.query, ""))


def _to_records(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []

    try:
        if hasattr(payload, "to_df"):
            df = payload.to_df()
            return df.to_dict(orient="records")
        if hasattr(payload, "to_dataframe"):
            df = payload.to_dataframe()
            return df.to_dict(orient="records")
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump()
    except Exception as ex:
        _log(f"OpenBB payload conversion failed: {ex}")

    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]

    if isinstance(payload, dict):
        for key in ("results", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        return [payload]

    return []


def _pick_first(item: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def _normalize_item(raw: dict[str, Any]) -> dict[str, Any] | None:
    base = raw.get("content") if isinstance(raw.get("content"), dict) else raw

    title = _pick_first(base, "title", "headline", "name")
    url_value = _pick_first(
        base,
        "url",
        "link",
        "article_url",
        "story_url",
        "canonical_url",
        "canonicalUrl",
    )

    if isinstance(url_value, dict):
        url_value = _pick_first(url_value, "url", "link")

    canonical_url = _canonicalize_url(url_value)
    if not title or not canonical_url:
        return None

    source = _pick_first(base, "source", "publisher", "provider", "site", "domain")
    if isinstance(source, dict):
        source = _pick_first(source, "name", "title")

    published = _pick_first(
        base,
        "publishDateUtc",
        "published_at",
        "publishedAt",
        "published",
        "pubDate",
        "providerPublishTime",
        "datetime",
        "time",
    )
    publish_date_utc = _parse_iso_utc(published)

    summary = _pick_first(base, "summary", "description", "snippet", "content")
    if summary is None:
        summary = ""
    elif not isinstance(summary, str):
        summary = str(summary)

    return {
        "title": str(title).strip(),
        "url": canonical_url,
        "source": str(source).strip() if source else "",
        "publishDateUtc": publish_date_utc,
        "summary": summary.strip(),
    }


def _dedupe(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    for item in items:
        url = item.get("url")
        if not isinstance(url, str) or not url:
            continue
        if url in seen:
            continue
        seen.add(url)
        output.append(item)
        if len(output) >= limit:
            break

    return output


def _fetch_openbb(symbol: str | None, limit: int) -> tuple[list[dict[str, Any]], str | None]:
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from openbb import obb  # type: ignore
    except Exception as ex:
        _log(f"OpenBB import failed: {ex}")
        return [], "OpenBB import failed"

    attempts: list[tuple[str, Any]] = []
    if symbol:
        attempts.extend([
            ("obb.news.company(symbol)", lambda: obb.news.company(symbol=symbol, limit=limit)),
            ("obb.equity.news(symbol)", lambda: obb.equity.news(symbol=symbol, limit=limit)),
            ("obb.news.search(query)", lambda: obb.news.search(query=symbol, limit=limit)),
        ])

    attempts.extend([
        ("obb.news.latest()", lambda: obb.news.latest(limit=limit)),
        ("obb.news.world()", lambda: obb.news.world(limit=limit)),
    ])

    last_error = "OpenBB returned no items"
    for label, fn in attempts:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                payload = fn()
            records = _to_records(payload)
            normalized = []
            for rec in records:
                item = _normalize_item(rec)
                if item is not None:
                    normalized.append(item)
            if normalized:
                _log(f"OpenBB succeeded via {label} ({len(normalized)} items)")
                return _dedupe(normalized, limit), None
            _log(f"OpenBB attempt yielded no usable items: {label}")
        except Exception as ex:
            last_error = f"{label} failed: {ex}"
            _log(last_error)

    return [], last_error


def _fetch_yfinance(symbol: str, limit: int) -> tuple[list[dict[str, Any]], str | None]:
    try:
        import yfinance as yf  # type: ignore
    except Exception as ex:
        _log(f"yfinance import failed: {ex}")
        return [], "yfinance import failed"

    try:
        ticker = yf.Ticker(symbol)
        raw_items = ticker.news or []
    except Exception as ex:
        _log(f"yfinance fetch failed: {ex}")
        return [], f"yfinance fetch failed: {ex}"

    normalized = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        item = _normalize_item(raw)
        if item is not None:
            normalized.append(item)

    deduped = _dedupe(normalized, limit)
    if deduped:
        _log(f"yfinance produced {len(deduped)} items")
        return deduped, None
    return [], "yfinance returned no usable items"


def _fetch_rss(symbol: str | None, limit: int) -> tuple[list[dict[str, Any]], str | None]:
    try:
        import feedparser  # type: ignore
    except Exception as ex:
        _log(f"feedparser import failed: {ex}")
        return [], "feedparser import failed"

    feeds: list[str] = []
    if symbol:
        query = urllib.parse.quote_plus(f"{symbol} stock")
        feeds.append(f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en")
    else:
        feeds.extend([
            "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
            "https://feeds.bbci.co.uk/news/business/rss.xml",
        ])

    collected: list[dict[str, Any]] = []
    for feed_url in feeds:
        try:
            parsed = feedparser.parse(feed_url)
            entries = getattr(parsed, "entries", [])
            _log(f"RSS feed loaded: {feed_url} ({len(entries)} entries)")
            for entry in entries:
                if not isinstance(entry, dict):
                    entry = dict(entry)
                item = _normalize_item(entry)
                if item is not None:
                    collected.append(item)
        except Exception as ex:
            _log(f"RSS feed failed ({feed_url}): {ex}")

    deduped = _dedupe(collected, limit)
    if deduped:
        return deduped, None
    return [], "RSS returned no usable items"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deep Blue News Headlines Worker")
    parser.add_argument("--symbol", default="", help="Optional ticker symbol (e.g., AMD)")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Headlines limit (1-30)")

    symbol: str | None = None
    limit = DEFAULT_LIMIT
    result: dict[str, Any]

    try:
        args = parser.parse_args(argv)
        symbol = (args.symbol or "").strip().upper() or None
        limit = max(MIN_LIMIT, min(MAX_LIMIT, int(args.limit)))

        openbb_items, openbb_err = _fetch_openbb(symbol, limit)
        if openbb_items:
            result = {
                "ok": True,
                "providerUsed": "openbb",
                "symbol": symbol,
                "items": openbb_items,
                "error": None,
            }
            print(json.dumps(result))
            return

        yfinance_items: list[dict[str, Any]] = []
        yfinance_err: str | None = None
        if symbol:
            yfinance_items, yfinance_err = _fetch_yfinance(symbol, limit)
            if yfinance_items:
                result = {
                    "ok": True,
                    "providerUsed": "yfinance",
                    "symbol": symbol,
                    "items": yfinance_items,
                    "error": None,
                }
                print(json.dumps(result))
                return

        rss_items, rss_err = _fetch_rss(symbol, limit)
        if rss_items:
            result = {
                "ok": True,
                "providerUsed": "rss",
                "symbol": symbol,
                "items": rss_items,
                "error": None,
            }
            print(json.dumps(result))
            return

        errors = [x for x in [openbb_err, yfinance_err, rss_err] if x]
        message = "; ".join(errors) if errors else "No headlines found."
        result = _format_error(symbol, message)
        print(json.dumps(result))
    except SystemExit:
        result = _format_error(symbol, "Invalid arguments.")
        print(json.dumps(result))
    except Exception as ex:
        _log(f"Unhandled error: {ex}")
        result = _format_error(symbol, f"Unhandled error: {ex}")
        print(json.dumps(result))


if __name__ == "__main__":
    main()
