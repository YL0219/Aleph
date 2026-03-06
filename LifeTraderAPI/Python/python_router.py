"""
python_router.py — Single Python entrypoint for Deep Blue.

C# (PythonDispatcherService) always invokes this script.
This router is a thin argparse dispatcher — no heavy logic, no network calls.
Real logic lives in modular worker modules under Workers/.

Stdout contract: exactly ONE JSON object/array per execution (from the worker).
Stderr: all logging.

Routing:
    python python_router.py <domain> <action> [flags...]

Market domain actions:
    ingest          Batch OHLCV ingestion (OpenBB + yfinance fallback)
    fetch-quote     Real-time quote for a single symbol
    fetch-candles   OHLCV candle data for a single symbol
    parquet-read    Read local Parquet data lake

News domain actions:
    headlines       Fetch recent headlines (OpenBB -> yfinance -> RSS fallback)
    scrape          Scrape website text with SSRF protections

Legacy domain actions:
    fetch-news      RSI + news sentiment report (stdout is human-readable text, NOT JSON)

Examples:
    python python_router.py market ingest --symbols AMD,AAPL --interval 1d --lookbackDays 365 --outRoot data_lake/market/ohlcv
    python python_router.py market fetch-quote --symbol AMD
    python python_router.py market fetch-candles --symbol AMD --tf 1d --range 180d --limit 500
    python python_router.py market parquet-read --symbol AMD --days 7
    python python_router.py news headlines --symbol AMD --limit 5
    python python_router.py news scrape --url https://example.com --timeoutSec 12
    python python_router.py legacy fetch-news AMD
"""

import sys
import os
import json
import argparse


def _error_exit(msg):
    """Print a generic JSON error to stdout and exit with code 1."""
    print(json.dumps({"ok": False, "error": msg}))
    sys.exit(1)


def _route_market(action, remaining):
    """Dispatch market domain actions to worker modules."""
    if action == "ingest":
        from Workers.market_ingest import main as worker_main
        worker_main(remaining)

    elif action == "fetch-quote":
        from Workers.fetchmarketdata import main as worker_main
        worker_main(["quote"] + remaining)

    elif action == "fetch-candles":
        from Workers.fetchmarketdata import main as worker_main
        worker_main(["candles"] + remaining)

    elif action == "parquet-read":
        from Workers.parquet_read import main as worker_main
        worker_main(remaining)

    else:
        _error_exit(f"Unknown market action: '{action}'. "
                     f"Valid: ingest, fetch-quote, fetch-candles, parquet-read")


def _route_legacy(action, remaining):
    """Dispatch legacy domain actions. These may output non-JSON text."""
    if action == "fetch-news":
        from Legacy.fetch_news import main as worker_main
        worker_main(remaining)
    else:
        _error_exit(f"Unknown legacy action: '{action}'. Valid: fetch-news")


def _route_news(action, remaining):
    """Dispatch news domain actions to worker modules."""
    if action == "headlines":
        from Workers.news_headlines import main as worker_main
        worker_main(remaining)
    elif action == "scrape":
        from Workers.scrape_text import main as worker_main
        worker_main(remaining)
    else:
        _error_exit(f"Unknown news action: '{action}'. Valid: headlines, scrape")


def main():
    parser = argparse.ArgumentParser(
        description="Deep Blue Python Router — thin dispatcher to worker modules")
    parser.add_argument("domain", help="Domain (e.g., 'market')")
    parser.add_argument("action", help="Action within the domain")

    # parse_known_args: domain+action are consumed, remaining flags pass to worker
    args, remaining = parser.parse_known_args()

    # Ensure Workers package is importable from this script's directory
    router_dir = os.path.dirname(os.path.abspath(__file__))
    if router_dir not in sys.path:
        sys.path.insert(0, router_dir)

    if args.domain == "market":
        _route_market(args.action, remaining)
    elif args.domain == "news":
        _route_news(args.action, remaining)
    elif args.domain == "legacy":
        _route_legacy(args.action, remaining)
    else:
        _error_exit(f"Unknown domain: '{args.domain}'. Valid: market, news, legacy")


if __name__ == "__main__":
    # Required on Windows for multiprocessing (used by market ingestion worker)
    import multiprocessing
    multiprocessing.freeze_support()

    main()
