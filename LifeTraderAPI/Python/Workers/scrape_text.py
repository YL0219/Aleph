"""
scrape_text.py - Atomic website text scraper for Deep Blue.

Primary extractor: trafilatura
Optional fallback: newspaper3k (if installed)

Stdout contract:
  Exactly ONE JSON object per invocation.
Stderr:
  Logs only.
"""

from __future__ import annotations

import argparse
import contextlib
import ipaddress
import json
import socket
import sys
import urllib.parse
from typing import Any


DEFAULT_TIMEOUT_SEC = 12
MIN_TIMEOUT_SEC = 1
MAX_TIMEOUT_SEC = 30
MAX_TEXT_CHARS = 200000


def _log(msg: str) -> None:
    print(f"[scrape_text] {msg}", file=sys.stderr)


def _build_result(
    ok: bool,
    url: str,
    method: str,
    success: bool,
    text: str | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "ok": ok,
        "url": url,
        "extraction": {
            "method": method,
            "success": success,
        },
        "text": text,
        "error": error,
    }


def _is_blocked_ip(addr: Any) -> bool:
    if (
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local
        or addr.is_unspecified
        or addr.is_multicast
        or addr.is_reserved
    ):
        return True

    if isinstance(addr, ipaddress.IPv6Address):
        first = addr.packed[0]
        # fc00::/7 unique local
        if (first & 0xFE) == 0xFC:
            return True

    return False


def _validate_url_ssrf(url: str) -> tuple[bool, str]:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False, "Invalid URL format."

    if parsed.scheme.lower() not in ("http", "https"):
        return False, "Only http/https URLs are allowed."

    host = (parsed.hostname or "").strip().lower()
    if not host:
        return False, "URL host is missing."

    if host == "localhost" or host.endswith(".localhost"):
        return False, "Localhost is not allowed."

    try:
        literal = ipaddress.ip_address(host)
        if _is_blocked_ip(literal):
            return False, "Target IP is in a blocked range."
    except ValueError:
        pass

    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except Exception as ex:
        return False, f"DNS resolution failed: {ex}"

    if not infos:
        return False, "DNS resolution returned no addresses."

    resolved_ips: list[str] = []
    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        resolved_ips.append(ip_str)

        try:
            addr = ipaddress.ip_address(ip_str)
        except ValueError:
            return False, f"Resolved invalid IP: {ip_str}"

        if _is_blocked_ip(addr):
            return False, f"Resolved blocked IP: {ip_str}"

    _log(f"Resolved {host} -> {', '.join(sorted(set(resolved_ips)))}")
    return True, ""


def _cap_text(text: str | None) -> str | None:
    if text is None:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None
    if len(cleaned) > MAX_TEXT_CHARS:
        return cleaned[:MAX_TEXT_CHARS]
    return cleaned


def _extract_trafilatura(url: str, timeout_sec: int) -> tuple[bool, str | None, str | None]:
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import trafilatura  # type: ignore
    except Exception as ex:
        return False, None, f"trafilatura import failed: {ex}"

    try:
        with contextlib.redirect_stdout(sys.stderr):
            downloaded = trafilatura.fetch_url(url, timeout=timeout_sec)
    except TypeError:
        # Backward compatibility for trafilatura versions without timeout parameter.
        with contextlib.redirect_stdout(sys.stderr):
            downloaded = trafilatura.fetch_url(url)
    except Exception as ex:
        return False, None, f"trafilatura fetch failed: {ex}"

    if not downloaded:
        return False, None, "trafilatura fetch returned no content."

    try:
        with contextlib.redirect_stdout(sys.stderr):
            extracted = trafilatura.extract(downloaded)
    except Exception as ex:
        return False, None, f"trafilatura extract failed: {ex}"

    text = _cap_text(extracted)
    if text:
        return True, text, None
    return False, None, "trafilatura extracted empty text."


def _extract_newspaper(url: str, timeout_sec: int) -> tuple[bool, str | None, str | None]:
    try:
        with contextlib.redirect_stdout(sys.stderr):
            from newspaper import Article  # type: ignore
    except Exception as ex:
        return False, None, f"newspaper3k import failed: {ex}"

    try:
        with contextlib.redirect_stdout(sys.stderr):
            article = Article(url, fetch_images=False, request_timeout=timeout_sec)
            article.download()
            article.parse()
        text = _cap_text(article.text)
        if text:
            return True, text, None
        return False, None, "newspaper3k extracted empty text."
    except Exception as ex:
        return False, None, f"newspaper3k extraction failed: {ex}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deep Blue Website Text Scraper")
    parser.add_argument("--url", required=True, help="Target URL")
    parser.add_argument(
        "--timeoutSec",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help="Fetch timeout in seconds (1-30)",
    )

    url = ""
    timeout_sec = DEFAULT_TIMEOUT_SEC

    try:
        args = parser.parse_args(argv)
        url = (args.url or "").strip()
        timeout_sec = max(MIN_TIMEOUT_SEC, min(MAX_TIMEOUT_SEC, int(args.timeoutSec)))

        ok_ssrf, ssrf_error = _validate_url_ssrf(url)
        if not ok_ssrf:
            print(json.dumps(_build_result(False, url, "none", False, None, ssrf_error)))
            return

        ok, text, error = _extract_trafilatura(url, timeout_sec)
        if ok:
            print(json.dumps(_build_result(True, url, "trafilatura", True, text, None)))
            return

        _log(error or "trafilatura failed")

        fb_ok, fb_text, fb_error = _extract_newspaper(url, timeout_sec)
        if fb_ok:
            print(json.dumps(_build_result(True, url, "newspaper3k", True, fb_text, None)))
            return

        _log(fb_error or "newspaper3k failed")
        final_error = fb_error if fb_error and "import failed" not in fb_error else (error or fb_error)
        print(json.dumps(_build_result(False, url, "trafilatura", False, None, final_error)))
    except SystemExit:
        print(json.dumps(_build_result(False, url, "none", False, None, "Invalid arguments.")))
    except Exception as ex:
        _log(f"Unhandled error: {ex}")
        print(json.dumps(_build_result(False, url, "none", False, None, f"Unhandled error: {ex}")))


if __name__ == "__main__":
    main()
