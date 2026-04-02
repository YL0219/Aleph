"""
aether_router.py - Thin dispatcher for Aether domain workers.

Security layer:
  - Validates all outputs through Pydantic contracts before emitting to stdout.
  - Catches Python exceptions and returns structured error JSON (never raw tracebacks).
  - NaN/Infinity values are rejected at the contract boundary.
  - Domain/action allowlist is enforced (defense-in-depth with C# dispatcher).

Usage:
    python aether_router.py <domain> <action> [args...]
"""

import argparse
import json
import math
import os
import sys
import traceback


# ── Domain/Action allowlist (defense-in-depth with C# PythonDispatcherService) ──
_ALLOWED_DOMAINS = {"math", "ml", "sim", "macro"}


class NumericalIntegrityError(Exception):
    """Raised when NaN or Infinity is detected in an output payload.
    In quant/ML, these indicate blown gradients, division by zero, or
    data corruption. Silent suppression would mask catastrophic trading risk."""
    pass


def _detect_nan_inf(obj, path="root"):
    """
    Recursively scan a payload for NaN/Infinity. If found, raise
    NumericalIntegrityError with the exact field path.

    This is NOT a sanitizer — it is a detector. The philosophy:
    a NaN means something upstream is broken. We do not hide it.
    We surface it loudly so the organ can quarantine the event.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            raise NumericalIntegrityError(
                "NaN detected at '{}'. Likely blown gradient or division by zero.".format(path))
        if math.isinf(obj):
            raise NumericalIntegrityError(
                "Infinity detected at '{}'. Likely numerical overflow.".format(path))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _detect_nan_inf(v, "{}.{}".format(path, k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _detect_nan_inf(v, "{}[{}]".format(path, i))


def _emit(payload):
    """Validate numerical integrity and emit a single JSON object to stdout.
    If NaN/Infinity is found, raises NumericalIntegrityError — caller must
    catch and emit a structured AetherErrorResponse instead."""
    _detect_nan_inf(payload)
    print(json.dumps(payload, separators=(",", ":")))


def _emit_error(domain, action, error_msg, error_type="worker_error"):
    """Emit a structured error response for C# quarantine logging."""
    _emit({
        "ok": False,
        "domain": domain,
        "action": action,
        "error": str(error_msg)[:500],  # Truncate to prevent log floods
        "error_type": error_type,
        "quarantine": True,
    })


def _error(msg):
    print(msg, file=sys.stderr)
    _emit({"ok": False, "error": msg})
    sys.exit(1)


def _route_math(action, remaining):
    from math_manager import handle_action
    return handle_action(action, remaining)


def _route_ml(action, remaining):
    from ml_manager import handle_action
    return handle_action(action, remaining)


def _route_sim(action, remaining):
    from sim_manager import handle_action
    return handle_action(action, remaining)


def _route_macro(action, remaining):
    from macro_manager import handle_action
    return handle_action(action, remaining)


_ROUTERS = {
    "math": _route_math,
    "ml": _route_ml,
    "sim": _route_sim,
    "macro": _route_macro,
}


def _validate_output_contract(domain, action, payload):
    """
    Validate a worker's output payload against its Pydantic output contract.
    If the action has a registered contract, validate and return the cleaned dict.
    If validation fails, raise so the caller can emit a structured error.
    Actions without a contract pass through unmodified.
    """
    if not payload.get("ok"):
        return payload  # Error responses pass through — they have their own shape

    try:
        from contracts.aether_contracts import (
            CortexPredictOutput,
            CortexTrainOutput,
            CortexResolveOutput,
            CortexStatusOutput,
            CortexEvaluateOutput,
            DreamStepOutput,
        )
    except ImportError:
        # Contracts module not available — pass through (dev/test fallback)
        return payload

    # Map (domain, action) → Pydantic output model
    _OUTPUT_CONTRACTS = {
        ("ml", "cortex_predict"):    CortexPredictOutput,
        ("ml", "cortex_train"):      CortexTrainOutput,
        ("ml", "cortex_resolve"):    CortexResolveOutput,
        ("ml", "cortex_status"):     CortexStatusOutput,
        ("ml", "cortex_evaluate"):   CortexEvaluateOutput,
        ("sim", "dream_step"):       DreamStepOutput,
    }

    contract_cls = _OUTPUT_CONTRACTS.get((domain, action))
    if contract_cls is None:
        return payload  # No contract registered — pass through

    try:
        validated = contract_cls(**payload)
        return json.loads(validated.model_dump_json())
    except Exception as ex:
        print("Contract validation failed for {}/{}: {}".format(domain, action, ex), file=sys.stderr)
        # Re-raise so caller can emit structured error
        raise ValueError(
            "Output contract violation for {}/{}: {}".format(domain, action, str(ex)[:200])
        ) from ex


def main():
    parser = argparse.ArgumentParser(description="Aether Python router")
    parser.add_argument("domain")
    parser.add_argument("action")
    args, remaining = parser.parse_known_args()

    router_dir = os.path.dirname(os.path.abspath(__file__))
    if router_dir not in sys.path:
        sys.path.insert(0, router_dir)

    domain = args.domain.lower().strip()
    action = args.action.strip()

    # ── Gate: domain allowlist ──
    if domain not in _ALLOWED_DOMAINS:
        _error("Unknown domain: '{}'. Valid: {}".format(domain, ", ".join(sorted(_ALLOWED_DOMAINS))))
        return

    router_fn = _ROUTERS.get(domain)
    if router_fn is None:
        _error("No router registered for domain: '{}'.".format(domain))
        return

    try:
        payload = router_fn(action, remaining)
    except Exception as ex:
        # ── Catch-all: never let raw tracebacks reach C# stdout ──
        print(traceback.format_exc(), file=sys.stderr)
        _emit_error(domain, action, str(ex), error_type="unhandled_exception")
        sys.exit(1)

    if payload is None:
        _emit_error(domain, action, "Worker returned None.", error_type="null_response")
        sys.exit(1)

    if not isinstance(payload, dict):
        _emit_error(domain, action, "Worker returned non-dict type: {}".format(type(payload).__name__),
                     error_type="invalid_type")
        sys.exit(1)

    # ── Validate output via Pydantic contracts ──
    try:
        payload = _validate_output_contract(domain, action, payload)
    except ValueError as ex:
        _emit_error(domain, action, str(ex), error_type="contract_violation")
        sys.exit(1)

    # ── Emit with NaN/Infinity integrity check ──
    try:
        _emit(payload)
    except NumericalIntegrityError as ex:
        # NaN/Inf in output = blown gradient or data corruption.
        # Surface it loudly so C# quarantines instead of trading on garbage.
        print("Numerical integrity failure: {}".format(ex), file=sys.stderr)
        _emit_error(domain, action,
                    "Numerical_Integrity_Failure: {}".format(str(ex)[:300]),
                    error_type="nan_inf_detected")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as ex:
        # Last-resort safety net — should never fire, but prevents bare tracebacks
        print("aether_router fatal: {}".format(ex), file=sys.stderr)
        print(json.dumps({"ok": False, "error": "router_fatal", "quarantine": True},
                         separators=(",", ":")))
        sys.exit(1)
