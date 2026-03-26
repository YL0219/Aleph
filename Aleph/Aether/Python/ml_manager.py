import argparse
import json
import sys


def _ok(action, symbol, extra=None):
    payload = {
        "ok": True,
        "domain": "ml",
        "action": action,
        "status": "placeholder",
        "message": "ML manager wired successfully.",
        "symbol": symbol,
    }
    if extra:
        payload.update(extra)
    return payload


def handle_action(action, argv):
    # ── Cortex actions (Phase 8+ brain) ──
    if action in ("cortex_predict", "cortex_train", "cortex_status", "cortex_resolve", "cortex_evaluate", "cortex_operational_status"):
        return _handle_cortex(action, argv)

    # ── Legacy ML actions ──
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--symbol", default="")
    parser.add_argument("--horizonDays", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1)
    args, _ = parser.parse_known_args(argv)

    symbol = (args.symbol or "").strip().upper()

    if action == "status":
        return _ok(action, symbol or "GLOBAL")

    if not symbol:
        return {"ok": False, "domain": "ml", "action": action, "error": "--symbol is required."}

    if action == "predict":
        return _ok(action, symbol, {"horizonDays": max(args.horizonDays, 1)})

    if action == "train":
        return _ok(action, symbol, {"epochs": max(args.epochs, 1)})

    return {"ok": False, "domain": "ml", "action": action, "error": "Unknown ml action."}


def _handle_cortex(action, argv):
    """Route cortex actions to the ml.ml_cortex module."""
    import os

    # Ensure the ml package is importable
    router_dir = os.path.dirname(os.path.abspath(__file__))
    if router_dir not in sys.path:
        sys.path.insert(0, router_dir)

    from ml.ml_cortex import cortex_predict, cortex_train, cortex_status, cortex_resolve, cortex_evaluate

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--symbol", default="")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--horizon", default="1d")
    parser.add_argument("--asof", default="")
    parser.add_argument("--payload", default="{}")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--challengers", default="")
    args, _ = parser.parse_known_args(argv)

    symbol = (args.symbol or "").strip().upper()
    if not symbol:
        return {"ok": False, "domain": "ml", "action": action, "error": "--symbol is required."}

    if action == "cortex_predict":
        # Parse the metabolic payload JSON
        try:
            payload = json.loads(args.payload)
        except json.JSONDecodeError as ex:
            return {"ok": False, "domain": "ml", "action": action, "error": f"Invalid payload JSON: {ex}"}

        return cortex_predict(
            symbol=symbol,
            interval=args.interval,
            horizon=args.horizon,
            asof_utc=args.asof,
            payload=payload,
        )

    elif action == "cortex_train":
        return cortex_train(
            symbol=symbol,
            horizon=args.horizon,
            max_samples=max(args.max_samples, 1),
        )

    elif action == "cortex_resolve":
        return cortex_resolve(
            symbol=symbol,
            horizon=args.horizon,
            interval=args.interval,
        )

    elif action == "cortex_status":
        return cortex_status(
            symbol=symbol,
            horizon=args.horizon,
        )

    elif action == "cortex_evaluate":
        return cortex_evaluate(
            symbol=symbol,
            horizon=args.horizon,
            challengers_json=args.challengers,
        )

    elif action == "cortex_operational_status":
        from ml.operational_status import compute_operational_snapshot, format_operational_snapshot
        op = compute_operational_snapshot(symbol, args.horizon, args.interval)
        return {"ok": True, "domain": "ml", "action": action, **format_operational_snapshot(op)}

    return {"ok": False, "domain": "ml", "action": action, "error": "Unknown cortex action."}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aether ML manager")
    parser.add_argument("action")
    args, remaining = parser.parse_known_args(argv)

    payload = handle_action(args.action, remaining)
    print(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print("ml_manager error: {}".format(ex), file=sys.stderr)
        print(json.dumps({"ok": False, "domain": "ml", "error": "manager_exception"}, separators=(",", ":")))
