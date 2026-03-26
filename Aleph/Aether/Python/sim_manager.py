import argparse
import json
import os
import sys


def _ensure_path():
    """Ensure the ml package is importable."""
    router_dir = os.path.dirname(os.path.abspath(__file__))
    if router_dir not in sys.path:
        sys.path.insert(0, router_dir)


def handle_action(action, argv):
    # ── Dream-state actions (Phase 10 simulation layer) ──
    if action == "dream_create":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--symbol", required=True)
        parser.add_argument("--horizon", default="1d")
        parser.add_argument("--interval", default="1h")
        parser.add_argument("--start", required=True)  # ISO UTC
        parser.add_argument("--end", required=True)     # ISO UTC
        parser.add_argument("--model-key", default="cortex_sgd_1h_24bar")
        parser.add_argument("--feature-version", default="v2.0.0")
        parser.add_argument("--warm-start", action="store_true")
        args, _ = parser.parse_known_args(argv)

        _ensure_path()
        from ml.dream_state import create_dream
        manifest = create_dream(
            symbol=args.symbol.upper(),
            horizon=args.horizon,
            interval=args.interval,
            replay_start=args.start,
            replay_end=args.end,
            model_key=args.model_key,
            feature_version=args.feature_version,
            config={"warm_start": args.warm_start},
        )
        return {"ok": True, "domain": "sim", "action": action, "dream": manifest}

    elif action == "dream_step":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--dream-id", required=True)
        args, _ = parser.parse_known_args(argv)

        _ensure_path()
        from ml.dream_state import dream_step
        result = dream_step(dream_id=args.dream_id)
        return {"ok": True, "domain": "sim", "action": action, **result}

    elif action == "dream_resolve":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--dream-id", required=True)
        args, _ = parser.parse_known_args(argv)

        _ensure_path()
        from ml.dream_state import dream_resolve
        result = dream_resolve(dream_id=args.dream_id)
        return {"ok": True, "domain": "sim", "action": action, **result}

    elif action == "dream_train":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--dream-id", required=True)
        args, _ = parser.parse_known_args(argv)

        _ensure_path()
        from ml.dream_state import dream_train
        result = dream_train(dream_id=args.dream_id)
        return {"ok": True, "domain": "sim", "action": action, **result}

    elif action == "dream_evaluate":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--dream-id", required=True)
        args, _ = parser.parse_known_args(argv)

        _ensure_path()
        from ml.dream_state import dream_evaluate
        result = dream_evaluate(dream_id=args.dream_id)
        return {"ok": True, "domain": "sim", "action": action, **result}

    elif action == "dream_status":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--dream-id", required=True)
        args, _ = parser.parse_known_args(argv)

        _ensure_path()
        from ml.dream_state import dream_status
        result = dream_status(dream_id=args.dream_id)
        return {"ok": True, "domain": "sim", "action": action, **result}

    elif action == "dream_list":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--symbol", default="")
        args, _ = parser.parse_known_args(argv)

        _ensure_path()
        from ml.dream_state import dream_list
        symbol = (args.symbol or "").strip().upper() or None
        result = dream_list(symbol=symbol)
        return {"ok": True, "domain": "sim", "action": action, **result}

    elif action == "dream_abort":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--dream-id", required=True)
        args, _ = parser.parse_known_args(argv)

        _ensure_path()
        from ml.dream_state import dream_abort
        result = dream_abort(dream_id=args.dream_id)
        return {"ok": True, "domain": "sim", "action": action, **result}

    # ── Legacy backtest placeholder ──
    elif action == "backtest":
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--symbol", default="")
        parser.add_argument("--days", type=int, default=180)
        parser.add_argument("--strategy", default="baseline")
        args, _ = parser.parse_known_args(argv)

        symbol = (args.symbol or "").strip().upper()
        if not symbol:
            return {"ok": False, "domain": "sim", "action": action, "error": "--symbol is required."}

        return {
            "ok": True,
            "domain": "sim",
            "action": action,
            "status": "placeholder",
            "message": "Simulation manager wired successfully.",
            "symbol": symbol,
            "days": max(args.days, 1),
            "strategy": (args.strategy or "baseline").strip().lower(),
        }

    return {"ok": False, "domain": "sim", "action": action, "error": "Unknown sim action."}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Aether simulation manager")
    parser.add_argument("action")
    args, remaining = parser.parse_known_args(argv)

    payload = handle_action(args.action, remaining)
    print(json.dumps(payload, separators=(",", ":")))


if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        print("sim_manager error: {}".format(ex), file=sys.stderr)
        print(json.dumps({"ok": False, "domain": "sim", "error": "manager_exception"}, separators=(",", ":")))
