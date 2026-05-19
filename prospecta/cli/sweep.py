"""`prospecta sweep PATH...` — run the sweeper."""
from __future__ import annotations

import sys
import time


def cmd_sweep(args) -> int:
    from prospecta.cli import _common
    from prospecta._sweeper import SweeperConfig, run_one_pass

    try:
        memory = _common.make_memory(args)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1

    # Default to --once when neither --once nor --watch is given.
    do_watch = bool(args.watch)
    do_once = bool(args.once) or not do_watch

    if do_once:
        config = SweeperConfig(
            corpus_paths=list(args.path),
            sweep_interval_seconds=float(args.interval),
        )
        try:
            results = run_one_pass(memory, config)
        except Exception as e:
            print(f"error: sweep failed: {e}", file=sys.stderr)
            return 2
        for r in results:
            _common.pretty_print_sweep_result(r)
        return 0

    # --watch: start daemon, block until SIGINT.
    try:
        memory.start_sweeper(
            corpus_paths=list(args.path),
            sweep_interval_seconds=float(args.interval),
        )
    except Exception as e:
        print(f"error: failed to start sweeper: {e}", file=sys.stderr)
        return 2
    print(f"sweeper running; interval={args.interval}s — Ctrl-C to stop")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nshutting down...")
    finally:
        memory.shutdown()
    return 0
