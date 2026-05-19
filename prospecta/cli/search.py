"""`prospecta search QUERY` — hybrid retrieval."""
from __future__ import annotations

import sys


def cmd_search(args) -> int:
    from prospecta.cli import _common

    try:
        memory = _common.make_memory(args)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1

    try:
        results = memory.search(
            args.query,
            mode=args.mode,
            limit=args.limit,
            rrf_k=args.rrf_k,
        )
    except Exception as e:
        print(f"error: search failed: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(_common.recalled_to_json(results))
    else:
        _common.pretty_print_recalled(results)
    return 0
