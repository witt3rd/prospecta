"""`prospecta index PATH` — index a corpus directory."""
from __future__ import annotations

import sys


def cmd_index(args) -> int:
    from prospecta.cli import _common

    try:
        memory = _common.make_memory(args)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1

    try:
        stats = memory.index_directory(
            args.path,
            source_prefix=args.source_prefix,
        )
    except Exception as e:
        print(f"error: index failed: {e}", file=sys.stderr)
        return 2

    _common.pretty_print_index_stats(stats)
    return 0
