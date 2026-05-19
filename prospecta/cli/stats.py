"""`prospecta stats` — show counters across substrate."""
from __future__ import annotations

import sys


def cmd_stats(args) -> int:
    from prospecta.cli import _common
    from prospecta.db.queries import stats as stats_query

    try:
        memory = _common.make_memory(args)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1

    bank_id = memory.default_bank_id if args.bank_only else None
    try:
        with memory._pool.connection() as conn:  # type: ignore[attr-defined]
            counters = stats_query(conn, bank_id=bank_id)
    except Exception as e:
        print(f"error: stats failed: {e}", file=sys.stderr)
        return 2

    print(
        f"bank: {memory.default_bank_id}"
        + (" (scoped)" if args.bank_only else "")
    )
    _common.pretty_print_stats_table(counters)
    return 0
