"""`prospecta drop-links [--yes]`: remove a bank's link rows (no table lock)."""
from __future__ import annotations

import sys


def _mb(n: int) -> str:
    return f"{n / 1048576:.1f} MiB"


def cmd_drop_links(args) -> int:
    from prospecta._linkdrop import link_rows_and_bytes
    from prospecta.cli import _common

    try:
        memory = _common.make_memory(args)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    bank = memory.default_bank_id
    try:
        if not args.yes:
            with memory._pool.dedicated() as conn:
                rows, size = link_rows_and_bytes(conn, bank)
            print(f"drop-links: bank {bank!r} has {rows} link rows (about {_mb(size)}). "
                  "Nothing removed: run again with --yes to remove them.", file=sys.stderr)
            print(f"rows={rows} bytes={size}")
            return 2
        out = memory.drop_links(bank, yes=True, batch=args.batch)
    except Exception as e:
        print(f"error: drop-links failed: {e}", file=sys.stderr)
        return 2
    print(f"drop-links: removed {out['rows']} link rows from bank {bank!r}, freeing about "
          f"{_mb(out['bytes'])} for reuse (VACUUM lets the table reuse it; returning it to "
          "the disk needs a rewrite: pg_repack, or VACUUM FULL in a window).")
    return 0
