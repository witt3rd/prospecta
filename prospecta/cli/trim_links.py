"""`prospecta trim-links [--keep N] [--dry-run]` — trim the bank to the strongest N
similarity links per note. Row locks only, one small transaction per note."""
from __future__ import annotations

import os
import sys


def _mb(n: int) -> str:
    return f"{n / 1024 / 1024:.1f} MiB"


def cmd_trim_links(args) -> int:
    from prospecta._linkcap import DEFAULT_MAX_LINKS_PER_NOTE, trim_bank
    from prospecta.cli import _common

    keep = args.keep if args.keep is not None else int(
        os.environ.get("PROSPECTA_MAX_LINKS_PER_NOTE", DEFAULT_MAX_LINKS_PER_NOTE))
    if keep < 1:
        print("error: --keep must be at least 1", file=sys.stderr)
        return 2
    try:
        memory = _common.make_memory(args)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1

    def progress(p: dict) -> None:
        if p["visited"] % 100 == 0:
            print(f"trim-links: notes={p['visited']} remaining={p['remaining']} "
                  f"rows_removed={p['rows_removed']}", flush=True)
    try:
        with memory._pool.connection() as conn:  # type: ignore[attr-defined]
            conn.autocommit = True
            out = trim_bank(conn, memory.default_bank_id, keep, args.dry_run, progress)
    except Exception as e:
        print(f"error: trim-links failed: {e}", file=sys.stderr)
        return 2
    verb = "would remove" if args.dry_run else "removed"
    print(f"trim-links: keep={keep} notes_over={out['notes_over']} pairs {verb}={out['pairs_removed']} "
          f"rows {verb}={out['rows_removed']}")
    print(f"space the rows take (table + indexes, estimate): {_mb(out['bytes_reclaimable'])}; "
          "VACUUM lets the table reuse it, pg_repack or VACUUM FULL gives it back to the disk")
    return 0
