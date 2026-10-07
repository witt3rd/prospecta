"""`prospecta link-pass --mode all-pairs [--resume] [--limit N]` — upgrade linked
documents to all-pairs after the import. Resumable, never blocks retain."""
from __future__ import annotations

import sys


def cmd_link_pass(args) -> int:
    from prospecta.cli import _common

    try:
        memory = _common.make_memory(args)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    if getattr(memory, "_linker", None) is None:
        print("error: link-pass needs the linker (OPENROUTER_API_KEY set, PROSPECTA_LINKER on)",
              file=sys.stderr)
        return 2

    def progress(p: dict) -> None:
        print(f"link-pass: documents={p['documents']} pairs_judged={p['judged']} "
              f"pairs_cached={p['cached']} failed={p['failed']} remaining={p['remaining']}",
              flush=True)
    try:
        out = memory.link_pass(args.mode, limit=args.limit, on_progress=progress)
    except Exception as e:
        print(f"error: link-pass failed: {e}", file=sys.stderr)
        return 2
    print(f"link-pass done: {out}")
    return 1 if out["failed"] else 0
