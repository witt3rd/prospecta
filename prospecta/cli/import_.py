"""`prospecta import hindsight SOURCE_URL` — import a Hindsight bank."""
from __future__ import annotations

import json
import sys


def cmd_import(args) -> int:
    from prospecta import _import_hindsight as ih
    from prospecta.cli import _common

    try:
        banks = ih.list_banks(args.source)
    except Exception as e:
        print(f"error: cannot read Hindsight source: {e}", file=sys.stderr)
        return 1
    if args.all_banks:
        wanted = banks
    else:
        hb = args.hindsight_bank or (banks[0] if len(banks) == 1 else None)
        if hb is None:
            print(f"error: source has banks {banks}; pass --hindsight-bank or --all-banks",
                  file=sys.stderr)
            return 1
        wanted = [hb]

    reports, ok = [], True
    for hb in wanted:
        if args.all_banks:
            args.bank = hb  # same-named prospecta bank per Hindsight bank
        try:
            memory = _common.make_memory(args)
        except SystemExit as e:
            return int(e.code) if isinstance(e.code, int) else 1
        try:
            rep = ih.import_bank(memory, args.source, hb, batch=args.batch)
        except Exception as e:
            print(f"error: import of bank {hb!r} failed: {e}", file=sys.stderr)
            return 2
        ok = ok and rep.ok
        reports.append(rep)
    if args.json:
        print(json.dumps([r.to_dict() for r in reports], indent=2))
    else:
        print("\n\n".join(r.render() for r in reports))
    return 0 if ok else 3
