"""`prospecta import hindsight DUMP` — import a Hindsight bank dump."""
from __future__ import annotations

import json
import sys


def cmd_import(args) -> int:
    from prospecta import _import_hindsight as ih
    from prospecta.cli import _common

    try:
        dump = ih.load_dump(args.dump)
    except (OSError, ValueError) as e:
        print(f"error: cannot read dump {args.dump}: {e}", file=sys.stderr)
        return 1
    try:
        memory = _common.make_memory(args)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    report = ih.import_hindsight(memory, dump, synthesize=args.synthesize)
    print(json.dumps(report.to_dict(), indent=2) if args.json else report.render())
    return 3 if (report.units.failed or report.links.failed or report.entities.failed) else 0
