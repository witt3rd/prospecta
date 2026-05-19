"""`prospecta config` — print loaded config (with secrets redacted)."""
from __future__ import annotations

import os


def cmd_config(args) -> int:
    from prospecta.cli._common import redact_database_url

    rows = {
        "database_url": redact_database_url(args.database_url),
        "bank":         args.bank,
        "PROSPECTA_BANK env":  os.environ.get("PROSPECTA_BANK", "(unset)"),
        "PROSPECTA_USE_DEFAULTS env": os.environ.get(
            "PROSPECTA_USE_DEFAULTS", "(unset)"
        ),
    }
    width = max(len(k) for k in rows)
    for k, v in rows.items():
        print(f"  {k.ljust(width)}  {v}")
    return 0
