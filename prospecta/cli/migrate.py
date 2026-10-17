"""`prospecta migrate` subcommand."""
from __future__ import annotations

import sys

from prospecta.db.migrate import get_schema_version, run_migrations


def run(args) -> int:
    if not args.database_url:
        print("error: --database-url or DATABASE_URL env var required", file=sys.stderr)
        return 2
    result = run_migrations(args.database_url)
    version = get_schema_version(args.database_url)
    print(f"Applied: {result['applied']}")
    print(f"Skipped: {result['skipped']}")
    print(f"Schema version: {version}")
    return 0
