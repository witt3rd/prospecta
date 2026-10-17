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


def run_keyword_index(args) -> int:
    """`prospecta keyword-index [--rebuild]`: see docs/exact-tsv.md ("Legacy keyword index")."""
    from prospecta.db.keyword_index import ensure_keyword_index
    from prospecta.db.schema import resolve_schema, scoped_url
    if not args.database_url:
        print("error: --database-url or DATABASE_URL env var required", file=sys.stderr)
        return 2
    url = scoped_url(args.database_url, resolve_schema(None))
    r = ensure_keyword_index(url, rebuild=args.rebuild)
    state = "built" if r["created"] else "rebuilt" if r["rebuilt"] else "already present"
    print(f"Keyword index {r['index']}: {state}")
    return 0
