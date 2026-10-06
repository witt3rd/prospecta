"""`prospecta backup <path>` / `prospecta restore <path>`.

Plain-SQL pg_dump of the whole database (all banks, pgvector data included),
replayed with psql. Connection comes from --database-url / DATABASE_URL.

A newer pg_dump client emits `SET transaction_timeout`, which older servers
reject; restore drops that line from the dump header (never COPY data) so client/server version skew is harmless.
"""
from __future__ import annotations

import subprocess
import sys


def _url(args) -> str | None:
    if not args.database_url:
        print("error: --database-url or DATABASE_URL env var required", file=sys.stderr)
        return None
    return args.database_url


def cmd_backup(args) -> int:
    url = _url(args)
    if not url:
        return 1
    r = subprocess.run(
        ["pg_dump", "--dbname", url, "--no-owner", "--no-privileges",
         "--file", str(args.path)]
    )
    if r.returncode:
        print("error: pg_dump failed", file=sys.stderr)
        return 2
    print(f"backup written: {args.path}")
    return 0


def cmd_restore(args) -> int:
    url = _url(args)
    if not url:
        return 1
    with open(args.path, encoding="utf-8") as f:
        lines = []
        in_header = True
        for line in f:
            if in_header and line.startswith("COPY "):
                in_header = False
            if in_header and line.startswith("SET transaction_timeout"):
                continue
            lines.append(line)
        sql = "".join(lines)
    r = subprocess.run(
        ["psql", "--dbname", url, "-v", "ON_ERROR_STOP=1", "--single-transaction",
         "--quiet", "--file", "-"],
        input=sql, text=True, stdout=subprocess.DEVNULL,
    )
    if r.returncode:
        print("error: restore failed (target must be an empty database)", file=sys.stderr)
        return 2
    print(f"restored: {args.path}")
    return 0
