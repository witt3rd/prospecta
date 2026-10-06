"""`prospecta health` — one line; non-zero + `PROSPECTA DOWN:` if DB or embedder is unavailable."""
from __future__ import annotations

import sys


def _check_db(url: str | None) -> str:
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    import psycopg

    with psycopg.connect(url, connect_timeout=10) as conn:
        conn.execute("SELECT 1")
    return "ok"


def _check_embedder() -> int:
    from prospecta.cli import _common

    embed = _common._resolve_embedder()
    vecs = embed(["prospecta health probe"])
    return len(vecs[0])


def cmd_health(args) -> int:
    problems = []
    try:
        _check_db(args.database_url)
    except (Exception, SystemExit) as e:
        problems.append(f"database unavailable ({type(e).__name__}: {e})")
    dim = None
    try:
        dim = _check_embedder()
    except (Exception, SystemExit) as e:
        problems.append(f"embedder unavailable ({type(e).__name__}: {e})")
    if problems:
        msg = "; ".join(problems).replace("\n", " ")
        print(f"PROSPECTA DOWN: {msg}")
        return 1
    print(f"PROSPECTA OK: database reachable, embedder dim={dim}")
    return 0
