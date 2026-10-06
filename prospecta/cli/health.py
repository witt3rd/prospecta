"""`prospecta health` — one line; non-zero + `PROSPECTA DOWN:` if DB or embedder is unavailable."""
from __future__ import annotations

import os


def _check_db(url: str | None) -> str:
    if not url:
        raise RuntimeError("DATABASE_URL not set")
    import psycopg

    with psycopg.connect(url, connect_timeout=10) as conn:
        conn.execute("SELECT 1")
    return "ok"


def _embedder_kind() -> str:
    kind = os.environ.get("PROSPECTA_EMBEDDER", "default").strip().lower()
    if kind in ("sentence-transformers", "sentence_transformers", "st"):
        return "sentence-transformers"
    return "litellm" if kind in ("", "default", "litellm") else kind


def _check_embedder() -> tuple[str, int]:
    from prospecta.cli import _common

    kind = _embedder_kind()
    if kind == "sentence-transformers":
        import sentence_transformers  # noqa: F401
    embed = _common._resolve_embedder()
    vecs = embed(["prospecta health probe"])
    return kind, len(vecs[0])


def cmd_health(args) -> int:
    problems = []
    try:
        _check_db(args.database_url)
    except (Exception, SystemExit) as e:
        problems.append(f"database unavailable ({type(e).__name__}: {e})")
    probed = None
    try:
        probed = _check_embedder()
    except (Exception, SystemExit) as e:
        problems.append(f"embedder unavailable ({type(e).__name__}: {e})")
    if problems:
        msg = "; ".join(problems).replace("\n", " ")
        print(f"PROSPECTA DOWN: {msg}")
        return 1
    print(f"PROSPECTA OK: database reachable, embedder {probed[0]} dim={probed[1]}")
    return 0
