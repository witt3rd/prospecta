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


def _bank_dim(args) -> int | None:
    from prospecta.cli import _common
    return _common.bank_embedding_dim(args.database_url, args.bank)


def _check_embedder(dimensions: int | None = None) -> tuple[str, int]:
    from prospecta.cli import _common

    kind = _embedder_kind()
    if kind == "sentence-transformers":
        import sentence_transformers  # noqa: F401
    embed = _common._resolve_embedder(dimensions)
    vecs = embed(["prospecta health probe"])
    return kind, len(vecs[0])


def cmd_health(args) -> int:
    problems = []
    try:
        _check_db(args.database_url)
    except (Exception, SystemExit) as e:
        problems.append(f"database unavailable ({type(e).__name__}: {e})")
    probed = None
    bank_dim = _bank_dim(args)
    try:
        probed = _check_embedder(bank_dim)
        if bank_dim is not None and probed[1] != bank_dim:
            problems.append(
                f"embedder {probed[0]} dim={probed[1]} does not match bank "
                f"{args.bank!r} embedding_dim={bank_dim}")
    except (Exception, SystemExit) as e:
        problems.append(f"embedder unavailable ({type(e).__name__}: {e})")
    if problems:
        msg = "; ".join(problems).replace("\n", " ")
        print(f"PROSPECTA DOWN: {msg}")
        return 1
    print(f"PROSPECTA OK: database reachable, embedder {probed[0]} dim={probed[1]}")
    return 0
