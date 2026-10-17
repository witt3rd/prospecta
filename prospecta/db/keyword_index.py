"""Stored keyword index: a GIN expression index over memory_items.original_chunk.

The keyword channel's word rule is one SQL expression (`word_vector`): Postgres's
'simple' text-search parser (stemming off; versions, paths, ids stay whole words)
plus the pieces of a path split at slashes. The index is built over exactly that
expression and the channel's query uses exactly that expression, so index and scan
cannot drift and the database keeps the index current on every write (no
application code path can forget it).

It is created without blocking writes (CREATE INDEX CONCURRENTLY) by
`ensure_keyword_index`, which `run_migrations` calls after migration 0018 and
`prospecta keyword-index` calls on demand. Stemmed queries and banks without a
valid index use the plain scan with the same expression.

The Stage 1 exact retriever does not use this index: it reads the stored column
`memory_items.exact_tsv` (migration 0020), generated from `indexed_expression()` below
and indexed by its own GIN index (prospecta/db/exact_tsv.py). This index still serves
the `keyword` channel.
"""
from __future__ import annotations

import time

import psycopg

INDEX_NAME = "memory_items_keyword_gin"
# MIGRATE_LOCK_KEY + 3 (migrate.py uses +0..+2); kept literal to avoid an import cycle.
_BASE_KEY = 0x70726F73706563
_LOCK_KEY = _BASE_KEY + 3
# The advisory keys of the post-commit index builders: finish_0004 +1, finish_0005 +2, this
# index +3, the exact_tsv index +4 (prospecta/db/exact_tsv.py).
_BUILDER_KEYS = tuple(_BASE_KEY + n for n in (1, 2, 3, 4))


def take_other_builder_locks(conn: psycopg.Connection, own_key: int) -> list[int]:
    """Take (polling, never blocking) the advisory keys of the OTHER post-commit index builders
    and return them for the caller to release.

    Two CREATE INDEX CONCURRENTLY on one table can deadlock: the statement waiting for the
    table's lock already holds a snapshot, which the building one waits to end in its last
    phase (seen between finish_0004's `CREATE INDEX CONCURRENTLY IF NOT EXISTS` and this
    index's build when four processes migrate at once). Holding every builder's key while
    building makes them run one at a time. Polled, because a peer blocked inside
    pg_advisory_lock would deadlock the build the same way (the 0004 caveat)."""
    taken: list[int] = []
    for key in _BUILDER_KEYS:
        if key == own_key:
            continue
        while not conn.execute("SELECT pg_try_advisory_lock(%s)", (key,)).fetchone()[0]:
            time.sleep(0.05)
        taken.append(key)
    return taken


def word_vector(cfg: str, expr: str) -> str:
    """The one word rule. `cfg` is a SQL expression for the regconfig."""
    return (f"(to_tsvector({cfg}::regconfig, {expr}) || "
            f"to_tsvector({cfg}::regconfig, translate({expr}, '/\\', '  ')))")


def indexed_expression() -> str:
    return word_vector("'simple'", "original_chunk")


def has_index(conn: psycopg.Connection, name: str) -> bool:
    """True when a valid index of that name exists on memory_items in the connection's schema."""
    row = conn.execute(
        "SELECT 1 FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
        "WHERE i.indrelid = to_regclass('memory_items') AND c.relname = %s "
        "AND i.indisvalid AND i.indisready AND c.relnamespace = "
        "(SELECT relnamespace FROM pg_class WHERE oid = to_regclass('memory_items'))",
        (name,)).fetchone()
    return row is not None


def has_keyword_index(conn: psycopg.Connection) -> bool:
    """True when a valid keyword index exists in the connection's schema."""
    return has_index(conn, INDEX_NAME)


def ensure_keyword_index(database_url: str, *, rebuild: bool = False) -> dict:
    """Create the keyword index concurrently (writes are never blocked).

    Idempotent. An invalid leftover (an interrupted build) is dropped and rebuilt;
    `rebuild=True` rebuilds a valid one too, building the new index under a
    temporary name first so the channel keeps using the old index until the swap.
    """
    with psycopg.connect(database_url, autocommit=True) as conn:
        while not conn.execute("SELECT pg_try_advisory_lock(%s)", (_LOCK_KEY,)).fetchone()[0]:
            time.sleep(0.05)
        others: list[int] = []
        try:
            existed = has_keyword_index(conn)
            if existed and not rebuild:
                return {"index": INDEX_NAME, "created": False, "rebuilt": False}
            others = take_other_builder_locks(conn, _LOCK_KEY)
            expr = indexed_expression()
            if existed:
                tmp = INDEX_NAME + "_new"
                conn.execute(f'DROP INDEX CONCURRENTLY IF EXISTS "{tmp}"')
                conn.execute(f'CREATE INDEX CONCURRENTLY "{tmp}" '
                             f"ON memory_items USING GIN (({expr}))")
                conn.execute(f'DROP INDEX CONCURRENTLY "{INDEX_NAME}"')
                conn.execute(f'ALTER INDEX "{tmp}" RENAME TO "{INDEX_NAME}"')
                return {"index": INDEX_NAME, "created": False, "rebuilt": True}
            conn.execute(f'DROP INDEX CONCURRENTLY IF EXISTS "{INDEX_NAME}"')  # invalid leftover
            conn.execute(f'CREATE INDEX CONCURRENTLY "{INDEX_NAME}" '
                         f"ON memory_items USING GIN (({expr}))")
            return {"index": INDEX_NAME, "created": True, "rebuilt": False}
        finally:
            for key in reversed(others):
                conn.execute("SELECT pg_advisory_unlock(%s)", (key,))
            conn.execute("SELECT pg_advisory_unlock(%s)", (_LOCK_KEY,))
