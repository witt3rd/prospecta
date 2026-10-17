"""The stored exact vector: `memory_items.exact_tsv` and its GIN index (migration 0020).

The Stage 1 exact / phrase retriever (queries.stage1_search) matches and ranks the 'simple'
word vector of `original_chunk`. Computed per row at query time, that vector re-tokenized
every chunk holding a query word, so a question made only of very common words ("the") cost
about 0.15 ms per matching chunk. It is now a STORED generated column, filled by the database
on every write, with its own GIN index; the query reads the column and nothing is tokenized.

One definition: the column is generated from `keyword_index.indexed_expression()`, the same
word rule migration 0018 indexes for the `keyword` channel. Migration 0020's SQL carries a
placeholder that `migrate.run_migrations` fills from that function, so the migration, the
column and the `keyword` index cannot drift, and the query names only the column.

Adding the column REWRITES `memory_items` (every row, every index of the table), once, inside
the migration. The GIN index is then built without blocking writes (CREATE INDEX
CONCURRENTLY, which cannot run in the migration's transaction) by `ensure_exact_tsv_index`,
which `run_migrations` calls after migration 0020 and on every later run (so re-running
`prospecta migrate` repairs an interrupted build). See docs/exact-tsv.md for the operator
runbook.
"""
from __future__ import annotations

import time

import psycopg

from prospecta.db.keyword_index import has_index, indexed_expression

COLUMN = "exact_tsv"
INDEX_NAME = "memory_items_exact_tsv_gin"
# MIGRATE_LOCK_KEY + 4 (migrate.py uses +0..+2, keyword_index.py +3); kept literal to avoid an
# import cycle.
LOCK_KEY = 0x70726F73706563 + 4


def column_expression() -> str:
    """The SQL the column is generated from: the one word rule, over original_chunk."""
    return indexed_expression()


def has_exact_tsv_index(conn: psycopg.Connection) -> bool:
    """True when a valid exact_tsv index exists in the connection's schema."""
    return has_index(conn, INDEX_NAME)


def ensure_exact_tsv_index(database_url: str, *, rebuild: bool = False) -> dict:
    """Create the exact_tsv GIN index concurrently (writes are never blocked).

    Idempotent. An invalid leftover (an interrupted build) is dropped and rebuilt;
    `rebuild=True` rebuilds a valid one too, building the new index under a temporary name
    first so queries keep using the old index until the swap. A new index is followed by
    ANALYZE of the column (a rewrite leaves it without statistics, and the planner needs
    them to cost a very common word).

    The lock is polled with pg_try_advisory_lock, never waited on: CREATE INDEX
    CONCURRENTLY waits on every running statement and transaction, so a peer blocked inside
    pg_advisory_lock would deadlock it (the 0004/0018 caveat).
    """
    with psycopg.connect(database_url, autocommit=True) as conn:
        while not conn.execute("SELECT pg_try_advisory_lock(%s)", (LOCK_KEY,)).fetchone()[0]:
            time.sleep(0.05)
        try:
            existed = has_exact_tsv_index(conn)
            if existed and not rebuild:
                return {"index": INDEX_NAME, "created": False, "rebuilt": False}
            if existed:
                tmp = INDEX_NAME + "_new"
                conn.execute(f'DROP INDEX CONCURRENTLY IF EXISTS "{tmp}"')
                conn.execute(f'CREATE INDEX CONCURRENTLY "{tmp}" '
                             f"ON memory_items USING GIN ({COLUMN})")
                conn.execute(f'DROP INDEX CONCURRENTLY "{INDEX_NAME}"')
                conn.execute(f'ALTER INDEX "{tmp}" RENAME TO "{INDEX_NAME}"')
                return {"index": INDEX_NAME, "created": False, "rebuilt": True}
            conn.execute(f'DROP INDEX CONCURRENTLY IF EXISTS "{INDEX_NAME}"')  # invalid leftover
            conn.execute(f'CREATE INDEX CONCURRENTLY "{INDEX_NAME}" '
                         f"ON memory_items USING GIN ({COLUMN})")
            conn.execute(f"ANALYZE memory_items ({COLUMN})")
            return {"index": INDEX_NAME, "created": True, "rebuilt": False}
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
