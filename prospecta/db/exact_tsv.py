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

import psycopg

from prospecta.db.builder_locks import hold_builder_locks
from prospecta.db.keyword_index import has_index, indexed_expression

COLUMN = "exact_tsv"
INDEX_NAME = "memory_items_exact_tsv_gin"
# This build's own advisory key (MIGRATE_LOCK_KEY + 4); it takes every builder's key, this one
# included, in ascending order (prospecta/db/builder_locks.py).
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
    first so queries keep using the old index until the swap.

    No ANALYZE here: it takes a snapshot, then waits for the table's ShareUpdateExclusiveLock
    that a peer's CREATE INDEX CONCURRENTLY holds while it waits for that snapshot to end,
    which deadlocks two concurrent `run_migrations`. The Stage 1 plan does not depend on
    statistics of the column (its tsquery is a parameter, costed at the default selectivity
    whatever the statistics say), and autovacuum analyzes it in due course.

    The build holds every index builder's advisory key, its own (+4) included, taken in the
    one ascending order and polled with pg_try_advisory_lock, never waited on: CREATE INDEX
    CONCURRENTLY waits on every running statement and transaction, so a peer blocked inside
    pg_advisory_lock would deadlock it (the 0004/0018 caveat), and no two CREATE INDEX
    CONCURRENTLY run on the table at once (prospecta/db/builder_locks.py; a builder that
    held only its own key while polling the others' livelocked against a peer doing the same).
    """
    with psycopg.connect(database_url, autocommit=True) as conn:
        if not rebuild and has_exact_tsv_index(conn):      # the usual case: nothing to take locks for
            return {"index": INDEX_NAME, "created": False, "rebuilt": False}
        with hold_builder_locks(conn):
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
            return {"index": INDEX_NAME, "created": True, "rebuilt": False}
