"""The advisory keys of the post-commit index builders, taken in one fixed order.

Two CREATE INDEX CONCURRENTLY on one table can deadlock: the statement waiting for the
table's lock already holds a snapshot that the building one waits to end in its last phase
(seen between finish_0004's `CREATE INDEX CONCURRENTLY IF NOT EXISTS` and the keyword index
build, since retired, when four processes migrate at once). So the builders run one at a time,
each holding a session advisory key while it builds:

    MIGRATE_LOCK_KEY + 1   finish_0004   (migrate.py)
    MIGRATE_LOCK_KEY + 2   finish_0005   (migrate.py)
    MIGRATE_LOCK_KEY + 3   the keyword index (retired; the key stays so that a process still running the
                           previous library, which builds that index, keeps taking turns with this one)
    MIGRATE_LOCK_KEY + 4   the exact_tsv index (exact_tsv.py)

The rule that makes that safe: every builder takes ALL the keys, its own included, in ONE
ascending order, and releases them all. (Holding its own key and then polling the others'
is a livelock when two builders run at once in different processes: each holds its own and
waits for the other's forever.) The keys are polled with pg_try_advisory_lock and never
waited on: CREATE INDEX CONCURRENTLY waits on every running statement and transaction, so a
peer blocked inside pg_advisory_lock would deadlock the build (the 0004/0018 caveat).

The base is the literal of migrate.MIGRATE_LOCK_KEY (a test pins them equal), kept here to
avoid an import cycle.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

import psycopg

BASE_KEY = 0x70726F73706563
BUILDER_KEYS = tuple(BASE_KEY + n for n in (1, 2, 3, 4))   # ascending: the one acquisition order
POLL_SECONDS = 0.05


@contextmanager
def hold_builder_locks(conn: psycopg.Connection) -> Iterator[None]:
    """Hold every index-builder advisory key (session level) for the block: taken in
    ascending order, each polled for, all released afterwards (in reverse) whatever happens.
    `conn` must be an autocommit connection."""
    taken: list[int] = []
    try:
        for key in BUILDER_KEYS:
            while not conn.execute("SELECT pg_try_advisory_lock(%s)", (key,)).fetchone()[0]:
                time.sleep(POLL_SECONDS)
            taken.append(key)
        yield
    finally:
        for key in reversed(taken):
            conn.execute("SELECT pg_advisory_unlock(%s)", (key,))
