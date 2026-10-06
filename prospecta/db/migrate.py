"""Migration runner.

Applies numbered SQL files from prospecta/db/migrations/ in order.
Tracks applied migrations in prospecta_schema_version table.
Serializes concurrent migration attempts via pg_advisory_xact_lock.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import psycopg

logger = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).parent / "migrations"

# Advisory-lock key for migration serialization.
# 0x70726F73706563 = ASCII "prospec" (7 bytes, 56 bits) — safely fits in signed bigint.
# Verified during schema ralplan: longer values overflowed.
MIGRATE_LOCK_KEY = 0x70726F73706563


def _list_migrations() -> list[tuple[int, Path]]:
    """Find all NNNN_*.sql migration files in dependency order."""
    files: list[tuple[int, Path]] = []
    for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
        m = re.match(r"^(\d+)_", f.name)
        if m:
            files.append((int(m.group(1)), f))
    return files


def _ensure_version_table(conn: psycopg.Connection) -> None:
    """Create prospecta_schema_version if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS prospecta_schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                description TEXT NOT NULL DEFAULT ''
            )
            """
        )


def _applied_versions(conn: psycopg.Connection) -> set[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT version FROM prospecta_schema_version")
        return {row[0] for row in cur.fetchall()}


def run_migrations(database_url: str) -> dict:
    """Apply pending migrations idempotently.

    Uses pg_advisory_xact_lock to serialize concurrent migration attempts.
    Returns {"applied": [versions], "skipped": [versions]}.
    """
    applied: list[int] = []
    skipped: list[int] = []

    with psycopg.connect(database_url, autocommit=False) as conn:
        # Acquire advisory lock for the duration of this transaction.
        # If another process holds it, this BLOCKS until it's released.
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATE_LOCK_KEY,))

        _ensure_version_table(conn)
        current = _applied_versions(conn)

        for version, path in _list_migrations():
            if version in current:
                skipped.append(version)
                logger.debug("Migration %d already applied: %s", version, path.name)
                continue

            sql = path.read_text()
            logger.info("Applying migration %d: %s", version, path.name)
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO prospecta_schema_version (version, description) VALUES (%s, %s)",
                    (version, path.stem),
                )
            applied.append(version)

        conn.commit()

    if 4 in applied:
        finish_0004(database_url)

    return {"applied": applied, "skipped": skipped}


BACKFILL_BATCH = 5000


def finish_0004(database_url: str, *, batch_size: int = BACKFILL_BATCH) -> dict:
    """Post-commit half of migration 0004; idempotent, safe to re-run.

    1. Batched backfill (one short transaction per batch): rows the
       directory-index path wrote carry chunk_index/start_char/end_char in
       their metadata (Chunk.to_metadata); they become kind='chunk' with
       ordinal/char_start/char_end filled. Everything else stays 'question'.
    2. CREATE INDEX CONCURRENTLY on (bank_id, kind).
    3. Per bank and per kind, a partial HNSW index (embedding::vector(N)).
    """
    from prospecta.db.queries import KIND_INDEX_KINDS, ensure_kind_hnsw_indexes

    backfilled = 0
    with psycopg.connect(database_url, autocommit=True) as conn:
        while True:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH batch AS (
                        SELECT id FROM memory_items
                        WHERE kind = 'question' AND metadata ? 'chunk_index'
                        LIMIT %s
                        FOR UPDATE SKIP LOCKED
                    )
                    UPDATE memory_items m
                    SET kind = 'chunk',
                        ordinal = (m.metadata->>'chunk_index')::integer,
                        char_start = (m.metadata->>'start_char')::integer,
                        char_end = (m.metadata->>'end_char')::integer
                    FROM batch WHERE m.id = batch.id
                    """,
                    (batch_size,),
                )
                n = cur.rowcount
            backfilled += n
            if n == 0:
                break
        with conn.cursor() as cur:
            cur.execute(
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS memory_items_bank_kind_idx "
                "ON memory_items (bank_id, kind)"
            )
            cur.execute("SELECT bank_id, embedding_dim FROM banks")
            banks = cur.fetchall()
        for bank_id, dim in banks:
            ensure_kind_hnsw_indexes(conn, bank_id, dim)
    return {"backfilled": backfilled, "banks": len(banks), "kinds": list(KIND_INDEX_KINDS)}


def get_schema_version(database_url: str) -> int | None:
    """Return the highest applied version, or None if no migrations applied."""
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'prospecta_schema_version'
                )
                """
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return None
            cur.execute("SELECT MAX(version) FROM prospecta_schema_version")
            row = cur.fetchone()
            return row[0] if row else None
