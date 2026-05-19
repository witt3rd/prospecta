"""Centralized SQL for prospecta operations.

All SQL lives here; psycopg-3 parameterized form. Extended by future
tasks (T9 vertical slice, T11 retain, T12 recall_synth, etc.).
"""
from __future__ import annotations

import re as _re

# ---------------------------------------------------------------------------
# Bank lifecycle
# ---------------------------------------------------------------------------

INSERT_BANK = """
    INSERT INTO banks (bank_id, embedding_dim, embedding_model_id, mission, retain_mission)
    VALUES (%(bank_id)s, %(embedding_dim)s, %(embedding_model_id)s, %(mission)s, %(retain_mission)s)
    ON CONFLICT (bank_id) DO NOTHING
    RETURNING bank_id, embedding_dim
"""

SELECT_BANK = """
    SELECT bank_id, embedding_dim, embedding_model_id, mission, retain_mission,
           created_at, updated_at
    FROM banks
    WHERE bank_id = %(bank_id)s
"""

SELECT_BANK_STATS = """
    SELECT
        b.bank_id,
        (SELECT COUNT(*) FROM documents d WHERE d.bank_id = b.bank_id) AS documents,
        (SELECT COUNT(*) FROM memory_items m WHERE m.bank_id = b.bank_id) AS memory_items,
        (SELECT MAX(created_at) FROM retain_events r WHERE r.bank_id = b.bank_id) AS last_retain_at
    FROM banks b
    WHERE b.bank_id = %(bank_id)s
"""

# Per-bank HNSW partial index (schema.md §4).
# Index name MUST be: memory_items_embedding_<safe_bank_id>_idx
# bank_id is interpolated as identifier (not parameterized) since CREATE INDEX
# doesn't accept parameters for index names. Caller MUST validate bank_id is
# a safe identifier before formatting.
# Note: Postgres requires the WHERE clause of a partial index to use literal
# values (parameters yield "could not determine data type"). bank_id is
# inlined here as a SQL string literal; the caller MUST validate it first via
# validate_bank_id().
HNSW_INDEX_TEMPLATE = """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name}
    ON memory_items USING hnsw ((embedding::vector({embedding_dim})) vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE bank_id = '{bank_id_literal}'
"""

# Validate bank_id is safe for use in identifier position.
# Allowed: ASCII letters, digits, underscores, hyphens. Length 1-63.
_BANK_ID_PATTERN = _re.compile(r"^[a-zA-Z0-9_-]{1,63}$")


def validate_bank_id(bank_id: str) -> None:
    if not isinstance(bank_id, str) or not _BANK_ID_PATTERN.match(bank_id):
        raise ValueError(
            f"Invalid bank_id {bank_id!r}: must match {_BANK_ID_PATTERN.pattern}"
        )


def hnsw_index_name(bank_id: str) -> str:
    """Construct the canonical per-bank HNSW index name.

    Format: memory_items_embedding_<bank_id>_idx
    Hyphens in bank_id are converted to underscores for SQL identifier safety.
    """
    validate_bank_id(bank_id)
    safe = bank_id.replace("-", "_")
    return f"memory_items_embedding_{safe}_idx"
