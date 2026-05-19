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


# ---------------------------------------------------------------------------
# T9 — Document + memory_item write helpers
# ---------------------------------------------------------------------------

def _vec_literal(vec):
    """Serialize a list[float] to pgvector text form (caller casts ::vector)."""
    return "[" + ",".join(repr(float(x)) for x in vec) + "]"


def upsert_document(
    conn,
    *,
    bank_id: str,
    source: str | None,
    content_hash: str,
    original_text: str,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> tuple[str, bool, str | None]:
    """Insert or update a documents row.

    Returns (document_id, was_replaced, prior_source).
    - If (bank_id, content_hash) is new → insert, was_replaced=False.
    - If exists with matching source (incl. both NULL) → no-op fetch, was_replaced=False.
    - If exists with divergent source → updates source bookkeeping but still
      treats it as a replace event for caller's stats.

    Replace-on-source-match per schema.md §6: source-divergence at the
    document layer is *not* raised here — T9's slice indexes by file path,
    where content_hash collisions across different source paths are caller
    error best surfaced higher up. T11's retain() will impose the strict
    DocumentSourceConflictError contract.
    """
    import json as _json

    metadata_json = _json.dumps(metadata or {})
    tags_arr = list(tags or [])

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, source FROM documents "
            "WHERE bank_id = %(bank_id)s AND content_hash = %(content_hash)s",
            {"bank_id": bank_id, "content_hash": content_hash},
        )
        row = cur.fetchone()
        if row is not None:
            doc_id, existing_source = row
            return str(doc_id), False, existing_source

        cur.execute(
            """
            INSERT INTO documents
                (bank_id, source, original_text, content_hash,
                 tags, document_metadata)
            VALUES (%(bank_id)s, %(source)s, %(original_text)s, %(content_hash)s,
                    %(tags)s, %(metadata)s::jsonb)
            RETURNING id
            """,
            {
                "bank_id": bank_id,
                "source": source,
                "original_text": original_text,
                "content_hash": content_hash,
                "tags": tags_arr,
                "metadata": metadata_json,
            },
        )
        doc_id = cur.fetchone()[0]
        return str(doc_id), False, None


def delete_memory_items_for_document(conn, *, document_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM memory_items WHERE document_id = %(doc)s",
            {"doc": document_id},
        )
        return cur.rowcount or 0


def update_document_source(conn, *, document_id: str, source: str | None,
                           original_text: str, metadata: dict | None,
                           tags: list[str] | None) -> None:
    import json as _json
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE documents
            SET source = %(source)s,
                original_text = %(original_text)s,
                document_metadata = %(metadata)s::jsonb,
                tags = %(tags)s,
                updated_at = now()
            WHERE id = %(id)s
            """,
            {
                "id": document_id,
                "source": source,
                "original_text": original_text,
                "metadata": _json.dumps(metadata or {}),
                "tags": list(tags or []),
            },
        )


def upsert_memory_items(
    conn,
    *,
    bank_id: str,
    document_id: str,
    items: list[dict],
) -> int:
    """Bulk-insert memory_items rows. content_tsv is generated by Postgres.

    Each item: {content, original_chunk, embedding, metadata, tags, update_mode}.
    Returns number of rows inserted.
    """
    import json as _json
    if not items:
        return 0
    with conn.cursor() as cur:
        for it in items:
            emb_lit = _vec_literal(it["embedding"])
            cur.execute(
                """
                INSERT INTO memory_items
                    (bank_id, document_id, content, original_chunk, embedding,
                     metadata, tags, update_mode, llm_generated)
                VALUES
                    (%(bank_id)s, %(doc)s, %(content)s, %(orig)s, %(emb)s::vector,
                     %(metadata)s::jsonb, %(tags)s, %(update_mode)s, %(llm_gen)s)
                """,
                {
                    "bank_id": bank_id,
                    "doc": document_id,
                    "content": it["content"],
                    "orig": it.get("original_chunk", it["content"]),
                    "emb": emb_lit,
                    "metadata": _json.dumps(it.get("metadata") or {}),
                    "tags": list(it.get("tags") or []),
                    "update_mode": it.get("update_mode", "append"),
                    "llm_gen": bool(it.get("llm_generated", False)),
                },
            )
    return len(items)


def append_retain_event(
    conn,
    *,
    bank_id: str,
    document_id: str | None,
    items_count: int,
    index_text_caller_supplied: bool,
    duration_ms: int,
    raw_llm_response: str | None = None,
    error: str | None = None,
) -> None:
    """INSERT a row into retain_events.

    Schema invariants (schema.md §1):
      - bank_id, items_count, index_text_caller_supplied, duration_ms are NOT NULL.
      - document_id is NULL-able (deletion sets it NULL via ON DELETE SET NULL).
      - created_at defaults to now().
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO retain_events
                (bank_id, document_id, items_count,
                 index_text_caller_supplied, duration_ms,
                 raw_llm_response, error)
            VALUES
                (%(bank_id)s, %(doc)s, %(items_count)s,
                 %(caller_supplied)s, %(duration_ms)s,
                 %(raw)s, %(error)s)
            """,
            {
                "bank_id": bank_id,
                "doc": document_id,
                "items_count": int(items_count),
                "caller_supplied": bool(index_text_caller_supplied),
                "duration_ms": int(duration_ms),
                "raw": raw_llm_response,
                "error": error,
            },
        )


def append_recall_event(
    conn,
    *,
    bank_id: str,
    queries: list[str],
    mode: str,
    n_results: int,
    duration_ms: int,
    trace: dict | None = None,
) -> None:
    """INSERT a row into recall_events.

    Schema invariants (schema.md §1):
      - bank_id, queries (JSONB), mode, n_results, duration_ms are NOT NULL.
      - query_timestamp, trace are nullable.
      - created_at defaults to now().
    """
    import json as _json
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO recall_events
                (bank_id, queries, mode, n_results, duration_ms, trace)
            VALUES
                (%(bank_id)s, %(queries)s::jsonb, %(mode)s,
                 %(n_results)s, %(duration_ms)s, %(trace)s::jsonb)
            """,
            {
                "bank_id": bank_id,
                "queries": _json.dumps(list(queries)),
                "mode": mode,
                "n_results": int(n_results),
                "duration_ms": int(duration_ms),
                "trace": _json.dumps(trace) if trace is not None else None,
            },
        )


def append_formulate_event(
    conn,
    *,
    bank_id: str,
    message: str,
    n_queries: int,
    parse_fallback: bool,
    raw_response: str | None,
    error_kind: str | None,
    duration_ms: int,
    json_mode_used: bool = True,
    metadata: dict | None = None,
) -> None:
    """INSERT a row into formulate_events.

    Schema invariants (schema.md §1; migration 0001_initial.sql):
      - bank_id, message, n_queries_out, json_mode_used, parse_fallback,
        raw_response, duration_ms are NOT NULL.
      - created_at defaults to now().

    parse_fallback is canonical observability per schema.md §13. Raw LLM
    output is preserved verbatim. error_kind is passed in for symmetry
    with the in-memory FormulateOutcome but is NOT currently persisted —
    the table shape does not yet have an error_kind column; if needed,
    T14 sweeper can extend the schema.
    """
    # error_kind is not yet persisted (no column); accept for forward
    # compatibility and to keep the Memory wrapper symmetric.
    _ = error_kind
    _ = metadata
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO formulate_events
                (bank_id, message, n_queries_out, json_mode_used,
                 parse_fallback, raw_response, duration_ms)
            VALUES
                (%(bank_id)s, %(message)s, %(n_queries)s, %(json_mode_used)s,
                 %(parse_fallback)s, %(raw_response)s, %(duration_ms)s)
            """,
            {
                "bank_id": bank_id,
                "message": message,
                "n_queries": int(n_queries),
                "json_mode_used": bool(json_mode_used),
                "parse_fallback": bool(parse_fallback),
                # schema column is NOT NULL; coerce None → empty string.
                "raw_response": raw_response if raw_response is not None else "",
                "duration_ms": int(duration_ms),
            },
        )


def append_sweep_pass(
    conn,
    *,
    bank_id: str,
    corpus_path: str,
    started_at,
    ended_at,
    files_seen: int,
    files_indexed: int,
    files_pruned: int = 0,
    errors_count: int = 0,
    duration_ms: int,
    error: str | None = None,
    pass_metadata: dict | None = None,
) -> int:
    """Append a row to sweep_passes (A4 append-only history).

    Returns the inserted row's BIGSERIAL id.
    """
    import json as _json
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sweep_passes
                (bank_id, corpus_path, started_at, ended_at,
                 files_seen, files_indexed, files_pruned, errors_count,
                 duration_ms, error, pass_metadata)
            VALUES
                (%(bank_id)s, %(corpus_path)s, %(started_at)s, %(ended_at)s,
                 %(files_seen)s, %(files_indexed)s, %(files_pruned)s, %(errors_count)s,
                 %(duration_ms)s, %(error)s, %(pass_metadata)s::jsonb)
            RETURNING id
            """,
            {
                "bank_id": bank_id,
                "corpus_path": corpus_path,
                "started_at": started_at,
                "ended_at": ended_at,
                "files_seen": int(files_seen),
                "files_indexed": int(files_indexed),
                "files_pruned": int(files_pruned),
                "errors_count": int(errors_count),
                "duration_ms": int(duration_ms),
                "error": error,
                "pass_metadata": _json.dumps(pass_metadata or {}),
            },
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def upsert_sweeper_state(
    conn,
    *,
    bank_id: str,
    corpus_path: str,
    last_pass_started_at,
    last_pass_ended_at,
    last_pass_files_seen: int,
    last_pass_files_indexed: int,
    last_pass_files_pruned: int,
    last_pass_errors: int,
    last_pass_duration_ms: int,
    last_error: str | None = None,
) -> None:
    """Upsert sweeper_state (current-state cache, schema.md §9)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO sweeper_state
                (bank_id, corpus_path,
                 last_pass_started_at, last_pass_ended_at,
                 last_pass_files_seen, last_pass_files_indexed,
                 last_pass_files_pruned, last_pass_errors,
                 last_pass_duration_ms, last_error)
            VALUES
                (%(bank_id)s, %(corpus_path)s,
                 %(started)s, %(ended)s,
                 %(seen)s, %(indexed)s, %(pruned)s, %(errors)s,
                 %(duration_ms)s, %(error)s)
            ON CONFLICT (bank_id, corpus_path) DO UPDATE
            SET last_pass_started_at = EXCLUDED.last_pass_started_at,
                last_pass_ended_at   = EXCLUDED.last_pass_ended_at,
                last_pass_files_seen = EXCLUDED.last_pass_files_seen,
                last_pass_files_indexed = EXCLUDED.last_pass_files_indexed,
                last_pass_files_pruned  = EXCLUDED.last_pass_files_pruned,
                last_pass_errors     = EXCLUDED.last_pass_errors,
                last_pass_duration_ms = EXCLUDED.last_pass_duration_ms,
                last_error           = EXCLUDED.last_error
            """,
            {
                "bank_id": bank_id,
                "corpus_path": corpus_path,
                "started": last_pass_started_at,
                "ended": last_pass_ended_at,
                "seen": int(last_pass_files_seen),
                "indexed": int(last_pass_files_indexed),
                "pruned": int(last_pass_files_pruned),
                "errors": int(last_pass_errors),
                "duration_ms": int(last_pass_duration_ms),
                "error": last_error,
            },
        )


def read_sweeper_state(conn, *, bank_id: str, corpus_path: str) -> dict | None:
    """Return current sweeper_state row for (bank_id, corpus_path) as a dict,
    or None if no row exists."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT bank_id, corpus_path,
                   last_pass_started_at, last_pass_ended_at,
                   last_pass_files_seen, last_pass_files_indexed,
                   last_pass_files_pruned, last_pass_errors,
                   last_pass_duration_ms, last_error
            FROM sweeper_state
            WHERE bank_id = %(bank_id)s AND corpus_path = %(corpus_path)s
            """,
            {"bank_id": bank_id, "corpus_path": corpus_path},
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "bank_id": row[0],
            "corpus_path": row[1],
            "last_pass_started_at": row[2],
            "last_pass_ended_at": row[3],
            "last_pass_files_seen": row[4],
            "last_pass_files_indexed": row[5],
            "last_pass_files_pruned": row[6],
            "last_pass_errors": row[7],
            "last_pass_duration_ms": row[8],
            "last_error": row[9],
        }


def delete_documents_by_source(
    conn, *, bank_id: str, sources: list[str]
) -> int:
    """Delete documents (and cascade memory_items) by source list."""
    if not sources:
        return 0
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM documents "
            "WHERE bank_id = %(bank_id)s AND source = ANY(%(sources)s)",
            {"bank_id": bank_id, "sources": list(sources)},
        )
        return cur.rowcount or 0


# ---------------------------------------------------------------------------
# T9 — Retrieval SQL (hybrid / semantic / lexical)
# ---------------------------------------------------------------------------

# A6: COALESCE on scores → never NULL in returned rows. Plain SQL using
# parameterized bank_id, query_embedding, query_text, limit, rrf_k.
HYBRID_SQL = """
WITH
  semantic AS (
    SELECT m.id AS id, m.document_id, m.content, m.original_chunk, d.source,
           m.metadata, m.tags,
           1 - (m.embedding <=> %(qe)s::vector) AS sem_score,
           ROW_NUMBER() OVER (ORDER BY m.embedding <=> %(qe)s::vector) AS sem_rank
    FROM memory_items m
    JOIN documents d ON d.id = m.document_id
    WHERE m.bank_id = %(bank_id)s
      AND (%(meta)s::jsonb IS NULL OR m.metadata @> %(meta)s::jsonb)
    ORDER BY m.embedding <=> %(qe)s::vector
    LIMIT %(per_side)s
  ),
  lexical AS (
    SELECT m.id AS id, m.document_id, m.content, m.original_chunk, d.source,
           m.metadata, m.tags,
           ts_rank_cd(m.content_tsv, websearch_to_tsquery('english', %(qt)s)) AS lex_score,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(m.content_tsv, websearch_to_tsquery('english', %(qt)s)) DESC
           ) AS lex_rank
    FROM memory_items m
    JOIN documents d ON d.id = m.document_id
    WHERE m.bank_id = %(bank_id)s
      AND m.content_tsv @@ websearch_to_tsquery('english', %(qt)s)
      AND (%(meta)s::jsonb IS NULL OR m.metadata @> %(meta)s::jsonb)
    ORDER BY lex_score DESC
    LIMIT %(per_side)s
  ),
  fused AS (
    SELECT
      COALESCE(s.id, l.id) AS id,
      COALESCE(s.document_id, l.document_id) AS document_id,
      COALESCE(s.content, l.content) AS content,
      COALESCE(s.original_chunk, l.original_chunk) AS original_chunk,
      COALESCE(s.source, l.source) AS source,
      COALESCE(s.metadata, l.metadata) AS metadata,
      COALESCE(s.tags, l.tags) AS tags,
      COALESCE(s.sem_score, 0.0) AS sem_score,
      COALESCE(l.lex_score, 0.0) AS lex_score,
      (COALESCE(1.0 / (%(rrf_k)s + s.sem_rank), 0.0)
       + COALESCE(1.0 / (%(rrf_k)s + l.lex_rank), 0.0)) AS rrf_score
    FROM semantic s
    FULL OUTER JOIN lexical l ON s.id = l.id
  )
SELECT id, document_id, content, original_chunk, source, metadata, tags,
       sem_score, lex_score, rrf_score
FROM fused
ORDER BY rrf_score DESC
LIMIT %(limit)s
"""

SEMANTIC_SQL = """
SELECT m.id, m.document_id, m.content, m.original_chunk, d.source,
       m.metadata, m.tags,
       1 - (m.embedding <=> %(qe)s::vector) AS sem_score
FROM memory_items m
JOIN documents d ON d.id = m.document_id
WHERE m.bank_id = %(bank_id)s
  AND (%(meta)s::jsonb IS NULL OR m.metadata @> %(meta)s::jsonb)
ORDER BY m.embedding <=> %(qe)s::vector
LIMIT %(limit)s
"""

LEXICAL_SQL = """
SELECT m.id, m.document_id, m.content, m.original_chunk, d.source,
       m.metadata, m.tags,
       ts_rank_cd(m.content_tsv, websearch_to_tsquery('english', %(qt)s)) AS lex_score
FROM memory_items m
JOIN documents d ON d.id = m.document_id
WHERE m.bank_id = %(bank_id)s
  AND m.content_tsv @@ websearch_to_tsquery('english', %(qt)s)
  AND (%(meta)s::jsonb IS NULL OR m.metadata @> %(meta)s::jsonb)
ORDER BY lex_score DESC
LIMIT %(limit)s
"""


def _meta_param(metadata_filter: dict | None) -> str | None:
    import json as _json
    if metadata_filter is None:
        return None
    return _json.dumps(metadata_filter)


def hybrid_search(
    conn, *, bank_id: str, query_text: str, query_embedding: list[float],
    limit: int = 10, rrf_k: int = 60,
    metadata_filter: dict | None = None,
    per_side: int = 50,
) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            HYBRID_SQL,
            {
                "bank_id": bank_id,
                "qe": _vec_literal(query_embedding),
                "qt": query_text,
                "limit": int(limit),
                "rrf_k": int(rrf_k),
                "meta": _meta_param(metadata_filter),
                "per_side": int(per_side),
            },
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def semantic_search(
    conn, *, bank_id: str, query_embedding: list[float],
    limit: int = 10, metadata_filter: dict | None = None,
) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            SEMANTIC_SQL,
            {
                "bank_id": bank_id,
                "qe": _vec_literal(query_embedding),
                "limit": int(limit),
                "meta": _meta_param(metadata_filter),
            },
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


def lexical_search(
    conn, *, bank_id: str, query_text: str,
    limit: int = 10, metadata_filter: dict | None = None,
) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            LEXICAL_SQL,
            {
                "bank_id": bank_id,
                "qt": query_text,
                "limit": int(limit),
                "meta": _meta_param(metadata_filter),
            },
        )
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# T15 — Global stats (CLI `prospecta stats`)
# ---------------------------------------------------------------------------

def stats(conn, *, bank_id: str | None = None) -> dict:
    """Return aggregate counters across substrate.

    If bank_id is provided, scopes counts to that bank where applicable.
    Returns a dict with: banks, documents, memory_items, retain_events,
    recall_events, formulate_events, sweep_passes.
    """
    out: dict = {}
    with conn.cursor() as cur:
        if bank_id is None:
            cur.execute("SELECT COUNT(*) FROM banks")
            out["banks"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM documents")
            out["documents"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM memory_items")
            out["memory_items"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM retain_events")
            out["retain_events"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM recall_events")
            out["recall_events"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM formulate_events")
            out["formulate_events"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM sweep_passes")
            out["sweep_passes"] = cur.fetchone()[0]
        else:
            cur.execute("SELECT COUNT(*) FROM banks WHERE bank_id = %s", (bank_id,))
            out["banks"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM documents WHERE bank_id = %s", (bank_id,))
            out["documents"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM memory_items WHERE bank_id = %s", (bank_id,))
            out["memory_items"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM retain_events WHERE bank_id = %s", (bank_id,))
            out["retain_events"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM recall_events WHERE bank_id = %s", (bank_id,))
            out["recall_events"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM formulate_events WHERE bank_id = %s", (bank_id,))
            out["formulate_events"] = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM sweep_passes WHERE bank_id = %s", (bank_id,))
            out["sweep_passes"] = cur.fetchone()[0]
    return out
