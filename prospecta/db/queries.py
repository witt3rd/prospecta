"""Centralized SQL for prospecta operations.

All SQL lives here; psycopg-3 parameterized form. Extended by future
tasks (T9 vertical slice, T11 retain, T12 recall_synth, etc.).
"""
from __future__ import annotations

import re as _re
from prospecta._filters import filter_fields
from prospecta.db.exact_tsv import COLUMN as EXACT_TSV_COLUMN

# ---------------------------------------------------------------------------
# Bank lifecycle
# ---------------------------------------------------------------------------

# The one place a bank's config is written at creation (prospecta/bank_defaults.py): a new
# bank runs hybrid_chunk + cross_encoder with linking off. ON CONFLICT DO NOTHING: a bank
# that exists is never touched. The columns' own DEFAULTs ('[]', '{}', '{}') stay legacy.
INSERT_BANK = """
    INSERT INTO banks (bank_id, embedding_dim, embedding_model_id, mission, retain_mission,
                       channel_config, recall_config, link_config)
    VALUES (%(bank_id)s, %(embedding_dim)s, %(embedding_model_id)s, %(mission)s, %(retain_mission)s,
            %(channel_config)s::jsonb, %(recall_config)s::jsonb, %(link_config)s::jsonb)
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

# Per-bank HNSW partial index (docs/schema.md, "memory_items").
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


# Per-bank, per-kind partial HNSW indexes (design 8.3): the predicate repeats
# bank_id and kind literally so the planner can use them, and a kind-filtered
# scan is not starved by the other kind sharing one index.
KIND_INDEX_KINDS = ("question", "chunk")

HNSW_KIND_INDEX_TEMPLATE = """
    CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name}
    ON memory_items USING hnsw ((embedding::vector({embedding_dim})) vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE bank_id = '{bank_id_literal}' AND kind = '{kind}'
"""


def hnsw_kind_index_name(bank_id: str, kind: str) -> str:
    """memory_items_embedding_<bank>_<kind>_idx, kept within Postgres' 63 bytes.

    When the natural name is too long the bank part is cut and an md5 of the
    full bank_id is appended so two long bank ids never collide.
    """
    import hashlib

    validate_bank_id(bank_id)
    if kind not in KIND_INDEX_KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    safe = bank_id.replace("-", "_")
    name = f"memory_items_embedding_{safe}_{kind}_idx"
    if len(name) <= 63:
        return name
    digest = hashlib.md5(bank_id.encode()).hexdigest()[:8]
    fixed = f"memory_items_embedding__{digest}_{kind}_idx"
    room = 63 - len(fixed)
    return f"memory_items_embedding_{safe[:room]}_{digest}_{kind}_idx"


def ensure_kind_hnsw_indexes(conn, bank_id: str, embedding_dim: int) -> None:
    """Create both per-kind partial HNSW indexes. `conn` MUST be autocommit."""
    validate_bank_id(bank_id)
    for kind in KIND_INDEX_KINDS:
        sql = HNSW_KIND_INDEX_TEMPLATE.format(
            index_name=hnsw_kind_index_name(bank_id, kind),
            bank_id_literal=bank_id,
            embedding_dim=int(embedding_dim),
            kind=kind,
        )
        with conn.cursor() as cur:
            cur.execute(sql)


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

    Replace-on-source-match (docs/schema.md, "Re-retain"): source-divergence at the
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
                 tags, document_metadata, created_on, person, source_kind)
            VALUES (%(bank_id)s, %(source)s, %(original_text)s, %(content_hash)s,
                    %(tags)s, %(metadata)s::jsonb,
                    %(created_on)s, %(person)s, %(source_kind)s)
            RETURNING id
            """,
            {
                "bank_id": bank_id,
                "source": source,
                "original_text": original_text,
                "content_hash": content_hash,
                "tags": tags_arr,
                "metadata": metadata_json,
                **filter_fields(metadata),
            },
        )
        doc_id = cur.fetchone()[0]
        return str(doc_id), False, None


def delete_memory_items_for_document(conn, *, document_id: str) -> int:
    with conn.cursor() as cur:
        # the items' links and entity rows cascade; the Linker must run again
        cur.execute("DELETE FROM memory_link_state WHERE document_id = %(doc)s",
                    {"doc": document_id})
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
                created_on = %(created_on)s, person = %(person)s,
                source_kind = %(source_kind)s,
                updated_at = now()
            WHERE id = %(id)s
            """,
            {
                "id": document_id,
                "source": source,
                "original_text": original_text,
                "metadata": _json.dumps(metadata or {}),
                "tags": list(tags or []),
                **filter_fields(metadata),
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
                     metadata, tags, update_mode, llm_generated,
                     kind, ordinal, char_start, char_end)
                VALUES
                    (%(bank_id)s, %(doc)s, %(content)s, %(orig)s, %(emb)s::vector,
                     %(metadata)s::jsonb, %(tags)s, %(update_mode)s, %(llm_gen)s,
                     %(kind)s, %(ordinal)s, %(char_start)s, %(char_end)s)
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
                    "kind": it.get("kind", "question"),
                    "ordinal": it.get("ordinal"),
                    "char_start": it.get("char_start"),
                    "char_end": it.get("char_end"),
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
    index_text_generated: list[str] | None = None,
) -> None:
    """INSERT a row into retain_events.

    Schema invariants (docs/schema.md, migration 0003):
      - bank_id, items_count, index_text_caller_supplied, duration_ms are NOT NULL.
      - document_id is NULL-able (deletion sets it NULL via ON DELETE SET NULL).
      - index_text_generated is the verbatim LLM-authored list (NULL when caller
        supplied index_text or pre-0003).
      - created_at defaults to now().
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO retain_events
                (bank_id, document_id, items_count,
                 index_text_caller_supplied, duration_ms,
                 raw_llm_response, error, index_text_generated)
            VALUES
                (%(bank_id)s, %(doc)s, %(items_count)s,
                 %(caller_supplied)s, %(duration_ms)s,
                 %(raw)s, %(error)s, %(idx_gen)s)
            """,
            {
                "bank_id": bank_id,
                "doc": document_id,
                "items_count": int(items_count),
                "caller_supplied": bool(index_text_caller_supplied),
                "duration_ms": int(duration_ms),
                "raw": raw_llm_response,
                "error": error,
                "idx_gen": list(index_text_generated)
                if index_text_generated is not None
                else None,
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
    results: list[dict] | None = None,
    synthesis: str | None = None,
    plan: dict | None = None,
    channels: list[dict] | None = None,
    fusion: dict | None = None,
    rerank: dict | None = None,
    hops: dict | None = None,
    candidates: list[dict] | None = None,
    cost_usd: float | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    n_llm_calls: int | None = None,
    citations: list[dict] | None = None,
    depth: str | None = None,
) -> int:
    """INSERT a row into recall_events (and its recall_event_candidates).

    plan/channels/fusion/rerank/hops and candidates (migration 0007) are
    written by channel-pipeline recalls; legacy recalls leave them NULL.
    Returns the new recall_events.id.

    Schema invariants (docs/schema.md, migration 0003):
      - bank_id, queries (JSONB), mode, n_results, duration_ms are NOT NULL.
      - query_timestamp, trace are nullable.
      - results (JSONB): {source, document_id, rank, scores, content_preview}[]
        — inspection record (200-char preview per element). Full content stays
        in memory_items. NULL on pre-0003 rows.
      - synthesis (TEXT): RAG synthesis output; NULL for plain recall().
      - created_at defaults to now().
    """
    import json as _json
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO recall_events
                (bank_id, queries, mode, n_results, duration_ms, trace,
                 results, synthesis, plan, channels, fusion, rerank, hops,
                 cost_usd, tokens_in, tokens_out, n_llm_calls, citations, depth)
            VALUES
                (%(bank_id)s, %(queries)s::jsonb, %(mode)s,
                 %(n_results)s, %(duration_ms)s, %(trace)s::jsonb,
                 %(results)s::jsonb, %(synthesis)s, %(plan)s::jsonb,
                 %(channels)s::jsonb, %(fusion)s::jsonb, %(rerank)s::jsonb,
                 %(hops)s::jsonb, %(cost_usd)s, %(tokens_in)s,
                 %(tokens_out)s, %(n_llm_calls)s, %(citations)s::jsonb, %(depth)s)
            RETURNING id
            """,
            {
                "bank_id": bank_id,
                "queries": _json.dumps(list(queries)),
                "mode": mode,
                "n_results": int(n_results),
                "duration_ms": int(duration_ms),
                "trace": _json.dumps(trace) if trace is not None else None,
                "results": _json.dumps(results) if results is not None else None,
                "synthesis": synthesis,
                "plan": _dumps(plan),
                "channels": _dumps(channels),
                "fusion": _dumps(fusion),
                "rerank": _dumps(rerank),
                "hops": _dumps(hops),
                "cost_usd": cost_usd,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "n_llm_calls": n_llm_calls,
                "citations": _dumps(citations),
                "depth": depth,
            },
        )
        event_id = cur.fetchone()[0]
        for c in candidates or []:
            cur.execute(
                """
                INSERT INTO recall_event_candidates
                    (recall_event_id, query_index, document_id, item_id,
                     channel, rank, score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (event_id, int(c.get("query_index", 0)), c.get("document_id"),
                 c.get("item_id"), c["channel"], int(c["rank"]), float(c["score"])),
            )
    return event_id


def _dumps(obj):
    import json as _json
    return _json.dumps(obj) if obj is not None else None


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

    Schema invariants (docs/schema.md; migrations 0001 + 0003):
      - bank_id, message, n_queries_out, json_mode_used, parse_fallback,
        raw_response, duration_ms are NOT NULL.
      - error_kind is nullable; ``'malformed_json' | 'schema_mismatch' | NULL``.
        Persisted via 0003 (was dropped on the floor pre-0003).
      - created_at defaults to now().
    """
    _ = metadata
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO formulate_events
                (bank_id, message, n_queries_out, json_mode_used,
                 parse_fallback, raw_response, duration_ms, error_kind)
            VALUES
                (%(bank_id)s, %(message)s, %(n_queries)s, %(json_mode_used)s,
                 %(parse_fallback)s, %(raw_response)s, %(duration_ms)s,
                 %(error_kind)s)
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
                "error_kind": error_kind,
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
    """Upsert sweeper_state (current-state cache, docs/schema.md)."""
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


def append_llm_call(
    conn,
    *,
    bank_id: str | None,
    prompt_name: str,
    messages_count: int,
    json_mode: bool,
    duration_ms: int,
    error: str | None = None,
    prompt_text: str | None = None,
    response_text: str | None = None,
    model: str | None = None,
    tokens_in: int | None = None,
    tokens_out: int | None = None,
    cost_usd: float | None = None,
) -> None:
    """Append a row to llm_calls (docs/schema.md; migration 0003).

    bank_id is nullable (provider/setup-time calls may not have one).
    prompt_text and response_text are nullable; PostgresSink's
    persist_llm_text flag may force them to NULL even when the tracer
    payload carries them.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO llm_calls
                (bank_id, prompt_name, messages_count, json_mode,
                 duration_ms, error, prompt_text, response_text,
                 model, tokens_in, tokens_out, cost_usd)
            VALUES
                (%(bank_id)s, %(prompt_name)s, %(messages_count)s,
                 %(json_mode)s, %(duration_ms)s, %(error)s,
                 %(prompt_text)s, %(response_text)s,
                 %(model)s, %(tokens_in)s, %(tokens_out)s, %(cost_usd)s)
            """,
            {
                "bank_id": bank_id,
                "prompt_name": prompt_name,
                "messages_count": int(messages_count),
                "json_mode": bool(json_mode),
                "duration_ms": int(duration_ms),
                "error": error,
                "prompt_text": prompt_text,
                "response_text": response_text,
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": cost_usd,
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

# A6: COALESCE on scores → never NULL in returned rows. Three-channel RRF
# fusion: semantic (cosine over embedding) + lexical_content (ts_rank_cd over
# content_tsv = index_text / questions) + lexical_body (ts_rank_cd over
# body_tsv = original_chunk / source body). Three CTEs FULL OUTER JOIN'd;
# RRF score = sum of three COALESCE'd reciprocals. All score components
# always-numeric per A6 invariant.
#
# The lexical_body channel implements the P14 body-fallback safety net:
# when LLM-generated index_text drifts hard from query language, body
# content still rescues the doc via RRF fusion of its ts_rank_cd rank.
# Lexical channels use websearch_to_tsquery, which requires EVERY query
# term to match (AND semantics) and ts_rank_cd is cover-density rank, not
# BM25 (no IDF, no length normalisation). The safety net therefore fires
# on short queries only; long natural-language questions rarely match.
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
      AND (%(meta)s::jsonb IS NULL OR m.metadata @> (%(meta)s::jsonb - '_allow'))
  AND (%(meta)s::jsonb->'_allow' IS NULL OR (
        (%(meta)s::jsonb->'_allow'->'audience_key' IS NULL OR COALESCE(m.metadata->>'audience_key','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'audience_key')))
        AND (%(meta)s::jsonb->'_allow'->'project' IS NULL OR COALESCE(m.metadata->>'project','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'project')))))
    ORDER BY m.embedding <=> %(qe)s::vector
    LIMIT %(per_side)s
  ),
  lexical_content AS (
    SELECT m.id AS id, m.document_id, m.content, m.original_chunk, d.source,
           m.metadata, m.tags,
           ts_rank_cd(m.content_tsv, websearch_to_tsquery('english', %(qt)s)) AS lex_content_score,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(m.content_tsv, websearch_to_tsquery('english', %(qt)s)) DESC
           ) AS lex_content_rank
    FROM memory_items m
    JOIN documents d ON d.id = m.document_id
    WHERE m.bank_id = %(bank_id)s
      AND m.content_tsv @@ websearch_to_tsquery('english', %(qt)s)
      AND (%(meta)s::jsonb IS NULL OR m.metadata @> (%(meta)s::jsonb - '_allow'))
  AND (%(meta)s::jsonb->'_allow' IS NULL OR (
        (%(meta)s::jsonb->'_allow'->'audience_key' IS NULL OR COALESCE(m.metadata->>'audience_key','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'audience_key')))
        AND (%(meta)s::jsonb->'_allow'->'project' IS NULL OR COALESCE(m.metadata->>'project','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'project')))))
    ORDER BY lex_content_score DESC
    LIMIT %(per_side)s
  ),
  lexical_body AS (
    SELECT m.id AS id, m.document_id, m.content, m.original_chunk, d.source,
           m.metadata, m.tags,
           ts_rank_cd(m.body_tsv, websearch_to_tsquery('english', %(qt)s)) AS lex_body_score,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(m.body_tsv, websearch_to_tsquery('english', %(qt)s)) DESC
           ) AS lex_body_rank
    FROM memory_items m
    JOIN documents d ON d.id = m.document_id
    WHERE m.bank_id = %(bank_id)s
      AND m.body_tsv @@ websearch_to_tsquery('english', %(qt)s)
      AND (%(meta)s::jsonb IS NULL OR m.metadata @> (%(meta)s::jsonb - '_allow'))
  AND (%(meta)s::jsonb->'_allow' IS NULL OR (
        (%(meta)s::jsonb->'_allow'->'audience_key' IS NULL OR COALESCE(m.metadata->>'audience_key','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'audience_key')))
        AND (%(meta)s::jsonb->'_allow'->'project' IS NULL OR COALESCE(m.metadata->>'project','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'project')))))
    ORDER BY lex_body_score DESC
    LIMIT %(per_side)s
  ),
  fused AS (
    SELECT
      COALESCE(s.id, lc.id, lb.id) AS id,
      COALESCE(s.document_id, lc.document_id, lb.document_id) AS document_id,
      COALESCE(s.content, lc.content, lb.content) AS content,
      COALESCE(s.original_chunk, lc.original_chunk, lb.original_chunk) AS original_chunk,
      COALESCE(s.source, lc.source, lb.source) AS source,
      COALESCE(s.metadata, lc.metadata, lb.metadata) AS metadata,
      COALESCE(s.tags, lc.tags, lb.tags) AS tags,
      COALESCE(s.sem_score, 0.0) AS sem_score,
      COALESCE(lc.lex_content_score, 0.0) AS lex_content_score,
      COALESCE(lb.lex_body_score, 0.0) AS lex_body_score,
      (COALESCE(1.0 / (%(rrf_k)s + s.sem_rank), 0.0)
       + COALESCE(1.0 / (%(rrf_k)s + lc.lex_content_rank), 0.0)
       + COALESCE(1.0 / (%(rrf_k)s + lb.lex_body_rank), 0.0)) AS rrf_score
    FROM semantic s
    FULL OUTER JOIN lexical_content lc ON s.id = lc.id
    FULL OUTER JOIN lexical_body lb ON COALESCE(s.id, lc.id) = lb.id
  )
SELECT id, document_id, content, original_chunk, source, metadata, tags,
       sem_score, lex_content_score, lex_body_score, rrf_score
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
  AND (%(meta)s::jsonb IS NULL OR m.metadata @> (%(meta)s::jsonb - '_allow'))
  AND (%(meta)s::jsonb->'_allow' IS NULL OR (
        (%(meta)s::jsonb->'_allow'->'audience_key' IS NULL OR COALESCE(m.metadata->>'audience_key','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'audience_key')))
        AND (%(meta)s::jsonb->'_allow'->'project' IS NULL OR COALESCE(m.metadata->>'project','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'project')))))
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
  AND (%(meta)s::jsonb IS NULL OR m.metadata @> (%(meta)s::jsonb - '_allow'))
  AND (%(meta)s::jsonb->'_allow' IS NULL OR (
        (%(meta)s::jsonb->'_allow'->'audience_key' IS NULL OR COALESCE(m.metadata->>'audience_key','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'audience_key')))
        AND (%(meta)s::jsonb->'_allow'->'project' IS NULL OR COALESCE(m.metadata->>'project','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'project')))))
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
    """Lexical search over content_tsv (the index_text / questions channel).

    Note: mode='lexical' is preserved as content-only per v0.1 option A.
    The body channel (body_tsv) only fires through mode='hybrid' (RRF fusion).
    See body_lexical_search() for the parallel helper over body_tsv.
    """
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


BODY_LEXICAL_SQL = """
SELECT m.id, m.document_id, m.content, m.original_chunk, d.source,
       m.metadata, m.tags,
       ts_rank_cd(m.body_tsv, websearch_to_tsquery('english', %(qt)s)) AS lex_body_score
FROM memory_items m
JOIN documents d ON d.id = m.document_id
WHERE m.bank_id = %(bank_id)s
  AND m.body_tsv @@ websearch_to_tsquery('english', %(qt)s)
  AND (%(meta)s::jsonb IS NULL OR m.metadata @> (%(meta)s::jsonb - '_allow'))
  AND (%(meta)s::jsonb->'_allow' IS NULL OR (
        (%(meta)s::jsonb->'_allow'->'audience_key' IS NULL OR COALESCE(m.metadata->>'audience_key','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'audience_key')))
        AND (%(meta)s::jsonb->'_allow'->'project' IS NULL OR COALESCE(m.metadata->>'project','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'project')))))
ORDER BY lex_body_score DESC
LIMIT %(limit)s
"""


def body_lexical_search(
    conn, *, bank_id: str, query_text: str,
    limit: int = 10, metadata_filter: dict | None = None,
) -> list[dict]:
    """Lexical search over body_tsv (the original_chunk / source body channel).

    Parallel to lexical_search but matches against the body_tsv GENERATED
    column over original_chunk. Used internally by hybrid_search; not
    currently surfaced as its own Memory.search mode in v0.1 (the body
    rescue fires through mode='hybrid' RRF fusion). Exposed here for
    forward-compat and direct callers.
    """
    with conn.cursor() as cur:
        cur.execute(
            BODY_LEXICAL_SQL,
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
# Stage 1 — Reserved-Slot Hybrid over kind='chunk' items
# (docs/postgres-hybrid.md section 6). One query: a semantic, a stemmed lexical
# and an unstemmed exact/phrase retriever each keep their own top slots, so an
# exact hit always reaches the reranker; fused by FULL OUTER JOIN into the top
# `fused_limit` with an RRF fallback score, and ts_headline() runs on those
# winners only. Needs pgvector >= 0.8 (hnsw.iterative_scan).
#
# A question whose English reading is empty (blank, punctuation, only stop words) and that
# has a date window has nothing to match: the window itself is the question, and Stage 1
# lists the chunks inside it, newest note first, as `date` candidates (_STAGE1_DATE_SQL).
# No window and no words at all is empty.
# ---------------------------------------------------------------------------

STAGE1_SEMANTIC_LIMIT = 25
STAGE1_STEMMED_LIMIT = 20
STAGE1_EXACT_LIMIT = 20
STAGE1_FUSED_LIMIT = 50
STAGE1_HNSW_EF_SEARCH = 100
STAGE1_HEADLINE_OPTIONS = ('StartSel=<<, StopSel=>>, MaxWords=35, MinWords=15, '
                           'MaxFragments=2, FragmentDelimiter=" … "')

_NOTE_DATE = "COALESCE(d.created_on, d.created_at::date)"

# Pushed into all three retrievers, before their LIMIT.
_STAGE1_FILTERS = """
      AND (%(date_from)s::date IS NULL OR __NOTE_DATE__ >= %(date_from)s::date)
      AND (%(date_to)s::date IS NULL OR __NOTE_DATE__ <= %(date_to)s::date)
      AND (%(excluded)s::text[] IS NULL OR d.source IS NULL
           OR NOT (d.source = ANY(%(excluded)s::text[])))
      AND (%(meta)s::jsonb IS NULL OR m.metadata @> (%(meta)s::jsonb - '_allow'))
      AND (%(meta)s::jsonb->'_allow' IS NULL OR (
            (%(meta)s::jsonb->'_allow'->'audience_key' IS NULL OR COALESCE(m.metadata->>'audience_key','') IN
                (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'audience_key')))
            AND (%(meta)s::jsonb->'_allow'->'project' IS NULL OR COALESCE(m.metadata->>'project','') IN
                (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'project')))))
"""

# (SELECT stem FROM q) is one parsed tsquery, evaluated once and usable as an index
# condition. __DIM__: the bank's embedding width, so the per-bank partial HNSW index
# (an index on embedding::vector(N), see HNSW_KIND_INDEX_TEMPLATE) can serve the
# semantic retriever. __EXACT_VEC__: memory_items.exact_tsv, the stored 'simple' word
# vector (migration 0020, memory_items_exact_tsv_gin): matched and ranked as stored, so
# a chunk is never re-tokenized, however many match.
_STAGE1_SQL = """
WITH
  q AS (
    SELECT __STEM__ AS stem,
           websearch_to_tsquery('simple', %(qx)s) AS exact
  ),
  semantic_hits AS (
    SELECT m.id,
           ROW_NUMBER() OVER (ORDER BY m.embedding::vector(__DIM__) <=> %(qe)s::vector(__DIM__)) AS rank_ix
    FROM memory_items m JOIN documents d ON d.id = m.document_id
    WHERE m.bank_id = %(bank_id)s AND m.kind = 'chunk'
      __FILTERS__
    ORDER BY m.embedding::vector(__DIM__) <=> %(qe)s::vector(__DIM__)
    LIMIT %(semantic_limit)s
  ),
  stemmed_hits AS (
    SELECT m.id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(m.body_tsv, (SELECT stem FROM q)) DESC,
                                       __NOTE_DATE__ DESC, m.id) AS rank_ix
    FROM memory_items m JOIN documents d ON d.id = m.document_id
    WHERE m.bank_id = %(bank_id)s AND m.kind = 'chunk'
      AND m.body_tsv @@ (SELECT stem FROM q)
      __FILTERS__
    ORDER BY ts_rank_cd(m.body_tsv, (SELECT stem FROM q)) DESC, __NOTE_DATE__ DESC, m.id
    LIMIT %(stemmed_limit)s
  ),
  exact_hits AS (
    SELECT m.id,
           ROW_NUMBER() OVER (ORDER BY ts_rank_cd(__EXACT_VEC__, (SELECT exact FROM q), 4) DESC,
                                       __NOTE_DATE__ DESC, m.id) AS rank_ix
    FROM memory_items m JOIN documents d ON d.id = m.document_id
    WHERE m.bank_id = %(bank_id)s AND m.kind = 'chunk'
      AND __EXACT_VEC__ @@ (SELECT exact FROM q)
      __FILTERS__
    ORDER BY ts_rank_cd(__EXACT_VEC__, (SELECT exact FROM q), 4) DESC, __NOTE_DATE__ DESC, m.id
    LIMIT %(exact_limit)s
  ),
  fused AS (
    SELECT COALESCE(s.id, st.id, ex.id) AS id,
           s.rank_ix AS sem_rank, st.rank_ix AS stem_rank, ex.rank_ix AS exact_rank,
           (COALESCE(0.45 / (60 + s.rank_ix), 0.0)
            + COALESCE(0.25 / (60 + st.rank_ix), 0.0)
            + COALESCE(0.30 / (60 + ex.rank_ix), 0.0)) AS rrf_score
    FROM semantic_hits s
    FULL OUTER JOIN stemmed_hits st ON s.id = st.id
    FULL OUTER JOIN exact_hits ex ON COALESCE(s.id, st.id) = ex.id
    ORDER BY rrf_score DESC, COALESCE(s.id, st.id, ex.id)
    LIMIT %(fused_limit)s
  )
SELECT c.id, c.document_id, d.source, c.content, c.original_chunk, c.metadata,
       __NOTE_DATE__ AS created_on,
       f.sem_rank, f.stem_rank, f.exact_rank, f.rrf_score,
       CASE
         WHEN f.exact_rank IS NOT NULL THEN
           ts_headline('simple', c.original_chunk, (SELECT exact FROM q), %(headline)s)
         WHEN f.stem_rank IS NOT NULL THEN
           ts_headline('english', c.original_chunk, (SELECT stem FROM q), %(headline)s)
       END AS exact_snippet,
       NULL::bigint AS date_rank
FROM fused f
JOIN memory_items c ON c.id = f.id
JOIN documents d ON d.id = c.document_id
ORDER BY f.rrf_score DESC, c.id
"""


# A plain question (no search operator) matches the stemmed retriever as the OR of its
# stemmed words: websearch_to_tsquery ANDs them, so a natural-language question matched
# only a chunk holding every one of its words. The words come out of the parser
# (to_tsvector: stemmed, stop words gone), never out of the text, and each is quoted for
# tsquery (the backslash and the single quote doubled), so no text can add an operator.
# Every word is used: there is no cap (a cap dropped words without saying so); a longer
# question costs more per matching row (docs/exact-tsv.md has the measured cost).
_STEM_OR = r"""(SELECT COALESCE(string_agg(t.quoted, ' | '), '')::tsquery FROM (
          SELECT '''' || replace(replace(w.lexeme, '\', '\\'), '''', '''''') || '''' AS quoted
          FROM unnest(to_tsvector('english', %(qt)s)) AS w
          ORDER BY w.lexeme) t)"""
_STEM_WEBSEARCH = "websearch_to_tsquery('english', %(qt)s)"
# The documented search-box operators: a straight double quote, a standalone OR, a word led
# by a hyphen (-exclude).
_OPERATORS = _re.compile(r'"|(?:^|\s)-\w|(?:^|\s)OR(?:\s|$)')


def has_websearch_operators(query_text: str) -> bool:
    """True when the question uses a search operator ("quoted phrase", a standalone OR, a
    -excluded word): its words then keep websearch semantics (AND) in both lexical
    retrievers. Without one the stemmed retriever reads the words as an OR."""
    return _OPERATORS.search(query_text) is not None


def _stage1_statement(
    *, bank_id: str, query_text: str, query_embedding: list[float],
    date_from=None, date_to=None, excluded_sources: list[str] | None = None,
    metadata_filter: dict | None = None,
    semantic_limit: int = STAGE1_SEMANTIC_LIMIT, stemmed_limit: int = STAGE1_STEMMED_LIMIT,
    exact_limit: int = STAGE1_EXACT_LIMIT, fused_limit: int = STAGE1_FUSED_LIMIT,
    exact_text: str | None = None,
) -> tuple[str, dict]:
    sql = (_STAGE1_SQL
           .replace("__STEM__", _STEM_WEBSEARCH if has_websearch_operators(query_text) else _STEM_OR)
           .replace("__FILTERS__", _STAGE1_FILTERS)
           .replace("__NOTE_DATE__", _NOTE_DATE)
           .replace("__EXACT_VEC__", f"m.{EXACT_TSV_COLUMN}")
           .replace("__DIM__", str(len(query_embedding))))
    params = {
        "bank_id": bank_id, "qt": query_text, "qe": _vec_literal(query_embedding),
        "qx": query_text if exact_text is None else exact_text,
        "date_from": date_from, "date_to": date_to,
        "excluded": list(excluded_sources) if excluded_sources is not None else None,
        "meta": _meta_param(metadata_filter), "headline": STAGE1_HEADLINE_OPTIONS,
        "semantic_limit": int(semantic_limit), "stemmed_limit": int(stemmed_limit),
        "exact_limit": int(exact_limit), "fused_limit": int(fused_limit),
    }
    return sql, params


# The date-only listing: the newest notes' chunks inside the window, every filter of the
# three retrievers applied. Ties of one note date: the note created last, then the note's
# chunks in order. date_rank is the position (1 = newest); rrf_score descends with it so the
# order survives any fallback that sorts by score.
_STAGE1_DATE_SQL = """
WITH win AS (
  SELECT m.id, m.document_id, d.source, m.content, m.original_chunk, m.metadata,
         __NOTE_DATE__ AS created_on, d.created_at AS note_created_at, m.ordinal
  FROM memory_items m JOIN documents d ON d.id = m.document_id
  WHERE m.bank_id = %(bank_id)s AND m.kind = 'chunk'
    __FILTERS__
  ORDER BY __NOTE_DATE__ DESC, d.created_at DESC, m.document_id, m.ordinal, m.id
  LIMIT %(fused_limit)s
)
SELECT id, document_id, source, content, original_chunk, metadata, created_on,
       NULL::bigint AS sem_rank, NULL::bigint AS stem_rank, NULL::bigint AS exact_rank,
       1.0 / (60 + ROW_NUMBER() OVER w) AS rrf_score,
       NULL::text AS exact_snippet,
       ROW_NUMBER() OVER w AS date_rank
FROM win
WINDOW w AS (ORDER BY created_on DESC, note_created_at DESC, document_id, ordinal, id)
ORDER BY date_rank
"""

# Which Stage 1 shape a question gets. no_stem: its English reading is empty (blank,
# punctuation, only stop words); no_simple: its unstemmed reading is empty too.
_STAGE1_WORDS_SQL = """
SELECT numnode(websearch_to_tsquery('english', %(qt)s)) = 0,
       numnode(websearch_to_tsquery('simple', %(qt)s)) = 0,
       numnode(websearch_to_tsquery('simple', %(qx)s)) = 0
"""


def _stage1_mode(cur, query_text: str, date_from, date_to, exact_text: str | None = None) -> str:
    """'words' (the three retrievers), 'date' (list the window) or 'none' (nothing to ask).
    `exact_text` (the exact retriever's own text, see stage1_search) that holds words is
    something to ask even when `query_text` has none."""
    cur.execute(_STAGE1_WORDS_SQL, {"qt": query_text,
                                    "qx": query_text if exact_text is None else exact_text},
                prepare=False)
    no_stem, no_simple, no_exact = cur.fetchone()
    if not no_stem:
        return "words"
    if exact_text is not None and not no_exact:
        return "words"
    if date_from is not None or date_to is not None:
        return "date"
    return "none" if no_simple else "words"     # no window: stop words still match as written


def _stage1_date_statement(
    *, bank_id: str, date_from=None, date_to=None, excluded_sources: list[str] | None = None,
    metadata_filter: dict | None = None, fused_limit: int = STAGE1_FUSED_LIMIT, **_unused,
) -> tuple[str, dict]:
    sql = _STAGE1_DATE_SQL.replace("__FILTERS__", _STAGE1_FILTERS).replace("__NOTE_DATE__", _NOTE_DATE)
    params = {
        "bank_id": bank_id, "date_from": date_from, "date_to": date_to,
        "excluded": list(excluded_sources) if excluded_sources is not None else None,
        "meta": _meta_param(metadata_filter), "fused_limit": int(fused_limit),
    }
    return sql, params


def stage1_search(conn, *, hnsw_ef_search: int = STAGE1_HNSW_EF_SEARCH, **query) -> list[dict]:
    """The Stage 1 candidates: at most `fused_limit` chunk rows, best first.

    `query` is bank_id, query_text, query_embedding (the vector, or a callable that returns
    it: called only when the semantic retriever runs) and optionally date_from /
    date_to (the note's date, inclusive), excluded_sources (the anti-echo list of
    documents.source values), metadata_filter (the metadata / `_allow` filter of the
    other channels) and the slot counts semantic_limit / stemmed_limit / exact_limit /
    fused_limit (defaults 25 / 20 / 20 / 50). The filters apply inside every retriever,
    before its slots are cut. `query_text` is read by websearch_to_tsquery:
    "exact phrase", OR, -exclude. A plain question (no such operator, see
    has_websearch_operators) is read by the stemmed retriever as the OR of its stemmed
    words (all of them: no cap), ranked by ts_rank_cd; the
    exact retriever always keeps websearch semantics (AND). `exact_text`, when given, is what
    the exact retriever reads INSTEAD of `query_text` (and the snippet it headlines): the
    names, identifiers and "quoted phrases" a caller copied verbatim (query expansion:
    `query_text` then feeds the semantic and stemmed retrievers only); None (the default)
    leaves the exact retriever reading `query_text`. The semantic retriever reads at least
    `hnsw_ef_search` index candidates and keeps scanning (pgvector iterative scan)
    until its slots are full.

    Each row: id, document_id, source, content, original_chunk, metadata, created_on
    (the note's date: documents.created_on, else its created_at date), sem_rank /
    stem_rank / exact_rank (the chunk's rank in each retriever, None where it did not
    match), rrf_score and exact_snippet (ts_headline with <<matches>>; None for a
    purely semantic hit), and date_rank (None except in the date-only listing below).

    A question with no searchable words (its English reading is empty: blank, punctuation,
    only stop words) and a date window gets no retriever: the rows are the chunks inside the
    window, newest note first (by `created_on`, then the note created last, then the note's
    chunks in order), up to `fused_limit`, each with date_rank 1, 2, ... and a descending
    rrf_score, no other rank and no snippet; every other filter still applies. A question
    with no words at all (not even stop words) and no window is empty.
    """
    with conn.transaction(), conn.cursor() as cur:
        mode = _stage1_mode(cur, query["query_text"], query.get("date_from"), query.get("date_to"),
                            query.get("exact_text"))
        if mode == "none":
            return []
        if mode == "date":
            sql, params = _stage1_date_statement(**query)
            cur.execute(sql, params, prepare=False)
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
        if callable(query["query_embedding"]):
            query = {**query, "query_embedding": query["query_embedding"]()}
        sql, params = _stage1_statement(**query)
        cur.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        cur.execute(f"SET LOCAL hnsw.ef_search = {int(hnsw_ef_search)}")
        # prepare=False: a fresh plan per call, so the bank and kind literals select the
        # per-bank partial HNSW index.
        cur.execute(sql, params, prepare=False)
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
