"""Migration 0003: observability completeness — durable persistence of
verbatim LLM-generated index_text, recall results, synthesis, llm
prompt/response, and formulate error_kind.

Pairs the spec's eight acceptance tests with the columns added in
prospecta/db/migrations/0003_observability_completeness.sql.
"""
from __future__ import annotations

import psycopg

from prospecta._types import Query


def _one(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def _all(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


# ---------------------------------------------------------------------------
# A. retain_events.index_text_generated
# ---------------------------------------------------------------------------

def test_retain_persists_generated_index_text(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    mem._llm.canned = "q1\nq2\nq3"  # type: ignore[attr-defined]
    mem.retain("body", source="t1")
    with psycopg.connect(mem.database_url) as conn:
        row = _one(
            conn,
            "SELECT index_text_generated FROM retain_events "
            "WHERE bank_id = %s ORDER BY id DESC LIMIT 1",
            ("test",),
        )
    assert row is not None
    assert row[0] == ["q1", "q2", "q3"]


def test_retain_caller_supplied_leaves_generated_null(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    mem.retain("body", source="t1", index_text="my question")
    with psycopg.connect(mem.database_url) as conn:
        row = _one(
            conn,
            "SELECT index_text_generated, index_text_caller_supplied "
            "FROM retain_events WHERE bank_id = %s ORDER BY id DESC LIMIT 1",
            ("test",),
        )
    assert row is not None
    assert row[0] is None
    assert row[1] is True
    # P4: mock_llm.canned default would have produced 3 lines; calls must be empty.
    assert mem._llm.calls == []  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# B + C. recall_events.results + synthesis
# ---------------------------------------------------------------------------

def test_recall_persists_results_and_synthesis(populated_corpus):
    mem = populated_corpus
    mem._llm.set_response(  # type: ignore[attr-defined]
        lambda messages, *, json_mode: (
            '{"queries":[{"text":"kelly"}]}' if json_mode else "Kelly was born March 4th."
        )
    )
    mem.recall_synth("query about kelly")
    with psycopg.connect(mem.database_url) as conn:
        row = _one(
            conn,
            "SELECT results, synthesis FROM recall_events "
            "WHERE bank_id = %s ORDER BY id DESC LIMIT 1",
            ("test",),
        )
    assert row is not None
    results, synthesis = row
    assert results is not None
    assert isinstance(results, list)
    assert len(results) >= 1
    first = results[0]
    assert "source" in first
    assert "document_id" in first
    assert "rank" in first
    assert "scores" in first
    assert "content_preview" in first
    assert isinstance(first["scores"], dict)
    # A6 invariant: scores dict has the four numeric keys.
    for key in ("semantic", "lexical", "lexical_body", "rrf"):
        assert key in first["scores"]
    assert synthesis is not None
    assert len(synthesis) > 0


def test_recall_no_synth_leaves_synthesis_null(populated_corpus):
    mem = populated_corpus
    mem.recall([Query(text="kelly")], limit=3)
    with psycopg.connect(mem.database_url) as conn:
        row = _one(
            conn,
            "SELECT results, synthesis FROM recall_events "
            "WHERE bank_id = %s ORDER BY id DESC LIMIT 1",
            ("test",),
        )
    assert row is not None
    results, synthesis = row
    assert synthesis is None
    assert results is not None
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# D. llm_calls.prompt_text + response_text
# ---------------------------------------------------------------------------

def test_llm_calls_persists_prompt_and_response_by_default(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    mem._llm.canned = "q1"  # type: ignore[attr-defined]
    mem.retain("body", source="t1")
    with psycopg.connect(mem.database_url) as conn:
        rows = _all(
            conn,
            "SELECT prompt_name, prompt_text, response_text FROM llm_calls "
            "WHERE bank_id = %s ORDER BY id",
            ("test",),
        )
    # Should have one row for index_text generation.
    idx_rows = [r for r in rows if r[0] == "index_text"]
    assert len(idx_rows) == 1
    _, prompt_text, response_text = idx_rows[0]
    assert prompt_text is not None and len(prompt_text) > 0
    assert response_text == "q1"


def test_llm_calls_text_disabled_when_sink_flag_off(fresh_db, mock_llm):
    """When PostgresSink(persist_llm_text=False), the columns are written NULL
    even though the tracer payload carries the prompt and response."""
    from prospecta.memory import Memory
    from prospecta.observability import PostgresSink
    from tests._stub_embedder import EMBED_DIM, stub_embed

    mem = Memory(
        database_url=fresh_db,
        bank_id="testbank",
        llm=mock_llm,
        embed=stub_embed,
    )
    # Swap in a sink with persist_llm_text=False before any work.
    mem._tracer = PostgresSink(
        pool=mem._pool, bank_id="testbank", persist_llm_text=False
    )
    mem.create_bank("testbank", embedding_dim=EMBED_DIM)
    mock_llm.canned = "q"
    mem.retain("body", source="t")
    try:
        with psycopg.connect(mem.database_url) as conn:
            row = _one(
                conn,
                "SELECT prompt_text, response_text FROM llm_calls "
                "WHERE bank_id = %s AND prompt_name = 'index_text' "
                "ORDER BY id DESC LIMIT 1",
                ("testbank",),
            )
        assert row is not None
        assert row[0] is None
        assert row[1] is None
    finally:
        mem.close()


# ---------------------------------------------------------------------------
# E. formulate_events.error_kind
# ---------------------------------------------------------------------------

def test_formulate_persists_error_kind_on_malformed(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    mem._llm.canned = "not json {{"  # type: ignore[attr-defined]
    mem.formulate_queries("message")
    with psycopg.connect(mem.database_url) as conn:
        row = _one(
            conn,
            "SELECT error_kind, parse_fallback FROM formulate_events "
            "WHERE bank_id = %s ORDER BY id DESC LIMIT 1",
            ("test",),
        )
    assert row is not None
    assert row[0] == "malformed_json"
    assert row[1] is True


def test_formulate_persists_error_kind_on_schema_mismatch(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    mem._llm.canned = '{"foo": "bar"}'  # type: ignore[attr-defined]
    mem.formulate_queries("message")
    with psycopg.connect(mem.database_url) as conn:
        row = _one(
            conn,
            "SELECT error_kind, parse_fallback FROM formulate_events "
            "WHERE bank_id = %s ORDER BY id DESC LIMIT 1",
            ("test",),
        )
    assert row is not None
    assert row[0] == "schema_mismatch"
    assert row[1] is True
