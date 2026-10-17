"""Migration 0003: observability completeness — durable persistence of the recall results
and the llm prompt / response.

Pairs the columns added in prospecta/db/migrations/0003_observability_completeness.sql with what is
written now. (`retain_events.index_text_generated`, `recall_events.synthesis` and `formulate_events` hold
rows of calls that were removed; nothing writes them any more.)
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
# A. retain_events: a write calls no model and says so
# ---------------------------------------------------------------------------

def test_retain_event_columns_that_stay_are_filled_and_no_model_is_called(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    mem.retain("body", source="t1")
    with psycopg.connect(mem.database_url) as conn:
        row = _one(
            conn,
            "SELECT items_count, index_text_generated, index_text_caller_supplied, raw_llm_response "
            "FROM retain_events WHERE bank_id = %s ORDER BY id DESC LIMIT 1",
            ("test",),
        )
    assert row == (1, None, False, None)
    assert mem._llm.calls == []  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# B. recall_events.results
# ---------------------------------------------------------------------------

def test_recall_persists_results(populated_corpus):
    mem = populated_corpus
    mem.recall([Query(text="kelly")], limit=3)
    with psycopg.connect(mem.database_url) as conn:
        row = _one(
            conn,
            "SELECT results, synthesis, mode FROM recall_events "
            "WHERE bank_id = %s ORDER BY id DESC LIMIT 1",
            ("test",),
        )
    assert row is not None
    results, synthesis, mode = row
    assert isinstance(results, list) and len(results) >= 1
    first = results[0]
    for key in ("source", "document_id", "rank", "scores", "content_preview"):
        assert key in first
    assert isinstance(first["scores"], dict)
    # the numeric scores of Stage 1's one channel and the fused score
    for key in ("hybrid_chunk", "rrf"):
        assert isinstance(first["scores"][key], float)
    assert synthesis is None and mode == "hybrid"       # the answer of `ask` is an llm_call, not a recall column


# ---------------------------------------------------------------------------
# C. llm_calls.prompt_text + response_text
# ---------------------------------------------------------------------------

def test_llm_calls_persists_prompt_and_response_by_default(populated_corpus):
    mem = populated_corpus
    reply = "Kelly was born March 4th [1]."
    mem._llm.canned = reply  # type: ignore[attr-defined]
    mem.ask("when was kelly born")
    with psycopg.connect(mem.database_url) as conn:
        rows = _all(
            conn,
            "SELECT prompt_name, prompt_text, response_text FROM llm_calls "
            "WHERE bank_id = %s ORDER BY id",
            ("test",),
        )
    ask_rows = [r for r in rows if r[0] == "ask_cited"]
    assert len(ask_rows) == 1
    _, prompt_text, response_text = ask_rows[0]
    assert prompt_text is not None and "when was kelly born" in prompt_text
    assert response_text == reply


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
    mem.retain("Kelly was born on March 4th", source="kelly")
    mock_llm.canned = "an answer [1]"
    mem.ask("when was kelly born")
    try:
        with psycopg.connect(mem.database_url) as conn:
            row = _one(
                conn,
                "SELECT prompt_text, response_text FROM llm_calls "
                "WHERE bank_id = %s AND prompt_name = 'ask_cited' "
                "ORDER BY id DESC LIMIT 1",
                ("testbank",),
            )
        assert row is not None
        assert row[0] is None
        assert row[1] is None
    finally:
        mem.close()
