"""T16 — PostgresSink integration tests.

Asserts the default-tracer PostgresSink preserves the row-write semantics
that prior tests (T11, T12, T13, T14) depend on. No DB-row count regression
must occur after T16's rip-and-replace.
"""
from __future__ import annotations

import psycopg


def _count(conn, table: str, bank_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {table} WHERE bank_id = %s", (bank_id,))
        return int(cur.fetchone()[0])


def test_postgres_sink_writes_retain_events(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    mem._llm.canned = "q1"  # type: ignore[attr-defined]
    mem.retain("content", source="test")
    with psycopg.connect(mem.database_url) as conn:
        assert _count(conn, "retain_events", "test") == 1


def test_postgres_sink_writes_recall_events(populated_corpus):
    mem = populated_corpus
    mem._llm.set_response(  # type: ignore[attr-defined]
        lambda messages, *, json_mode: (
            '{"queries":[{"text":"q"}]}' if json_mode else "synth"
        )
    )
    mem.recall_synth("query")
    with psycopg.connect(mem.database_url) as conn:
        assert _count(conn, "recall_events", "test") >= 1


def test_postgres_sink_writes_formulate_events(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    mem._llm.canned = '{"queries":[{"text":"q"}]}'  # type: ignore[attr-defined]
    mem.formulate_queries("msg")
    with psycopg.connect(mem.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT parse_fallback FROM formulate_events WHERE bank_id = %s",
                ("test",),
            )
            rows = cur.fetchall()
            assert len(rows) == 1
            assert rows[0][0] is False


def test_postgres_sink_writes_sweep_pass(memory_with_bank_and_mock_llm, tmp_corpus):
    from prospecta._sweeper import SweeperConfig, run_one_pass
    mem = memory_with_bank_and_mock_llm
    run_one_pass(mem, SweeperConfig(corpus_paths=[tmp_corpus]))
    with psycopg.connect(mem.database_url) as conn:
        assert _count(conn, "sweep_passes", "test") == 1
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM sweeper_state WHERE bank_id = %s",
                ("test",),
            )
            assert int(cur.fetchone()[0]) == 1


def test_postgres_sink_writes_llm_call(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    mem._llm.canned = "q1"  # type: ignore[attr-defined]
    mem.retain("content", source="test")
    with psycopg.connect(mem.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT prompt_name FROM llm_calls WHERE bank_id = %s",
                ("test",),
            )
            rows = cur.fetchall()
            assert any(r[0] == "index_text" for r in rows)


def test_composite_tracer_combines_postgres_sink_and_recording(fresh_db, mock_llm):
    """CompositeTracer(PostgresSink, RecordingTracer) preserves DB writes
    AND captures events in-memory for test assertions."""
    from prospecta._tracer import CompositeTracer, RecordingTracer
    from prospecta.memory import Memory
    from prospecta.observability import PostgresSink
    from tests._stub_embedder import EMBED_DIM, stub_embed

    rec = RecordingTracer()
    mem = Memory(
        database_url=fresh_db,
        bank_id="test",
        llm=mock_llm,
        embed=stub_embed,
    )
    mem.create_bank("test", embedding_dim=EMBED_DIM)
    sink = PostgresSink(pool=mem._pool, bank_id="test")
    mem._tracer = CompositeTracer(sink, rec)
    mock_llm.canned = "q1"
    mem.retain("content", source="t1")
    # Both sides observed the event.
    assert rec.by_name("retain")
    with psycopg.connect(mem.database_url) as conn:
        assert _count(conn, "retain_events", "test") == 1
    mem.close()
