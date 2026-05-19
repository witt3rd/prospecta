"""T12: read-side bilateral spine — Memory.recall + recall_synth + stub formulate.

Acceptance criteria:
  - recall_synth chains formulate→recall→synthesize, returns RAGResult
  - synth_prompt_override honored (P4)
  - recall accepts list[Query] OR list[str]
  - recall_events row appended on recall_synth
  - queries_to_results keyed by query.text
  - full content preserved through chunks (P5)
"""
from __future__ import annotations

import psycopg

from prospecta._types import Query, RAGResult, RecalledMemory


def _recall_events(database_url: str) -> list[dict]:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT bank_id, queries, mode, n_results, duration_ms "
                "FROM recall_events ORDER BY id"
            )
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# 1. recall_synth chains formulate→recall→synthesize
# ---------------------------------------------------------------------------

def test_recall_synth_calls_synthesize(populated_corpus, mock_llm):
    mock_llm.canned = "Kelly's birthday is March 4th."
    result = populated_corpus.recall_synth("what about kelly?")
    assert isinstance(result, RAGResult)
    assert result.synthesis  # non-empty
    assert result.synthesis == "Kelly's birthday is March 4th."
    assert len(result.queries) == 1  # stub returns single query
    assert result.queries_to_results  # dict per query
    # Exactly one synthesize call
    assert len(mock_llm.calls) == 1
    assert mock_llm.calls[-1]["json_mode"] is False


# ---------------------------------------------------------------------------
# 2. synth_prompt_override (P4)
# ---------------------------------------------------------------------------

def test_recall_synth_prompt_override(memory_with_bank_and_mock_llm, mock_llm):
    mem = memory_with_bank_and_mock_llm
    # No data needed — empty corpus is fine; we just verify the override
    # is rendered with our variables and sent to the LLM.
    result = mem.recall_synth(
        "what",
        synth_prompt_override="custom: {{ query }} chunks={{ chunks|length }}",
    )
    assert isinstance(result, RAGResult)
    last = mock_llm.calls[-1]
    user_msg = last["messages"][-1]["content"]
    assert "custom:" in user_msg
    assert "custom: what chunks=0" in user_msg


# ---------------------------------------------------------------------------
# 3. recall low-level — list[Query] input → flat list[RecalledMemory]
# ---------------------------------------------------------------------------

def test_recall_low_level_caller_queries(populated_corpus):
    queries = [Query(text="kelly"), Query(text="beach")]
    results = populated_corpus.recall(queries, limit=3)
    assert isinstance(results, list)
    assert len(results) <= 6
    assert all(isinstance(r, RecalledMemory) for r in results)


# ---------------------------------------------------------------------------
# 4. recall accepts list[str] (convenience coercion)
# ---------------------------------------------------------------------------

def test_recall_accepts_str_queries(populated_corpus):
    results = populated_corpus.recall(["kelly", "beach"], limit=3)
    assert isinstance(results, list)
    assert all(isinstance(r, RecalledMemory) for r in results)


# ---------------------------------------------------------------------------
# 5. recall_events row appended
# ---------------------------------------------------------------------------

def test_recall_appends_recall_event(populated_corpus):
    populated_corpus.recall_synth("query")
    events = _recall_events(populated_corpus.database_url)
    assert len(events) == 1
    e = events[0]
    assert e["bank_id"] == "test"
    assert e["mode"] == "hybrid"
    assert isinstance(e["duration_ms"], int)
    assert e["duration_ms"] >= 0
    assert isinstance(e["n_results"], int)
    assert e["n_results"] >= 0
    # queries column is JSONB-decoded by psycopg to python list
    assert e["queries"] == ["query"]


# ---------------------------------------------------------------------------
# 6. queries_to_results keyed by query.text
# ---------------------------------------------------------------------------

def test_recall_synth_queries_to_results_keyed_by_query_text(populated_corpus):
    result = populated_corpus.recall_synth("kelly birthday")
    assert "kelly birthday" in result.queries_to_results


# ---------------------------------------------------------------------------
# 7. P5: full content preserved through queries_to_results
# ---------------------------------------------------------------------------

def test_recall_synth_full_content_preserved(populated_corpus):
    result = populated_corpus.recall_synth("kelly")
    # Should have at least one query bucket
    assert result.queries_to_results
    for query_text, chunks in result.queries_to_results.items():
        for c in chunks:
            assert c.content
            assert len(c.content) > 0
            # original_chunk preserves the full document body (P5)
            assert c.original_chunk
            assert len(c.original_chunk) > 0


# ---------------------------------------------------------------------------
# 8. formulate_queries stub returns single-query echo
# ---------------------------------------------------------------------------

def test_formulate_queries_stub(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    out = mem.formulate_queries("anything")
    assert len(out) == 1
    assert isinstance(out[0], Query)
    assert out[0].text == "anything"
    # Stub does NOT call the LLM (T13 owns that)
    # mock_llm.calls would have grown if formulate touched LLM
