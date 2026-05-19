"""T13 — read-side bilateral spine: formulate_queries (JSON-mode + parse-fallback).

Acceptance criteria per task spec. Library NEVER raises on malformed LLM JSON
(schema.md §13 — parse-fallback is canonical observability).
"""
from __future__ import annotations

import psycopg
import pytest

from prospecta._types import Query


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_formulate_events(mem) -> list[dict]:
    """Read all formulate_events rows for the test bank."""
    with psycopg.connect(mem.database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT bank_id, message, n_queries_out, json_mode_used, "
                "parse_fallback, raw_response, duration_ms "
                "FROM formulate_events WHERE bank_id = %s "
                "ORDER BY id ASC",
                (mem.default_bank_id,),
            )
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_formulate_queries_returns_multiple(memory_with_bank_and_mock_llm, mock_llm):
    mem = memory_with_bank_and_mock_llm
    mock_llm.canned = '{"queries": [{"text": "q1"}, {"text": "q2"}, {"text": "q3"}]}'
    queries = mem.formulate_queries("did kelly's bday work out?", context="DM")
    assert len(queries) == 3
    assert all(isinstance(q, Query) for q in queries)
    assert [q.text for q in queries] == ["q1", "q2", "q3"]


def test_formulate_queries_uses_json_mode(memory_with_bank_and_mock_llm, mock_llm):
    mem = memory_with_bank_and_mock_llm
    mock_llm.canned = '{"queries": [{"text": "x"}]}'
    mem.formulate_queries("message")
    assert mock_llm.calls
    assert mock_llm.calls[-1].get("json_mode") is True


# ---------------------------------------------------------------------------
# Parse fallback
# ---------------------------------------------------------------------------

def test_formulate_queries_malformed_json_fallback(memory_with_bank_and_mock_llm, mock_llm):
    mem = memory_with_bank_and_mock_llm
    mock_llm.canned = "not json {{"
    queries = mem.formulate_queries("original message")
    assert len(queries) == 1
    assert queries[0].text == "original message"
    rows = _fetch_formulate_events(mem)
    assert len(rows) == 1
    assert rows[0]["parse_fallback"] is True
    assert rows[0]["raw_response"] == "not json {{"
    assert rows[0]["n_queries_out"] == 1


def test_formulate_queries_schema_mismatch_fallback(memory_with_bank_and_mock_llm, mock_llm):
    mem = memory_with_bank_and_mock_llm
    mock_llm.canned = '{"foo": "bar"}'  # well-formed JSON, wrong shape
    queries = mem.formulate_queries("original message")
    assert len(queries) == 1
    assert queries[0].text == "original message"
    rows = _fetch_formulate_events(mem)
    assert len(rows) == 1
    assert rows[0]["parse_fallback"] is True
    assert rows[0]["raw_response"] == '{"foo": "bar"}'


def test_formulate_queries_empty_array_fallback(memory_with_bank_and_mock_llm, mock_llm):
    mem = memory_with_bank_and_mock_llm
    mock_llm.canned = '{"queries": []}'
    queries = mem.formulate_queries("original message")
    assert len(queries) == 1
    assert queries[0].text == "original message"
    rows = _fetch_formulate_events(mem)
    assert len(rows) == 1
    assert rows[0]["parse_fallback"] is True


# ---------------------------------------------------------------------------
# Prompt override (P4: caller wins)
# ---------------------------------------------------------------------------

def test_formulate_queries_prompt_override(memory_with_bank_and_mock_llm, mock_llm):
    mem = memory_with_bank_and_mock_llm
    mock_llm.canned = '{"queries": [{"text": "x"}]}'
    mem.formulate_queries("msg", prompt_override="custom prompt: {{ message }}")
    last = mock_llm.calls[-1]
    assert "custom prompt: msg" in last["messages"][-1]["content"]
    # json_mode is still requested even on override path
    assert last.get("json_mode") is True


# ---------------------------------------------------------------------------
# Event logging
# ---------------------------------------------------------------------------

def test_formulate_queries_appends_formulate_event_on_success(
    memory_with_bank_and_mock_llm, mock_llm
):
    mem = memory_with_bank_and_mock_llm
    mock_llm.canned = '{"queries": [{"text": "a"}, {"text": "b"}]}'
    mem.formulate_queries("msg")
    rows = _fetch_formulate_events(mem)
    assert len(rows) == 1
    row = rows[0]
    assert row["parse_fallback"] is False
    assert row["n_queries_out"] == 2
    assert row["json_mode_used"] is True
    assert isinstance(row["duration_ms"], int)
    assert row["duration_ms"] >= 0
    assert row["message"] == "msg"
    assert row["raw_response"] == '{"queries": [{"text": "a"}, {"text": "b"}]}'


# ---------------------------------------------------------------------------
# recall_synth now uses real formulate (no longer stub)
# ---------------------------------------------------------------------------

def test_recall_synth_uses_real_formulate(populated_corpus, mock_llm):
    mem = populated_corpus

    def respond(messages, *, json_mode):
        if json_mode:
            return '{"queries": [{"text": "kelly"}, {"text": "birthday"}]}'
        return "synthesis text"

    mock_llm.set_response(respond)
    result = mem.recall_synth("did kelly's bday work?")
    assert len(result.queries) == 2  # not 1 — the stub is gone
    assert result.synthesis == "synthesis text"
