"""T11: write-side bilateral spine via Memory.retain().

Acceptance criteria:
  - caller-supplied index_text bypasses LLM (P4)
  - retain_events row appended with correct columns
  - replace-on-source-match (docs/schema.md, "Re-retain")
  - DocumentSourceConflictError on hash collision with different source
"""
from __future__ import annotations

import psycopg
import pytest

from prospecta._types import DocumentSourceConflictError


def _rows_for_document(database_url: str, document_id: str) -> list[dict]:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.content, m.metadata, m.original_chunk, m.llm_generated,
                       m.update_mode
                FROM memory_items m
                WHERE m.document_id = %s
                ORDER BY m.content
                """,
                (document_id,),
            )
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


def _document_row(database_url: str, document_id: str) -> dict:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, source, original_text, content_hash, tags, "
                "document_metadata FROM documents WHERE id = %s",
                (document_id,),
            )
            row = cur.fetchone()
            cols = [c.name for c in cur.description]
            return dict(zip(cols, row))


def _count_documents(database_url: str) -> int:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM documents")
            return cur.fetchone()[0]


def _retain_events(database_url: str) -> list[dict]:
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT bank_id, document_id, items_count, "
                "index_text_caller_supplied, duration_ms, raw_llm_response, error, "
                "created_at FROM retain_events ORDER BY id"
            )
            cols = [c.name for c in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# 1. Caller-supplied index_text bypasses LLM
# ---------------------------------------------------------------------------

def test_retain_caller_supplied_index_text(memory_with_bank_and_mock_llm, mock_llm):
    mem = memory_with_bank_and_mock_llm
    body = "Kelly was born on March 4th, 1990."
    doc_id = mem.retain(
        body,
        index_text="When is Kelly's birthday?",
        source="kelly-bio",
    )
    assert isinstance(doc_id, str)
    assert mock_llm.calls == []  # LLM NEVER called when override supplied

    rows = _rows_for_document(mem.database_url, doc_id)
    assert len(rows) == 1
    assert rows[0]["content"] == "When is Kelly's birthday?"
    assert rows[0]["metadata"].get("index_text_caller_supplied") is True
    assert rows[0]["llm_generated"] is False
    # P5: body preserved
    assert rows[0]["original_chunk"] == body
    doc = _document_row(mem.database_url, doc_id)
    assert doc["original_text"] == body
    assert doc["source"] == "kelly-bio"


# ---------------------------------------------------------------------------
# 4. retain_events row appended with correct columns + duration_ms numeric
# ---------------------------------------------------------------------------

def test_retain_appends_retain_event(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    doc_id = mem.retain("payload", index_text="q?", source="s1")
    events = _retain_events(mem.database_url)
    assert len(events) == 1
    e = events[0]
    assert e["bank_id"] == "test"
    assert str(e["document_id"]) == doc_id
    assert e["items_count"] == 1
    assert e["index_text_caller_supplied"] is True
    assert isinstance(e["duration_ms"], int)
    assert e["duration_ms"] >= 0
    assert e["error"] is None
    assert e["created_at"] is not None


# ---------------------------------------------------------------------------
# 5. Re-retain same source REPLACES
# ---------------------------------------------------------------------------

def test_reretain_same_source_replaces(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    body = "stable content"
    doc_id_1 = mem.retain(body, index_text=["a?", "b?"], source="same-src")
    doc_id_2 = mem.retain(body, index_text=["c?", "d?", "e?"], source="same-src")
    # Same content_hash + same source → same document_id, replaced items.
    assert doc_id_1 == doc_id_2
    assert _count_documents(mem.database_url) == 1

    rows = _rows_for_document(mem.database_url, doc_id_2)
    contents = sorted(r["content"] for r in rows)
    assert contents == ["c?", "d?", "e?"]


# ---------------------------------------------------------------------------
# 6. Re-retain DIFFERENT source raises
# ---------------------------------------------------------------------------

def test_reretain_different_source_raises(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    body = "same body"
    mem.retain(body, index_text="q?", source="src-A")
    with pytest.raises(DocumentSourceConflictError) as ei:
        mem.retain(body, index_text="q?", source="src-B")
    err = ei.value
    assert err.existing_source == "src-A"
    assert err.attempted_source == "src-B"
    # content_hash exposes the sha256 hex
    assert len(err.content_hash) == 64


# ---------------------------------------------------------------------------
# 7. list index_text produces multiple rows (T10 parity via retain API)
# ---------------------------------------------------------------------------

def test_retain_list_index_text_produces_multiple_rows(memory_with_bank_and_mock_llm, mock_llm):
    mem = memory_with_bank_and_mock_llm
    doc_id = mem.retain(
        "body",
        index_text=["q1?", "q2?", "q3?"],
        source="list-src",
    )
    assert mock_llm.calls == []
    rows = _rows_for_document(mem.database_url, doc_id)
    assert sorted(r["content"] for r in rows) == ["q1?", "q2?", "q3?"]
    for r in rows:
        assert r["metadata"].get("index_text_caller_supplied") is True


# ---------------------------------------------------------------------------
# 8. Default source derived from content_hash
# ---------------------------------------------------------------------------

def test_retain_source_default_derived_from_content_hash(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    doc_id = mem.retain("default-source-body", index_text="q?")
    doc = _document_row(mem.database_url, doc_id)
    assert doc["source"] is not None
    assert doc["source"].startswith("retain://")


# ---------------------------------------------------------------------------
# 9. retain() writes content to path when supplied + doesn't exist
# ---------------------------------------------------------------------------

def test_retain_writes_path_when_supplied(memory_with_bank_and_mock_llm, tmp_path):
    mem = memory_with_bank_and_mock_llm
    target = tmp_path / "subdir" / "note.md"
    assert not target.exists()
    mem.retain(
        "file content from retain()",
        index_text="q?",
        path=target,
    )
    assert target.exists()
    assert target.read_text() == "file content from retain()"
