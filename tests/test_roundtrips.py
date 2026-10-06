"""Round trips: retain -> recall/search and import (index_directory) -> search.

Provenance (source, original_chunk, metadata, document_id, bank_id) and
timestamps must survive the trip intact.
"""
from __future__ import annotations


import psycopg


def _db_now(url: str) -> datetime:
    with psycopg.connect(url) as conn:
        return conn.execute("SELECT now()").fetchone()[0]


def _doc_times(url: str, document_id: str):
    with psycopg.connect(url) as conn:
        return conn.execute(
            "SELECT created_at, updated_at, tags FROM documents WHERE id = %s",
            (document_id,),
        ).fetchone()


def _item_times(url: str, document_id: str):
    with psycopg.connect(url) as conn:
        return conn.execute(
            "SELECT created_at, updated_at FROM memory_items WHERE document_id = %s",
            (document_id,),
        ).fetchall()


def test_retain_then_recall_keeps_provenance_and_timestamps(
    memory_with_bank_and_mock_llm,
):
    mem = memory_with_bank_and_mock_llm
    body = "The anvil sings at dawn when Forge strikes the cold iron."
    before = _db_now(mem.database_url)
    doc_id = mem.retain(
        body,
        index_text="when does the anvil sing",
        source="roundtrip-note",
        tags=["forge", "dawn"],
        metadata={"origin": "unit-test"},
    )
    after = _db_now(mem.database_url)

    results = mem.recall(["anvil sing"], limit=5)
    assert results, "retained memory not recalled"
    hit = next(r for r in results if r.document_id == doc_id)
    assert hit.source == "roundtrip-note"
    assert hit.original_chunk == body
    assert hit.content == "when does the anvil sing"
    assert hit.bank_id == "test"
    assert hit.metadata.get("origin") == "unit-test"

    created, updated, tags = _doc_times(mem.database_url, doc_id)
    assert before <= created <= after
    assert created <= updated <= after
    assert sorted(tags) == ["dawn", "forge"]
    for c, u in _item_times(mem.database_url, doc_id):
        assert before <= c <= after

    # Reading must not alter stored timestamps.
    mem.search("anvil sing", limit=5)
    assert _doc_times(mem.database_url, doc_id)[:2] == (created, updated)


def test_retain_then_search_lexical_and_semantic(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    doc_id = mem.retain(
        "Cookie runs DJ Babycakes from the kitchen.",
        index_text="who is DJ Babycakes",
        source="cookie",
    )
    for mode in ("lexical", "semantic", "hybrid"):
        res = mem.search("DJ Babycakes", mode=mode, limit=5)
        assert any(r.document_id == doc_id for r in res), mode
        hit = next(r for r in res if r.document_id == doc_id)
        assert hit.source == "cookie"
        assert hit.original_chunk == "Cookie runs DJ Babycakes from the kitchen."


def test_retain_replace_on_source_updates_recall(memory_with_bank_and_mock_llm):
    mem = memory_with_bank_and_mock_llm
    mem.retain("old body about granite", index_text="granite", source="s1")
    mem.retain("new body about basalt", index_text="basalt", source="s1")
    res = mem.recall(["basalt"], limit=5)
    assert any(r.original_chunk == "new body about basalt" for r in res)
    assert not any(r.original_chunk == "old body about granite" for r in mem.recall(["granite"], limit=5))


def test_import_round_trip_keeps_source_and_body(memory_with_bank, tmp_path):
    body = "# Kelly\n\nKelly's birthday is April 30. She loves sunshine.\n"
    f = tmp_path / "kelly.md"
    f.write_text(body)
    before = _db_now(memory_with_bank.database_url)
    stats = memory_with_bank.index_directory(tmp_path)
    after = _db_now(memory_with_bank.database_url)
    assert stats.documents_added == 1

    res = memory_with_bank.search("kelly birthday", limit=5)
    assert res
    hit = res[0]
    assert "kelly.md" in hit.source
    assert "April 30" in hit.original_chunk
    created, _, _ = _doc_times(memory_with_bank.database_url, hit.document_id)
    assert before <= created <= after

    # Re-import unchanged: no new docs, timestamp preserved.
    stats2 = memory_with_bank.index_directory(tmp_path)
    assert stats2.documents_added == 0
    assert _doc_times(memory_with_bank.database_url, hit.document_id)[0] == created
