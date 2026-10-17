"""Hybrid plan slice 1: the write path indexes a note's chunks directly.

retain() writes kind='chunk' items (content = original_chunk = the chunk) with
no model call and stamps documents.created_on on every note. The directory
indexer writes its chunks as kind='chunk' as well. (Question items, written by
a call that was removed, stay in an old bank; tests/test_legacy_removed.py.)"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import psycopg

# Several paragraphs of invented text: more than one 1,000-character chunk.
BODY = "\n\n".join(
    f"Paragraph {i} about the harbour lighthouse. " + "The beacon turns slowly. " * 12
    for i in range(8)
)


def _items(url, doc_id):
    with psycopg.connect(url) as conn:
        return conn.execute(
            "SELECT kind, content, original_chunk, ordinal, char_start, char_end, "
            "llm_generated, embedding IS NOT NULL, "
            "body_tsv @@ websearch_to_tsquery('english', 'lighthouse') "
            "FROM memory_items WHERE document_id = %s ORDER BY ordinal NULLS LAST, content",
            (doc_id,)).fetchall()


def _created_on(url, doc_id):
    with psycopg.connect(url) as conn:
        return conn.execute(
            "SELECT created_on FROM documents WHERE id = %s", (doc_id,)).fetchone()[0]


def _utc_today():
    return dt.datetime.now(dt.timezone.utc).date()


# --------------------------------------------------------------------------
# retain(): chunks only, no model call
# --------------------------------------------------------------------------

def test_default_retain_makes_no_llm_calls(memory_with_bank_and_mock_llm, mock_llm):
    mem = memory_with_bank_and_mock_llm
    mem.retain(BODY, source="harbour.md")
    assert mock_llm.calls == []
    with psycopg.connect(mem.database_url) as conn:
        assert conn.execute("SELECT count(*) FROM llm_calls").fetchone()[0] == 0


def test_default_retain_writes_searchable_chunk_items(memory_with_bank):
    doc = memory_with_bank.retain(BODY, source="harbour.md")
    rows = _items(memory_with_bank.database_url, doc)
    assert len(rows) > 1
    for i, (kind, content, chunk, ordinal, start, end, generated, has_vec, has_tsv) in enumerate(rows):
        assert kind == "chunk" and generated is False
        assert content == chunk == BODY[start:end] and ordinal == i
        assert has_vec and has_tsv      # an embedding and a body_tsv hit on 'lighthouse'
    assert rows[0][4] == 0 and rows[-1][5] == len(BODY.rstrip())   # the chunks cover the note


def test_retain_is_found_by_stage_1(memory_with_bank):
    mem = memory_with_bank
    mem.retain(BODY, source="harbour.md")
    mem.retain("Orchids need humid air.", source="orchids.md")
    hit = mem.search("lighthouse", limit=1)[0]
    assert hit.source == "harbour.md" and hit.match_kind == "exact+semantic"
    assert mem.search("harbour lighthouse beacon", limit=1)[0].source == "harbour.md"


def test_retain_replace_regenerates_chunks_only(memory_with_bank):
    first = memory_with_bank.retain(BODY, source="harbour.md")
    again = memory_with_bank.retain(BODY, source="harbour.md")
    assert first == again
    rows = _items(memory_with_bank.database_url, again)
    assert {r[0] for r in rows} == {"chunk"}
    with psycopg.connect(memory_with_bank.database_url) as conn:
        assert conn.execute("SELECT count(*) FROM memory_items").fetchone()[0] == len(rows)


# --------------------------------------------------------------------------
# retain(): documents.created_on is always stamped
# --------------------------------------------------------------------------

def test_undated_note_is_stamped_with_todays_date_utc(memory_with_bank, monkeypatch):
    monkeypatch.delenv("PROSPECTA_TIMEZONE", raising=False)
    before = _utc_today()
    doc = memory_with_bank.retain("An undated note.", source="plain.md")
    assert _created_on(memory_with_bank.database_url, doc) in {before, _utc_today()}


def test_undated_note_is_stamped_in_the_configured_timezone(memory_with_bank, monkeypatch):
    zone = "Pacific/Kiritimati"   # UTC+14: its date differs from UTC's for 14 hours a day
    monkeypatch.setenv("PROSPECTA_TIMEZONE", zone)
    before = dt.datetime.now(ZoneInfo(zone)).date()
    doc = memory_with_bank.retain("An undated note.", source="plain.md")
    assert _created_on(memory_with_bank.database_url, doc) in {
        before, dt.datetime.now(ZoneInfo(zone)).date()}


def test_note_date_comes_from_frontmatter_then_metadata_kwarg(memory_with_bank):
    fm = "---\ncreated: 2024-03-05\n---\nA dated note."
    a = memory_with_bank.retain(fm, source="a.md")
    b = memory_with_bank.retain(fm + " Again.", source="b.md", metadata={"created_on": "2023-01-02"})
    assert str(_created_on(memory_with_bank.database_url, a)) == "2024-03-05"
    assert str(_created_on(memory_with_bank.database_url, b)) == "2023-01-02"


def test_every_write_is_stamped(memory_with_bank_and_mock_llm, tmp_path):
    mem = memory_with_bank_and_mock_llm
    docs = [mem.retain("one", source="1"),
            mem.retain("two", source="2", tags=["t"], metadata={"created_on": "2023-01-02"}),
            mem.retain("three", path=tmp_path / "three.md")]
    assert all(_created_on(mem.database_url, d) is not None for d in docs)


def test_re_retain_keeps_the_note_date_and_stamps_a_null(memory_with_bank):
    mem = memory_with_bank
    doc = mem.retain("A note retained twice.", source="twice.md")
    with psycopg.connect(mem.database_url) as conn:
        conn.execute("UPDATE documents SET created_on = '2020-05-06' WHERE id = %s", (doc,))
        conn.commit()
    mem.retain("A note retained twice.", source="twice.md")
    assert str(_created_on(mem.database_url, doc)) == "2020-05-06"
    with psycopg.connect(mem.database_url) as conn:
        conn.execute("UPDATE documents SET created_on = NULL WHERE id = %s", (doc,))
        conn.commit()
    mem.retain("A note retained twice.", source="twice.md")
    assert _created_on(mem.database_url, doc) is not None


# --------------------------------------------------------------------------
# the date helper
# --------------------------------------------------------------------------

def test_today_is_utc_unless_a_timezone_is_configured(monkeypatch):
    from prospecta._filters import today
    monkeypatch.delenv("PROSPECTA_TIMEZONE", raising=False)
    assert today() in {_utc_today(), _utc_today() - dt.timedelta(days=1)}
    monkeypatch.setenv("PROSPECTA_TIMEZONE", "Etc/GMT+12")
    assert today() == dt.datetime.now(ZoneInfo("Etc/GMT+12")).date()


def test_today_falls_back_to_utc_on_an_unknown_timezone(monkeypatch, caplog):
    from prospecta._filters import today
    monkeypatch.setenv("PROSPECTA_TIMEZONE", "Mars/Olympus_Mons")
    assert today() in {_utc_today(), _utc_today() - dt.timedelta(days=1)}
    assert "PROSPECTA_TIMEZONE" in caplog.text


# --------------------------------------------------------------------------
# index_directory(): chunks are kind='chunk' too, and never call a model
# --------------------------------------------------------------------------

def test_directory_index_writes_chunk_kind_items_without_a_model(
        memory_with_bank_and_mock_llm, mock_llm, tmp_path):
    mem = memory_with_bank_and_mock_llm
    (tmp_path / "harbour.md").write_text(BODY)
    mem.index_directory(tmp_path)
    with psycopg.connect(mem.database_url) as conn:
        rows = conn.execute(
            "SELECT kind, content, original_chunk, ordinal, char_start, char_end "
            "FROM memory_items ORDER BY ordinal").fetchall()
    assert len(rows) > 1 and mock_llm.calls == []
    for i, (kind, content, chunk, ordinal, start, end) in enumerate(rows):
        assert kind == "chunk" and content == chunk and ordinal == i
        assert BODY[start:end].strip() == content
