"""Migration 0005 and the filter fields: the note's date, person and kind, filled at retain and index
time from frontmatter or the `metadata=` kwarg, and the date window of Stage 1."""
from __future__ import annotations

import datetime as dt
import json

import psycopg
import pytest

from prospecta.db.migrate import finish_0005, run_migrations
from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM, stub_embed

# ------------------------------------------------------- db integration

@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    notes = [
        ("a1.md", "---\nperson: Alice\ncreated: 2024-03-05\ntype: journal\n---\nAlice walked the dog by the river."),
        ("a2.md", "---\nperson: Alice\ncreated: 2024-04-09\n---\nAlice repaired the bicycle chain."),
        ("b1.md", "---\nperson: Bob Stone\ncreated: 2024-03-20\n---\nBob walked the dog near the park."),
        ("n1.md", "No frontmatter, a note about walking a dog in the park."),
    ]
    for src, text in notes:
        m.retain(text, source=src)
    yield m
    m.close()


def rows(url):
    with psycopg.connect(url) as conn:
        return {s: r for s, *r in conn.execute(
            "SELECT source, created_on, person, source_kind FROM documents ORDER BY source")}


def test_retain_fills_filter_columns_from_frontmatter(mem):
    r = rows(mem.database_url)
    assert r["a1.md"] == [dt.date(2024, 3, 5), "Alice", "journal"]
    assert r["b1.md"] == [dt.date(2024, 3, 20), "Bob Stone", None]
    assert r["n1.md"][1:] == [None, None]
    assert r["n1.md"][0] is not None   # an undated note is stamped at retain


def test_retain_metadata_kwarg_wins_and_replace_refreshes(mem):
    mem.retain("---\nperson: Alice\n---\nbody", source="x.md",
               metadata={"person": "Carol", "created_on": "2020-01-02"})
    assert rows(mem.database_url)["x.md"][:2] == [dt.date(2020, 1, 2), "Carol"]
    mem.retain("---\nperson: Dave\n---\nbody changed", source="x.md")
    assert rows(mem.database_url)["x.md"][1] == "Dave"


def test_index_directory_fills_filter_columns(mem, tmp_path):
    (tmp_path / "n.md").write_text("---\nperson: Alice\ncreated: 2023-07-01\n---\n\nIndexed body text here.\n")
    mem.index_directory(tmp_path)
    r = rows(mem.database_url)
    key = next(k for k in r if k.endswith("n.md"))
    assert r[key][:2] == [dt.date(2023, 7, 1), "Alice"]


def test_0005_backfill_and_indexes_are_idempotent(fresh_db):
    with psycopg.connect(fresh_db, autocommit=True) as conn:
        conn.execute("INSERT INTO banks (bank_id, embedding_dim) VALUES ('b', 32)")
        for i, meta in enumerate([
            {"person": "Zed", "created": "2022-02-03"}, {"created": "2022-13-45"},
            {"other": 1}, {"date": "2021-05-06T10:00:00", "type": "log"}]):
            conn.execute(
                "INSERT INTO documents (bank_id, source, original_text, content_hash, document_metadata) "
                "VALUES ('b', %s, 't', %s, %s::jsonb)", (f"s{i}", f"h{i}", json.dumps(meta)))
    out = finish_0005(fresh_db, batch_size=1)
    assert out["backfilled"] == 2
    assert finish_0005(fresh_db)["backfilled"] == 0
    r = rows(fresh_db)
    assert r["s0"] == [dt.date(2022, 2, 3), "Zed", None]
    assert r["s1"] == [None, None, None] and r["s2"] == [None, None, None]
    assert r["s3"] == [dt.date(2021, 5, 6), None, "log"]
    with psycopg.connect(fresh_db) as conn:
        idx = {n for (n,) in conn.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename='documents'")}
        assert {"documents_bank_created_on_idx", "documents_bank_person_idx"} <= idx
    assert run_migrations(fresh_db)["applied"] == []


def _sources(res):
    return [r.source for r in res]


def test_the_note_date_filters_stage_1_and_an_empty_window_excludes_nothing_else(mem):
    """The date window is a Stage 1 filter over documents.created_on, inclusive; the person and kind columns
    are not search filters any more (nothing reads them but the citation fields)."""
    in_march = mem.search("dog", limit=10, date_from="2024-03-01", date_to="2024-03-31")
    assert sorted(_sources(in_march)) == ["a1.md", "b1.md"]
    assert {r.created_on for r in in_march} == {"2024-03-05", "2024-03-20"}
    assert next(r for r in in_march if r.source == "a1.md").kind == "journal"
    assert "n1.md" not in _sources(in_march)            # an undated note is stamped today, outside the window
    assert _sources(mem.search("dog", limit=10, date_from="2030-01-01")) == []
