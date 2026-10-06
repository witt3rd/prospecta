"""Migration 0005, filter fields at retain/index, extraction (stub LLM and
regex fallback), the MetadataScope channel and scope promotion."""
from __future__ import annotations

import datetime as dt
import json

import psycopg
import pytest

from prospecta.channels import (
    DEFAULT_CHANNEL_CONFIG, Candidate, Filters, QueryPlan, extract_filters,
    promote_scope, regex_filters, scope_members,
)
from prospecta.channels.fusion import FusedDoc
from prospecta.db.migrate import finish_0005, run_migrations
from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM, stub_embed

NOW = dt.datetime(2025, 6, 15, tzinfo=dt.timezone.utc)
VOCAB = ["Alice", "Bob Stone"]


def llm_returning(payload):
    calls = []

    def llm(messages, *, json_mode=False):
        calls.append(messages[0]["content"])
        return payload if isinstance(payload, str) else json.dumps(payload)
    llm.calls = calls
    return llm


# ------------------------------------------------------------ extraction

def test_llm_extraction_validates_against_vocab_and_dates():
    llm = llm_returning({"people": ["alice", "Mallory"], "date_from": "2024-03-01",
                         "date_to": "2024-03-31", "hard": True})
    f = extract_filters("what did Alice do in March 2024", llm=llm, people_vocab=VOCAB, now=NOW)
    assert f == Filters(people=["Alice"], date_from="2024-03-01", date_to="2024-03-31", hard=True)
    assert "Alice, Bob Stone" in llm.calls[0] and "2025-06-15" in llm.calls[0]


def test_llm_hard_without_any_filter_is_not_hard():
    llm = llm_returning({"people": [], "date_from": None, "date_to": None, "hard": True})
    f = extract_filters("hello", llm=llm, people_vocab=VOCAB, now=NOW)
    assert f == Filters()


@pytest.mark.parametrize("bad", ["not json", "[1,2]", '{"people": 5}'])
def test_bad_llm_output_falls_back_to_regex(bad):
    f = extract_filters("Bob Stone in March 2024?", llm=llm_returning(bad),
                        people_vocab=VOCAB, now=NOW)
    assert f.people == ["Bob Stone"] and f.date_from == "2024-03-01" and f.hard is False


def test_llm_exception_and_no_llm_fall_back_to_regex():
    def boom(messages, *, json_mode=False):
        raise RuntimeError("down")
    for llm in (boom, None):
        f = extract_filters("notes from 2023", llm=llm, people_vocab=VOCAB, now=NOW)
        assert (f.date_from, f.date_to, f.hard) == ("2023-01-01", "2023-12-31", False)


def test_regex_baseline_cases():
    assert regex_filters("on 2024-02-29 we met", [], NOW).date_from == "2024-02-29"
    assert regex_filters("in february", [], NOW).date_to == "2025-02-28"
    assert regex_filters("you may go", [], NOW).date_from is None  # bare 'may' is a verb
    assert regex_filters("may 2024 trip", [], NOW).date_to == "2024-05-31"
    assert regex_filters("Alicedale", VOCAB, NOW).people == []  # word boundary
    assert regex_filters("what did ALICE say", VOCAB, NOW).people == ["Alice"]
    assert regex_filters("nothing here", VOCAB, NOW) == Filters()


# ------------------------------------------------------- promotion (pure)

def C(doc, rank, channel="meta"):
    return Candidate(document_id=doc, item_id=None, source=doc, channel=channel,
                     rank=rank, score=0.5, evidence=None)


def F(doc, score):
    return FusedDoc(document_id=doc, source=doc, score=score, best=C(doc, 1, "dense_chunk"))


def test_promotion_moves_members_to_front_and_never_excludes():
    fused = [F("a", 3), F("b", 2), F("c", 1)]
    members = [C("c", 1), C("z", 2)]
    out = promote_scope(fused, members)
    assert [f.document_id for f in out] == ["c", "z", "a", "b"]  # z added, none removed


def test_scope_members_gates():
    hard = QueryPlan(text="q", filters=Filters(people=["A"], hard=True))
    soft = QueryPlan(text="q", filters=Filters(people=["A"], hard=False))
    few = [C(f"d{i}", i + 1) for i in range(12)]
    assert scope_members(hard, few, 50) == few
    assert scope_members(soft, few, 50) == []
    assert scope_members(hard, [C(f"d{i}", i + 1) for i in range(13)], 50) == []
    assert scope_members(hard, few, 12) == []  # set may be truncated by the limit
    assert scope_members(hard, [], 50) == []


# ------------------------------------------------------- db integration

@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=llm_returning("{}"), embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    notes = [
        ("a1.md", "---\nperson: Alice\ncreated: 2024-03-05\ntype: journal\n---\nAlice walked the dog by the river.",
         "walk river"),
        ("a2.md", "---\nperson: Alice\ncreated: 2024-04-09\n---\nAlice repaired the bicycle chain.",
         "bicycle repair"),
        ("b1.md", "---\nperson: Bob Stone\ncreated: 2024-03-20\n---\nBob walked the dog near the park.",
         "bob dog walk"),
        ("n1.md", "No frontmatter, a note about walking a dog in the park.", "dog park"),
    ]
    for src, text, it in notes:
        m.retain(text, source=src, index_text=it)
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
    assert r["n1.md"] == [None, None, None]


def test_retain_metadata_kwarg_wins_and_replace_refreshes(mem):
    mem.retain("---\nperson: Alice\n---\nbody", source="x.md", index_text="x",
               metadata={"person": "Carol", "created_on": "2020-01-02"})
    assert rows(mem.database_url)["x.md"][:2] == [dt.date(2020, 1, 2), "Carol"]
    mem.retain("---\nperson: Dave\n---\nbody changed", source="x.md", index_text="x")
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


def test_meta_channel_ranks_filter_set_and_soft_filter_adds_no_promotion(mem):
    mem.set_channel_config(DEFAULT_CHANNEL_CONFIG)
    mem._llm = llm_returning({"people": ["Alice"], "date_from": None, "date_to": None, "hard": False})
    traces: list = []
    res = mem.search("walk the dog", limit=10, _trace=traces)
    meta = [c for c in traces[0]["candidates"] if c["channel"] == "meta"]
    assert {c["document_id"] for c in meta} == {
        r.document_id for r in res if r.source in ("a1.md", "a2.md")}
    assert "scope_promoted" not in traces[0]["fusion"]
    assert traces[0]["plan"]["filters"]["people"] == ["Alice"]
    assert traces[0]["fusion"]["weights"]["meta"] == 3.0
    assert len(res) == 4  # nothing excluded


def test_hard_scope_promotes_whole_set_even_beyond_limit(mem):
    mem.set_channel_config(DEFAULT_CHANNEL_CONFIG)
    mem._llm = llm_returning({"people": ["Alice"], "date_from": "2024-03-01",
                              "date_to": "2024-04-30", "hard": True})
    traces: list = []
    res = mem.search("walk the dog by the park", limit=1, _trace=traces)
    assert set(_sources(res)) == {"a1.md", "a2.md"}  # all members despite limit=1
    assert len(traces[0]["fusion"]["scope_promoted"]) == 2


def test_date_filter_and_empty_scope_never_excludes(mem):
    mem.set_channel_config(DEFAULT_CHANNEL_CONFIG)
    mem._llm = llm_returning({"people": [], "date_from": "2024-03-10",
                              "date_to": "2024-03-31", "hard": True})
    res = mem.search("walk the dog", limit=10)
    assert _sources(res)[0] == "b1.md" and len(res) == 4
    # a filter matching nothing leaves the ordinary result intact
    mem._llm = llm_returning({"people": [], "date_from": "1999-01-01",
                              "date_to": "1999-12-31", "hard": True})
    assert len(mem.search("walk the dog", limit=10)) == 4


def test_no_llm_failure_means_regex_and_recall_still_works(mem):
    mem.set_channel_config(DEFAULT_CHANNEL_CONFIG)
    mem._llm = llm_returning("garbage")
    res = mem.search("what did Bob Stone do", limit=10)
    assert len(res) == 4
