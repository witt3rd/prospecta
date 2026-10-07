"""Migration 0005, filter fields at retain/index, extraction (stub LLM and
regex fallback), the MetadataScope channel and scope promotion."""
from __future__ import annotations

from prospecta._test_helpers import no_cut

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


GOOD = {"people": ["Alice"], "date_from": "2024-03-01", "date_to": "2024-03-31", "hard": True}
WANT = Filters(people=["Alice"], date_from="2024-03-01", date_to="2024-03-31", hard=True)


@pytest.mark.parametrize("wrap", [
    "```json\n{j}\n```",
    "```\n{j}\n```",
    "Here is the extraction:\n{j}",
    "Sure! Here you go:\n```json\n{j}\n```\nLet me know if you need more.",
    "{j}\n\nNote: hard is true because the question names a month.",
])
def test_llm_extraction_tolerates_prose_and_fences(wrap):
    llm = llm_returning(wrap.format(j=json.dumps(GOOD)))
    assert extract_filters("what did Alice do in March 2024", llm=llm,
                           people_vocab=VOCAB, now=NOW) == WANT


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


def test_promotion_boosts_members_and_never_excludes():
    fused = [F("a", 3), F("b", 2), F("c", 1)]
    members = [C("c", 1), C("z", 2)]   # cosine 0.5 each
    out = promote_scope(fused, members, weight=4.0)
    ids = [f.document_id for f in out]
    assert set(ids) == {"a", "b", "c", "z"}          # z added, none removed
    assert ids[:2] == ["c", "z"] or ids[:2] == ["z", "c"]  # boost lifts them past a, b


def test_promotion_weight_decides_not_count():
    fused = [F("a", 3), F("b", 2)]
    assert [f.document_id for f in promote_scope(fused, [C("b", 1)], weight=0.1)] == ["a", "b"]
    assert [f.document_id for f in promote_scope(fused, [C("b", 1)], weight=2.0)] == ["b", "a"]
    assert promote_scope(fused, [C("b", 1)], weight=0) == fused


def test_scope_members_gates_and_has_no_size_limit():
    hard = QueryPlan(text="q", filters=Filters(people=["A"], hard=True))
    soft = QueryPlan(text="q", filters=Filters(people=["A"], hard=False))
    many = [C(f"d{i}", i + 1) for i in range(200)]
    assert scope_members(hard, many) == many
    assert scope_members(soft, many) == []
    assert scope_members(hard, []) == []


def test_q096_style_large_filter_set_cover_at_10():
    """Large filter set (40 notes, 3 gold, all gold in the set but buried in the
    fused pool): the old 12-note cap did nothing; the boost lifts gold."""
    gold = {"g1", "g2", "g3"}
    outside = [F(f"x{i}", 1.0 - i * 0.005) for i in range(60)]
    inset = [F(f"m{i}", 0.30 - i * 0.001) for i in range(37)] + [F(g, 0.2) for g in sorted(gold)]
    fused = sorted(outside + inset, key=lambda f: -f.score)
    def cos(d):  # gold are the closest members to the query
        return 0.9 if d in gold else 0.3
    members = [Candidate(document_id=f.document_id, item_id=None, source=f.document_id,
                         channel="meta", rank=i + 1, score=cos(f.document_id), evidence=None)
               for i, f in enumerate(inset)]
    def cover(order):
        return len(gold & {f.document_id for f in order[:10]}) / len(gold)
    old_cap = fused if len(members) > 12 else promote_scope(fused, members)  # PROMOTE_MAX=12
    new = promote_scope(fused, members, weight=1.0)
    print(f"cover@10 old cap={cover(old_cap):.2f} boost={cover(new):.2f}")
    assert cover(old_cap) == 0.0 and cover(new) == 1.0


def test_q096_style_weak_cosine_members():
    """Members with weak cosine still rise above non-members of equal fused score
    only in proportion to cosine; a cosine-0 member gets no boost (the boost
    multiplies a real score)."""
    gold = {"g1", "g2", "g3"}
    outside = [F(f"x{i}", 1.0 - i * 0.005) for i in range(60)]
    inset = [F(f"m{i}", 0.30 - i * 0.001) for i in range(37)] + [F(g, 0.2) for g in sorted(gold)]
    fused = sorted(outside + inset, key=lambda f: -f.score)
    def mk(cos):
        return [Candidate(document_id=f.document_id, item_id=None, source=f.document_id,
                          channel="meta", rank=i + 1, score=cos(f.document_id), evidence=None)
                for i, f in enumerate(inset)]
    def cover(order):
        return len(gold & {f.document_id for f in order[:10]}) / len(gold)
    weak = promote_scope(fused, mk(lambda d: 0.85 if d in gold else 0.05), weight=1.0)
    zero = promote_scope(fused, mk(lambda d: 0.0), weight=1.0)
    print(f"cover@10 weak-cosine boost={cover(weak):.2f} zero-cosine={cover(zero):.2f}")
    assert cover(weak) == 1.0 and cover(zero) == 0.0


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
    mem.set_channel_config(no_cut(DEFAULT_CHANNEL_CONFIG))
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


def _promote_cfg(on):
    cfg = no_cut(DEFAULT_CHANNEL_CONFIG)
    for e in cfg:
        if e["name"] == "meta":
            e["params"] = {**(e.get("params") or {}), "promote": on}
    return cfg


def test_scope_promotion_is_off_by_default(mem):
    mem.set_channel_config(no_cut(DEFAULT_CHANNEL_CONFIG))
    mem._llm = llm_returning({"people": ["Alice"], "date_from": "2024-03-01",
                              "date_to": "2024-04-30", "hard": True})
    traces: list = []
    mem.search("walk the dog by the park", limit=2, _trace=traces)
    assert "scope_promoted" not in traces[0]["fusion"]


def test_hard_scope_promotes_set_members(mem):
    mem.set_channel_config(_promote_cfg(True))
    mem._llm = llm_returning({"people": ["Alice"], "date_from": "2024-03-01",
                              "date_to": "2024-04-30", "hard": True})
    traces: list = []
    res = mem.search("walk the dog by the park", limit=2, _trace=traces)
    assert set(_sources(res)) == {"a1.md", "a2.md"}  # boosted members lead
    assert len(traces[0]["fusion"]["scope_promoted"]) == 2


def test_date_filter_and_empty_scope_never_excludes(mem):
    mem.set_channel_config(no_cut(DEFAULT_CHANNEL_CONFIG))
    mem._llm = llm_returning({"people": [], "date_from": "2024-03-10",
                              "date_to": "2024-03-31", "hard": True})
    res = mem.search("walk the dog", limit=10)
    assert _sources(res)[0] == "b1.md" and len(res) == 4
    # a filter matching nothing leaves the ordinary result intact
    mem._llm = llm_returning({"people": [], "date_from": "1999-01-01",
                              "date_to": "1999-12-31", "hard": True})
    assert len(mem.search("walk the dog", limit=10)) == 4


def test_no_llm_failure_means_regex_and_recall_still_works(mem):
    mem.set_channel_config(no_cut(DEFAULT_CHANNEL_CONFIG))
    mem._llm = llm_returning("garbage")
    res = mem.search("what did Bob Stone do", limit=10)
    assert len(res) == 4


# ------------------------------------------------------ extraction model

def _recording_llm(seen):
    def llm(messages, *, json_mode=False, model=None):
        seen.append(model)
        return "{}"
    return llm


def test_default_extract_model_reaches_llm(mem):
    from prospecta.channels.extract import DEFAULT_EXTRACT_MODEL
    seen: list = []
    mem.set_channel_config(no_cut(DEFAULT_CHANNEL_CONFIG))
    mem._llm = _recording_llm(seen)
    mem.search("walk the dog", limit=5)
    assert seen == [DEFAULT_EXTRACT_MODEL] == ["anthropic/claude-sonnet-5.5"]


def test_extract_model_param_override_reaches_llm(mem):
    seen: list = []
    cfg = [dict(e) for e in no_cut(DEFAULT_CHANNEL_CONFIG)]
    for e in cfg:
        if e["name"] == "meta":
            e["params"] = {**e["params"], "extract_model": "x/small"}
    mem.set_channel_config(cfg)
    mem._llm = _recording_llm(seen)
    mem.search("walk the dog", limit=5)
    assert seen == ["x/small"]


def test_kwargs_llm_receives_model_and_plain_llm_still_works():
    seen: dict = {}

    def kw_llm(messages, **kwargs):
        seen.update(kwargs)
        return "{}"
    extract_filters("hi", llm=kw_llm, people_vocab=[], now=NOW, model="m1")
    assert seen == {"json_mode": True, "model": "m1"}
    f = extract_filters("Alice", llm=llm_returning({"people": ["Alice"]}),
                        people_vocab=VOCAB, now=NOW, model="m1")
    assert f.people == ["Alice"]
