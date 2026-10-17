"""Mode 1 of docs/postgres-hybrid.md, the merge: near-copies collapse BY MEANING too.

One 'session' search of a live recall returned the same bug note in four or five wordings: the
wordings differ, so the token-Jaccard collapse (0.88) keeps them all. After it the merge now compares
the records' chunk EMBEDDINGS and collapses a record whose cosine similarity to an already kept (better
scored) record is at or above a threshold. Fakes with synthetic vectors; invented text."""
from __future__ import annotations

import dataclasses
import hashlib
import math

import pytest

from prospecta._expand import (
    NEAR_COPY_COSINE, merge_records, resolve_near_copy_cosine,
)
from prospecta._types import RecalledMemory
from prospecta.cited import format_cited_rag_block
from prospecta.stages import deduplicate_by_meaning, validate_recall_config
from tests._stub_embedder import EMBED_DIM
from tests.test_expand import FakeRerank, FakeWriter, S, build, sources


def rec(doc, text, score, item="auto", source=None):
    return RecalledMemory(content=text, original_chunk=text, source=source or doc + ".md", score=0.01, scores={},
                          metadata={}, bank_id="b", document_id=doc, relevance_score=score,
                          item_id=f"i-{doc}" if item == "auto" else item)


def unit(*c):
    n = math.sqrt(sum(x * x for x in c))
    return [x / n for x in c]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (math.hypot(*a) * math.hypot(*b))


def near(i, dim=6):
    """The i-th of several vectors close to one meaning (cosine 0.99 to each other): axis 0, and a small bump
    on its own other axis."""
    v = [0.0] * dim
    v[0], v[1 + i] = 1.0, 0.1
    return unit(*v)


# five notes about the same thing in five wordings (different words: the Jaccard collapse keeps all five)
WORDINGS = ["the header of the context block was hardcoded in the template file",
            "a template file hardcodes the context block header, which should come from settings",
            "found it: context header hardcoded inside template, not read from config",
            "bug report: hardcoded header in the context template instead of the configured one",
            "the session context header sits hardcoded in a template; make it configurable"]


def five(scores=(0.5, 0.9, 0.6, 0.7, 0.4), found=((1,), (2,), (1, 3), (2,), (3,))):
    """The five wordings as five records, found by searches 1, 2, 1+3, 2 and 3; best scored: d1."""
    recs = [rec(f"d{i}", WORDINGS[i], scores[i]) for i in range(5)]
    return {n: [r for r, f in zip(recs, found) if n in f] for n in (1, 2, 3)}, recs


SAME = {f"i-d{i}": near(i) for i in range(5)}           # one meaning, five wordings


def test_the_five_wordings_share_a_meaning_and_not_their_words():
    assert min(cos(SAME[f"i-d{i}"], SAME[f"i-d{j}"]) for i in range(5) for j in range(5) if i != j) > 0.95
    assert len(merge_records(five()[0])) == 5                         # no vectors: the Jaccard collapse keeps all five


def test_five_same_meaning_notes_in_different_wordings_collapse_to_the_best_one():
    per, recs = five()
    merged = merge_records(per, vectors=SAME, near_copy_cosine=0.9)
    assert [(m.document_id, m.relevance_score) for m in merged] == [("d1", 0.9)]
    assert merged[0].original_chunk == WORDINGS[1]                     # the best-scored record is the one kept
    assert merged[0].found_by == (1, 2, 3)                             # every search that found any of the five
    assert set(merged[0].collapsed_ids) == {"d0", "d2", "d3", "d4"} and len(merged[0].collapsed_ids) == 4
    assert merged[0].collapsed_sources == tuple(f"d{i}.md" for i in (3, 2, 0, 4))      # best scored first, each once


def test_found_by_of_the_collapsed_records_joins_the_kept_one():
    a, b, c = rec("a", "alpha one wording", 0.9), rec("b", "beta other wording here", 0.8), rec("c", "gamma last", 0.7)
    vecs = {"i-a": unit(1, 0, 0), "i-b": unit(1, 0.05, 0), "i-c": unit(1, 0.02, 0.02)}
    [m] = merge_records({1: [a], 4: [b], 7: [c, b]}, vectors=vecs, near_copy_cosine=0.9)
    assert m.document_id == "a" and m.found_by == (1, 4, 7) and set(m.collapsed_ids) == {"b", "c"}


def test_distinct_notes_with_moderate_similarity_are_kept():
    vecs = {"i-a": unit(1, 0, 0, 0),
            "i-b": unit(1, 1, 0, 0),                                   # cosine 0.71 to a: related, not the same
            "i-d": unit(0.8, -0.6, 0, 0)}                              # cosine 0.80 to a, 0.14 to b
    assert 0.70 < cos(vecs["i-a"], vecs["i-b"]) < 0.72 and 0.79 < cos(vecs["i-a"], vecs["i-d"]) < 0.81
    per = {1: [rec("a", "kappa lambda", 0.9), rec("b", "mu nu xi", 0.8), rec("d", "omicron pi rho", 0.7)]}
    merged = merge_records(per, vectors=vecs)                          # the default threshold
    assert [m.document_id for m in merged] == ["a", "b", "d"] and all(not m.collapsed_ids for m in merged)
    low = merge_records(per, vectors=vecs, near_copy_cosine=0.7)       # a lower threshold takes b and d into a
    assert [m.document_id for m in low] == ["a"] and set(low[0].collapsed_ids) == {"b", "d"}


def test_the_threshold_is_configurable_and_at_or_above_collapses():
    vecs = {"i-a": [1.0, 0.0, 0.0], "i-b": [3.0, 4.0, 0.0]}            # cosine exactly 0.6
    per = {1: [rec("a", "kappa lambda", 0.9), rec("b", "mu nu xi", 0.8)]}
    assert [m.document_id for m in merge_records(per, vectors=vecs, near_copy_cosine=0.6)] == ["a"]      # at: collapses
    assert [m.document_id for m in merge_records(per, vectors=vecs, near_copy_cosine=0.61)] == ["a", "b"]
    assert [m.document_id for m in merge_records(per, vectors=vecs, near_copy_cosine=0.2)] == ["a"]


def test_a_record_is_compared_with_the_kept_ones_only_so_a_chain_does_not_collapse():
    a = [1.0, 0.0]
    b = [math.cos(math.radians(30)), math.sin(math.radians(30))]       # a~b 0.87, b~c 0.87, a~c 0.50
    c = [math.cos(math.radians(60)), math.sin(math.radians(60))]
    assert cos(a, b) > 0.8 and cos(b, c) > 0.8 and cos(a, c) < 0.8
    per = {1: [rec("a", "kappa lambda", 0.9), rec("b", "mu nu xi", 0.8), rec("c", "omicron pi", 0.7)]}
    merged = merge_records(per, vectors={"i-a": a, "i-b": b, "i-c": c}, near_copy_cosine=0.8)
    assert [m.document_id for m in merged] == ["a", "c"] and merged[0].collapsed_ids == ("b",)


def test_records_without_an_embedding_are_kept_as_they_are():
    a, b = rec("a", "kappa lambda", 0.9), rec("b", "mu nu xi", 0.8)
    no_id, missing, zero, short, nan = (rec("e", "eta theta iota", 0.7, item=None), rec("f", "pi rho sigma", 0.6),
                                        rec("g", "tau upsilon phi", 0.5), rec("h", "chi psi omega", 0.4),
                                        rec("i", "alpha beta gamma", 0.3))
    vecs = {"i-a": [1.0, 0.0], "i-b": [1.0, 0.0], "i-g": [0.0, 0.0], "i-h": [1.0], "i-i": [float("nan"), 1.0]}
    merged = merge_records({1: [a, b, no_id, missing, zero, short, nan]}, vectors=vecs, near_copy_cosine=0.9)
    assert [m.document_id for m in merged] == ["a", "e", "f", "g", "h", "i"]    # only b, a copy of a, collapsed
    assert all(not m.collapsed_ids for m in merged[1:]) and merged[0].collapsed_ids == ("b",)


def test_no_vectors_at_all_is_the_old_merge():
    per = {1: [rec("a", "kappa lambda", 0.9), rec("b", "mu nu xi", 0.8)]}
    assert [m.document_id for m in merge_records(per, vectors={})] == ["a", "b"]
    assert [m.document_id for m in merge_records(per, vectors=None)] == ["a", "b"]
    assert merge_records({}, vectors=SAME) == []


def test_the_lexical_collapse_comes_first_and_both_kinds_of_copy_are_listed():
    wording = "one two three four five six seven eight nine ten"
    a, lex, mean = rec("a", wording, 0.9), rec("l", wording + " eleven", 0.8), rec("m", "other words entirely", 0.7)
    vecs = {"i-a": [1.0, 0.0], "i-m": [1.0, 0.01], "i-l": [0.0, 1.0]}  # the lexical copy is never compared
    asked = []

    def loader(ids):
        asked.append(list(ids))
        return vecs
    [m] = merge_records({1: [a, mean], 2: [lex]}, vectors=loader, near_copy_cosine=0.9)
    assert asked == [["i-a", "i-m"]]                                   # only the survivors of the Jaccard collapse
    assert set(m.collapsed_ids) == {"l", "m"} and m.found_by == (1, 2)


def test_the_loader_is_not_asked_for_fewer_than_two_records():
    asked = []
    merge_records({1: [rec("a", "kappa lambda", 0.9)]}, vectors=lambda ids: asked.append(ids) or {})
    assert asked == []


def test_a_numpy_free_environment_gives_the_same_answer(monkeypatch):
    import prospecta.stages as stages
    monkeypatch.setattr(stages, "_numpy", lambda: None)
    per, _ = five()
    assert [m.document_id for m in merge_records(per, vectors=SAME, near_copy_cosine=0.9)] == ["d1"]


def test_deduplicate_by_meaning_unit():
    vecs = [[1.0, 0.0], [0.99, 0.1], None, [0.0, 1.0], [1.0, 0.0]]
    survivors, collapsed = deduplicate_by_meaning(vecs, [4, 1, 0, 2, 3], 0.9)
    assert survivors == [4, 2, 3] and collapsed == {1: 4, 0: 4}


def test_a_collapsed_record_shows_in_the_block_header():
    a = dataclasses.replace(rec("a", "kappa lambda", 0.9), collapsed_ids=("b", "c"),
                            collapsed_sources=("b.md", "c.md"), found_by=(1, 2))
    block = format_cited_rag_block([a, rec("z", "mu nu", 0.5)], heading=None)
    head = block.splitlines()[0]
    assert "found_by=1,2" in head and "similar=2" in head and block.count("similar=") == 1


# --------------------------------------------------------------------------
# settings: call argument, bank recall_config, environment, default
# --------------------------------------------------------------------------

def test_the_threshold_is_per_call_then_bank_then_environment_then_default(monkeypatch):
    monkeypatch.delenv("PROSPECTA_NEAR_COPY_COSINE", raising=False)
    assert resolve_near_copy_cosine() == NEAR_COPY_COSINE
    monkeypatch.setenv("PROSPECTA_NEAR_COPY_COSINE", "0.91")
    assert resolve_near_copy_cosine() == 0.91
    assert resolve_near_copy_cosine(None, 0.8) == 0.8                  # the bank beats the environment
    assert resolve_near_copy_cosine(0.7, 0.8) == 0.7                   # the call beats the bank
    monkeypatch.setenv("PROSPECTA_NEAR_COPY_COSINE", "garbage")
    assert resolve_near_copy_cosine() == NEAR_COPY_COSINE              # a bad environment value is ignored, not fatal
    for bad in (0, 1.5, -0.2, True, "0.9", float("nan")):
        with pytest.raises(ValueError):
            resolve_near_copy_cosine(bad)
        with pytest.raises(ValueError):
            resolve_near_copy_cosine(None, bad)


def test_recall_config_checks_the_threshold():
    validate_recall_config({"expansion": {"near_copy_cosine": 0.9}})
    for bad in (0, 1.1, "x", True):
        with pytest.raises(ValueError, match="near_copy_cosine"):
            validate_recall_config({"expansion": {"near_copy_cosine": bad}})


# --------------------------------------------------------------------------
# through the entry point, on a real bank with synthetic embeddings
# --------------------------------------------------------------------------

TOPICS = ("alpha", "beta", "gamma")


def topic_embed(texts):
    """One axis per topic word, plus a small hash bump per text: notes of one topic are about 0.99 alike,
    a note of two topics is 0.71 from either."""
    out = []
    for t in texts:
        v = [0.0] * EMBED_DIM
        for i, topic in enumerate(TOPICS):
            if topic in t.lower():
                v[i] += 1.0
        if not any(v):
            v[3] = 1.0
        h = hashlib.sha256(t.encode()).digest()
        for k in range(8, EMBED_DIM):
            v[k] += (h[k] - 128) / 128 * 0.04
        n = math.sqrt(sum(x * x for x in v))
        out.append([x / n for x in v])
    return out


LEDGER = {
    **{f"ledger-{i}.md": (text, {"project": "atlas", "created_on": f"2024-05-0{i + 1}"})
       for i, text in enumerate([
           "alpha ledger entry: the header block of the context template was hardcoded in the file",
           "alpha ledger: a template file hardcodes the header of the context block instead of the setting",
           "ledger alpha, found it: header hardcoded in the context template, not read from config",
           "alpha ledger bug: configured header ignored, the context template has it written in",
           "the alpha ledger says the context header is hardcoded in a template; make it configurable"])},
    "beta-ledger.md": ("beta ledger: the context block is cached for ten minutes per thread", {"project": "atlas"}),
    "mixed-ledger.md": ("alpha beta ledger: header text and cache time are both settings", {"project": "atlas"}),
    "gamma-ledger.md": ("gamma ledger: the footer of the block lists the searches", {"project": "atlas"}),
}


def ledger_scores(q, d):
    return 0.9 if "ledger-1" in d or "a template file hardcodes" in d else 0.6


@pytest.fixture
def made(fresh_db):
    mems = []

    def make(**kw):
        m = build(fresh_db, embed=topic_embed, **kw)
        mems.append(m)
        return m
    yield make
    for m in mems:
        m.close()


def make_mem(made, **kw):
    mem = made(llm=FakeWriter([S("alpha ledger header hardcoded context template"),
                               S("ledger context block settings")]),
               rerank=FakeRerank(score=ledger_scores), notes=LEDGER, rerank_cfg={"deduplicate_near_copies": False},
               **kw)
    return mem


QUESTION = "what do the ledger notes say about the context header?"


def test_five_wordings_of_one_note_come_back_once_through_the_entry_point(made):
    mem = make_mem(made)
    got = mem.expand_and_recall(QUESTION, gate=False)
    five_ = [s for s in sources(got) if s.startswith("ledger-")]
    assert len(five_) == 1 and got.error is None
    kept = next(i for i in got.items if i.source.startswith("ledger-"))
    assert kept.relevance_score == pytest.approx(0.9) and len(kept.collapsed_ids) == 4
    assert set(kept.found_by) == {1, 2} and len(set(kept.collapsed_sources)) == 4
    assert {"beta-ledger.md", "mixed-ledger.md"} <= set(sources(got))   # the distinct notes stay
    assert "similar=4" in got.block and sum(1 for line in got.block.splitlines() if line.startswith("[")) == len(got.items)


def test_the_call_argument_turns_the_meaning_collapse_off_or_on(made):
    mem = make_mem(made)
    off = mem.expand_and_recall(QUESTION, gate=False, near_copy_cosine=1.0)
    assert len([s for s in sources(off) if s.startswith("ledger-")]) == 5
    on = mem.expand_and_recall(QUESTION, gate=False, near_copy_cosine=0.6)       # 0.71: the mixed note joins too
    assert len([s for s in sources(on) if s.startswith("ledger-") or s == "mixed-ledger.md"]) == 1


def test_the_bank_setting_applies_and_the_call_argument_beats_it(made, monkeypatch):
    mem = make_mem(made)
    mem.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0,
                                      "deduplicate_near_copies": False},
                           "expansion": {"near_copy_cosine": 1.0}})
    assert len([s for s in sources(mem.expand_and_recall(QUESTION, gate=False)) if s.startswith("ledger-")]) == 5
    assert len([s for s in sources(mem.expand_and_recall(QUESTION, gate=False, near_copy_cosine=0.9))
                if s.startswith("ledger-")]) == 1
    monkeypatch.setenv("PROSPECTA_NEAR_COPY_COSINE", "0.2")                      # the bank beats the environment
    assert len([s for s in sources(mem.expand_and_recall(QUESTION, gate=False)) if s.startswith("ledger-")]) == 5


def test_the_environment_applies_when_neither_call_nor_bank_say(made, monkeypatch):
    mem = make_mem(made)
    monkeypatch.setenv("PROSPECTA_NEAR_COPY_COSINE", "1.0")
    assert len([s for s in sources(mem.expand_and_recall(QUESTION, gate=False)) if s.startswith("ledger-")]) == 5


def test_an_invalid_threshold_is_an_error_result_not_a_crash(made):
    got = make_mem(made).expand_and_recall(QUESTION, gate=False, near_copy_cosine=2)
    assert got.error and "near_copy_cosine" in got.error and got.block == ""


def test_when_the_vectors_cannot_be_read_the_wording_collapse_stands_and_the_run_says_so(made, monkeypatch):
    import prospecta._expand_run as run
    mem = make_mem(made)

    def boom(*a, **k):
        raise RuntimeError("no vectors today")
    monkeypatch.setattr(run, "load_chunk_vectors", boom)
    got = mem.expand_and_recall(QUESTION, gate=False)
    assert got.error is None and len([s for s in sources(got) if s.startswith("ledger-")]) == 5
    assert any("meaning" in f and "no vectors today" in f for f in got.fallbacks)


def test_some_missing_vectors_keep_those_records_as_they_are(made, monkeypatch):
    import prospecta._expand_run as run
    mem = make_mem(made)
    real = run.load_chunk_vectors

    def partial(conn, bank_id, ids):
        out = real(conn, bank_id, ids)
        for k in sorted(out)[:2]:
            del out[k]                                                           # two chunks have no vector
        return out
    monkeypatch.setattr(run, "load_chunk_vectors", partial)
    got = mem.expand_and_recall(QUESTION, gate=False)
    assert got.error is None and 1 <= len([s for s in sources(got) if s.startswith("ledger-")]) <= 3


def test_the_vectors_cost_one_primary_key_read_of_the_survivors_only(made, monkeypatch):
    import prospecta._expand_run as run
    mem = make_mem(made)
    real, asked = run.load_chunk_vectors, []

    def spy(conn, bank_id, ids):
        asked.append(list(ids))
        return real(conn, bank_id, ids)
    monkeypatch.setattr(run, "load_chunk_vectors", spy)
    got = mem.expand_and_recall(QUESTION, gate=False)
    assert len(asked) == 1 and len(asked[0]) == len(set(asked[0])) <= len(LEDGER)
    assert got.timings_ms.get("vectors") is not None
