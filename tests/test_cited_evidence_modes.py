"""Hybrid plan slice 4 (library half): the cited-evidence contract and the three
modes that use it, end to end with fakes for the models.

- Mode 2A: Memory.search / POST /lookup hits carry source, created_on, kind, match_kind,
  relevance_score, exact_snippet and original_chunk; Stage 1 filters and the model
  overrides are per-call.
- Mode 1, task: Memory.recall_for_task / POST /lookup {"task": ...}: a thin wrapper of
  expand_and_recall (the brief becomes N searches, each runs Stage 1 + Stage 2, the notes merge
  into one cited block); the entry point's own tests are tests/test_expand.py.
- Mode 2B: Memory.ask: a grounded answer with [n] citations and verbatim snippets, and
  the optional Stage 3 sufficiency gate.
Invented text; a stub embedder, a fake rerank transport, a fake Jev, a fake LLM."""
from __future__ import annotations

import json
import re

import psycopg
import pytest

from prospecta._tracer import RecordingTracer
from prospecta._types import RecalledMemory
from prospecta.cited import (
    build_citations, format_cited_rag_block, match_kind, render_answer, supporting_snippet,
)
from prospecta.memory import Memory
from prospecta.service import ApiError, Service
from prospecta.stages import JEV_MODEL, JevScore, RERANKER_MODEL, validate_recall_config
from tests._stub_embedder import EMBED_DIM, stub_embed

PHRASE = "The night crew called the bridge lamps electric dreams. Nobody remembers who started it."
NOTES = {
    "decision.md": ("We decided the archive stays append-only. Deletes are tombstones.",
                    {"project": "atlas", "created_on": "2024-03-05", "source_kind": "decision"}),
    "phrase.md": (PHRASE, {"project": "atlas", "created_on": "2024-04-01", "source_kind": "note"}),
    "tree.md": ("The library tree view loads folders lazily. Large folders page in.",
                {"project": "borealis", "created_on": "2023-09-09", "source_kind": "note"}),
    "orchid.md": ("Orchids need humid air and a shaded window.",
                  {"project": "atlas", "created_on": "2022-01-02"}),
}


class FakeRerank:
    """A rerank transport: the score grows with the words the document shares with the query."""

    def __init__(self):
        self.requests = []

    def __call__(self, req, timeout):
        self.requests.append(req)
        asked = set(re.findall(r"\w{3,}", req["query"].split("Query:", 1)[-1].lower()))
        results = [{"index": i, "relevance_score": min(0.99, 0.2 + 0.2 * len(
            asked & set(re.findall(r"\w{3,}", d.lower()))))} for i, d in enumerate(req["documents"])]
        return {"model": req["model"], "results": results, "usage": {"total_tokens": 10, "cost": 0.0001}}


class FakeJev:
    """A System One transport that answers the scripted evidence_sufficient scores in turn
    (the last one repeats); an Exception in the script is raised."""

    def __init__(self, *scores):
        self.scores, self.requests = list(scores), []

    def __call__(self, req, timeout):
        self.requests.append(req)
        score = self.scores.pop(0) if len(self.scores) > 1 else self.scores[0]
        if isinstance(score, Exception):
            raise score
        return {"model": req["model"], "usage": {"input_tokens": 5, "output_tokens": 1, "cost": 0.00001},
                "answers": {qid: {"type": "score", "score": score} for qid in req["questions"]}}


class FakeLLM:
    """The query writer (json_mode: the scripted `searches`) and the synthesis model (`answer`), in one
    callable that accepts the optional `model` hint."""

    def __init__(self, searches=None, answer="not in memory"):
        self.searches, self.answer, self.calls = searches, answer, []

    def __call__(self, messages, *, json_mode=False, model=None):
        self.calls.append({"json_mode": json_mode, "model": model, "prompt": messages[-1]["content"]})
        if json_mode:
            return json.dumps({"searches": self.searches})
        return self.answer


def build(fresh_db, *, llm=None, jev=None, rerank=None, notes=NOTES, **kw):
    mem = Memory(database_url=fresh_db, bank_id="b", embed=stub_embed, llm=llm,
                 jev=JevScore(jev) if jev is not None else None, rerank_transport=rerank,
                 tracer=RecordingTracer(), **kw)
    mem.create_bank("b", embedding_dim=EMBED_DIM)
    mem.set_channel_config([{"name": "hybrid_chunk", "weight": 1}])
    for source, (text, metadata) in notes.items():
        mem.retain(text, source=source, metadata=metadata)
    return mem


@pytest.fixture
def mem(fresh_db):
    rerank = FakeRerank()
    m = build(fresh_db, rerank=rerank)
    m.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0}})
    m.rerank = rerank
    yield m
    m.close()


def hit(hits, source):
    return next(h for h in hits if h.source == source)


# --------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------

def test_search_hits_carry_the_cited_evidence_contract(mem):
    hits = mem.search('"electric dreams"', limit=4)
    h = hit(hits, "phrase.md")
    assert (h.source, h.created_on, h.kind) == ("phrase.md", "2024-04-01", "note")
    assert h.match_kind == "exact+semantic"
    assert "<<electric>> <<dreams>>" in h.exact_snippet
    assert h.original_chunk == PHRASE
    request = mem.rerank.requests[-1]
    assert 0 < h.relevance_score <= 1 and request["model"] == RERANKER_MODEL
    other = hit(hits, "orchid.md")      # found by meaning only
    assert (other.match_kind, other.exact_snippet) == ("semantic", None)
    assert other.created_on == "2022-01-02" and other.kind is None


def test_relevance_is_the_cross_encoder_score_and_otherwise_the_fused_score(fresh_db):
    plain = build(fresh_db)
    try:
        h = hit(plain.search('"electric dreams"', limit=4), "phrase.md")
        assert h.relevance_score == h.score                 # no stage 2: the fused RRF score
        assert h.match_kind == "exact+semantic" and h.created_on == "2024-04-01"
    finally:
        plain.close()


def test_hits_of_a_bank_without_channels_still_say_where_and_when(fresh_db):
    legacy = build(fresh_db)
    try:
        legacy.set_channel_config([])
        h = hit(legacy.search("electric dreams", limit=4), "phrase.md")
        assert (h.created_on, h.kind) == ("2024-04-01", "note") and h.relevance_score == h.score
        assert h.match_kind is None and h.exact_snippet is None
    finally:
        legacy.close()


def test_match_kind_names_why_a_note_matched():
    assert match_kind(exact=True, lexical=True, semantic=True) == "exact+semantic"
    assert match_kind(exact=True, lexical=False, semantic=False) == "exact"
    assert match_kind(exact=False, lexical=True, semantic=True) == "lexical"
    assert match_kind(exact=False, lexical=False, semantic=True) == "semantic"
    assert match_kind(exact=False, lexical=False, semantic=False) is None


def _item(source, text, **kw):
    return RecalledMemory(content=text, original_chunk=text, source=source, score=0.1, scores={},
                          metadata=kw.pop("metadata", {}), bank_id="b", document_id=source, **kw)


def test_format_cited_rag_block():
    items = [_item("a.md", "Alpha text.", created_on="2024-03-05", kind="note",
                   match_kind="exact", exact_snippet="<<Alpha>> text", metadata={"project": "atlas"}),
             _item("b.md", "Beta text.", metadata={"project": "borealis"})]
    assert format_cited_rag_block([]) == ""
    assert format_cited_rag_block(items, heading="## Context", own_project="atlas") == (
        '## Context\n\n'
        '[1] source=a.md | date=2024-03-05 | kind=note | match=exact\n'
        '    Exact match snippet: "<<Alpha>> text"\n    Alpha text.\n\n'
        '[2] source=b.md | date=unknown | project=borealis (another project)\n    Beta text.')
    plain = format_cited_rag_block(items[:1])
    assert plain.startswith("Background from memory, gathered for this turn") and "(another project)" not in plain
    assert format_cited_rag_block(items[:1], heading=None).startswith("[1] source=a.md")


# --------------------------------------------------------------------------
# Mode 2A: Stage 1 filters and model overrides, per call
# --------------------------------------------------------------------------

def test_search_pushes_the_date_window_and_excluded_sources_into_stage_1(mem):
    q = "archive tree orchid crew"
    assert {h.source for h in mem.search(q, limit=10)} == set(NOTES)
    assert {h.source for h in mem.search(q, limit=10, date_from="2024-01-01")} == {"decision.md", "phrase.md"}
    assert {h.source for h in mem.search(q, limit=10, date_to="2023-12-31")} == {"tree.md", "orchid.md"}
    assert "phrase.md" not in {h.source for h in mem.recall([q], limit=10, excluded_sources=["phrase.md"])}


def test_stage_1_filters_need_the_hybrid_channel(fresh_db):
    other = build(fresh_db)
    try:
        other.set_channel_config([{"name": "dense_chunk", "weight": 1}])
        with pytest.raises(ValueError, match="hybrid_chunk"):
            other.search("crew", excluded_sources=["phrase.md"])
        other.set_channel_config([])
        with pytest.raises(ValueError, match="hybrid_chunk"):
            other.search("crew", date_from="2024-01-01")
        assert other.search("crew")      # no filter, no problem
    finally:
        other.close()


def test_the_reranker_model_is_per_call_then_memory_then_bank_then_environment(fresh_db, monkeypatch):
    rerank = FakeRerank()
    m = build(fresh_db, rerank=rerank, reranker_model="memory/rr")
    try:
        m.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0,
                                        "model": "bank/rr"}})
        seen = []
        for call in ("call/rr", None):
            m.search("crew", reranker_model=call)
            seen.append(rerank.requests[-1]["model"])
        m._reranker_model = None
        m.search("crew")
        seen.append(rerank.requests[-1]["model"])
        m.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0}})
        monkeypatch.setenv("PROSPECTA_RERANKER_MODEL", "env/rr")
        m.search("crew")
        seen.append(rerank.requests[-1]["model"])
        monkeypatch.delenv("PROSPECTA_RERANKER_MODEL")
        m.search("crew")
        seen.append(rerank.requests[-1]["model"])
        assert seen == ["call/rr", "memory/rr", "bank/rr", "env/rr", RERANKER_MODEL]
    finally:
        m.close()


# --------------------------------------------------------------------------
# POST /lookup
# --------------------------------------------------------------------------

@pytest.fixture
def svc(mem):
    return Service(mem, None)


def test_lookup_hits_carry_the_citation_fields(mem):
    # a bank of source records: lookup returns record references
    with psycopg.connect(mem.database_url) as conn:
        conn.execute("UPDATE memory_items SET metadata = metadata || "
                     "'{\"kind\": \"note\", \"record_id\": \"7\", \"revision\": \"1\"}' "
                     "WHERE document_id = (SELECT id FROM documents WHERE source = 'phrase.md')")
        conn.commit()
    out = Service(mem, None).lookup({"questions": ['"electric dreams"'], "limit": 3})
    [hit_] = out["results"][0]["results"]
    assert (hit_["kind"], hit_["record_id"], hit_["created_on"]) == ("note", "7", "2024-04-01")
    assert hit_["match_kind"] == "exact+semantic" and "<<electric>>" in hit_["exact_snippet"]
    assert 0 < hit_["relevance_score"] <= 1


def test_lookup_passes_filters_and_models_and_refuses_bad_ones(svc, mem):
    svc.lookup({"questions": ["crew"], "date_from": "2024-01-01", "excluded_sources": ["phrase.md"],
                "reranker_model": "call/rr"})
    assert mem.rerank.requests[-1]["model"] == "call/rr"
    for bad in ({"date_from": "last week"}, {"date_to": 5}, {"excluded_sources": "phrase.md"},
                {"reranker_model": ""}, {"decision_model": 3}):
        with pytest.raises(ApiError) as e:
            svc.lookup({"questions": ["crew"], **bad})
        assert e.value.status == 400
    mem.set_channel_config([])
    with pytest.raises(ApiError, match="hybrid_chunk"):
        svc.lookup({"questions": ["crew"], "date_from": "2024-01-01"})


# --------------------------------------------------------------------------
# Mode 1, task: recall_for_task is a thin wrapper of expand_and_recall (tests/test_expand.py has
# the entry point's own tests); here the old behaviours that must still hold, through the wrapper.
# --------------------------------------------------------------------------

BRIEF = dict(title="Show the library tree view", project="atlas",
             ask='Build the tree view for the library; keep "electric dreams" as the demo folder name.',
             done="Large folders page in")


def test_recall_for_task_returns_a_cited_block_with_its_own_project_first(fresh_db):
    llm = FakeLLM(searches=[{"query": "library tree view decisions", "terms": ['"electric dreams"']},
                            {"query": "append-only archive conventions"}])
    rerank = FakeRerank()
    m = build(fresh_db, llm=llm, rerank=rerank)
    try:
        m.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0}})
        got = m.recall_for_task(**BRIEF)
        assert got.queries == ["library tree view decisions", "append-only archive conventions"]
        sources = [i.source for i in got.items]
        assert len(sources) == len(set(sources))                     # each note once
        assert sources.index("tree.md") > max(sources.index(s) for s in ("phrase.md", "decision.md"))
        assert got.block.startswith("## Recalled Project Context\n\n")
        assert "source=phrase.md | date=2024-04-01 | kind=note | match=exact" in got.block
        assert 'Exact match snippet: "' in got.block and "<<electric>> <<dreams>>" in got.block
        assert "source=tree.md | date=2023-09-09 | kind=note" in got.block
        assert "| project=borealis (another project)" in got.block
        assert "project=atlas" not in got.block          # the task's own project is not labeled
        assert len(rerank.requests) == 2                             # a Stage 2 request per search
    finally:
        m.close()


def test_recall_for_task_without_a_query_writer_searches_the_brief_itself(fresh_db):
    m = build(fresh_db)
    try:
        got = m.recall_for_task(**BRIEF)
        assert got.queries == [" ".join([BRIEF["title"], BRIEF["ask"], BRIEF["done"]])] and got.items
        assert got.searches[0].terms == ('"electric dreams"',)       # the verbatim terms of the brief
        for bad in ({"title": " "}, {"ask": ""}):
            with pytest.raises(ValueError):
                m.recall_for_task(**{**BRIEF, **bad})
    finally:
        m.close()


def test_lookup_task_form_returns_the_context_block_and_cited_results(fresh_db):
    searches = [{"query": 'tree view "electric dreams"', "terms": ['"electric dreams"']},
                {"query": "append-only archive"}]
    m = build(fresh_db, llm=FakeLLM(searches=searches))
    try:
        out = Service(m, None).lookup({"task": BRIEF, "max_tokens": 0})["task"]
        assert out["queries"] == [s["query"] for s in searches]
        assert out["searches"][0] == {"query": 'tree view "electric dreams"', "terms": ['"electric dreams"'],
                                      "date_from": None, "date_to": None}
        assert out["context"].startswith("## Recalled Project Context") and out["error"] is None
        first = out["results"][0]
        assert {"source", "created_on", "kind", "project", "match_kind", "relevance_score", "found_by"} <= set(first)
        for bad in ({"title": 3}, {"ask": ""}, {"done": 1}):
            with pytest.raises(ApiError) as e:
                Service(m, None).lookup({"task": {**BRIEF, **bad}})
            assert e.value.status == 400
        with pytest.raises(ApiError, match="sized by relevance") as e:      # no count cap on a task
            Service(m, None).lookup({"task": BRIEF, "limit": 3})
        assert e.value.status == 400
        with pytest.raises(ApiError):
            Service(m, None).lookup({"task": BRIEF, "max_tokens": -1})
    finally:
        m.close()


# --------------------------------------------------------------------------
# Mode 2B: ask
# --------------------------------------------------------------------------

def test_ask_answers_with_inline_citations_and_verbatim_snippets(fresh_db):
    llm = FakeLLM(answer="The crew called the bridge lamps electric dreams [1]. A tree view exists [9].")
    m = build(fresh_db, llm=llm, rerank=FakeRerank())
    try:
        m.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0}})
        got = m.ask('who called the bridge lamps "electric dreams"', limit=3)
        prompt = llm.calls[0]["prompt"]
        assert not llm.calls[0]["json_mode"]
        assert "[1] source=phrase.md | date=2024-04-01 | kind=note | match=exact" in prompt
        assert "Exact match snippet:" in prompt and "Background from memory" not in prompt
        assert got.items[0].source == "phrase.md"
        first, ghost = got.citations
        assert first == {"n": 1, "known": True, "source": "phrase.md", "created_on": "2024-04-01",
                         "snippet": "The night crew called the bridge lamps electric dreams.",
                         "exact_snippet": got.items[0].exact_snippet}
        assert first["snippet"] in PHRASE and ghost == {"n": 9, "known": False}
        assert got.rendered == got.answer + (
            '\n\nSources:\n[1] [phrase.md | 2024-04-01] "The night crew called the bridge lamps '
            'electric dreams."')
        assert (got.sufficient, got.gate, got.retried) == (None, None, False)
        assert m._tracer.by_name("llm_call")[-1]["purpose"] == "ask_cited"
    finally:
        m.close()


def test_ask_with_no_evidence_answers_not_in_memory_without_calling_the_model(fresh_db):
    llm = FakeLLM()
    m = build(fresh_db, llm=llm, notes={})
    try:
        got = m.ask("what is the plan?")
        assert (got.answer, got.rendered, got.items, got.citations) == ("not in memory",) * 2 + ([], [])
        assert llm.calls == []
        m._llm = None
        with pytest.raises(RuntimeError, match="llm"):
            m.ask("x")
    finally:
        m.close()


def test_supporting_snippet_is_a_verbatim_sentence_and_citations_follow_the_markers():
    chunk = "Alpha opened the gate. Beta watched the harbour lights! Gamma slept."
    assert supporting_snippet(chunk, "who watched the harbour") == "Beta watched the harbour lights!"
    assert supporting_snippet(chunk, "nothing in common") == "Alpha opened the gate."
    assert supporting_snippet("", "x") == ""
    items = [_item("a.md", chunk, created_on="2024-01-01"), _item("b.md", "Other text.")]
    cites = build_citations("Beta watched the lights [1][2]. Again [1, 2]. Missing [3].", items)
    assert [(c["n"], c["known"]) for c in cites] == [(1, True), (2, True), (3, False)]
    assert cites[0]["snippet"] == "Beta watched the harbour lights!"
    assert render_answer("no markers", []) == "no markers"
    assert render_answer("x [2].", cites).endswith('[2] [b.md | unknown] "Other text."')


# ---- Stage 3 ----

def gated(fresh_db, jev, **kw):
    llm = FakeLLM(answer="Deletes are tombstones [1].")
    m = build(fresh_db, llm=llm, jev=jev, rerank=FakeRerank(), **kw)
    m.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0}})
    return m


QUESTION = "how are deletes handled in the archive and the tree view?"


def test_stage_3_is_off_unless_asked_for(fresh_db):
    jev = FakeJev(3)
    m = gated(fresh_db, jev)
    try:
        got = m.ask(QUESTION)
        assert jev.requests == [] and (got.sufficient, got.gate, got.retried) == (None, None, False)
        got = m.ask(QUESTION, sufficiency=True)
        assert len(jev.requests) == 1 and got.sufficient is True and not got.retried
        assert got.gate["score"] == 1.0 and got.gate["threshold"] == 0.70 and got.gate["model"] == JEV_MODEL
        question = next(iter(jev.requests[0]["questions"].values()))
        assert question["instructions"]["title"] == "evidence_sufficient"
        assert "append-only" in question["instructions"]["text"]       # the excerpts under judgement
        assert jev.requests[0]["state"] == {"query": QUESTION}
        m.set_recall_config({"stage3_sufficiency": {"enabled": True}})
        m.ask(QUESTION)
        assert len(jev.requests) == 2                                   # on by the bank's config
        m.ask(QUESTION, sufficiency=False)
        assert len(jev.requests) == 2                                   # the call wins
    finally:
        m.close()


def test_insufficient_evidence_runs_one_broadened_retrieval_then_answers(fresh_db):
    jev = FakeJev(1, 3)           # 0.33 (insufficient), then 1.0 once the retry has widened the evidence
    m = gated(fresh_db, jev)
    try:
        got = m.ask(QUESTION, limit=1, date_from="2024-01-01", sufficiency=True)
        assert [i.source for i in got.items][0] == "decision.md"
        assert got.retried and got.sufficient is True and len(jev.requests) == 2
        first, second = m._tracer.by_name("recall")
        assert first["n_results"] == 1 and second["n_results"] > 1       # twice the limit, deeper pool
        assert len(got.items) > 1 and len({i.document_id for i in got.items}) == len(got.items)
        assert any(i.created_on < "2024-01-01" for i in got.items)        # the date window was dropped
        assert [c["purpose"] for c in m._tracer.by_name("llm_call")].count("stage3_sufficiency") == 2
        # still insufficient after the retry: it is not repeated, the model answers
        jev.scores = [0]
        got = m.ask(QUESTION, sufficiency=True)
        assert got.retried and got.sufficient is False and len(jev.requests) == 4
        assert got.answer == "Deletes are tombstones [1]."
    finally:
        m.close()


def test_stage_3_threshold_is_per_call_then_bank_then_default(fresh_db):
    jev = FakeJev(1.5)           # 0.5
    m = gated(fresh_db, jev)
    try:
        assert m.ask(QUESTION, sufficiency=True).retried                 # 0.5 < 0.70
        m.set_recall_config({"stage3_sufficiency": {"enabled": True, "threshold": 0.4}})
        assert not m.ask(QUESTION).retried                               # 0.5 >= 0.4
        assert m.ask(QUESTION, sufficiency_threshold=0.9).retried        # the call wins
    finally:
        m.close()


def test_a_failing_or_missing_gate_never_blocks_the_answer(fresh_db):
    m = gated(fresh_db, FakeJev(RuntimeError("provider down")))
    try:
        got = m.ask(QUESTION, sufficiency=True)
        assert got.sufficient is None and not got.retried and "provider down" in got.gate["error"]
        assert got.answer == "Deletes are tombstones [1]." and got.gate["threshold"] == 0.70
        m._jev = None
        got = m.ask(QUESTION, sufficiency=True)
        assert got.sufficient is None and "needs Memory(jev=" in got.gate["error"]
    finally:
        m.close()


def test_the_decision_model_is_per_call_then_memory_then_bank_then_environment_then_default(
        fresh_db, monkeypatch):
    jev = FakeJev(3)
    m = gated(fresh_db, jev, decision_model="memory/dm")
    try:
        seen = []
        for kw in ({"decision_model": "call/dm"}, {}):
            m.ask(QUESTION, sufficiency=True, **kw)
            seen.append(jev.requests[-1]["model"])
        m._decision_model = None
        m.set_recall_config({"models": {"decision_model": "bank/dm"}})
        m.ask(QUESTION, sufficiency=True)
        seen.append(jev.requests[-1]["model"])
        m.set_recall_config({})
        m.ask(QUESTION, sufficiency=True)
        seen.append(jev.requests[-1]["model"])
        assert seen == ["call/dm", "memory/dm", "bank/dm", JEV_MODEL]     # env is read once, at import
    finally:
        m.close()


def test_recall_config_accepts_the_new_sections_and_refuses_bad_ones():
    validate_recall_config({"stage3_sufficiency": {"enabled": True, "threshold": 0.7},
                            "models": {"decision_model": "a/b", "query_writer_model": "c/d"}})
    for bad in ({"stage3_sufficiency": {"enabled": "yes"}}, {"stage3_sufficiency": {"threshold": 1.5}},
                {"stage3_sufficiency": {"limit": 3}}, {"stage3_sufficiency": []},
                {"models": {"query_writer_model": ""}}, {"models": {"embedding_model": "x"}}):
        with pytest.raises(ValueError):
            validate_recall_config(bad)
