"""Mode 1 of docs/postgres-hybrid.md: automatic recall by query expansion.

`Memory.expand_and_recall` is the one entry point for a chat message and for a task: the gate
(chat only), the writer (the prompt becomes N structured searches), every search through Stage 1
and Stage 2 against its own query in parallel, one merge, one cited block, all inside a deadline.
Fakes for the writer, the decision, the Jev transport and the reranker; a real Postgres bank for
Stage 1 (the exact retriever's terms and the date filters are checked on real rows). Invented text."""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import date

import pytest

from prospecta import _expand as ex
from prospecta._expand import (
    DEFAULT_QUERY_WRITER_MODEL, apply_maximum, exact_query, guard_terms, merge_records, normalize_decision,
    parse_searches, render_writer_prompt, suggest_searches, verbatim_terms,
)
from prospecta._tracer import RecordingTracer
from prospecta._types import ExpandedRecall, GateDecision, RecalledMemory, SearchSpec, TaskBrief
from prospecta.cited import CITE_LINE, HEADING, TASK_HEADING, format_cited_rag_block
from prospecta.memory import Memory
from prospecta.stages import JEV_MODEL, JevScore, LLMResult, deduplicate_near_copies, validate_recall_config
from tests._stub_embedder import EMBED_DIM, stub_embed

TODAY = date(2024, 6, 1)
NOTES = {
    "lamps-decision.md": ("We decided the harbour lamps get a warm filter. The decision stands until the "
                          "winter storm review.", {"project": "atlas", "created_on": "2024-03-05",
                                                   "source_kind": "decision"}),
    "lamps-incident.md": ("The bridge lamps flickered for two nights; the night crew traced it to the old "
                          "relay.", {"project": "atlas", "created_on": "2024-04-01", "source_kind": "note"}),
    "phrase.md": ("The night crew called the bridge lamps electric dreams. Nobody remembers who started it.",
                  {"project": "atlas", "created_on": "2024-04-02", "source_kind": "note"}),
    "pool.md": ("PROSPECTA_DB_POOL_MAX caps the database connections of the gateway.",
                {"project": "atlas", "created_on": "2023-11-11", "source_kind": "note"}),
    "survey.md": ("Dan Maycock said the pontoon survey is due in May.",
                  {"project": "atlas", "created_on": "2024-02-10", "source_kind": "note"}),
    "tree.md": ("The library tree view loads folders lazily. Large folders page in.",
                {"project": "borealis", "created_on": "2023-09-09", "source_kind": "note"}),
    "orchid.md": ("Orchids need humid air and a shaded window.", {"project": "atlas", "created_on": "2022-01-02"}),
}


# --------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------

class FakeWriter:
    """The query-writer model (an LLM callable accepting the `model` hint). `searches` is the list the
    reply carries; `fn(prompt)` computes it from the prompt; `raw` replaces the whole reply."""

    def __init__(self, searches=None, *, fn=None, raw=None, delay=0.0, raises=None):
        self.searches, self.fn, self.raw, self.delay, self.raises = searches, fn, raw, delay, raises
        self.calls: list[dict] = []

    def __call__(self, messages, *, json_mode=False, model=None):
        prompt = messages[-1]["content"]
        self.calls.append({"json_mode": json_mode, "model": model, "prompt": prompt})
        if self.delay:
            time.sleep(self.delay)
        if self.raises:
            raise self.raises
        if self.raw is not None:
            return self.raw
        searches = self.fn(prompt) if self.fn else self.searches
        return json.dumps({"searches": searches})


def scaling_writer(prompt: str) -> list[dict]:
    """A writer that honors the scaling rule of its prompt: as many distinct searches as the prompt
    suggests, never more than the maximum it names."""
    n = int(re.search(r"Suggested number of searches: (\d+)", prompt).group(1))
    cap = int(re.search(r"Never write more than (\d+) searches", prompt).group(1))
    body = re.split(r"## The (?:message|task brief)\n\n", prompt)[1].split("\n\n## Output format")[0]
    words = list(dict.fromkeys(w for w in re.findall(r"[A-Za-z]{5,}", body.lower())))
    n = min(n, cap)
    return [{"query": f"background on {words[i % len(words)]} part {i}", "terms": [], "date_from": None,
             "date_to": None} for i in range(n)]


class FakeRerank:
    """A rerank transport: the score is `score(query, document)`, by default 0.2 + 0.2 per word the
    document shares with the query. Thread-safe bookkeeping: every request, the concurrency reached."""

    def __init__(self, score=None, delay=0.0, slow=None, fail=None):
        self.score_fn, self.delay, self.slow, self.fail = score, delay, slow, fail
        self.requests: list[dict] = []
        self.lock = threading.Lock()
        self.now = self.peak = 0

    def __call__(self, req, timeout):
        query = req["query"].split("Query:", 1)[-1].strip()
        with self.lock:
            self.requests.append(req)
            self.now += 1
            self.peak = max(self.peak, self.now)
        try:
            if self.fail and self.fail in query:
                raise RuntimeError("rerank provider down")
            wait = self.delay + (self.slow[1] if self.slow and self.slow[0] in query else 0.0)
            if wait:
                time.sleep(wait)
            asked = set(re.findall(r"\w{3,}", query.lower()))
            results = []
            for i, d in enumerate(req["documents"]):
                if self.score_fn:
                    s = self.score_fn(query, d)
                else:
                    s = min(0.99, 0.2 + 0.2 * len(asked & set(re.findall(r"\w{3,}", d.lower()))))
                results.append({"index": i, "relevance_score": s})
            return {"model": req["model"], "results": results, "usage": {"total_tokens": 10, "cost": 0.0001}}
        finally:
            with self.lock:
                self.now -= 1

    def queries(self):
        return [r["query"].split("Query:", 1)[-1].strip() for r in self.requests]


class FakeJev:
    """A System One transport answering the needs_memory question with scripted noul values."""

    def __init__(self, *noul):
        self.noul, self.requests = list(noul), []

    def __call__(self, req, timeout):
        self.requests.append(req)
        v = self.noul.pop(0) if len(self.noul) > 1 else self.noul[0]
        if isinstance(v, Exception):
            raise v
        return {"model": req["model"], "usage": {"input_tokens": 7, "output_tokens": 1, "cost": 0.00002},
                "answers": {"needs_memory": {"type": "noul", "noul": v}}}


def build(fresh_db, *, llm=None, rerank=None, jev=None, notes=NOTES, rerank_cfg=None, embed=stub_embed, **kw):
    mem = Memory(database_url=fresh_db, bank_id="b", embed=embed, llm=llm,
                 jev=JevScore(jev) if jev is not None else None, rerank_transport=rerank,
                 tracer=RecordingTracer(), **kw)
    mem.create_bank("b", embedding_dim=EMBED_DIM)
    mem.set_channel_config([{"name": "hybrid_chunk", "weight": 1}])
    if rerank is not None:
        mem.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0,
                                          **(rerank_cfg or {})}})
    for source, (text, metadata) in notes.items():
        mem.retain(text, source=source, metadata=metadata)
    return mem


def S(query, terms=(), df=None, dt=None):
    return {"query": query, "terms": list(terms), "date_from": df, "date_to": dt}


@pytest.fixture
def made(fresh_db):
    """build(...) with cleanup: made(llm=..., rerank=...)"""
    mems = []

    def make(**kw):
        m = build(fresh_db, **kw)
        mems.append(m)
        return m
    yield make
    for m in mems:
        m.close()


def sources(result: ExpandedRecall) -> list[str]:
    return [i.source for i in result.items]


# --------------------------------------------------------------------------
# the writer: prompt, scaling, maximum
# --------------------------------------------------------------------------

LONG_BRIEF = TaskBrief(
    title="Rework the harbour lamp subsystem and its night crew rota",
    ask=("Replace the relay boards of the bridge lamps, keep the warm filter decision, make the "
         "flicker incident report part of the handover, move the pontoon survey data into the new "
         "archive layout, check the connection limits of the gateway, add a regression check for the "
         "winter storm review, update the crew rota and the documentation of every lamp group, and "
         "report each step to the harbour office. " * 2),
    done="All lamp groups pass the nightly check; the rota and the handover notes are in the archive.",
    project="atlas")


def test_a_short_chat_message_expands_to_about_three_searches(made):
    writer = FakeWriter(fn=scaling_writer)
    mem = made(llm=writer, rerank=FakeRerank())
    got = mem.expand_and_recall("what did we decide about the harbour lamps?", gate=False)
    assert len(got.searches) == 3 and len(writer.calls) == 1
    assert got.error is None and got.items


def test_a_long_instruction_or_task_brief_expands_to_more_than_ten_searches(made):
    writer = FakeWriter(fn=scaling_writer)
    mem = made(llm=writer, rerank=FakeRerank())
    got = mem.expand_and_recall(LONG_BRIEF, kind="task")
    assert 10 < len(got.searches) <= 20
    long_chat = " ".join(f"Please also take care of the harbour lamp step {n} and its relay board." for n in range(25))
    got = mem.expand_and_recall(long_chat, gate=False)
    assert len(got.searches) > 10


def test_the_writer_prompt_carries_the_scaling_rule_the_maximum_and_the_prompt(made):
    writer = FakeWriter([S("harbour lamps")])
    mem = made(llm=writer, rerank=FakeRerank())
    mem.expand_and_recall("what did we decide about the harbour lamps?", agent_last="We could repaint them.",
                          gate=False, max_searches=12)
    p = writer.calls[0]["prompt"]
    assert "about 3 searches" in p and "up to 10 to 20" in p and "Never write more than 12 searches" in p
    assert "SURROUNDING CONTEXT" in p and "not questions" in p and "VERBATIM" in p
    assert "what did we decide about the harbour lamps?" in p
    assert "The agent's last turn" in p and "We could repaint them." in p       # follow-ups: agent_last
    assert writer.calls[0]["json_mode"] is True
    mem.expand_and_recall(LONG_BRIEF, kind="task")
    t = writer.calls[1]["prompt"]
    assert "Title: Rework the harbour lamp subsystem" in t and "Project: atlas" in t
    assert "Done when: All lamp groups pass" in t and "The agent's last turn" not in t
    assert "Never write more than 20 searches" in t


def test_the_suggested_count_scales_with_the_prompt():
    assert suggest_searches("chat", 6, 20) == 3
    assert suggest_searches("chat", 60, 20) == 5
    assert suggest_searches("chat", 500, 20) == 12
    assert 10 < suggest_searches("task", 250, 20) <= 20 and suggest_searches("task", 10, 20) == 8
    assert suggest_searches("task", 400, 6) == 6                      # never over the maximum
    p = render_writer_prompt("one two three", kind="chat", agent_last=None, max_searches=20, today=TODAY)
    assert "3 words. Suggested number of searches: 3." in p and "Today is 2024-06-01" in p


def test_a_writer_that_writes_too_many_searches_is_cut_at_the_hard_maximum(made):
    many = [S(f"search number {n}") for n in range(45)]
    writer = FakeWriter(many)
    mem = made(llm=writer, rerank=FakeRerank())
    got = mem.expand_and_recall("harbour lamps", gate=False)
    assert len(got.searches) == 20 and got.searches[0].query == "search number 0"       # the default maximum
    assert any("45 searches: cut to the maximum of 20" in f for f in got.fallbacks)
    assert len(mem.expand_and_recall("harbour lamps", gate=False, max_searches=5).searches) == 5
    mem.set_recall_config({"expansion": {"max_searches": 7},
                           "rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0}})
    assert len(mem.expand_and_recall("harbour lamps", gate=False).searches) == 7             # the bank's
    assert len(mem.expand_and_recall("harbour lamps", gate=False, max_searches=3).searches) == 3   # call wins


def test_apply_maximum():
    specs = [SearchSpec(query=str(n)) for n in range(5)]
    assert apply_maximum(specs, 5) == (specs, None) and apply_maximum(specs, 2) == (specs[:2], 5)


def test_the_query_writer_model_is_per_call_then_bank_then_environment_then_default(made, monkeypatch):
    writer = FakeWriter([S("tree view")])
    mem = made(llm=writer, rerank=FakeRerank())
    models = []
    mem.expand_and_recall("tree view", gate=False, query_writer_model="call/qw")
    models.append(writer.calls[-1]["model"])
    mem.set_recall_config({"models": {"query_writer_model": "bank/qw"}})
    mem.expand_and_recall("tree view", gate=False)
    models.append(writer.calls[-1]["model"])
    mem.set_recall_config({})
    monkeypatch.setenv("PROSPECTA_QUERY_WRITER_MODEL", "env/qw")
    mem.expand_and_recall("tree view", gate=False)
    models.append(writer.calls[-1]["model"])
    monkeypatch.delenv("PROSPECTA_QUERY_WRITER_MODEL")
    mem.expand_and_recall("tree view", gate=False)
    models.append(writer.calls[-1]["model"])
    assert models == ["call/qw", "bank/qw", "env/qw", DEFAULT_QUERY_WRITER_MODEL]
    assert DEFAULT_QUERY_WRITER_MODEL == "openrouter/google/gemini-2.5-flash-lite"


# --------------------------------------------------------------------------
# parsing the writer's reply, defensively
# --------------------------------------------------------------------------

def test_parse_searches_is_defensive():
    good = ('Here you go:\n```json\n{"searches": [{"query": "lamp decisions", "terms": ["v1.14", "v1.14", " "], '
            '"date_from": "2024-03", "date_to": "2024-03-31"}, "plain query", {"terms": ["Dan Maycock"]}, '
            '{"query": "lamp decisions", "terms": ["v1.14"], "date_from": "2024-03-01", "date_to": "2024-03-31"}, '
            '42, {"query": ""}]}\n```')
    specs, why = parse_searches(good, TODAY)
    assert why is None and specs == [
        SearchSpec("lamp decisions", ("v1.14",), "2024-03-01", "2024-03-31"),    # YYYY-MM -> the month; duplicates drop
        SearchSpec("plain query"),
        SearchSpec("Dan Maycock", ("Dan Maycock",))]                              # a search with terms only
    assert parse_searches('[{"query": "q"}]', TODAY)[0] == [SearchSpec("q")]      # a bare array
    for bad in ("not json at all", "{}", '{"searches": []}', '{"searches": "x"}', "", '{"searches": [1, null]}'):
        specs, why = parse_searches(bad, TODAY)
        assert specs == [] and why
    # dates the writer got wrong are no window; the search stays
    for df, dt in (("1900-01-01", "2024-01-01"), ("2024-05-01", "2024-01-01"), ("2099-01-01", None),
                   ("last week", "yesterday"), ("2024-02-31", None)):
        [s] = parse_searches(json.dumps({"searches": [{"query": "q", "date_from": df, "date_to": dt}]}), TODAY)[0]
        assert (s.date_from, s.date_to) == (None, None), (df, dt)
    [s] = parse_searches('{"searches": [{"query": "q", "date_from": "2023-01-01"}]}', TODAY)[0]
    assert (s.date_from, s.date_to) == ("2023-01-01", None)                       # an open end is fine


@pytest.mark.parametrize("raw,expected", [
    (FakeWriter(raises=RuntimeError("provider down")), "RuntimeError"),
    (FakeWriter(raw="I am sorry, I cannot do that."), "not usable JSON"),
    (FakeWriter(raw='{"searches": []}'), "no usable search"),
    (FakeWriter(raw='{"answer": "the lamps are warm"}'), "no 'searches' list"),
])
def test_a_writer_failure_falls_back_to_one_search_with_the_prompt_and_its_verbatim_terms(made, raw, expected):
    mem = made(llm=raw, rerank=FakeRerank())
    prompt = 'Which note says "electric dreams" about the bridge lamps and PROSPECTA_DB_POOL_MAX?'
    got = mem.expand_and_recall(prompt, gate=False)
    assert got.searches == [SearchSpec(query=prompt, terms=('"electric dreams"', "PROSPECTA_DB_POOL_MAX"))]
    assert got.fallbacks[0].startswith("writer:") and expected in got.fallbacks[0]
    assert got.error is None and got.items and got.block
    assert got.calls[0]["purpose"] == "expand_searches"


def test_without_a_writer_model_the_prompt_itself_is_the_one_search(made):
    mem = made(llm=None, rerank=FakeRerank())
    got = mem.expand_and_recall("the bridge lamps electric dreams", gate=False)
    assert [s.query for s in got.searches] == ["the bridge lamps electric dreams"]
    assert "no query-writer model" in got.fallbacks[0] and "phrase.md" in sources(got)


# --------------------------------------------------------------------------
# the verbatim-terms guard
# --------------------------------------------------------------------------

def test_verbatim_terms_finds_quotes_identifiers_and_names():
    text = ('Greg asked about the Hawthorne Lane project: v1.14 broke "electric dreams" in src/main.rs; '
            "set PROSPECTA_DB_POOL_MAX. Dan Maycock called on Monday about error 0x80070005. Remind me.")
    assert verbatim_terms(text) == ['"electric dreams"', "v1.14", "src/main.rs", "PROSPECTA_DB_POOL_MAX",
                                    "0x80070005", "Hawthorne Lane", "Dan Maycock"]
    assert verbatim_terms("ok, ship it") == [] and verbatim_terms("") == []
    assert verbatim_terms("The 3rd of March, 12 pm, a 2nd look at e.g. this.") == []      # ordinals, dates, abbreviations
    assert verbatim_terms("run it with --force on HybridChunks, see config.yaml.") == [
        "--force", "HybridChunks", "config.yaml"]
    assert verbatim_terms('He said "ship it" and "ship it".') == ['"ship it"']            # once


def test_a_term_the_writer_dropped_is_put_back_into_the_first_search():
    prompt = 'Look at "electric dreams", PROSPECTA_DB_POOL_MAX and what Dan Maycock said about v1.14.'
    searches = [SearchSpec("bridge lamp lore", ("v1.14",)), SearchSpec("survey dates", ("Dan Maycock",))]
    guarded, back = guard_terms(searches, prompt)
    assert back == ['"electric dreams"', "PROSPECTA_DB_POOL_MAX"]
    assert guarded[0].terms == ("v1.14", '"electric dreams"', "PROSPECTA_DB_POOL_MAX")
    assert guarded[1] == searches[1] and guarded[0].query == "bridge lamp lore"
    # carried in any search, in any case, with or without its quotes: nothing is added
    carried = [SearchSpec("a", ("electric dreams",)), SearchSpec("b", ("prospecta_db_pool_max", "dan maycock", "V1.14"))]
    assert guard_terms(carried, prompt) == (carried, [])
    assert guard_terms([], prompt) == ([], [])


def test_the_guard_runs_on_the_writers_searches_in_a_recall(made):
    writer = FakeWriter([S("bridge lamp lore", ["v1.14"]), S("harbour lamp decisions")])
    mem = made(llm=writer, rerank=FakeRerank())
    got = mem.expand_and_recall('Why do the bridge lamps carry the name "electric dreams" since v1.14?',
                                gate=False)
    assert got.searches[0].terms == ("v1.14", '"electric dreams"') and got.searches[1].terms == ()
    assert any("put back into search 1" in f and '"electric dreams"' in f for f in got.fallbacks)
    assert "electric dreams" in got.block


def test_exact_query():
    assert exact_query(['"electric dreams"', "v1.14", "Dan Maycock", "-x", "OR", " ", '"']) == (
        '"electric dreams" OR v1.14 OR "Dan Maycock" OR x')
    assert exact_query([]) is None and exact_query(["OR", " "]) is None


def test_terms_feed_the_exact_retriever_and_the_query_the_others(made):
    """The query shares no word with the note holding the identifier: only the terms can find it by
    the exact retriever, and the snippet is the matched identifier."""
    writer = FakeWriter([S("capacity planning", ["PROSPECTA_DB_POOL_MAX"])])
    mem = made(llm=writer, rerank=FakeRerank())
    got = mem.expand_and_recall("capacity PROSPECTA_DB_POOL_MAX", gate=False)
    hit = next(i for i in got.items if i.source == "pool.md")
    assert "exact" in hit.match_kind and "<<PROSPECTA>>" in hit.exact_snippet
    assert [i.source for i in got.items if "exact" in (i.match_kind or "")] == ["pool.md"]     # only that note
    # without terms the exact retriever reads the query itself (Stage 1 as it always was)
    writer.searches = [S("capacity planning")]
    got = mem.expand_and_recall("capacity planning", gate=False)
    assert all("exact" not in (i.match_kind or "") for i in got.items)


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------

class Counting:
    """A decision callable that answers `answer` and counts its calls."""

    def __init__(self, answer):
        self.answer, self.calls = answer, []

    def __call__(self, prompt, agent_last):
        self.calls.append((prompt, agent_last))
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def test_the_gate_skips_an_acknowledgement_before_any_model_or_search(made):
    writer, rerank, decide = FakeWriter([S("anything")]), FakeRerank(), Counting(0.05)
    mem = made(llm=writer, rerank=rerank)
    got = mem.expand_and_recall("ok, ship it", agent_last="Shall I deploy?", decision=decide)
    assert got.skipped and got.block == "" and got.items == [] and got.searches == []
    assert "needs_memory 0.05 < 0.5" in got.skipped_reason and got.gate["needs_memory"] == 0.05
    assert decide.calls == [("ok, ship it", "Shall I deploy?")]
    assert writer.calls == [] and rerank.requests == []               # no writer, no search, no rerank
    assert got.error is None and got.accounting["n_llm_calls"] == 1    # only the gate was asked


@pytest.mark.parametrize("answer,expect_skip", [
    (0.5, False), (0.49, True), (1.0, False), (0.0, True), (True, False), (False, True),
    (GateDecision(needs_memory=0.9, skip=True, reason="a greeting"), True),
    (GateDecision(needs_memory=0.2), True), ({"needs_memory": 0.7}, False),
])
def test_the_gate_threshold_is_at_or_above_one_half(made, answer, expect_skip):
    mem = made(llm=FakeWriter([S("harbour lamps")]), rerank=FakeRerank())
    got = mem.expand_and_recall("what about the harbour lamps?", decision=Counting(answer))
    assert got.skipped is expect_skip and bool(got.items) is not expect_skip


def test_a_task_is_never_gated(made):
    decide = Counting(0.0)                                              # would skip any chat message
    mem = made(llm=FakeWriter([S("harbour lamps")]), rerank=FakeRerank())
    got = mem.expand_and_recall(TaskBrief(title="Lamps", ask="Fix the harbour lamps"), kind="task", decision=decide)
    assert not got.skipped and got.items and decide.calls == [] and got.gate is None
    got = mem.expand_and_recall("Fix the harbour lamps", kind="task", decision=decide)    # a task as plain text
    assert not got.skipped and decide.calls == []
    assert mem.recall_for_task(title="Lamps", ask="Fix the harbour lamps", decision=decide).items
    assert decide.calls == []


def test_the_gate_threshold_and_switch_are_configurable(made):
    mem = made(llm=FakeWriter([S("harbour lamps")]), rerank=FakeRerank())
    assert mem.expand_and_recall("lamps", decision=Counting(0.6), needs_memory_threshold=0.7).skipped
    assert not mem.expand_and_recall("lamps", decision=Counting(0.6), needs_memory_threshold=0.6).skipped
    mem.set_recall_config({"stage0_gate": {"needs_memory_threshold": 0.8},
                           "rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0}})
    assert mem.expand_and_recall("lamps", decision=Counting(0.7)).skipped                      # the bank's
    assert not mem.expand_and_recall("lamps", decision=Counting(0.7), needs_memory_threshold=0.5).skipped
    off = Counting(0.0)
    assert not mem.expand_and_recall("lamps", decision=off, gate=False).skipped and off.calls == []
    mem.set_recall_config({"stage0_gate": {"enabled": False},
                           "rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0}})
    assert not mem.expand_and_recall("lamps", decision=off).skipped and off.calls == []


def test_a_failing_or_malformed_gate_lets_the_message_through(made):
    mem = made(llm=FakeWriter([S("harbour lamps")]), rerank=FakeRerank())
    for answer in (RuntimeError("provider down"), "yes", 7, None, {"needs": 1}):
        got = mem.expand_and_recall("what about the harbour lamps?", decision=Counting(answer))
        assert not got.skipped and got.items and any(f.startswith("gate failed") for f in got.fallbacks)
        assert "error" in got.gate
    slow = lambda prompt, agent_last: time.sleep(2.5) or 0.0             # past the gate's share of the deadline
    t = time.monotonic()
    got = mem.expand_and_recall("what about the harbour lamps?", decision=slow, deadline=2.0)
    assert not got.skipped and got.items and time.monotonic() - t < 2.5
    assert any("gate failed" in f for f in got.fallbacks)


def test_the_default_gate_asks_the_decision_model_needs_memory(made, monkeypatch):
    jev = FakeJev(0.1)
    mem = made(llm=FakeWriter([S("harbour lamps")]), rerank=FakeRerank(), jev=jev)
    got = mem.expand_and_recall("thanks, that works", agent_last="I repainted the lamps.")
    [req] = jev.requests
    q = req["questions"]["needs_memory"]
    assert req["model"] == JEV_MODEL == "typesafe/jev-1.14" and set(req["questions"]) == {"needs_memory"}
    assert q["type"] == "noul" and set(q["criteria"]) == {"true", "false"}
    assert req["state"]["agent_last"] == "I repainted the lamps." and "thanks, that works" in req["state"]["new_lines"][0]
    assert got.skipped and got.gate == {"needs_memory": 0.1, "threshold": 0.5, "skip": False,
                                        "model": JEV_MODEL, "reason": None}
    assert got.accounting["cost_usd"] == pytest.approx(0.00002) and got.calls[0]["purpose"] == "expand_gate"
    jev.noul = [0.8]
    assert mem.expand_and_recall("what did we decide about the lamps?").items
    # the decision model: per call, then the Memory's, then the bank's
    mem.expand_and_recall("lamps?", decision_model="call/dm")
    mem._decision_model = "memory/dm"
    mem.expand_and_recall("lamps?")
    mem._decision_model = None
    mem.set_recall_config({"models": {"decision_model": "bank/dm"}})
    mem.expand_and_recall("lamps?")
    assert [r["model"] for r in jev.requests[2:]] == ["call/dm", "memory/dm", "bank/dm"]
    jev.noul = [RuntimeError("down")]                                    # a failing decision model: through
    assert mem.expand_and_recall("lamps?").items
    jev.noul = [1.5]                                                     # a malformed answer: through
    assert mem.expand_and_recall("lamps?").items


def test_a_chat_without_any_decision_goes_through_and_says_so(made):
    mem = made(llm=FakeWriter([S("harbour lamps")]), rerank=FakeRerank())
    got = mem.expand_and_recall("what about the harbour lamps?")
    assert got.items and any("no decision callable" in f for f in got.fallbacks)


def test_normalize_decision():
    assert normalize_decision(0.3) == GateDecision(needs_memory=0.3)
    assert normalize_decision(True).needs_memory == 1.0 and normalize_decision(False).needs_memory == 0.0
    assert normalize_decision({"needs_memory": 0.4, "skip": True, "junk": 1}) == GateDecision(0.4, skip=True)
    d = GateDecision(needs_memory=0.2)
    assert normalize_decision(d) is d
    for bad in ("0.3", None, 2, float("nan"), {}):
        with pytest.raises(ValueError):
            normalize_decision(bad)


# --------------------------------------------------------------------------
# stage 1 filters, per search
# --------------------------------------------------------------------------

def test_a_searchs_own_period_is_pushed_into_stage_1_and_the_others_use_the_callers_window(made):
    writer = FakeWriter([S("harbour lamps crew survey", df="2024-01-01", dt="2024-12-31"),
                         S("harbour lamps crew survey")])
    mem = made(llm=writer, rerank=FakeRerank(), rerank_cfg={"min_score": 0})
    got = mem.expand_and_recall("harbour lamps", gate=False)
    by_source = {i.source: i.found_by for i in got.items}
    assert by_source["lamps-decision.md"] == (1, 2) and by_source["survey.md"] == (1, 2)
    assert by_source["pool.md"] == (2,) and by_source["tree.md"] == (2,) and by_source["orchid.md"] == (2,)
    # the caller's window applies to the search without a period of its own, never over its own
    got = mem.expand_and_recall("harbour lamps", gate=False, date_from="2024-03-01")
    by_source = {i.source: i.found_by for i in got.items}
    assert by_source["survey.md"] == (1,)                                  # 2024-02-10: in search 1's own period only
    assert by_source["lamps-decision.md"] == (1, 2)
    assert "pool.md" not in by_source and "orchid.md" not in by_source
    got = mem.expand_and_recall("harbour lamps", gate=False, date_to="2023-12-31")
    assert {i.source for i in got.items if i.found_by == (2,)} == {"pool.md", "tree.md", "orchid.md"}


def test_the_same_filters_reach_every_search(made):
    rerank = FakeRerank()
    writer = FakeWriter([S(f"harbour lamps crew {w}") for w in ("alpha", "beta", "gamma", "delta")])
    mem = made(llm=writer, rerank=rerank)
    got = mem.expand_and_recall("harbour lamps", gate=False, excluded_sources=["phrase.md"],
                                metadata_filter={"project": "atlas"})
    assert "phrase.md" not in sources(got) and "tree.md" not in sources(got)       # anti-echo and scope
    assert "electric dreams" not in " ".join(d for r in rerank.requests for d in r["documents"])
    assert "Library tree view" not in " ".join(d for r in rerank.requests for d in r["documents"])
    assert sources(got)                                                              # and the rest is found


def test_each_search_is_reranked_against_its_own_query(made):
    rerank = FakeRerank()
    queries = ["bridge lamp relay incident", "pontoon survey due date", "database connection limits"]
    mem = made(llm=FakeWriter([S(q) for q in queries]), rerank=rerank)
    got = mem.expand_and_recall("lamps survey connections", gate=False)
    assert sorted(rerank.queries()) == sorted(queries)                  # one request per search, its own query
    assert all(r["model"] == "qwen/qwen3-reranker-8b" for r in rerank.requests)
    assert [r.reranked for r in got.runs] == [True, True, True] and all(r.status == "ok" for r in got.runs)
    assert got.accounting["n_llm_calls"] == 1 + 3                       # the writer and three reranks
    assert got.accounting["tokens_in"] == 30 and got.accounting["cost_usd"] == pytest.approx(0.0003)
    mem.expand_and_recall("lamps", gate=False, reranker_model="call/rr")
    assert rerank.requests[-1]["model"] == "call/rr"


def test_the_score_floor_cuts_what_does_not_clear_it(made):
    rerank = FakeRerank(score=lambda q, d: 0.9 if "harbour" in d else 0.1)
    mem = made(llm=FakeWriter([S("harbour lamps")]), rerank=rerank, rerank_cfg={"min_score": 0.15})
    got = mem.expand_and_recall("harbour lamps", gate=False)
    assert sources(got) == ["lamps-decision.md"]
    assert mem.expand_and_recall("harbour lamps", gate=False, min_score=0.0).items[1:]      # a lower floor keeps more


# --------------------------------------------------------------------------
# parallelism: bounded by the pool, the rerank holds no connection
# --------------------------------------------------------------------------

def _spy_stage1(monkeypatch, mem, hold=0.12):
    import prospecta.channels.hybrid as hybrid
    real, seen = hybrid.stage1_search, {"now": 0, "peak": 0, "in_use": 0, "lock": threading.Lock()}

    def spy(conn, **kw):
        with seen["lock"]:
            seen["now"] += 1
            seen["peak"] = max(seen["peak"], seen["now"])
            seen["in_use"] = max(seen["in_use"], mem._pool.in_use)
        try:
            time.sleep(hold)
            return real(conn, **kw)
        finally:
            with seen["lock"]:
                seen["now"] -= 1
    monkeypatch.setattr(hybrid, "stage1_search", spy)
    return seen


def test_searches_run_in_parallel_bounded_by_the_pool_and_max_concurrency(made, monkeypatch):
    rerank = FakeRerank(delay=0.3)
    mem = made(llm=FakeWriter([S(f"harbour lamps part {n}") for n in range(12)]), rerank=rerank)
    mem._pool.max_size = 3
    seen = _spy_stage1(monkeypatch, mem)
    got = mem.expand_and_recall("harbour lamps", gate=False, deadline=30)
    assert len(got.runs) == 12 and all(r.status == "ok" for r in got.runs)
    assert seen["peak"] == 3                                  # parallel, and never more than the pool has
    assert seen["in_use"] <= 3 and mem._pool.open_connections <= 3
    assert rerank.peak > 3                                    # the rerank holds no connection: not bound by the pool
    mem._pool.max_size = 10
    seen = _spy_stage1(monkeypatch, mem)
    mem.expand_and_recall("harbour lamps", gate=False, deadline=30, max_concurrency=2)
    assert seen["peak"] == 2                                  # max_concurrency, configurable
    seen = _spy_stage1(monkeypatch, mem)
    mem.expand_and_recall("harbour lamps", gate=False, deadline=30)
    assert seen["peak"] == 8                                  # the default maximum concurrency
    mem.set_recall_config({"expansion": {"max_concurrency": 4},
                           "rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0}})
    seen = _spy_stage1(monkeypatch, mem)
    mem.expand_and_recall("harbour lamps", gate=False, deadline=30)
    assert seen["peak"] == 4                                  # and the bank's setting


def test_twenty_searches_cost_about_the_slowest_not_the_sum(made):
    rerank = FakeRerank(delay=0.4)
    writer = FakeWriter([S(f"harbour lamps part {n}") for n in range(20)], delay=0.2)
    mem = made(llm=writer, rerank=rerank)
    t = time.monotonic()
    got = mem.expand_and_recall("harbour lamps", gate=False, deadline=30)
    took = time.monotonic() - t
    assert len(got.runs) == 20 and all(r.status == "ok" for r in got.runs)
    assert took < 0.2 + 0.4 + 1.2                              # the writer plus the slowest, with slack, not 20 x 0.4
    assert got.timings_ms["writer"] >= 200 and got.timings_ms["rerank"] >= 400
    assert got.timings_ms["slowest_search"] <= got.timings_ms["searches"] <= got.timings_ms["total"]


# --------------------------------------------------------------------------
# the merge
# --------------------------------------------------------------------------

def test_a_record_found_by_several_searches_appears_once_with_its_best_score(made):
    rerank = FakeRerank(score=lambda q, d: (0.9 if "alpha" in q else 0.5) if "electric dreams" in d else 0.3)
    writer = FakeWriter([S("bridge lamps beta"), S("bridge lamps alpha"), S("night crew gamma")])
    mem = made(llm=writer, rerank=rerank)
    got = mem.expand_and_recall("bridge lamps", gate=False)
    assert sources(got).count("phrase.md") == 1 and len(sources(got)) == len(set(sources(got)))
    hit = got.items[0]
    assert hit.source == "phrase.md" and hit.relevance_score == pytest.approx(0.9)
    assert hit.found_by == (1, 2, 3)
    scores = [i.relevance_score for i in got.items]
    assert scores == sorted(scores, reverse=True)                        # ordered by score
    assert "| found_by=1,2,3" in got.block.split("[2]")[0]


def test_near_copies_collapse_across_searches_into_the_best_scored_one(made):
    notes = {**NOTES,
             "copy-a.md": ("The pontoon survey was moved to May and the harbour office agreed to the new date.",
                           {"project": "atlas", "created_on": "2024-02-11"}),
             "copy-b.md": ("The pontoon survey was moved to May and the harbour office agreed to the new date today.",
                           {"project": "atlas", "created_on": "2024-02-12"})}
    rerank = FakeRerank(score=lambda q, d: (0.8 if "copy" in q else 0.6) if "pontoon survey was moved" in d else 0.3)
    mem = made(llm=FakeWriter([S("pontoon survey moved copy"), S("harbour office date")]), rerank=rerank,
               notes=notes, rerank_cfg={"deduplicate_near_copies": False})      # stage 2 does not collapse them
    got = mem.expand_and_recall("pontoon survey", gate=False)
    twins = [i for i in got.items if i.source in ("copy-a.md", "copy-b.md")]
    assert len(twins) == 1 and twins[0].relevance_score == pytest.approx(0.8)
    assert set(twins[0].found_by) == {1, 2} and twins[0].source == "copy-a.md"


def test_merge_records_unit():
    def rec(doc, text, score):
        return RecalledMemory(content=text, original_chunk=text, source=doc + ".md", score=0.01, scores={},
                              metadata={}, bank_id="b", document_id=doc, relevance_score=score)
    a_low, a_high = rec("a", "kappa lambda mu", 0.4), rec("a", "alpha chunk two", 0.7)
    b = rec("b", "something else entirely different here today", 0.5)
    d = rec("d", "alpha beta gamma delta epsilon zeta eta theta", 0.6)
    c = rec("c", "alpha beta gamma delta epsilon zeta eta theta iota", 0.55)      # a near-copy of d
    merged = merge_records({1: [a_low, b], 2: [a_high], 3: [d], 4: [c]})
    assert [(m.document_id, m.relevance_score, m.found_by) for m in merged] == [
        ("a", 0.7, (1, 2)), ("d", 0.6, (3, 4)), ("b", 0.5, (1,))]
    assert merged[0].original_chunk == "alpha chunk two"                          # the best-scored copy is kept
    assert merge_records({}) == [] and merge_records({1: []}) == []
    survivors, collapsed = deduplicate_near_copies(
        ["one two three four five six seven eight nine", "x y z", "one two three four five six seven eight nine ten"],
        [2, 0, 1])
    assert survivors == [2, 1] and collapsed == {0: 2}


def test_a_tasks_own_project_comes_first_and_other_projects_are_labeled(made):
    rerank = FakeRerank(score=lambda q, d: 0.9 if "tree view" in d else 0.4)     # the other project's note scores best
    mem = made(llm=FakeWriter([S("library tree view folders"), S("harbour lamps")]), rerank=rerank)
    got = mem.expand_and_recall(TaskBrief(title="Show the library tree view", ask="Build it", project="atlas"),
                                kind="task")
    projects = [(i.metadata or {}).get("project") for i in got.items]
    assert projects[0] == "atlas" and projects[-1] == "borealis" and sources(got)[-1] == "tree.md"
    assert got.items[-1].relevance_score == pytest.approx(0.9)            # first by project, not by score
    assert "| project=borealis (another project)" in got.block and "project=atlas" not in got.block
    chat = mem.expand_and_recall("library tree view", gate=False)         # a chat is ordered by score alone
    assert sources(chat)[0] == "tree.md"


def test_the_block_lists_the_searches_and_tells_the_agent_to_cite(made):
    mem = made(llm=FakeWriter([S("bridge lamp relay incident", ['"electric dreams"'], df="2024-03-01", dt="2024-04-30"),
                               S("pontoon survey due date")]), rerank=FakeRerank())
    got = mem.expand_and_recall("lamps", gate=False)
    head, *_ = got.block.split("\n\n")
    assert head == HEADING and CITE_LINE in got.block
    assert ('(1) bridge lamp relay incident | terms: "electric dreams" | dates: 2024-03-01 to 2024-04-30'
            in got.block)
    assert "(2) pontoon survey due date" in got.block and "Searches run" in got.block
    items = got.block.split(CITE_LINE)[1]
    assert re.search(r"\[1\] source=\S+ \| date=\d{4}-\d{2}-\d{2} \|( kind=\S+ \|)? match=\S+ \| found_by=", items)
    assert 'Exact match snippet: "' in got.block
    task = mem.expand_and_recall(TaskBrief(title="Lamps", ask="Check the lamps"), kind="task")
    assert task.block.startswith(TASK_HEADING + "\n\n") and CITE_LINE in task.block
    assert got.kind == "chat" and task.kind == "task"


def test_a_search_whose_rerank_fails_keeps_stage_1_order_and_the_others_are_unaffected(made):
    mem = made(llm=FakeWriter([S("bridge lamp relay incident"), S("pontoon survey due date")]),
               rerank=FakeRerank(fail="pontoon"))
    got = mem.expand_and_recall("lamps", gate=False)
    assert [r.status for r in got.runs] == ["ok", "ok"] and [r.reranked for r in got.runs] == [True, False]
    assert got.runs[1].fallback_reason and got.runs[1].n_kept > 0
    assert any("search 2" in f and "stage 2" in f and "rerank provider down" in f for f in got.fallbacks)
    assert any(2 in (i.found_by or ()) for i in got.items) and any(1 in (i.found_by or ()) for i in got.items)
    assert got.error is None and "(2) pontoon survey due date" in got.block


# --------------------------------------------------------------------------
# no count cap; the safety valve
# --------------------------------------------------------------------------

def test_the_block_is_sized_by_relevance_never_by_a_count(made):
    notes = {f"p{n}.md": (f"Pontoon survey record number {n} lists the unique words w{n}a w{n}b w{n}c.",
                          {"project": "atlas", "created_on": "2024-02-10"}) for n in range(40)}
    mem = made(llm=FakeWriter([S("pontoon survey records")]), rerank=FakeRerank(score=lambda q, d: 0.5),
               notes=notes)
    mem.set_channel_config([{"name": "hybrid_chunk", "weight": 1,
                             "params": {"semantic_limit": 50, "stemmed_limit": 50, "exact_limit": 50}}])
    got = mem.expand_and_recall("pontoon survey", gate=False)
    assert len(got.items) == 40 and got.block.count("source=p") == 40       # more than any default limit
    assert got.dropped_for_tokens == 0


def test_the_max_tokens_valve_is_off_by_default_and_trims_the_tail_when_set(made):
    notes = {f"p{n}.md": (f"Pontoon survey record number {n} " + ("lists the unique words " + f"w{n}x " * 40),
                          {"created_on": "2024-02-10"}) for n in range(12)}
    mem = made(llm=FakeWriter([S("pontoon survey records")]),
               rerank=FakeRerank(score=lambda q, d: 0.5), notes=notes)
    full = mem.expand_and_recall("pontoon survey", gate=False)
    assert len(full.items) == 12 and full.dropped_for_tokens == 0
    cut = mem.expand_and_recall("pontoon survey", gate=False, max_tokens=300)
    assert 1 <= len(cut.items) < 12 and cut.dropped_for_tokens == 12 - len(cut.items)
    assert f"({cut.dropped_for_tokens} more records matched less well and not shown.)" in cut.block
    assert [i.source for i in cut.items] == [i.source for i in full.items][:len(cut.items)]    # the best stay
    assert len(mem.expand_and_recall("pontoon survey", gate=False, max_tokens=1).items) == 1  # one is always kept
    mem.set_recall_config({"expansion": {"max_tokens": 300},
                           "rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0}})
    assert len(mem.expand_and_recall("pontoon survey", gate=False).items) == len(cut.items)  # the bank's


# --------------------------------------------------------------------------
# the deadline
# --------------------------------------------------------------------------

def test_a_search_still_running_at_the_deadline_is_dropped(made):
    rerank = FakeRerank(slow=("slow", 2.0))
    mem = made(llm=FakeWriter([S("bridge lamp relay incident"), S("slow pontoon survey"), S("harbour lamps")]),
               rerank=rerank)
    t = time.monotonic()
    got = mem.expand_and_recall("lamps", gate=False, deadline=1.0)
    took = time.monotonic() - t
    assert took < 1.5 and got.deadline_hit
    assert [r.status for r in got.runs] == ["ok", "late", "ok"]
    assert any("search 2" in f and "deadline" in f for f in got.fallbacks)
    assert "slow pontoon survey" not in got.block and all(2 not in (i.found_by or ()) for i in got.items)
    assert got.items and got.error is None
    time.sleep(1.3)                                                      # the late thread ends without touching the result
    assert [r.status for r in got.runs] == ["ok", "late", "ok"]


def test_a_slow_writer_falls_back_to_the_prompt_inside_the_deadline(made):
    mem = made(llm=FakeWriter([S("x")], delay=2.0), rerank=FakeRerank())
    t = time.monotonic()
    got = mem.expand_and_recall("the bridge lamps electric dreams", gate=False, deadline=1.0)
    assert time.monotonic() - t < 1.6
    assert got.searches == [SearchSpec(query="the bridge lamps electric dreams", terms=())]
    assert "writer failed (TimeoutError" in got.fallbacks[0] and got.items


def test_the_deadline_is_configurable_per_call_and_per_bank(made):
    mem = made(llm=FakeWriter([S("slow lamps")]), rerank=FakeRerank(slow=("slow", 1.5)))
    assert mem.expand_and_recall("lamps", gate=False, deadline=0.6).deadline_hit
    assert not mem.expand_and_recall("lamps", gate=False, deadline=5).deadline_hit
    mem.set_recall_config({"expansion": {"deadline_s": 0.6},
                           "rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0}})
    assert mem.expand_and_recall("lamps", gate=False).deadline_hit


# --------------------------------------------------------------------------
# never raises
# --------------------------------------------------------------------------

def test_expand_and_recall_never_raises(made):
    mem = made(llm=FakeWriter([S("harbour lamps")]), rerank=FakeRerank())
    for prompt in (None, 123, b"bytes", "", "   "):
        got = mem.expand_and_recall(prompt, gate=False)
        assert isinstance(got, ExpandedRecall) and got.skipped_reason == "empty prompt" and got.block == ""
    for kw, said in (({"kind": "bogus"}, "kind must be"), ({"date_from": "yesterday"}, "date_from"),
                     ({"deadline": -1}, "deadline"), ({"depth": "abyss"}, "depth"),
                     ({"max_searches": 0}, "max_searches")):
        got = mem.expand_and_recall("lamps", **{"gate": False, **kw})
        assert isinstance(got, ExpandedRecall) and said in got.error and got.block == "" and not got.items
    got = mem.expand_and_recall("harbour lamps", decision=object())        # a decision that is not callable
    assert got.items and any("gate failed" in f for f in got.fallbacks)
    # an embedder that is down: every search fails, the call does not
    mem._embed = lambda texts: (_ for _ in ()).throw(RuntimeError("embedder down"))
    got = mem.expand_and_recall("harbour lamps", gate=False)
    assert got.block == "" and got.items == [] and got.runs[0].status == "error" and got.error is None
    assert any("embedder down" in f for f in got.fallbacks)
    mem._embed = None
    assert "embed" in mem.expand_and_recall("harbour lamps", gate=False).error


def test_a_bank_without_the_hybrid_channel_is_an_error_result_before_any_model_call(made):
    writer = FakeWriter([S("harbour lamps")])
    mem = made(llm=writer, rerank=FakeRerank())
    mem.set_channel_config([{"name": "dense_chunk", "weight": 1}])
    got = mem.expand_and_recall("harbour lamps", gate=False)
    assert "no hybrid_chunk channel" in got.error and got.block == "" and writer.calls == []


def test_every_stage_failing_still_returns(made):
    rerank = FakeRerank(fail="lamps")
    mem = made(llm=FakeWriter(raises=RuntimeError("writer down")), rerank=rerank)
    got = mem.expand_and_recall("the bridge lamps electric dreams", gate=False, decision=None)
    assert got.error is None and got.items and got.runs[0].reranked is False        # Stage 1 order stood
    assert got.fallbacks and all(i.relevance_score > 0 for i in got.items)


def test_a_failing_tracer_does_not_fail_the_recall(made):
    mem = made(llm=FakeWriter([S("harbour lamps")]), rerank=FakeRerank())

    def boom(event, payload):
        raise RuntimeError("tracer down")
    mem._tracer = boom
    assert mem.expand_and_recall(TaskBrief(title="Lamps", ask="Check the lamps"), kind="task").items


# --------------------------------------------------------------------------
# stage 2 not configured, fallback order, accounting and events
# --------------------------------------------------------------------------

def test_without_stage_2_the_stage_1_order_stands_cut_by_relevance(made):
    mem = made(llm=FakeWriter([S("harbour lamps crew")]))                  # no rerank transport
    got = mem.expand_and_recall("harbour lamps", gate=False)
    assert got.items and got.runs[0].reranked is False and got.runs[0].status == "ok"
    assert any("stage 2 (cross_encoder) is not configured" in f for f in got.fallbacks)
    best = max(i.score for i in got.items)
    assert all(i.score >= 0.5 * best for i in got.items) and all(i.relevance_score == i.score for i in got.items)
    mem.set_recall_config({"expansion": {"fallback_min_rel": 0.0}})
    assert len(mem.expand_and_recall("harbour lamps", gate=False).items) >= len(got.items)


def test_accounting_sums_the_gate_the_writer_and_every_rerank(made):
    llm = FakeWriter(raw=None, fn=lambda p: [S("bridge lamp relay incident"), S("pontoon survey due date")])
    wrapped = lambda messages, *, json_mode=False, model=None: LLMResult(
        text=llm(messages, json_mode=json_mode, model=model), model="writer/m", tokens_in=100, tokens_out=40,
        cost_usd=0.001)
    mem = made(llm=wrapped, rerank=FakeRerank(), jev=FakeJev(0.9))
    got = mem.expand_and_recall("what happened to the lamps?")
    assert got.accounting == {"n_llm_calls": 4, "tokens_in": 100 + 7 + 2 * 10, "tokens_out": 41,
                              "cost_usd": pytest.approx(0.001 + 0.00002 + 2 * 0.0001)}
    assert [c["purpose"] for c in got.calls] == ["expand_gate", "expand_searches"] + ["cross_encoder_rerank"] * 2
    assert got.calls[1]["model"] == "writer/m" and got.calls[1]["tokens_out"] == 40
    assert set(got.timings_ms) >= {"gate", "writer", "embed", "stage1", "rerank", "merge", "total", "config"}


def test_a_task_records_its_calls_and_one_recall_event_and_a_chat_does_not_unless_asked(made):
    mem = made(llm=FakeWriter([S("bridge lamp relay incident"), S("harbour lamps")]), rerank=FakeRerank())
    got = mem.expand_and_recall(TaskBrief(title="Lamps", ask="Check the lamps"), kind="task")
    [event] = mem._tracer.by_name("recall")
    assert event["queries"] == got.queries == ["bridge lamp relay incident", "harbour lamps"]
    assert event["n_results"] == len(got.items) and [r["source"] for r in event["results"]] == sources(got)
    assert {e["purpose"] for e in mem._tracer.by_name("llm_call")} == {"expand_searches", "cross_encoder_rerank"}
    assert len(event["channels"]) == 2 and event["rerank"]["per_query"]
    mem._tracer.clear()
    mem.expand_and_recall("harbour lamps", gate=False)
    assert mem._tracer.by_name("recall") == [] and mem._tracer.by_name("llm_call") == []
    mem.expand_and_recall("harbour lamps", gate=False, record=True)
    assert len(mem._tracer.by_name("recall")) == 1
    mem._tracer.clear()
    mem.expand_and_recall(TaskBrief(title="Lamps", ask="Check the lamps"), kind="task", record=False)
    assert mem._tracer.by_name("recall") == []


# --------------------------------------------------------------------------
# recall_for_task: a thin wrapper
# --------------------------------------------------------------------------

BRIEF = dict(title="Show the library tree view", project="atlas",
             ask='Build the tree view for the library; keep "electric dreams" as the demo folder name.',
             done="Large folders page in")


def test_recall_for_task_is_the_entry_point_with_a_task_brief(made):
    writer = FakeWriter([S("library tree view decisions"), S("append-only archive conventions")])
    mem = made(llm=writer, rerank=FakeRerank())
    got = mem.recall_for_task(**BRIEF)
    assert isinstance(got, ExpandedRecall) and got.kind == "task"
    assert got.queries == ["library tree view decisions", "append-only archive conventions"]
    p = writer.calls[0]["prompt"]
    assert "Title: Show the library tree view" in p and "Project: atlas" in p and "Done when: Large folders page in" in p
    assert got.block.startswith(TASK_HEADING) and '"electric dreams"' in got.searches[0].terms     # the guard
    assert got.items and got.block == mem.expand_and_recall(
        TaskBrief(**BRIEF), kind="task").block
    for bad in ({"title": " "}, {"ask": ""}):
        with pytest.raises(ValueError):
            mem.recall_for_task(**{**BRIEF, **bad})
    with pytest.raises(TypeError):
        mem.recall_for_task(**BRIEF, max_facets=4)          # the old facet decomposition is gone
    with pytest.raises(TypeError):
        mem.recall_for_task(**BRIEF, limit=3)               # and so is the count cap


# --------------------------------------------------------------------------
# small units: config validation, the block, the guard on the real prompt
# --------------------------------------------------------------------------

def test_recall_config_accepts_and_checks_the_new_sections():
    validate_recall_config({"stage0_gate": {"enabled": False, "needs_memory_threshold": 0.4},
                            "expansion": {"max_searches": 20, "max_concurrency": 8, "deadline_s": 7,
                                          "max_tokens": 0, "fallback_min_rel": 0.5}})
    for bad in ({"stage0_gate": {"needs_memory_threshold": 2}}, {"stage0_gate": {"enabled": "yes"}},
                {"stage0_gate": {"x": 1}}, {"expansion": {"max_searches": 0}}, {"expansion": {"max_concurrency": 0}},
                {"expansion": {"max_tokens": -1}}, {"expansion": {"deadline_s": 0}},
                {"expansion": {"fallback_min_rel": 1.5}}, {"expansion": {"count": 5}}, {"expansion": []}):
        with pytest.raises(ValueError):
            validate_recall_config(bad)


def test_format_cited_rag_block_with_searches_cite_line_and_found_by():
    def item(n, **kw):
        return RecalledMemory(content="c", original_chunk=f"Text {n}.", source=f"s{n}.md", score=0.1, scores={},
                              metadata={}, bank_id="b", document_id=f"d{n}", created_on="2024-03-05",
                              kind="note", match_kind="exact", **kw)
    items = [item(1, found_by=(1, 3)), item(2)]
    searches = [(1, SearchSpec("lamps", ("v1.14",), None, "2024-04-30")), (3, SearchSpec("survey"))]
    assert format_cited_rag_block(items, heading="## H", searches=searches, cite=True, omitted=2) == (
        "## H\n\n"
        "Searches run (an item's header names the searches that found it):\n"
        "(1) lamps | terms: v1.14 | dates: ... to 2024-04-30\n(3) survey\n\n"
        f"{CITE_LINE}\n\n"
        "[1] source=s1.md | date=2024-03-05 | kind=note | match=exact | found_by=1,3\n    Text 1.\n\n"
        "[2] source=s2.md | date=2024-03-05 | kind=note | match=exact\n    Text 2.\n\n"
        "(2 more records matched less well and not shown.)")
    assert format_cited_rag_block([], searches=searches, cite=True) == ""


def test_the_real_writer_prompt_renders_for_every_shape():
    for kind, last in (("chat", None), ("chat", "Last turn."), ("task", None)):
        p = render_writer_prompt("a prompt with v1.14", kind=kind, agent_last=last, max_searches=20, today=TODAY)
        assert "{{" not in p and "{%" not in p and "a prompt with v1.14" in p
