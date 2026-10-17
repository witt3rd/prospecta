"""A date-only question (no searchable words, a date window) is a listing, newest first: the
cross-encoder is not called, no cost, the Stage 1 order stands. A question with words still reranks."""
from __future__ import annotations

import pytest

from tests.test_expand import NOTES, FakeRerank, FakeWriter, S, build, sources

WINDOW = {"date_from": "2024-01-01", "date_to": "2024-12-31"}
NEWEST_FIRST = ["phrase.md", "lamps-incident.md", "lamps-decision.md", "survey.md"]   # 2024 notes by date


@pytest.fixture
def made(fresh_db):
    mems = []

    def make(**kw):
        m = build(fresh_db, **kw)
        mems.append(m)
        return m
    yield make
    for m in mems:
        m.close()


def test_single_search_date_only_makes_no_rerank_call_and_keeps_newest_first(made):
    rerank = FakeRerank(score=lambda q, d: 0.01)      # would cut everything under any floor
    mem = made(rerank=rerank, rerank_cfg={"min_score": 0.9, "retries": 3})
    trace: list[dict] = []
    res = mem.search("the and of", limit=10, _trace=trace, **WINDOW)
    assert rerank.requests == []
    assert [r.source for r in res] == NEWEST_FIRST
    assert {r.match_kind for r in res} == {"date"}
    rec = trace[0]["rerank"]
    assert rec["skipped"] is True and rec["fallback_reason"] == "date-only listing: nothing to rank"
    assert rec["n_llm_calls"] == 0 and rec["cost_usd"] in (0, 0.0, None)


def test_mixed_search_still_reranks(made):
    rerank = FakeRerank()
    mem = made(rerank=rerank)
    res = mem.search("harbour lamps", limit=10, **WINDOW)
    assert rerank.requests and res


def test_expansion_date_only_search_skips_and_is_not_a_fallback(made):
    rerank = FakeRerank(score=lambda q, d: 0.01)
    mem = made(llm=FakeWriter([S("the and of", df="2024-01-01", dt="2024-12-31")]), rerank=rerank,
               rerank_cfg={"min_score": 0.9})
    got = mem.expand_and_recall("what happened this year", gate=False)
    assert rerank.requests == []
    assert sources(got) == NEWEST_FIRST
    run = got.runs[0]
    assert run.status == "ok" and run.reranked is False
    assert run.fallback_reason == "date-only listing: nothing to rank"
    assert got.fallbacks == [] and got.accounting["n_llm_calls"] == 1      # the writer only


def test_expansion_mixed_searches_rerank_only_the_worded_one(made):
    rerank = FakeRerank()
    mem = made(llm=FakeWriter([S("the and of", df="2024-01-01", dt="2024-12-31"),
                               S("bridge lamp relay incident")]), rerank=rerank)
    got = mem.expand_and_recall("lamps this year", gate=False)
    assert rerank.queries() == ["bridge lamp relay incident"]
    assert [r.reranked for r in got.runs] == [False, True] and got.fallbacks == []
