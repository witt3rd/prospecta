"""What the banks in use return, pinned.

Four fixture banks mirror the ones in use (tests/_live_banks.py): `augur-v2` and `forge-v2` run
`hybrid_chunk` + the cross-encoder, set by hand; `augur` and `forge` are old banks (`channel_config = []`,
question items beside their chunks, the rows of a linker that once ran): an empty config is the hybrid
default, so they search through `hybrid_chunk` and never see a question item. The expected values in
tests/fixtures/retrieval_golden.json were recorded from the code BEFORE the removal of all legacy for the
two hand-set banks (main at 4e2962d; the only differences are the removed `semantic` / `lexical` /
`lexical_body` score keys and the calls that were removed) and from the code after it for the two old
banks. Real Postgres, the stub embedder, a fake reranker and a scripted model.

Regenerate (only when a change is meant to move results): `PROSPECTA_WRITE_GOLDEN=1 pytest
tests/test_retrieval_golden.py`."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tests._live_banks import (
    CORPUS, QUERIES, TODAY, FakeRerank, ScriptedLLM, build_bank, retired_row_counts,
)

GOLDEN = Path(__file__).parent / "fixtures" / "retrieval_golden.json"
LIVE = {"augur": ("augur", True), "forge": ("forge", True), "augur-v2": ("augur", False),
        "forge-v2": ("forge", False)}


def snap(results) -> list[dict]:
    """What a caller sees of each hit. The `jev` score key is left out: it was always 0.0 on a bank
    that reranks with the cross-encoder, and went with the Jev rerank stage."""
    return [{
        "source": r.source, "score": round(r.score, 6),
        "scores": {k: round(float(v), 6) for k, v in sorted((r.scores or {}).items()) if k != "jev"},
        "match_kind": r.match_kind,
        "relevance": None if r.relevance_score is None else round(r.relevance_score, 6),
        "exact_snippet": r.exact_snippet, "created_on": r.created_on, "kind": r.kind,
        "found_by": list(r.found_by) if r.found_by else None,
        "collapsed": list(r.collapsed_sources) if r.collapsed_sources else None,
    } for r in results]


def snap_expanded(res) -> dict:
    return {
        "error": res.error, "skipped": res.skipped,
        "searches": [[s.query, list(s.terms), s.date_from, s.date_to] for s in res.searches],
        "runs": [[r.index, r.status, r.n_candidates, r.n_kept, r.reranked] for r in res.runs],
        "items": snap(res.items), "block": res.block, "fallbacks": list(res.fallbacks),
    }


def scenario(url: str, bank: str) -> dict:
    corpus, legacy = LIVE[bank]
    llm, rerank = ScriptedLLM(), FakeRerank()
    mem = build_bank(url, bank, corpus, legacy=legacy, llm=llm, rerank_transport=rerank)
    try:
        out: dict = {}
        qs = QUERIES[corpus]
        out["search"] = {q: snap(mem.search(q, limit=8)) for q in qs}
        out["recall"] = snap(mem.recall(qs[:3], limit=4))
        out["recall_parallel"] = snap(mem.recall(qs[:3], limit=4, parallel=True))
        out["search_dates"] = snap(mem.search(qs[0], limit=8, date_from="2026-03-01", date_to="2026-04-30"))
        out["search_anti_echo"] = snap(mem.search(
            qs[0], limit=8, excluded_sources=[CORPUS[corpus][0][0], CORPUS[corpus][1][0]]))
        out["search_deep"] = snap(mem.search(qs[4], limit=8, depth="deep"))
        out["expand_chat"] = snap_expanded(mem.expand_and_recall(
            "what did we decide about the lamps and the glaze?", "chat", gate=False, record=False, now=TODAY))
        out["expand_task"] = snap_expanded(mem.recall_for_task(
            title="Check the rollout", ask="Find the error and the decision behind it", record=False, now=TODAY))
        asked = mem.ask(qs[0], limit=4)
        out["ask"] = {"answer": asked.answer, "citations": asked.citations, "items": snap(asked.items),
                      "sufficient": asked.sufficient, "retried": asked.retried}
        out["llm_calls"] = list(llm.calls)
        out["rerank_requests"] = rerank.requests
        return out
    finally:
        mem.close()


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}


@pytest.mark.parametrize("bank", list(LIVE))
def test_a_live_shaped_bank_returns_what_it_returned(bank, fresh_db, golden):
    got = json.loads(json.dumps(scenario(fresh_db, bank), default=str))   # as it is stored
    if os.environ.get("PROSPECTA_WRITE_GOLDEN"):
        data = json.loads(GOLDEN.read_text()) if GOLDEN.exists() else {}
        data[bank] = got
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        return
    assert bank in golden, f"no recorded results for {bank}"
    assert got == golden[bank]


def test_the_fixture_banks_are_what_they_claim(fresh_db):
    """The old fixtures hold the rows of a linker that once ran; the hand-set ones are chunk-only."""
    legacy = build_bank(fresh_db, "augur", "augur", legacy=True)
    hybrid = build_bank(fresh_db, "augur-v2", "augur", legacy=False)
    try:
        assert all(n > 0 for n in retired_row_counts(fresh_db, "augur").values())
        assert not any(retired_row_counts(fresh_db, "augur-v2").values())
    finally:
        legacy.close()
        hybrid.close()
