"""Recall stages after fusion: Reranker (SonnetListwise, JevScore, Jev gate),
reader and agentic hop, accounting (migration 0009). LLM and Jev are stubs."""
from __future__ import annotations

import json
import re

import psycopg
import pytest

from prospecta.channels import DEFAULT_CHANNEL_CONFIG
from prospecta.memory import Memory
from prospecta.stages import JEV_CRITERIA, JEV_TASK, JevScore, LLMResult, validate_recall_config
from tests._stub_embedder import EMBED_DIM, stub_embed

NOTES = {
    "alpha.md": "alpha cat sat on the warm mat",
    "bravo.md": "bravo cat sat near the window",
    "charlie.md": "charlie cat sat under the table",
    "delta.md": "delta dog barked at the moon",
    "echo.md": "echo revenue grew twelve percent",
}
QUERY = "cat sat"


def vec(text):
    return "[" + ",".join(map(str, stub_embed([text])[0])) + "]"


class StubLLM:
    """Replies by the prompt it is asked: rerank vs reader."""

    def __init__(self, ranking=None, reader=None, fail=False, raw=None):
        self.ranking, self.reader, self.fail, self.raw = ranking, reader, fail, raw
        self.prompts: list[str] = []

    def __call__(self, messages, *, json_mode=False):
        assert json_mode
        text = messages[-1]["content"]
        self.prompts.append(text)
        if self.fail:
            raise RuntimeError("provider down")
        if "ranking notes" in text:
            n = len(re.findall(r"^\[(\d+)\] ", text, re.M))
            if self.raw is not None:
                return self.raw
            ranking = self.ranking(n) if callable(self.ranking) else (self.ranking or list(range(1, n + 1)))
            reply = json.dumps({"grades": {str(i): 3 - min(r, 3) for r, i in enumerate(ranking)},
                                "ranking": ranking})
        else:
            reply = json.dumps(self.reader or {"sufficient": True, "follow_ups": []})
        return LLMResult(reply, model="stub-sonnet", tokens_in=100, tokens_out=10, cost_usd=0.002)


@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=None, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    m.set_channel_config(DEFAULT_CHANNEL_CONFIG)
    for src, text in NOTES.items():
        m.retain(text, source=src, index_text=text)
    with psycopg.connect(fresh_db) as conn:
        for src, text in NOTES.items():
            conn.execute(
                "INSERT INTO memory_items (bank_id, document_id, content, original_chunk,"
                " embedding, kind, ordinal, char_start, char_end) "
                "SELECT 'b', d.id, %s, %s, %s::vector, 'chunk', 0, 0, %s FROM documents d "
                "WHERE d.source = %s", (text, text, vec(text), len(text), src))
        conn.commit()
    yield m
    m.close()


def sources(res):
    return [r.source for r in res]


def last_event(mem, cols):
    with psycopg.connect(mem.database_url) as conn:
        return conn.execute(f"SELECT {cols} FROM recall_events WHERE bank_id='b' "
                            "ORDER BY id DESC LIMIT 1").fetchone()


def test_stages_off_by_default_and_scores_unchanged(mem):
    res = mem.search(QUERY, limit=5)
    assert "rerank" not in res[0].scores and "jev" not in res[0].scores
    with psycopg.connect(mem.database_url) as conn:
        assert conn.execute("SELECT recall_config FROM banks WHERE bank_id='b'").fetchone()[0] == {}


def test_sonnet_listwise_reorders_grades_and_accounts(mem):
    base = mem.search(QUERY, limit=5)
    llm = StubLLM(ranking=lambda n: list(range(n, 0, -1)))   # reverse the fused order
    mem._rerank_llm = llm
    mem.set_recall_config({"rerank": {"enabled": True}})
    res = mem.recall([QUERY], limit=5)
    assert sources(res) == sources(base)[::-1]
    assert all({"rerank", "jev"} <= set(r.scores) for r in res)
    assert res[0].scores["rerank"] == 3.0
    rerank, cost, tin, tout, n = last_event(mem, "rerank, cost_usd, tokens_in, tokens_out, n_llm_calls")
    rec = rerank["per_query"][0]
    assert rec["stage"] == "sonnet_listwise" and rec["fallback_reason"] is None
    assert rec["order"] == [r.document_id for r in res]
    assert rec["n_candidates"] == 5 and rec["model"] == "anthropic/claude-sonnet-5.5"
    assert (float(cost), tin, tout, n) == (0.002, 100, 10, 1)
    with psycopg.connect(mem.database_url) as conn:
        row = conn.execute("SELECT prompt_name, model, tokens_in, tokens_out, cost_usd "
                           "FROM llm_calls WHERE prompt_name='rerank_listwise'").fetchone()
    assert row[:4] == ("rerank_listwise", "stub-sonnet", 100, 10) and float(row[4]) == 0.002


def test_evidence_in_prompt_is_best_chunk_capped_at_1000(mem):
    long = "cat sat " + "x" * 3000
    mem.retain(long, source="long.md", index_text="cat sat long")
    with psycopg.connect(mem.database_url) as conn:
        conn.execute("INSERT INTO memory_items (bank_id, document_id, content, original_chunk,"
                     " embedding, kind, ordinal, char_start, char_end) SELECT 'b', d.id, %s, %s,"
                     " %s::vector, 'chunk', 0, 0, 1 FROM documents d WHERE d.source='long.md'",
                     (long, long, vec(long)))
        conn.commit()
    llm = StubLLM()
    mem._rerank_llm = llm
    mem.set_recall_config({"rerank": {"enabled": True}})
    mem.search("cat sat " + "x" * 3000, limit=5)
    prompt = llm.prompts[0]
    assert "long.md | date:" in prompt
    assert all(len(block) < 1200 for block in prompt.split("\n\n[")[1:])


@pytest.mark.parametrize("llm", [StubLLM(fail=True), StubLLM(raw="no json here"),
                                 StubLLM(raw='{"ranking": "x"}')])
def test_failure_falls_back_to_fused_order_and_says_why(mem, llm):
    base = sources(mem.search(QUERY, limit=5))
    mem._rerank_llm = llm
    mem.set_recall_config({"rerank": {"enabled": True}})
    assert sources(mem.recall([QUERY], limit=5)) == base
    rerank, n = last_event(mem, "rerank, n_llm_calls")
    assert rerank["per_query"][0]["fallback_reason"].startswith("fused order stands")
    assert n == 1


def fake_jev(scores, calls=None, fail=False, key="test-key"):
    """Offline System One, strict like the real door: POST body keys exactly
    {model, state, questions}; answers keyed by question id."""
    def transport(req, timeout):
        assert set(req) == {"model", "state", "questions"}
        assert req["model"] == "typesafe/jev-1.13"
        if calls is not None:
            calls.append(req)
        if fail:
            raise TimeoutError("jev timed out")
        return {"answers": {qid: {"type": "score", "score": scores(i)}
                            for i, qid in enumerate(req["questions"])},
                "id": "gen-dec-1", "model": "typesafe/jev-1.13-20260917",
                "provider": "TypeSafe",
                "usage": {"cost": 2.5e-05, "input_tokens": 596, "output_tokens": 75}}
    return transport


jev_transport = fake_jev


def test_jev_score_as_reranker_in_spire_call_shape(mem):
    seen = []
    # the last candidate scores highest
    mem._jev = JevScore(jev_transport(lambda i: float(i) / 4, seen))
    base = sources(mem.search(QUERY, limit=5))
    mem.set_recall_config({"rerank": {"enabled": True, "stage": "jev_score"}})
    res = mem.recall([QUERY], limit=5)
    assert sources(res) == base[::-1]
    req = seen[0]
    assert req["state"] == {"query": QUERY}
    assert list(req["questions"]) == [f"p{i}" for i in range(len(req["questions"]))]
    assert all(q["type"] == "score" and q["criteria"] == JEV_CRITERIA
               and q["instructions"]["task"] == JEV_TASK
               and q["instructions"]["kind"] == "note" and q["instructions"]["title"]
               and q["instructions"]["text"] for q in req["questions"].values())
    assert res[0].scores["jev"] == max(r.scores["jev"] for r in res)
    n, cost = last_event(mem, "n_llm_calls, cost_usd")
    assert n == 1 and float(cost) == pytest.approx(2.5e-05)


def test_jev_failure_standalone_keeps_fused_order(mem):
    mem._jev = JevScore(jev_transport(None, fail=True))
    base = sources(mem.search(QUERY, limit=5))
    mem.set_recall_config({"rerank": {"enabled": True, "stage": "jev_score"}})
    assert sources(mem.recall([QUERY], limit=5)) == base
    rerank, = last_event(mem, "rerank")
    assert "TimeoutError" in rerank["per_query"][0]["fallback_reason"]


def test_jev_partial_answer_is_a_failure(mem):
    def transport(req, timeout):
        return {"answers": {"p0": {"type": "score", "score": 3}}}
    mem._jev = JevScore(transport)
    mem.set_recall_config({"rerank": {"enabled": True, "stage": "jev_score"}})
    mem.recall([QUERY], limit=5)
    rerank, = last_event(mem, "rerank")
    assert "answers are not exactly the asked ids" in rerank["per_query"][0]["fallback_reason"]


@pytest.mark.parametrize("answers", [
    {"p0": {"score": 1}, "p1": {"score": 1}, "p2": {"score": 1}, "p3": {"score": 1},
     "p4": {"score": 1}, "p99": {"score": 1}},                       # extra id
    {f"p{i}": {"score": 3.5} for i in range(5)},                     # out of range
    {f"p{i}": {"score": float("nan")} for i in range(5)},            # not finite
    {f"p{i}": {"score": "2"} for i in range(5)},                     # not a number
])
def test_jev_invalid_answers_mean_no_ranking(mem, answers):
    mem._jev = JevScore(lambda req, timeout: {"answers": answers})
    base = sources(mem.search(QUERY, limit=5))
    mem.set_recall_config({"rerank": {"enabled": True, "stage": "jev_score"}})
    assert sources(mem.recall([QUERY], limit=5)) == base
    assert last_event(mem, "rerank")[0]["per_query"][0]["fallback_reason"]


def test_jev_response_without_answers_is_no_ranking(mem):
    mem._jev = JevScore(lambda req, timeout: {"error": {"message": "overloaded"}})
    mem.set_recall_config({"rerank": {"enabled": True, "stage": "jev_score"}})
    mem.recall([QUERY], limit=5)
    assert "no answers" in last_event(mem, "rerank")[0]["per_query"][0]["fallback_reason"]


def test_jev_gate_decides_jev_or_sonnet(mem):
    gate = {"rerank": {"enabled": True}, "gate": {"enabled": True, "threshold": 2.95}}
    base = sources(mem.search(QUERY, limit=5))
    llm = StubLLM(ranking=lambda n: list(range(n, 0, -1)))
    mem._rerank_llm = llm
    mem.set_recall_config(gate)

    mem._jev = JevScore(jev_transport(lambda i: 3.0 - 0.1 * i))      # confident: Jev order = fused
    assert sources(mem.recall([QUERY], limit=5)) == base
    rec = last_event(mem, "rerank")[0]["per_query"][0]
    assert rec["gate"]["decision"] == "jev" and llm.prompts == []

    mem._jev = JevScore(jev_transport(lambda i: 1.0))                # not confident: Sonnet reranks
    assert sources(mem.recall([QUERY], limit=5)) == base[::-1]
    rec = last_event(mem, "rerank")[0]["per_query"][0]
    assert rec["gate"]["decision"] == "sonnet" and rec["stage"] == "sonnet_listwise"
    assert len(llm.prompts) == 1 and rec["jev"]

    mem._jev = JevScore(jev_transport(None, fail=True))              # Jev down: Sonnet reranks
    assert sources(mem.recall([QUERY], limit=5)) == base[::-1]
    assert last_event(mem, "rerank")[0]["per_query"][0]["gate"]["decision"] == "jev_failed"


def test_jev_requests_are_15_per_call_and_two_for_a_pool_of_30():
    from prospecta.channels.base import Candidate
    from prospecta.channels.fusion import FusedDoc
    from prospecta.stages import Item
    items = []
    for i in range(30):
        c = Candidate(f"d{i}", None, f"s{i}", "x", i + 1, 0.0, "e", {})
        items.append(Item(FusedDoc(f"d{i}", f"s{i}", 1.0, c), f"s{i}", "e" * 100))
    calls: list[dict] = []
    reqs: list[dict] = []
    s = JevScore(jev_transport(lambda i: 1.0, reqs)).score("q", items, calls)
    assert [len(r["questions"]) for r in reqs] == [15, 15] and len(s) == 30 and len(calls) == 2
    # the 60 KB request cap splits a large batch further
    big = [Item(it.doc, it.header, "e" * 2400) for it in items]
    reqs.clear()
    JevScore(jev_transport(lambda i: 1.0, reqs)).score("q", big, [])
    assert all(len(json.dumps(r).encode()) <= 60_000 for r in reqs) and sum(
        len(r["questions"]) for r in reqs) == 30


def test_reader_sufficient_runs_no_hop(mem):
    llm = StubLLM(reader={"sufficient": True, "follow_ups": ["ignored"]})
    mem._rerank_llm = llm
    mem.set_recall_config({"rerank": {"enabled": True}, "reader": {"enabled": True}})
    mem.recall([QUERY], limit=5)
    hops, n = last_event(mem, "hops, n_llm_calls")
    h = hops["per_query"][0]
    assert h["verdict"] == "sufficient" and h["follow_ups"] == [] and n == 2


def test_reader_follow_ups_join_new_notes_and_rerank_again(mem):
    # a top-1 pool: the follow-up must bring in notes outside the top 1
    llm = StubLLM(reader={"sufficient": False,
                          "follow_ups": ["dog barked moon", "revenue grew", "third is dropped"]})
    mem._rerank_llm = llm
    mem.set_recall_config({"rerank": {"enabled": True},
                           "reader": {"enabled": True, "join_top": 1, "top": 1, "max_new": 2}})
    res = mem.recall([QUERY], limit=10)
    hops, n = last_event(mem, "hops, n_llm_calls")
    h = hops["per_query"][0]
    assert h["verdict"] == "follow_up" and h["follow_ups"] == ["dog barked moon", "revenue grew"]
    assert len(h["new_candidates"]) == 2 and h["rerank"]["n_candidates"] == 3
    assert n == 3   # first rerank, reader, rerank of the joined set
    assert len({r.document_id for r in res}) == len(res)
    assert {"delta.md", "echo.md"} <= set(sources(res))


def test_reader_failure_is_recorded_and_order_stands(mem):
    class Bad(StubLLM):
        def __call__(self, messages, *, json_mode=False):
            if "ranking notes" in messages[-1]["content"]:
                return super().__call__(messages, json_mode=json_mode)
            raise RuntimeError("reader down")
    mem._rerank_llm = Bad()
    mem.set_recall_config({"rerank": {"enabled": True}, "reader": {"enabled": True}})
    assert mem.recall([QUERY], limit=5)
    h = last_event(mem, "hops")[0]["per_query"][0]
    assert h["verdict"] == "error" and "reader down" in h["error"]


def test_validate_recall_config():
    validate_recall_config({})
    validate_recall_config({"rerank": {"enabled": True}, "gate": {"enabled": True}})
    for bad in ({"nope": 1}, {"rerank": {"stage": "x"}}, {"gate": {"enabled": True}},
                {"rerank": {"enabled": True, "stage": "jev_score"}, "gate": {"enabled": True}},
                {"rerank": {"pool": 0}}, []):
        with pytest.raises(ValueError):
            validate_recall_config(bad)


def test_run_stages_misconfiguration_falls_back_to_fused_order():
    from prospecta.stages import StageDeps, run_stages

    fused = ["a", "b"]
    out, trace = run_stages(None, "q", fused, [], {"rerank": {"enabled": True}}, StageDeps())
    assert out == fused
    assert trace["fallback_reason"]
    assert trace["n_llm_calls"] == 0
