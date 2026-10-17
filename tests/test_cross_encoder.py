"""Stage 2 cross-encoder reranker (stage 'cross_encoder', OpenRouter /rerank):
ordering, cuts, near-copy collapse, clean fallback, accounting and the model
configuration precedence. The transport is a fake of the documented shape."""
from __future__ import annotations

import json
import subprocess
import sys
import urllib.error

import pytest

from prospecta.channels.base import Candidate
from prospecta.channels.fusion import FusedDoc
from prospecta.stages import (
    CrossEncoderReranker, Item, StageDeps, build_reranker, resolve_reranker_model, run_reranker,
    validate_recall_config, _with_decision_model, JevScore,
)

QUERY = "who fixed the build"


def make_items(texts):
    items = []
    for i, text in enumerate(texts):
        c = Candidate(f"d{i}", None, f"s{i}", "x", i + 1, 0.0, text, {})
        items.append(Item(FusedDoc(f"d{i}", f"s{i}", 1.0 / (i + 1), c), f"s{i}", text))
    return items


def fake_rerank(scores, reqs=None, usage=None):
    """Offline /rerank of the documented shape: results echo index and score."""
    def transport(req, timeout):
        assert set(req) == {"model", "query", "documents", "top_n"}
        if reqs is not None:
            reqs.append((req, timeout))
        results = [{"index": i, "relevance_score": scores(d), "document": {"text": d}}
                   for i, d in enumerate(req["documents"])]
        results.sort(key=lambda r: -r["relevance_score"])
        return {"id": "gen-rerank-1", "model": "accounts/fireworks/models/qwen3-reranker-8b",
                "provider": "Fireworks", "results": results[:req["top_n"]],
                "usage": usage if usage is not None else {"total_tokens": 334, "cost": 6.68e-05}}
    return transport


def by_marker(text):
    """Score from a '=0.8' marker in the document text."""
    return float(text.rsplit("=", 1)[1])


def ids(items, out):
    return [items[i].doc.document_id for i in out.order]


def test_success_orders_by_relevance_and_sends_the_documented_request():
    items = make_items(["a note=0.2", "b note=0.9", "c note=0.5"])
    reqs: list = []
    out = CrossEncoderReranker(fake_rerank(by_marker, reqs)).rerank(QUERY, items, [])
    assert ids(items, out) == ["d1", "d2", "d0"]
    assert out.grades == {"d1": 0.9, "d2": 0.5, "d0": 0.2}
    (req, timeout), = reqs
    assert req["model"] == "qwen/qwen3-reranker-8b" and timeout == 2.5
    assert req["query"].startswith("Instruct: ") and req["query"].endswith("\nQuery: " + QUERY)
    assert req["documents"] == ["s0\na note=0.2", "s1\nb note=0.9", "s2\nc note=0.5"]
    assert req["top_n"] == 3


def test_top_n_and_min_score_cut():
    items = make_items([f"n{i}=0.{9 - i}" for i in range(9)])   # 0.9 .. 0.1
    out = CrossEncoderReranker(fake_rerank(by_marker), top_n=3).rerank(QUERY, items, [])
    assert ids(items, out) == ["d0", "d1", "d2"]
    out = CrossEncoderReranker(fake_rerank(by_marker), min_score=0.5).rerank(QUERY, items, [])
    assert ids(items, out) == ["d0", "d1", "d2", "d3", "d4"]
    assert out.record["n_below_min_score"] == 4
    default = CrossEncoderReranker(fake_rerank(by_marker)).rerank(QUERY, items, [])
    assert len(default.order) == 8   # top_n 10 leaves room; the 0.15 floor drops 0.1


def test_five_near_copies_collapse_to_the_highest_scored_and_distinct_notes_stay():
    base = ("weekly summary the build pipeline was fixed by sam after the cache failed "
            "while mia reviewed logs and omar rolled back release nine then everyone agreed "
            "to pin the runner image and add a nightly canary job for the flaky suite")
    copies = [f"{base} run {i}=0.{5 + i}" for i in range(5)]       # scores .5 .. .9
    distinct = ["the garden needs water on friday=0.6", "invoice 4417 was paid=0.7"]
    items = make_items(copies + distinct)
    out = CrossEncoderReranker(fake_rerank(by_marker)).rerank(QUERY, items, [])
    assert ids(items, out) == ["d4", "d6", "d5"]          # best copy 0.9, then 0.7, 0.6
    assert out.record["collapsed"] == {"d0": "d4", "d1": "d4", "d2": "d4", "d3": "d4"}
    off = CrossEncoderReranker(fake_rerank(by_marker), deduplicate=False).rerank(QUERY, items, [])
    assert len(off.order) == 7 and off.record["collapsed"] == {}


def test_stage_never_adds_candidates_and_keeps_only_cut_survivors():
    items = make_items([f"note {i} unique words {i * 7919}=0.{i}" for i in range(1, 8)])
    out = CrossEncoderReranker(fake_rerank(by_marker), top_n=4).rerank(QUERY, items, [])
    kept = set(out.order)
    assert kept <= set(range(len(items))) and len(kept) == len(out.order) == 4


@pytest.mark.parametrize("fail, reason", [
    (TimeoutError("timed out"), "TimeoutError"),
    (urllib.error.HTTPError("u", 503, "service overloaded", {}, None), "HTTPError"),
    (urllib.error.HTTPError("u", 429, "rate limited", {}, None), "HTTPError"),
    (json.JSONDecodeError("bad", "{", 0), "JSONDecodeError"),
])
def test_provider_failure_falls_back_to_the_fused_order(fail, reason):
    def transport(req, timeout):
        raise fail
    items = make_items(["a=0.1", "b=0.9", "c=0.5"])
    calls: list = []
    ordered, out, rec = run_reranker(CrossEncoderReranker(transport), QUERY, items, calls)
    assert out is None and ordered == items            # every candidate, fused order
    assert rec["fallback_reason"].startswith("fused order stands: " + reason)
    assert rec["model"] == "qwen/qwen3-reranker-8b"
    assert calls[0]["error"].startswith(reason) and calls[0]["duration_ms"] >= 0


@pytest.mark.parametrize("resp", [
    "not json", [], {}, {"results": "x"},
    {"results": [{"index": 0, "relevance_score": 0.5}]},                       # not every candidate
    {"results": [{"index": 0, "relevance_score": 0.5}, {"index": 0, "relevance_score": 0.4}]},
    {"results": [{"index": 0, "relevance_score": 0.5}, {"index": 7, "relevance_score": 0.4}]},
    {"results": [{"index": 0, "relevance_score": "high"}, {"index": 1, "relevance_score": 0.4}]},
    {"results": [{"index": 0, "relevance_score": float("nan")}, {"index": 1, "relevance_score": 0.4}]},
    {"results": [{"index": True, "relevance_score": 0.5}, {"index": 1, "relevance_score": 0.4}]},
])
def test_malformed_response_falls_back(resp):
    items = make_items(["a=0.1", "b=0.9"])
    ordered, out, rec = run_reranker(CrossEncoderReranker(lambda req, t: resp), QUERY, items, [])
    assert out is None and ordered == items
    assert rec["fallback_reason"].startswith("fused order stands: ")


def test_all_below_floor_is_a_real_answer_not_a_fallback():
    items = make_items(["a=0.01", "b=0.02"])
    ordered, out, rec = run_reranker(CrossEncoderReranker(fake_rerank(by_marker)), QUERY, items, [])
    assert ordered == [] and rec["fallback_reason"] is None


def test_accounting_lands_in_the_call_and_the_stage_record():
    items = make_items(["a=0.1", "b=0.9"])
    calls: list = []
    _, _, rec = run_reranker(CrossEncoderReranker(fake_rerank(by_marker)), QUERY, items, calls)
    c, = calls
    assert c["purpose"] == "cross_encoder_rerank" and c["error"] is None
    assert c["model"] == "accounts/fireworks/models/qwen3-reranker-8b"
    assert c["tokens_in"] == 334 and c["cost_usd"] == 6.68e-05 and c["duration_ms"] >= 0
    assert rec["stage"] == "cross_encoder" and rec["fallback_reason"] is None
    assert rec["tokens_in"] == 334 and rec["cost_usd"] == 6.68e-05 and rec["n_llm_calls"] == 1
    assert rec["latency_ms"] >= 0 and rec["n_candidates"] == 2
    assert rec["model"] == "qwen/qwen3-reranker-8b"


def test_usage_may_be_absent():
    calls: list = []
    CrossEncoderReranker(fake_rerank(by_marker, usage={})).rerank(
        QUERY, make_items(["a=0.5"]), calls)
    assert calls[0]["tokens_in"] is None and calls[0]["cost_usd"] is None


def test_large_pools_split_into_requests_that_fit_the_context():
    from prospecta._scorecut import estimate_tokens
    from prospecta.stages import RERANK_BATCH_TOKENS
    items = make_items([f"unique {i} {'w' * 4000}=0.{i % 8 + 2}" for i in range(60)])
    reqs: list = []
    out = CrossEncoderReranker(fake_rerank(by_marker, reqs), top_n=60).rerank(QUERY, items, [])
    assert len(reqs) > 1 and sum(len(r["documents"]) for r, _ in reqs) == 60
    assert all(sum(estimate_tokens(d) for d in r["documents"]) <= RERANK_BATCH_TOKENS
               for r, _ in reqs)
    assert len(out.order) == 60            # every note was scored; scores merge across requests
    assert [out.grades[items[i].doc.document_id] for i in out.order] == sorted(
        out.grades.values(), reverse=True)


# --------------------------------------------------------------- configuration

def test_reranker_model_precedence(monkeypatch):
    monkeypatch.delenv("PROSPECTA_RERANKER_MODEL", raising=False)
    assert resolve_reranker_model() == "qwen/qwen3-reranker-8b"
    monkeypatch.setenv("PROSPECTA_RERANKER_MODEL", "env/model")
    assert resolve_reranker_model() == "env/model"
    assert resolve_reranker_model(None, "bank/model") == "bank/model"
    assert resolve_reranker_model("call/model", "bank/model") == "call/model"


def test_decision_model_default_env_and_alias():
    def model(**env):
        out = subprocess.run(
            [sys.executable, "-c", "from prospecta.stages import JEV_MODEL; print(JEV_MODEL)"],
            env={"PATH": "", **env}, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    assert model() == "typesafe/jev-1.14"
    assert model(PROSPECTA_JEV_MODEL="alias/jev") == "alias/jev"
    assert model(PROSPECTA_DECISION_MODEL="env/decision") == "env/decision"
    assert model(PROSPECTA_DECISION_MODEL="env/decision", PROSPECTA_JEV_MODEL="alias/jev") == "env/decision"


def test_decision_model_call_beats_bank_beats_the_jev_default():
    jev = JevScore(lambda req, t: {})
    assert _with_decision_model(jev, None, None) is jev
    assert _with_decision_model(jev, None, "bank/jev").model == "bank/jev"
    assert _with_decision_model(jev, "call/jev", "bank/jev").model == "call/jev"
    assert jev.model == "typesafe/jev-1.14"          # the shared Jev is never mutated
    assert _with_decision_model(None, "call/jev", None) is None


def rr_cfg(**kw):
    return {"rerank": {"enabled": True, "stage": "cross_encoder", **kw}}


def test_build_reranker_resolves_model_and_settings(monkeypatch):
    monkeypatch.setenv("PROSPECTA_RERANKER_MODEL", "env/model")
    t = fake_rerank(by_marker)
    rr = build_reranker(rr_cfg(), llm=None, jev=None, rerank_transport=t)
    assert (type(rr), rr.model, rr.top_n, rr.min_score, rr.timeout, rr.deduplicate) == (
        CrossEncoderReranker, "env/model", 10, 0.15, 2.5, True)
    rr = build_reranker(rr_cfg(model="bank/model", top_n=4, min_score=0.3, timeout_s=1.0,
                               deduplicate_near_copies=False),
                        llm=None, jev=None, rerank_transport=t)
    assert (rr.model, rr.top_n, rr.min_score, rr.timeout, rr.deduplicate) == (
        "bank/model", 4, 0.3, 1.0, False)
    rr = build_reranker(rr_cfg(model="bank/model"), llm=None, jev=None, rerank_transport=t,
                        reranker_model="call/model")
    assert rr.model == "call/model"


def test_cross_encoder_needs_a_transport():
    with pytest.raises(RuntimeError, match="rerank transport"):
        build_reranker(rr_cfg(), llm=None, jev=None)


def test_validate_cross_encoder_config():
    validate_recall_config({**rr_cfg(model="m", top_n=5, min_score=0.2, timeout_s=3,
                                     deduplicate_near_copies=True),
                            "models": {"decision_model": "m"}})
    for bad in (rr_cfg(top_n=0), rr_cfg(top_n=True), rr_cfg(min_score=2), rr_cfg(timeout_s=0),
                rr_cfg(timeout_s="1"), rr_cfg(model=""), rr_cfg(deduplicate_near_copies="yes"),
                {"models": {"nope": "x"}}, {"models": {"decision_model": 3}}, {"models": []},
                {"rerank": {"enabled": True, "stage": "cross_encoder"}, "gate": {"enabled": True}}):
        with pytest.raises(ValueError):
            validate_recall_config(bad)


def test_stagedeps_carries_the_call_level_overrides():
    d = StageDeps(reranker_model="r", decision_model="d", rerank_transport=None)
    assert (d.reranker_model, d.decision_model, d.rerank_transport) == ("r", "d", None)


def test_near_copy_threshold_is_above_088():
    ten = " ".join(f"w{i}" for i in range(10))
    eight_of_ten = " ".join(f"w{i}" for i in range(8)) + " x1 x2"      # Jaccard 8/12
    items = make_items([f"{ten}=0.9", f"{eight_of_ten}=0.8"])
    out = CrossEncoderReranker(fake_rerank(by_marker)).rerank(QUERY, items, [])
    assert ids(items, out) == ["d0", "d1"] and out.record["collapsed"] == {}
