"""Recall stages after fusion: the cross-encoder rerank (Stage 2), the decision model's holder
(JevScore), the validation of a recall_config. The reranker and Jev are stubs.
(tests/test_legacy_removed.py pins what a bank that stored a config of removed stages does.)"""
from __future__ import annotations

import psycopg
import pytest

from prospecta.memory import Memory
from prospecta.stages import JEV_MODEL, JevScore, validate_recall_config
from tests._stub_embedder import EMBED_DIM, stub_embed

NOTES = {
    "alpha.md": "alpha cat sat on the warm mat",
    "bravo.md": "bravo cat sat near the window",
    "charlie.md": "charlie cat sat under the table",
    "delta.md": "delta dog barked at the moon",
    "echo.md": "echo revenue grew twelve percent",
}
QUERY = "cat sat"


@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=None, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    m.set_recall_config({})        # no stage until a test sets one
    for src, text in NOTES.items():
        m.retain(text, source=src)
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


def cross_encoder_transport(scores, fail=None):
    """Offline /rerank (documented shape): the score of a document is looked up
    by the note name in its text."""
    def transport(req, timeout):
        if fail:
            raise fail
        return {"model": req["model"], "usage": {"total_tokens": 120, "cost": 2.4e-05},
                "results": [{"index": i, "relevance_score": next(
                    (v for k, v in scores.items() if k in d), 0.0)}
                    for i, d in enumerate(req["documents"])]}
    return transport


def test_cross_encoder_stage_through_recall(mem):
    mem._rerank_transport = cross_encoder_transport({"alpha": 0.9, "bravo": 0.6, "charlie": 0.05})
    mem.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder"}})
    res = mem.recall([QUERY], limit=5)
    assert sources(res) == ["alpha.md", "bravo.md"]          # charlie is under the 0.15 floor
    assert res[0].scores["rerank"] == 0.9
    rec, = last_event(mem, "rerank")[0]["per_query"]
    assert rec["stage"] == "cross_encoder" and rec["fallback_reason"] is None
    assert rec["tokens_in"] == 120 and rec["cost_usd"] == 2.4e-05
    assert rec["model"] == "qwen/qwen3-reranker-8b"


def test_cross_encoder_provider_failure_keeps_the_fused_order(mem):
    fused = sources(mem.recall([QUERY], limit=5))
    mem._rerank_transport = cross_encoder_transport({}, fail=TimeoutError("timed out"))
    mem.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder"}})
    assert sources(mem.recall([QUERY], limit=5)) == fused
    rec, = last_event(mem, "rerank")[0]["per_query"]
    assert rec["fallback_reason"].startswith("fused order stands: TimeoutError")


def test_models_are_configurable_per_call_bank_and_env(mem, monkeypatch):
    seen: list[str] = []

    def transport(req, timeout):
        seen.append(req["model"])
        return cross_encoder_transport({"alpha": 0.9})(req, timeout)
    mem._rerank_transport = transport
    monkeypatch.setenv("PROSPECTA_RERANKER_MODEL", "env/rr")
    for cfg_model, call_model in ((None, None), ("bank/rr", None), ("bank/rr", "call/rr")):
        mem._reranker_model = call_model
        mem.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder",
                                          **({"model": cfg_model} if cfg_model else {})}})
        mem.recall([QUERY], limit=5)
    assert seen == ["env/rr", "bank/rr", "call/rr"]


def test_the_decision_model_is_the_call_then_the_bank_then_the_default():
    from prospecta.stages import _with_decision_model
    jev = JevScore(lambda request, timeout: {})
    assert _with_decision_model(jev, None, None) is jev and jev.model == JEV_MODEL == "typesafe/jev-1.14"
    assert _with_decision_model(jev, None, "bank/jev").model == "bank/jev"
    assert _with_decision_model(jev, "call/jev", "bank/jev").model == "call/jev"
    assert jev.model == JEV_MODEL                          # the shared holder is never mutated
    assert _with_decision_model(None, "call/jev", None) is None


def test_validate_recall_config():
    validate_recall_config({})
    validate_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder"}})
    validate_recall_config({"rerank": {"enabled": False}})
    for bad in ({"nope": 1}, {"rerank": {"stage": "x"}}, {"rerank": {"pool": 30}},
                {"rerank": {"enabled": True, "stage": "cross_encoder", "min_rel_score": 2}}, [],
                {"gate": {"enabled": True}}, {"fusion": {"method": "rrf"}}, {"evidence": {}}):
        with pytest.raises(ValueError):
            validate_recall_config(bad)


def test_run_stages_misconfiguration_falls_back_to_fused_order():
    from prospecta.stages import StageDeps, run_stages

    fused = ["a", "b"]       # cross_encoder with no rerank transport
    out, trace = run_stages(None, "q", fused, {"rerank": {"enabled": True, "stage": "cross_encoder"}},
                            StageDeps())
    assert out == fused
    assert trace["fallback_reason"]
    assert trace["n_llm_calls"] == 0
