"""Recall depth: standard (rerank pool cut 0.15) vs deep (0.05)."""
from __future__ import annotations

import json

import pytest

from prospecta import _scorecut, evaluation as ev, stages
from prospecta.stages import validate_recall_config
from tests.test_stages import QUERY, StubLLM, last_event, mem  # noqa: F401  (fixture)


def test_pool_cut_precedence():
    assert _scorecut.DEPTHS == {"standard": 0.15, "deep": 0.05}
    assert _scorecut.pool_min_rel(None, {}) == 0.15
    assert _scorecut.pool_min_rel("deep", {}) == 0.05
    assert _scorecut.pool_min_rel(None, {"depth": "deep"}) == 0.05
    assert _scorecut.pool_min_rel("standard", {"depth": "deep"}) == 0.15      # per call wins
    assert _scorecut.pool_min_rel(None, {"rerank": {"min_rel_score": 0.3}}) == 0.3
    assert _scorecut.pool_min_rel("deep", {"rerank": {"min_rel_score": 0.3}}) == 0.05
    with pytest.raises(ValueError):
        _scorecut.check_depth("shallow")


def test_validate_depth_in_recall_config():
    validate_recall_config({"depth": "deep"})
    with pytest.raises(ValueError):
        validate_recall_config({"depth": "medium"})


@pytest.fixture
def cuts(monkeypatch):
    seen = []
    real = stages.rel_cut
    monkeypatch.setattr(stages, "rel_cut", lambda items, score, rel: (
        seen.append(rel) or real(items, score, rel)))
    return seen


def test_recall_depth_per_call_bank_default_and_trace(mem, cuts):
    mem._rerank_llm = StubLLM()
    mem.set_recall_config({"rerank": {"enabled": True}})
    mem.recall([QUERY], limit=5)
    assert cuts[-1] == 0.15 and last_event(mem, "depth")[0] == "standard"
    mem.recall([QUERY], limit=5, depth="deep")
    assert cuts[-1] == 0.05 and last_event(mem, "depth")[0] == "deep"
    mem.set_recall_config({"depth": "deep", "rerank": {"enabled": True}})
    mem.recall([QUERY], limit=5)
    assert cuts[-1] == 0.05 and last_event(mem, "depth")[0] == "deep"
    mem.recall([QUERY], limit=5, depth="standard")
    assert cuts[-1] == 0.15 and last_event(mem, "depth")[0] == "standard"
    with pytest.raises(ValueError):
        mem.recall([QUERY], depth="shallow")


def test_search_accepts_depth(mem, cuts):
    mem._rerank_llm = StubLLM()
    mem.set_recall_config({"rerank": {"enabled": True}})
    mem.search(QUERY, limit=5, depth="deep")
    assert cuts[-1] == 0.05


def test_grounded_set_question_defaults_to_deep(mem, cuts):
    mem._rerank_llm = StubLLM()
    mem.set_recall_config({"rerank": {"enabled": True}})
    mem._llm = lambda *a, **k: json.dumps({"answer": "x", "citations": []})
    try:
        mem.recall_synth(QUERY, grounded=True, scope=["alpha.md", "bravo.md"])
    except Exception:
        pass  # the stub reply need not be a valid grounded answer; the recall ran
    assert cuts[-1] == 0.05
    try:
        mem.recall_synth(QUERY, grounded=True, scope=["alpha.md"], depth="standard")
    except Exception:
        pass
    assert cuts[-1] == 0.15


def test_eval_reports_and_uses_depth(mem, cuts):
    mem._rerank_llm = StubLLM()
    mem.set_recall_config({"rerank": {"enabled": True}})
    qs = ev.parse_questions("## Q1\nquestion: cat sat\ngold: alpha.md\n")
    rep = ev.run_eval(mem, qs)
    assert rep["depth"] == "standard" and "depth: standard" in ev.format_report(rep)
    rep = ev.run_eval(mem, qs, depth="deep")
    assert rep["depth"] == "deep" and cuts[-1] == 0.05
    assert "rerank pool >= 0.05" in ev.format_report(rep) and "depth: deep" in ev.format_report(rep)
    assert json.loads(ev.report_json(rep))["depth"] == "deep"
