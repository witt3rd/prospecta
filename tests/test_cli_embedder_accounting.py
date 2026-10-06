"""F4/F6/F10/F14: embedder dimensions, health mismatch, accounted LLM contract, BM25 dir."""
from __future__ import annotations

import argparse
from datetime import datetime

from prospecta._formulate import formulate_queries
from prospecta._index_text import generate_index_text
from prospecta._rag import synthesize
from prospecta.channels import bm25
from prospecta.channels.extract import extract_filters
from prospecta.cli import _common, health
from prospecta.stages import LLMResult


def test_memory_importable_from_package():
    from prospecta import Memory
    from prospecta.memory import Memory as M2
    assert Memory is M2


def test_resolve_embedder_passes_dimensions(monkeypatch):
    from prospecta import defaults
    seen = {}
    monkeypatch.setattr(defaults, "make_default_embedder",
                        lambda **kw: seen.update(kw) or (lambda t: [[0.0]]))
    monkeypatch.delenv("PROSPECTA_EMBEDDER", raising=False)
    _common._resolve_embedder(1536)
    assert seen == {"dimensions": 1536}


def test_default_embedder_is_large_1536(monkeypatch):
    import litellm
    from prospecta.defaults import make_default_embedder
    monkeypatch.delenv("PROSPECTA_EMBED_MODEL", raising=False)
    monkeypatch.delenv("PROSPECTA_EMBED_DIM", raising=False)
    got = {}

    class R:
        data = [{"embedding": [0.0]}]
    monkeypatch.setattr(litellm, "embedding", lambda **kw: got.update(kw) or R())
    make_default_embedder()(["x"])
    assert got["model"] == "openai/text-embedding-3-large" and got["dimensions"] == 1536
    make_default_embedder(dimensions=3072)(["x"])
    assert got["dimensions"] == 3072


def test_health_dim_mismatch_is_down(monkeypatch, capsys):
    monkeypatch.setattr(health, "_check_db", lambda url: "ok")
    monkeypatch.setattr(health, "_bank_dim", lambda args: 1536)
    monkeypatch.setattr(health, "_check_embedder", lambda d=None: ("litellm", 3072))
    args = argparse.Namespace(database_url="x", bank="b")
    assert health.cmd_health(args) == 1
    out = capsys.readouterr().out
    assert out.startswith("PROSPECTA DOWN:") and "3072" in out and "1536" in out
    monkeypatch.setattr(health, "_check_embedder", lambda d=None: ("litellm", 1536))
    assert health.cmd_health(args) == 0


def test_stage_kwargs_need_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert _common._stage_kwargs() == {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-real")
    kw = _common._stage_kwargs()
    assert {"rerank_llm", "synth_llm", "jev", "linker"} <= set(kw)
    monkeypatch.setenv("PROSPECTA_JEV", "off")
    monkeypatch.setenv("PROSPECTA_LINKER", "off")
    assert "jev" not in _common._stage_kwargs() and "linker" not in _common._stage_kwargs()


def _accounted(text):
    return lambda messages, **kw: LLMResult(text=text, model="m", cost_usd=0.5)


def test_every_llm_user_accepts_llmresult():
    lines, _, raw = generate_index_text("body", _accounted("What is it?\nWho?"))
    assert lines == ["What is it?", "Who?"] and isinstance(raw, str)
    qs, out = formulate_queries("hello", llm=_accounted('{"queries":[{"text":"a"}]}'))
    assert [q.text for q in qs] == ["a"] and not out.parse_fallback
    assert synthesize("q", [], _accounted("answer"))[0] == "answer"
    calls: list = []
    f = extract_filters("what did greg say", llm=_accounted('{"people":["Greg"],"hard":true}'),
                        people_vocab=["Greg"], now=datetime(2026, 1, 1), calls=calls)
    assert f.people == ["Greg"] and f.hard
    assert calls[0]["purpose"] == "extract_filters" and calls[0]["cost_usd"] == 0.5


def test_bm25_dir_never_home_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("PROSPECTA_BM25_DIR", raising=False)
    monkeypatch.delenv("PROSPECTA_DATA_DIR", raising=False)
    assert bm25.index_dir() is None
    monkeypatch.setenv("PROSPECTA_DATA_DIR", str(tmp_path))
    assert bm25.index_dir() == tmp_path / "bm25"
    monkeypatch.setenv("PROSPECTA_BM25_DIR", str(tmp_path / "x"))
    assert bm25.index_dir() == tmp_path / "x"
    monkeypatch.delenv("PROSPECTA_BM25_DIR")
    monkeypatch.delenv("PROSPECTA_DATA_DIR")
    assert bm25.InProcessBm25()._path("b") is None
