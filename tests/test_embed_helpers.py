"""Tests for prospecta.embed minimal-deps helpers."""
from __future__ import annotations

import sys

import pytest


def test_all_three_helpers_importable():
    """Smoke test that prospecta.embed exposes all three factories."""
    from prospecta.embed import openai, openai_compatible, sentence_transformers
    assert callable(sentence_transformers)
    assert callable(openai)
    assert callable(openai_compatible)


def test_sentence_transformers_import_error_when_missing(monkeypatch):
    """If sentence-transformers is not installed, ImportError has install hint."""
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    from prospecta.embed import sentence_transformers
    with pytest.raises(ImportError, match=r"prospecta\[embed-sentence-transformers\]"):
        sentence_transformers("all-MiniLM-L6-v2")


def test_openai_import_error_when_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "openai", None)
    from prospecta.embed import openai
    with pytest.raises(ImportError, match=r"prospecta\[embed-openai\]"):
        openai(api_key="x")


def test_openai_raises_when_no_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    from prospecta.embed import openai
    with pytest.raises(RuntimeError, match="API key"):
        openai()


def test_openai_uses_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    captured = {}

    class FakeEmbeddings:
        def create(self, model, input):
            captured["model"] = model
            captured["input"] = input
            return type("R", (), {"data": [type("D", (), {"embedding": [0.1] * 4})()]})()

    class FakeClient:
        def __init__(self, *, api_key, **kw):
            captured["api_key"] = api_key
            self.embeddings = FakeEmbeddings()

    import openai as openai_pkg
    monkeypatch.setattr(openai_pkg, "OpenAI", FakeClient)

    from prospecta.embed import openai as openai_helper
    embed = openai_helper()
    result = embed(["hello"])

    assert captured["api_key"] == "sk-from-env"
    assert captured["model"] == "text-embedding-3-small"
    assert captured["input"] == ["hello"]
    assert result == [[0.1] * 4]


def test_openai_explicit_api_key_wins(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    captured = {}

    class FakeClient:
        def __init__(self, *, api_key, **kw):
            captured["api_key"] = api_key
            self.embeddings = type("E", (), {
                "create": lambda self, model, input: type(
                    "R", (), {"data": [type("D", (), {"embedding": [0.0]})()]}
                )()
            })()

    import openai as openai_pkg
    monkeypatch.setattr(openai_pkg, "OpenAI", FakeClient)

    from prospecta.embed import openai as openai_helper
    embed = openai_helper(api_key="sk-explicit")
    embed(["x"])
    assert captured["api_key"] == "sk-explicit"


def test_openai_compatible_passes_base_url(monkeypatch):
    captured = {}

    class FakeEmbeddings:
        def create(self, model, input):
            captured["model"] = model
            return type("R", (), {"data": [type("D", (), {"embedding": [0.0]})()]})()

    class FakeClient:
        def __init__(self, *, api_key, base_url, timeout, **kw):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["timeout"] = timeout
            self.embeddings = FakeEmbeddings()

    import openai as openai_pkg
    monkeypatch.setattr(openai_pkg, "OpenAI", FakeClient)

    from prospecta.embed import openai_compatible
    embed = openai_compatible(base_url="http://localhost:7997/v1", model="bge-small-en")
    embed(["x"])

    assert captured["base_url"] == "http://localhost:7997/v1"
    assert captured["api_key"] == "sk-not-needed"
    assert captured["timeout"] == 30.0
    assert captured["model"] == "bge-small-en"


def test_openai_compatible_explicit_api_key(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, *, api_key, base_url, timeout, **kw):
            captured["api_key"] = api_key
            self.embeddings = type("E", (), {
                "create": lambda self, model, input: type(
                    "R", (), {"data": [type("D", (), {"embedding": [0.0]})()]}
                )()
            })()

    import openai as openai_pkg
    monkeypatch.setattr(openai_pkg, "OpenAI", FakeClient)

    from prospecta.embed import openai_compatible
    embed = openai_compatible(base_url="http://x", model="m", api_key="custom-token")
    embed(["x"])

    assert captured["api_key"] == "custom-token"
