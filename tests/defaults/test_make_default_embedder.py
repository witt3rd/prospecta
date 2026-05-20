"""Tests for prospecta.defaults.make_default_embedder."""
from __future__ import annotations


def test_returns_callable_with_correct_signature():
    from prospecta.defaults import make_default_embedder

    embed = make_default_embedder()
    assert callable(embed)


def test_embedder_calls_litellm_embedding(monkeypatch):
    """Verify litellm.embedding is called with the expected model."""
    import litellm

    calls = []

    class FakeEmbeddingResponse:
        data = [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]

    def fake_embedding(model=None, input=None, **kwargs):
        calls.append({"model": model, "input": input})
        return FakeEmbeddingResponse()

    monkeypatch.setattr(litellm, "embedding", fake_embedding)

    from prospecta.defaults import make_default_embedder

    embed = make_default_embedder(model="openai/text-embedding-3-small")
    result = embed(["hello", "world"])

    assert calls == [
        {"model": "openai/text-embedding-3-small", "input": ["hello", "world"]}
    ]
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_env_var_resolves_model(monkeypatch):
    monkeypatch.setenv("PROSPECTA_EMBED_MODEL", "openai/text-embedding-3-large")
    import litellm

    captured = {}

    def fake(model=None, **kw):
        captured["model"] = model

        class R:
            data = [{"embedding": [0.0]}]

        return R()

    monkeypatch.setattr(litellm, "embedding", fake)

    from prospecta.defaults import make_default_embedder

    embed = make_default_embedder()
    embed(["x"])
    assert captured["model"] == "openai/text-embedding-3-large"


def test_explicit_model_overrides_env(monkeypatch):
    monkeypatch.setenv("PROSPECTA_EMBED_MODEL", "env/model")
    import litellm

    captured = {}

    def fake(model=None, **kw):
        captured["model"] = model

        class R:
            data = [{"embedding": [0.0]}]

        return R()

    monkeypatch.setattr(litellm, "embedding", fake)

    from prospecta.defaults import make_default_embedder

    embed = make_default_embedder(model="explicit/model")
    embed(["x"])
    assert captured["model"] == "explicit/model"
