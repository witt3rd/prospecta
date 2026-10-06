"""Tests for prospecta.defaults.make_default_llm."""
from __future__ import annotations


def test_returns_callable():
    from prospecta.defaults import make_default_llm

    llm = make_default_llm()
    assert callable(llm)


def test_llm_calls_litellm_completion(monkeypatch):
    import litellm

    calls = []

    class FakeMsg:
        content = "the response"

    class FakeChoice:
        message = FakeMsg()

    class FakeResponse:
        choices = [FakeChoice()]

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    monkeypatch.setattr(litellm, "completion", fake_completion)

    from prospecta.defaults import make_default_llm

    llm = make_default_llm(model="openai/gpt-4o-mini")
    result = llm(messages=[{"role": "user", "content": "hi"}])

    assert len(calls) == 1
    assert calls[0]["model"] == "openai/gpt-4o-mini"
    assert calls[0]["messages"] == [{"role": "user", "content": "hi"}]
    assert result == "the response"


def test_llm_json_mode_passes_response_format(monkeypatch):
    import litellm

    captured = {}

    class FakeResponse:
        choices = [
            type("C", (), {"message": type("M", (), {"content": '{"ok": true}'})})()
        ]

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(litellm, "completion", fake_completion)

    from prospecta.defaults import make_default_llm

    llm = make_default_llm()
    llm(messages=[{"role": "user", "content": "x"}], json_mode=True)

    assert captured.get("response_format") == {"type": "json_object"}


def test_llm_no_json_mode_does_not_set_response_format(monkeypatch):
    import litellm

    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return type(
            "R",
            (),
            {"choices": [type("C", (), {"message": type("M", (), {"content": "plain"})})()]},
        )()

    monkeypatch.setattr(litellm, "completion", fake_completion)

    from prospecta.defaults import make_default_llm

    llm = make_default_llm()
    llm(messages=[{"role": "user", "content": "x"}])

    assert "response_format" not in captured or captured.get("response_format") is None


def test_llm_env_var_resolves_model(monkeypatch):
    monkeypatch.setenv("PROSPECTA_LLM_MODEL", "anthropic/claude-3-5-haiku")
    import litellm

    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return type(
            "R",
            (),
            {"choices": [type("C", (), {"message": type("M", (), {"content": "x"})})()]},
        )()

    monkeypatch.setattr(litellm, "completion", fake_completion)

    from prospecta.defaults import make_default_llm

    llm = make_default_llm()
    llm(messages=[{"role": "user", "content": "x"}])

    assert captured["model"] == "anthropic/claude-3-5-haiku"


def test_per_call_model_overrides_bank_model(monkeypatch):
    import litellm

    calls = []

    class FakeResponse:
        choices = [type("C", (), {"message": type("M", (), {"content": "x"})})()]

    def fake_completion(**kwargs):
        calls.append(kwargs)
        return FakeResponse()

    monkeypatch.setattr(litellm, "completion", fake_completion)

    from prospecta.channels.extract import _accepts_model
    from prospecta.defaults import make_default_llm

    llm = make_default_llm(model="openai/gpt-4o-mini")
    assert _accepts_model(llm)
    llm(messages=[], model="anthropic/claude-sonnet-5.5")
    llm(messages=[])
    assert [c["model"] for c in calls] == ["anthropic/claude-sonnet-5.5", "openai/gpt-4o-mini"]
