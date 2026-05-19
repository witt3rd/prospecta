"""T15 — full CLI surface tests.

Each test that needs a Memory monkeypatches `_common.make_memory` to return
the fixture-built Memory rather than constructing one from env + LLM/embed
defaults (T23 not shipped). This exercises the CLI body in-process and keeps
tests fast.
"""
from __future__ import annotations

import json

import pytest


def test_cli_help(capsys):
    """`prospecta --help` lists all subcommands."""
    from prospecta.cli.__main__ import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    for sub in [
        "migrate",
        "create-bank",
        "index",
        "search",
        "retain",
        "stats",
        "config",
        "sweep",
    ]:
        assert sub in out


def test_cli_config_redacts_secrets(capsys, monkeypatch):
    monkeypatch.setenv(
        "DATABASE_URL", "postgres://alice:supersecret@localhost:5432/proddb"
    )
    from prospecta.cli.__main__ import main

    rc = main(["config"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "supersecret" not in out
    assert "alice" in out  # username NOT redacted
    assert "****" in out or "***" in out


def test_cli_config_no_database_url(capsys, monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from prospecta.cli.__main__ import main

    rc = main(["config"])
    assert rc == 0
    out = capsys.readouterr().out
    # Should still print something (the active bank, etc.)
    assert "bank" in out.lower()


def test_cli_search(populated_corpus, capsys, monkeypatch):
    from prospecta.cli import _common
    from prospecta.cli.__main__ import main

    monkeypatch.setattr(_common, "make_memory", lambda args: populated_corpus)
    rc = main(["search", "Kelly"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "kelly" in out.lower()


def test_cli_search_json(populated_corpus, capsys, monkeypatch):
    from prospecta.cli import _common
    from prospecta.cli.__main__ import main

    monkeypatch.setattr(_common, "make_memory", lambda args: populated_corpus)
    rc = main(["search", "Kelly", "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    if parsed:
        # Check shape
        assert "content" in parsed[0]
        assert "source" in parsed[0]
        assert "scores" in parsed[0]


def test_cli_retain(memory_with_bank_and_mock_llm, capsys, monkeypatch):
    from prospecta.cli import _common
    from prospecta.cli.__main__ import main

    mem = memory_with_bank_and_mock_llm
    mem._llm.canned = "q1\nq2\nq3"  # type: ignore[attr-defined]
    monkeypatch.setattr(_common, "make_memory", lambda args: mem)
    rc = main(
        ["retain", "long content body", "--source", "test", "--tags", "foo,bar"]
    )
    assert rc == 0
    out = capsys.readouterr().out.strip()
    # Output should include the UUID (UUIDs are 36 chars w/ hyphens, 32 hex)
    assert len(out) >= 32


def test_cli_retain_with_index_text_bypasses_llm(
    memory_with_bank_and_mock_llm, monkeypatch
):
    from prospecta.cli import _common
    from prospecta.cli.__main__ import main

    mem = memory_with_bank_and_mock_llm
    monkeypatch.setattr(_common, "make_memory", lambda args: mem)
    rc = main(
        [
            "retain",
            "body",
            "--index-text",
            "q1",
            "--index-text",
            "q2",
            "--source",
            "t",
        ]
    )
    assert rc == 0
    # P4: caller-supplied index_text bypasses LLM
    assert mem._llm.calls == []  # type: ignore[attr-defined]


def test_cli_retain_at_file(memory_with_bank_and_mock_llm, tmp_path, monkeypatch):
    from prospecta.cli import _common
    from prospecta.cli.__main__ import main

    mem = memory_with_bank_and_mock_llm
    monkeypatch.setattr(_common, "make_memory", lambda args: mem)
    body_file = tmp_path / "body.txt"
    body_file.write_text("loaded-from-disk")
    rc = main(
        [
            "retain",
            f"@{body_file}",
            "--index-text",
            "q",
            "--source",
            "disk-test",
        ]
    )
    assert rc == 0


def test_cli_stats(populated_corpus, capsys, monkeypatch):
    from prospecta.cli import _common
    from prospecta.cli.__main__ import main

    monkeypatch.setattr(_common, "make_memory", lambda args: populated_corpus)
    rc = main(["stats"])
    assert rc == 0
    out = capsys.readouterr().out
    for label in ["documents", "memory_items"]:
        assert label in out.lower()


def test_cli_sweep_once(
    memory_with_bank_and_mock_llm, tmp_corpus, capsys, monkeypatch
):
    from prospecta.cli import _common
    from prospecta.cli.__main__ import main

    mem = memory_with_bank_and_mock_llm
    monkeypatch.setattr(_common, "make_memory", lambda args: mem)
    rc = main(["sweep", str(tmp_corpus), "--once"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "files_indexed" in out or "indexed" in out.lower()


def test_redact_database_url():
    from prospecta.cli._common import redact_database_url

    assert (
        redact_database_url("postgres://alice:secret@localhost/db")
        == "postgres://alice:****@localhost/db"
    )
    # No password — passthrough
    assert (
        redact_database_url("postgres://alice@localhost/db")
        == "postgres://alice@localhost/db"
    )
    # None or empty
    assert redact_database_url(None) == "(unset)"
    assert redact_database_url("") == "(unset)"
