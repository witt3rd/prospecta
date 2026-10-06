from __future__ import annotations

from prospecta.cli import health
from prospecta.cli.__main__ import main


def test_health_ok(monkeypatch, capsys):
    monkeypatch.setattr(health, "_check_db", lambda url: "ok")
    monkeypatch.setattr(health, "_check_embedder", lambda d=None: ("litellm", 384))
    assert main(["--database-url", "postgres://x", "health"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("PROSPECTA OK")


def test_health_db_down(monkeypatch, capsys):
    def boom(url):
        raise RuntimeError("refused")
    monkeypatch.setattr(health, "_check_db", boom)
    monkeypatch.setattr(health, "_check_embedder", lambda d=None: ("litellm", 384))
    assert main(["--database-url", "postgres://x", "health"]) == 1
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("PROSPECTA DOWN:") and "database" in out[0]


def test_health_embedder_down(monkeypatch, capsys):
    def boom():
        raise ImportError("no sentence-transformers")
    monkeypatch.setattr(health, "_check_db", lambda url: "ok")
    monkeypatch.setattr(health, "_check_embedder", boom)
    assert main(["--database-url", "postgres://x", "health"]) == 1
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("PROSPECTA DOWN:") and "embedder" in out[0]


def test_health_ok_names_embedder(monkeypatch, capsys):
    monkeypatch.setattr(health, "_check_db", lambda url: "ok")
    monkeypatch.setattr(health, "_check_embedder", lambda d=None: ("sentence-transformers", 384))
    assert main(["--database-url", "postgres://x", "health"]) == 0
    assert "sentence-transformers" in capsys.readouterr().out


def test_check_embedder_st_import_missing(monkeypatch, capsys):
    import sys
    monkeypatch.setenv("PROSPECTA_EMBEDDER", "sentence-transformers")
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    monkeypatch.setattr(health, "_check_db", lambda url: "ok")
    assert main(["--database-url", "postgres://x", "health"]) == 1
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("PROSPECTA DOWN:") and "embedder" in out[0]


def test_check_embedder_default_reports_litellm(monkeypatch):
    from prospecta.cli import _common
    monkeypatch.delenv("PROSPECTA_EMBEDDER", raising=False)
    monkeypatch.setattr(_common, "_resolve_embedder", lambda d=None: (lambda xs: [[0.0] * 3 for _ in xs]))
    assert health._check_embedder() == ("litellm", 3)
