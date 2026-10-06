from __future__ import annotations

from prospecta.cli import health
from prospecta.cli.__main__ import main


def test_health_ok(monkeypatch, capsys):
    monkeypatch.setattr(health, "_check_db", lambda url: "ok")
    monkeypatch.setattr(health, "_check_embedder", lambda: 384)
    assert main(["--database-url", "postgres://x", "health"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("PROSPECTA OK")


def test_health_db_down(monkeypatch, capsys):
    def boom(url):
        raise RuntimeError("refused")
    monkeypatch.setattr(health, "_check_db", boom)
    monkeypatch.setattr(health, "_check_embedder", lambda: 384)
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
