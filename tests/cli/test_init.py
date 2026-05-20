"""Acceptance tests for `prospecta init` (T22)."""
from __future__ import annotations

import pytest


def test_init_fresh_creates_default_bank(fresh_db, monkeypatch, capsys):
    """`prospecta init --no-substrate` on a fresh DB applies migrations + creates 'default' bank."""
    monkeypatch.setenv("DATABASE_URL", fresh_db)
    from prospecta.cli.__main__ import main
    rc = main(["init", "--no-substrate", "--embedding-dim", "32"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "substrate" in out.lower()
    assert "migrations" in out.lower()
    assert "default" in out.lower()
    import psycopg
    with psycopg.connect(fresh_db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT bank_id, embedding_dim FROM banks WHERE bank_id = 'default'")
            row = cur.fetchone()
            assert row is not None
            assert row[1] == 32


def test_init_idempotent(fresh_db, monkeypatch, capsys):
    """Second invocation prints 'already' markers for migrations + bank."""
    monkeypatch.setenv("DATABASE_URL", fresh_db)
    from prospecta.cli.__main__ import main
    main(["init", "--no-substrate", "--embedding-dim", "32"])
    capsys.readouterr()
    rc = main(["init", "--no-substrate", "--embedding-dim", "32"])
    assert rc == 0
    out = capsys.readouterr().out.lower()
    assert out.count("already") >= 2


def test_init_no_docker_no_database_url_exits_1(monkeypatch, capsys):
    """Missing both docker AND DATABASE_URL → exit 1 with actionable message."""
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("PATH", "/nonexistent")
    from prospecta.cli.__main__ import main
    rc = main(["init"])
    assert rc == 1
    captured = capsys.readouterr()
    blob = (captured.out + captured.err).lower()
    assert "docker" in blob or "database_url" in blob


def test_init_embedding_dim_override(fresh_db, monkeypatch, capsys):
    """`--embedding-dim 384` overrides default."""
    monkeypatch.setenv("DATABASE_URL", fresh_db)
    from prospecta.cli.__main__ import main
    rc = main(["init", "--no-substrate", "--embedding-dim", "384"])
    assert rc == 0
    import psycopg
    with psycopg.connect(fresh_db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT embedding_dim FROM banks WHERE bank_id = 'default'")
            assert cur.fetchone()[0] == 384


def test_init_env_var_embedding_dim(fresh_db, monkeypatch, capsys):
    """`PROSPECTA_EMBEDDING_DIM=128` env var resolves dim."""
    monkeypatch.setenv("DATABASE_URL", fresh_db)
    monkeypatch.setenv("PROSPECTA_EMBEDDING_DIM", "128")
    from prospecta.cli.__main__ import main
    rc = main(["init", "--no-substrate"])
    assert rc == 0
    import psycopg
    with psycopg.connect(fresh_db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT embedding_dim FROM banks WHERE bank_id = 'default'")
            assert cur.fetchone()[0] == 128


def test_init_default_dim_warns(fresh_db, monkeypatch, capsys):
    """With neither flag nor env, defaults to 1536 + prints assumption warning."""
    monkeypatch.setenv("DATABASE_URL", fresh_db)
    monkeypatch.delenv("PROSPECTA_EMBEDDING_DIM", raising=False)
    from prospecta.cli.__main__ import main
    rc = main(["init", "--no-substrate"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1536" in out
    assert (
        "text-embedding-3-small" in out.lower()
        or "openai" in out.lower()
        or "default" in out.lower()
    )
