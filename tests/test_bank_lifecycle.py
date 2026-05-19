"""Tests for Memory.create_bank() + bank_stats() + CLI subcommands."""
import json
import subprocess
import time
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest

from prospecta.memory import BankConfigConflict, Memory


@pytest.fixture
def fresh_db(pg_container):
    """Fresh database with vector extension + migrations applied."""
    base_url = pg_container.get_connection_url()
    if "+psycopg2" in base_url:
        base_url = base_url.replace("+psycopg2", "")
    if "+psycopg" in base_url:
        base_url = base_url.replace("+psycopg", "")

    db_name = f"test_bank_{int(time.time() * 1000)}"
    with psycopg.connect(base_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{db_name}"')

    parsed = urlparse(base_url)
    new_url = urlunparse(parsed._replace(path=f"/{db_name}"))

    with psycopg.connect(new_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    from prospecta.db.migrate import run_migrations
    run_migrations(new_url)

    yield new_url

    with psycopg.connect(base_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (db_name,),
            )
            cur.execute(f'DROP DATABASE "{db_name}"')


def test_create_bank_writes_row(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="test")
    mem.create_bank("test", embedding_dim=384)
    with psycopg.connect(fresh_db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT bank_id, embedding_dim FROM banks WHERE bank_id = 'test'")
            row = cur.fetchone()
            assert row == ("test", 384)
    mem.close()


def test_create_bank_creates_hnsw_index(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="test")
    mem.create_bank("test", embedding_dim=384)
    with psycopg.connect(fresh_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'memory_items' AND indexname LIKE 'memory_items_embedding_%'"
            )
            indexes = [row[0] for row in cur.fetchall()]
    assert any("test" in idx for idx in indexes), (
        f"Expected per-bank HNSW index for 'test'; got {indexes}"
    )
    mem.close()


def test_create_bank_idempotent_same_config(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="test")
    mem.create_bank("test", embedding_dim=384)
    mem.create_bank("test", embedding_dim=384)  # no-op
    with psycopg.connect(fresh_db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM banks WHERE bank_id = 'test'")
            assert cur.fetchone()[0] == 1
    mem.close()


def test_create_bank_rejects_dim_mismatch(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="test")
    mem.create_bank("test", embedding_dim=384)
    with pytest.raises(BankConfigConflict):
        mem.create_bank("test", embedding_dim=1024)
    mem.close()


def test_create_bank_validates_bank_id(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="test")
    with pytest.raises(ValueError):
        mem.create_bank("bad/id!", embedding_dim=384)
    with pytest.raises(ValueError):
        mem.create_bank("", embedding_dim=384)
    with pytest.raises(ValueError):
        mem.create_bank("x" * 64, embedding_dim=384)
    mem.close()


def test_create_bank_rejects_invalid_dim(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="test")
    with pytest.raises(ValueError):
        mem.create_bank("test", embedding_dim=0)
    with pytest.raises(ValueError):
        mem.create_bank("test", embedding_dim=-1)
    mem.close()


def test_dim_check_trigger_fires_on_wrong_dim(fresh_db):
    """After create_bank, the dim-check trigger should fire on wrong-dim insert."""
    mem = Memory(database_url=fresh_db, bank_id="test")
    mem.create_bank("test", embedding_dim=384)
    with psycopg.connect(fresh_db, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (bank_id, original_text, content_hash) "
                "VALUES ('test', 'hello', 'h1') RETURNING id"
            )
            doc_id = cur.fetchone()[0]
            wrong_vec = "[" + ",".join(["0.1"] * 100) + "]"
            with pytest.raises(psycopg.errors.RaiseException):
                cur.execute(
                    "INSERT INTO memory_items (bank_id, document_id, content, original_chunk, embedding) "
                    "VALUES ('test', %s, 'c', 'o', %s::vector)",
                    (doc_id, wrong_vec),
                )
    mem.close()


def test_bank_stats_empty(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="test")
    mem.create_bank("test", embedding_dim=384)
    stats = mem.bank_stats("test")
    assert stats.bank_id == "test"
    assert stats.documents == 0
    assert stats.memory_items == 0
    assert stats.last_retain_at is None
    mem.close()


def test_bank_stats_missing_bank_raises(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="nope")
    with pytest.raises(ValueError, match="No such bank"):
        mem.bank_stats("nope")
    mem.close()


def test_bank_stats_default_bank_id(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="mybank")
    mem.create_bank("mybank", embedding_dim=384)
    stats = mem.bank_stats()
    assert stats.bank_id == "mybank"
    mem.close()


def test_memory_requires_database_url():
    with pytest.raises(ValueError):
        Memory(database_url="")


def test_cli_create_bank(fresh_db):
    """Smoke-test the CLI subcommand against a fresh DB."""
    result = subprocess.run(
        ["prospecta", "create-bank", "--database-url", fresh_db,
         "--id", "clitest", "--embedding-dim", "384"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"stdout={result.stdout!r} stderr={result.stderr!r}"
    assert "clitest" in result.stdout


def test_cli_stats(fresh_db):
    """Smoke-test prospecta stats."""
    subprocess.run(
        ["prospecta", "create-bank", "--database-url", fresh_db,
         "--id", "statbank", "--embedding-dim", "384"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = subprocess.run(
        ["prospecta", "stats", "--database-url", fresh_db, "--bank", "statbank"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    output = json.loads(result.stdout)
    assert output["bank_id"] == "statbank"
    assert output["documents"] == 0
