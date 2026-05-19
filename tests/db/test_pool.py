"""Tests for ConnectionPool."""
import pytest

from prospecta.db import ConnectionPool


def test_pool_requires_database_url(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="database_url required"):
        ConnectionPool()


def test_pool_uses_env_var(pg_url, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", pg_url)
    pool = ConnectionPool()
    assert pool.database_url == pg_url
    pool.close()


def test_connection_context_manager(pg_url):
    pool = ConnectionPool(database_url=pg_url)
    with pool.connection() as conn:
        assert not conn.closed
    pool.close()


def test_cursor_executes_select(pg_url):
    pool = ConnectionPool(database_url=pg_url)
    with pool.cursor() as cur:
        cur.execute("SELECT 1")
        row = cur.fetchone()
        assert row == (1,)
    pool.close()


def test_cursor_rolls_back_on_error(pg_url):
    pool = ConnectionPool(database_url=pg_url)
    with pool.cursor() as cur:
        cur.execute("CREATE TEMP TABLE _rollback_test (id INT)")
        cur.execute("INSERT INTO _rollback_test VALUES (1)")
    with pytest.raises(RuntimeError):
        with pool.cursor() as cur:
            cur.execute("INSERT INTO _rollback_test VALUES (2)")
            raise RuntimeError("force rollback")
    # Temp tables live for the connection session; pool reuses the same conn
    # so the table still exists. Verify only the (2) row was rolled back.
    with pool.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM _rollback_test")
        count = cur.fetchone()[0]
        assert count == 1
    pool.close()


def test_connection_reuse(pg_url):
    pool = ConnectionPool(database_url=pg_url)
    with pool.connection() as c1:
        id1 = id(c1)
    with pool.connection() as c2:
        id2 = id(c2)
    assert id1 == id2
    pool.close()
