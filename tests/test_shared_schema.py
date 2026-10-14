"""Shared-database mode: everything lives in one configured schema."""
from __future__ import annotations

import time
from urllib.parse import urlparse, urlunparse

import psycopg
import psycopg.conninfo
import pytest

from prospecta.db.migrate import get_schema_version, run_migrations
from prospecta.db.schema import (
    SchemaError, check_schema_writable, scoped_url, validate_schema_name,
)
from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM, stub_embed, stub_llm
from tests.conftest import _normalize_url


@pytest.fixture
def bare_db(pg_container):
    """An empty database: pgvector installed (as the host would), nothing else."""
    base = _normalize_url(pg_container.get_connection_url())
    name = f"test_shared_{int(time.time() * 1_000_000)}"
    with psycopg.connect(base, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{name}"')
    url = urlunparse(urlparse(base)._replace(path=f"/{name}"))
    with psycopg.connect(url, autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        c.execute("CREATE TABLE public.host_table (id int)")   # the host's own object
    yield url
    with psycopg.connect(base, autocommit=True) as c:
        c.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                  "WHERE datname = %s AND pid <> pg_backend_pid()", (name,))
        c.execute(f'DROP DATABASE "{name}"')


def _objects(url):
    """Every relation/function/type owned outside pg_catalog, with its schema."""
    with psycopg.connect(url) as c:
        rels = c.execute(
            "SELECT n.nspname, c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
            "WHERE n.nspname NOT IN ('pg_catalog','information_schema','pg_toast')").fetchall()
        funcs = c.execute(
            "SELECT n.nspname, p.proname FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace "
            "WHERE n.nspname NOT IN ('pg_catalog','information_schema') "
            "AND p.oid NOT IN (SELECT objid FROM pg_depend WHERE deptype = 'e')").fetchall()
        exts = [r[0] for r in c.execute("SELECT extname FROM pg_extension").fetchall()]
    return rels, funcs, exts


def test_validate_schema_name():
    assert validate_schema_name("prospecta_x1") == "prospecta_x1"
    for bad in ("", "Public", "public", "pg_x", 'a"b', "a b", "1a", "information_schema", "x" * 64):
        with pytest.raises(SchemaError):
            validate_schema_name(bad)


def test_scoped_url_forms():
    u = scoped_url("postgresql://u:p@h:5432/db?sslmode=require", "s1")
    assert "sslmode=require" in u and "search_path" in u and u.count("options=") == 1
    assert scoped_url(u, "s1").count("options=") == 1
    assert scoped_url("postgresql://h/db", None) == "postgresql://h/db"
    assert scoped_url("host=h dbname=d", "s1").startswith("host=h dbname=d options='")


def test_migrations_touch_only_the_schema(bare_db):
    run_migrations(bare_db, schema="prospecta_shared")
    rels, funcs, exts = _objects(bare_db)
    outside = [(n, r) for n, r in rels if n not in ("prospecta_shared",) and r != "host_table"]
    assert outside == [], outside
    assert ("public", "host_table") in rels
    assert any(n == "prospecta_shared" and r == "memory_items" for n, r in rels)
    assert any(n == "prospecta_shared" and r == "prospecta_schema_version" for n, r in rels)
    assert all(n == "prospecta_shared" for n, _ in funcs), funcs
    assert exts == ["plpgsql", "vector"] or sorted(exts) == ["plpgsql", "vector"]
    assert get_schema_version(bare_db, schema="prospecta_shared") == 17
    assert get_schema_version(bare_db) is None          # nothing in public
    again = run_migrations(bare_db, schema="prospecta_shared")
    assert again["applied"] == []


def test_two_schemas_side_by_side(bare_db):
    run_migrations(bare_db, schema="p_one")
    run_migrations(bare_db, schema="p_two")
    a = Memory(database_url=bare_db, bank_id="b", embed=stub_embed, schema="p_one")
    b = Memory(database_url=bare_db, bank_id="b", embed=stub_embed, schema="p_two")
    a.create_bank("b", embedding_dim=EMBED_DIM)
    a.retain("alpha only in one", index_text="what is alpha", source="s1")
    b.create_bank("b", embedding_dim=EMBED_DIM)
    assert a.bank_stats().documents == 1 and b.bank_stats().documents == 0
    assert [r.content for r in a.recall(["what is alpha"])] == ["what is alpha"]
    assert b.recall(["what is alpha"]) == []
    a.close(); b.close()


def test_env_schema_is_used(bare_db, monkeypatch):
    monkeypatch.setenv("PROSPECTA_SCHEMA", "p_env")
    run_migrations(bare_db)
    m = Memory(database_url=bare_db, bank_id="b", embed=stub_embed)
    assert m.schema == "p_env"
    m.close()
    assert get_schema_version(bare_db, schema="p_env") == 17


def test_refuses_when_schema_not_writable(bare_db):
    run_migrations(bare_db, schema="p_ro")
    with psycopg.connect(bare_db, autocommit=True) as c:
        c.execute("DROP ROLE IF EXISTS reader_only")
        c.execute("CREATE ROLE reader_only LOGIN")
        c.execute("GRANT USAGE ON SCHEMA p_ro TO reader_only")
        c.execute("GRANT USAGE ON SCHEMA public TO reader_only")
    ro_url = psycopg.conninfo.make_conninfo(bare_db, user="reader_only")
    with pytest.raises(SchemaError, match="not writable"):
        check_schema_writable(ro_url, "p_ro")
    with pytest.raises(SchemaError, match="not writable"):
        Memory(database_url=ro_url, bank_id="b", embed=stub_embed, schema="p_ro")


def test_refuses_missing_schema_and_missing_vector(bare_db):
    with pytest.raises(SchemaError, match="does not exist"):
        check_schema_writable(bare_db, "nope")
    with psycopg.connect(bare_db, autocommit=True) as c:
        c.execute("CREATE SCHEMA p_nv")
        c.execute("DROP EXTENSION vector")
    with pytest.raises(SchemaError, match="vector extension"):
        check_schema_writable(bare_db, "p_nv")
