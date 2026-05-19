"""Shared pytest fixtures."""
from __future__ import annotations

import time
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest
from testcontainers.postgres import PostgresContainer


def _normalize_url(url: str) -> str:
    """Strip driver hint so the URL is psycopg3-compatible."""
    if "+psycopg2" in url:
        url = url.replace("+psycopg2", "")
    if "+psycopg" in url:
        url = url.replace("+psycopg", "")
    return url


@pytest.fixture(scope="session")
def pg_container():
    """Session-scoped Postgres testcontainer with pgvector pre-enabled."""
    container = PostgresContainer("pgvector/pgvector:pg16")
    container.start()
    try:
        url = _normalize_url(container.get_connection_url())
        with psycopg.connect(url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            conn.commit()
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="session")
def pg_url(pg_container) -> str:
    return _normalize_url(pg_container.get_connection_url())


@pytest.fixture
def fresh_db(pg_container):
    """Per-test fresh database with migrations applied. Returns its URL."""
    base_url = _normalize_url(pg_container.get_connection_url())
    db_name = f"test_{int(time.time() * 1_000_000)}"
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


@pytest.fixture
def memory_with_bank(fresh_db):
    """Memory wired with the deterministic stub embedder + a created bank.

    32-dim bank named 'test'. Stub LLM attached but T9 paths don't call it.
    """
    from prospecta.memory import Memory
    from tests._stub_embedder import EMBED_DIM, stub_embed, stub_llm

    mem = Memory(
        database_url=fresh_db,
        bank_id="test",
        llm=stub_llm,
        embed=stub_embed,
    )
    mem.create_bank("test", embedding_dim=EMBED_DIM)
    yield mem
    mem.close()
