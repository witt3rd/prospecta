"""Shared pytest fixtures."""
from __future__ import annotations

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
        import psycopg

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
