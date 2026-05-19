"""Postgres connection pool wrapper for prospecta.

Thin wrapper around psycopg connection management. v0.1 sync-only;
async deferred to v0.2.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg import Connection, Cursor


class ConnectionPool:
    """Per-Memory connection wrapper. v0.1 = simple lazy connection."""

    def __init__(self, database_url: str | None = None):
        self._database_url = database_url or os.environ.get("DATABASE_URL")
        if not self._database_url:
            raise ValueError(
                "database_url required (or set DATABASE_URL env var)"
            )
        self._conn: Connection | None = None

    @property
    def database_url(self) -> str:
        assert self._database_url is not None
        return self._database_url

    def _ensure_open(self) -> Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._database_url, autocommit=False)
        return self._conn

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        """Yield a connection. Caller manages commit/rollback."""
        conn = self._ensure_open()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise

    @contextmanager
    def cursor(self) -> Iterator[Cursor]:
        """Yield a cursor with auto-commit on success."""
        conn = self._ensure_open()
        try:
            with conn.cursor() as cur:
                yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
        self._conn = None
