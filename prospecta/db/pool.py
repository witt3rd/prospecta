"""Postgres connection pool wrapper for prospecta.

Thin wrapper around psycopg connection management. v0.1 sync-only;
async deferred to v0.2.
"""
from __future__ import annotations

import os
import threading
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
        # One connection per thread: a psycopg connection is not safe to share
        # between threads (concurrent cursors on it raise ProgrammingError /
        # corrupt each other's transactions).
        self._local = threading.local()
        self._all: list[Connection] = []
        self._lock = threading.Lock()

    @property
    def _conn(self) -> Connection | None:
        return getattr(self._local, "conn", None)

    @property
    def database_url(self) -> str:
        assert self._database_url is not None
        return self._database_url

    def _ensure_open(self) -> Connection:
        conn = self._conn
        if conn is None or conn.closed:
            conn = psycopg.connect(self._database_url, autocommit=False)
            self._local.conn = conn
            with self._lock:
                self._all = [c for c in self._all if not c.closed] + [conn]
        return conn

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

    @contextmanager
    def dedicated(self) -> Iterator[Connection]:
        """Yield a private connection (not the shared one) and close it after.

        For work on another thread (the Linker worker): it never shares a
        transaction with the caller's connection. Caller commits/rolls back.
        """
        conn = psycopg.connect(self.database_url, autocommit=False)
        try:
            yield conn
        finally:
            conn.close()

    def close(self) -> None:
        with self._lock:
            conns, self._all = self._all, []
        for c in conns:
            if not c.closed:
                c.close()
        self._local = threading.local()
