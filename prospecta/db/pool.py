"""Bounded, shared Postgres connection pool for prospecta (sync).

One pool per Memory (so one per process for the HTTP service). Every use
borrows a connection and gives it back when the block ends, success or error;
no thread ever keeps a connection. The pool never holds more than `max_size`
connections open: a borrower that finds none free waits, up to `timeout`
seconds, then gets PoolTimeout.

Environment (read when the pool is built; arguments win):
    PROSPECTA_DB_POOL_MAX            maximum connections (default 10)
    PROSPECTA_DB_POOL_IDLE_SECONDS   close a connection idle this long (default 60)
    PROSPECTA_REQUEST_TIMEOUT        seconds a borrower may wait (default 30)

When 70 percent or more of the pool is in use, one warning line (at most once a
minute) names the pool size and the count in use.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg import Connection, Cursor

logger = logging.getLogger("prospecta.db.pool")

DEFAULT_POOL_MAX = 10
DEFAULT_IDLE_SECONDS = 60.0
DEFAULT_TIMEOUT_SECONDS = 30.0
WARN_FRACTION = 0.7
WARN_INTERVAL_SECONDS = 60.0


class PoolTimeout(RuntimeError):
    """No connection became free within the timeout."""


def _env_number(name: str, default: float, cast=float, minimum: float = 0) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        v = cast(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if v < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {raw!r}")
    return v


class ConnectionPool:
    """Bounded pool: `max_size` connections at most, idle ones closed after
    `idle_seconds`, waiters queue for up to `timeout` seconds."""

    def __init__(self, database_url: str | None = None, *,
                 max_size: int | None = None, idle_seconds: float | None = None,
                 timeout: float | None = None):
        self._database_url = database_url or os.environ.get("DATABASE_URL")
        if not self._database_url:
            raise ValueError(
                "database_url required (or set DATABASE_URL env var)"
            )
        self.max_size = int(max_size if max_size is not None
                            else _env_number("PROSPECTA_DB_POOL_MAX", DEFAULT_POOL_MAX, int, 1))
        if self.max_size < 1:
            raise ValueError("pool max_size must be >= 1")
        self.idle_seconds = float(idle_seconds if idle_seconds is not None else
                                  _env_number("PROSPECTA_DB_POOL_IDLE_SECONDS", DEFAULT_IDLE_SECONDS))
        self.timeout = float(timeout if timeout is not None else
                             _env_number("PROSPECTA_REQUEST_TIMEOUT", DEFAULT_TIMEOUT_SECONDS))
        self._cond = threading.Condition()
        self._idle: list[tuple[Connection, float]] = []   # (conn, returned-at), newest last
        self._in_use = 0
        self._open = 0                                     # idle + in use
        self._closed = False
        self._last_warn = float("-inf")
        self._held = threading.local()    # the connection this thread has borrowed right now
        self._reaper: threading.Thread | None = None

    # -- introspection ----------------------------------------------------
    @property
    def database_url(self) -> str:
        assert self._database_url is not None
        return self._database_url

    @property
    def in_use(self) -> int:
        with self._cond:
            return self._in_use

    @property
    def open_connections(self) -> int:
        with self._cond:
            return self._open

    # -- borrowing --------------------------------------------------------
    def _warn_if_busy(self) -> None:   # called with the lock held
        if self._in_use >= WARN_FRACTION * self.max_size:
            now = time.monotonic()
            if now - self._last_warn >= WARN_INTERVAL_SECONDS:
                self._last_warn = now
                logger.warning("connection pool size %d: %d in use",
                               self.max_size, self._in_use)

    def _acquire(self) -> Connection:
        deadline = time.monotonic() + self.timeout
        with self._cond:
            while True:
                if self._closed:
                    raise RuntimeError("connection pool is closed")
                while self._idle:
                    conn, _ = self._idle.pop()
                    if conn.closed:
                        self._open -= 1
                        continue
                    self._in_use += 1
                    self._warn_if_busy()
                    return conn
                if self._open < self.max_size:
                    self._open += 1          # reserve the slot, connect outside the lock
                    self._in_use += 1
                    self._warn_if_busy()
                    break
                left = deadline - time.monotonic()
                if left <= 0:
                    raise PoolTimeout(
                        f"no database connection free after {self.timeout:g}s "
                        f"(pool size {self.max_size}, {self._in_use} in use)")
                self._cond.wait(left)
        try:
            return psycopg.connect(self._database_url, autocommit=False)
        except BaseException:
            with self._cond:
                self._open -= 1
                self._in_use -= 1
                self._cond.notify()
            raise

    def _release(self, conn: Connection) -> None:
        keep = False
        if not conn.closed:
            try:
                if conn.info.transaction_status != psycopg.pq.TransactionStatus.IDLE:
                    conn.rollback()
                keep = conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
            except Exception:
                keep = False
        if not keep and not conn.closed:
            try:
                conn.close()
            except Exception:  # pragma: no cover
                pass
        with self._cond:
            self._in_use -= 1
            if keep and not self._closed:
                self._idle.append((conn, time.monotonic()))
                self._start_reaper()
            else:
                self._open -= 1
                if not conn.closed:
                    conn.close()
            self._cond.notify()

    @contextmanager
    def _borrow(self, reentrant: bool) -> Iterator[Connection]:
        held = getattr(self._held, "conn", None)
        if reentrant and held is not None and not held.closed:
            yield held       # nested use in the same thread: same borrowed connection
            return
        conn = self._acquire()
        if reentrant:
            self._held.conn = conn
        try:
            yield conn
        finally:
            if reentrant:
                self._held.conn = None
            self._release(conn)

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        """Borrow a connection for the block. Caller commits; whatever is
        uncommitted is rolled back when the connection goes back."""
        with self._borrow(True) as conn:
            try:
                yield conn
            except Exception:
                if not conn.closed:
                    conn.rollback()
                raise

    @contextmanager
    def cursor(self) -> Iterator[Cursor]:
        """Borrow a connection and yield a cursor, committing on success."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                yield cur
            conn.commit()

    @contextmanager
    def dedicated(self) -> Iterator[Connection]:
        """Borrow a connection that is never the one the calling thread already
        holds, for work that must not share a transaction with the caller (the
        former linker worker used it; the plugin's pool stand-in implements it too).
        It still counts against the pool. Caller commits."""
        with self._borrow(False) as conn:
            try:
                yield conn
            except Exception:
                if not conn.closed:
                    conn.rollback()
                raise

    # -- idle reaping -----------------------------------------------------
    def _start_reaper(self) -> None:   # lock held
        if self._reaper is not None or self.idle_seconds <= 0:
            return
        self._reaper = threading.Thread(target=self._reap_loop, name="prospecta-pool-reaper",
                                        daemon=True)
        self._reaper.start()

    def reap_idle(self) -> int:
        """Close connections idle longer than idle_seconds; returns how many."""
        cutoff = time.monotonic() - self.idle_seconds
        with self._cond:
            stale = [c for c, t in self._idle if t <= cutoff or c.closed]
            self._idle = [(c, t) for c, t in self._idle if t > cutoff and not c.closed]
            self._open -= len(stale)
            self._cond.notify_all()
        for c in stale:
            try:
                c.close()
            except Exception:  # pragma: no cover
                pass
        return len(stale)

    def _reap_loop(self) -> None:
        interval = max(0.05, min(self.idle_seconds / 2, 5.0))
        while True:
            with self._cond:
                if self._closed:
                    return
                self._cond.wait(interval)
                if self._closed:
                    return
            self.reap_idle()

    def close(self) -> None:
        """Close every idle connection; borrowed ones close as they return."""
        with self._cond:
            self._closed = True
            conns, self._idle = [c for c, _ in self._idle], []
            self._open -= len(conns)
            self._cond.notify_all()
        for c in conns:
            if not c.closed:
                c.close()
