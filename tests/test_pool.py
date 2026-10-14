"""The one bounded connection pool: requests borrow and return, nothing leaks."""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request

import psycopg
import pytest

from prospecta.db.migrate import run_migrations
from prospecta.db.pool import ConnectionPool, PoolTimeout
from prospecta.memory import Memory
from prospecta.service import Server, Service, make_handler
from prospecta.source import SourceConfig, SourceIndexer
from tests._stub_embedder import EMBED_DIM, stub_embed
from tests.test_shared_schema import bare_db  # noqa: F401  (fixture)
from tests.test_source_contract import HOST_DDL, _note

POOL_MAX = 5


def _db_name(url):
    return psycopg.conninfo.conninfo_to_dict(url)["dbname"]


class Counter:
    """Counts the server's open connections to the test database (not its own)."""

    def __init__(self, url):
        self._c = psycopg.connect(url, autocommit=True)
        self._name = _db_name(url)
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def now(self):
        return self._c.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE datname = %s AND pid <> pg_backend_pid()",
            (self._name,)).fetchone()[0]

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.now())
            time.sleep(0.005)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()
        self.peak = max(self.peak, self.now())
        self._c.close()


@pytest.fixture
def served(bare_db, monkeypatch):  # noqa: F811
    monkeypatch.setenv("PROSPECTA_DB_POOL_MAX", str(POOL_MAX))
    with psycopg.connect(bare_db, autocommit=True) as c:
        c.execute(HOST_DDL)
    run_migrations(bare_db, schema="prospecta_shared")
    state = {"boom": False}

    def llm(messages, **k):
        if state["boom"]:
            raise RuntimeError("model failed")
        return "What is the first thing?\nWhy does the second thing matter?"

    mem = Memory(database_url=bare_db, bank_id="b", embed=stub_embed, llm=llm,
                 schema="prospecta_shared")
    mem.create_bank("b", embedding_dim=EMBED_DIM)
    ix = SourceIndexer(mem, SourceConfig("host_app", ("records",)))
    srv = Server(("127.0.0.1", 0), make_handler(Service(mem, ix), None))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield bare_db, mem, base, state
    srv.shutdown()
    srv.server_close()
    mem.close()


def _post(base, path, body):
    req = urllib.request.Request(base + path, json.dumps(body).encode(), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def test_pool_size_comes_from_environment(served, monkeypatch):
    _, mem, _, _ = served
    assert mem._pool.max_size == POOL_MAX
    monkeypatch.delenv("PROSPECTA_DB_POOL_MAX")
    assert ConnectionPool("postgresql://x/y").max_size == 10          # default
    assert ConnectionPool("postgresql://x/y").idle_seconds == 60      # default


def test_500_sequential_requests_stay_within_pool_max(served):
    url, mem, base, _ = served
    for i in range(3):
        _note(url, i)
    with Counter(url) as cnt:
        codes = [_post(base, "/index", {"kind": "note", "record_id": str(i % 3)}) for i in range(500)]
    assert set(codes) == {200}
    assert cnt.peak <= POOL_MAX, cnt.peak
    assert mem._pool.in_use == 0


def test_189_parallel_index_calls_no_errors_bounded_connections(served):
    url, mem, base, _ = served
    for i in range(189):
        _note(url, i)
    codes = []
    lock = threading.Lock()

    def go(i):
        c = _post(base, "/index", {"kind": "note", "record_id": str(i)})
        with lock:
            codes.append(c)
    with Counter(url) as cnt:
        ts = [threading.Thread(target=go, args=(i,)) for i in range(189)]
        [t.start() for t in ts]
        [t.join(120) for t in ts]
    assert len(codes) == 189 and set(codes) == {200}, sorted(set(codes))
    assert cnt.peak <= POOL_MAX, cnt.peak
    assert mem._pool.in_use == 0


def test_erroring_request_returns_its_connection(served):
    url, mem, base, state = served
    _note(url, 1)
    state["boom"] = True
    for _ in range(3 * POOL_MAX):          # would exhaust the pool if any error kept one
        assert _post(base, "/index", {"kind": "note", "record_id": "1"}) == 500
    assert mem._pool.in_use == 0
    state["boom"] = False
    assert _post(base, "/index", {"kind": "note", "record_id": "1"}) == 200


def test_worker_pool_uses_the_shared_pool(served):
    url, mem, base, _ = served
    for i in range(60):
        _note(url, i)
    with Counter(url) as cnt:
        refs = [{"kind": "note", "record_id": str(i)} for i in range(60)]
        assert _post(base, "/reconcile", {"references": refs, "concurrency": 32}) == 200
        assert _post(base, "/index", {"kind": "note", "reindex": True, "concurrency": 32}) == 200
    assert cnt.peak <= POOL_MAX, cnt.peak
    assert mem._pool.in_use == 0


# -- the pool itself -------------------------------------------------------
def test_warns_once_a_minute_at_70_percent(bare_db, caplog):  # noqa: F811
    pool = ConnectionPool(bare_db, max_size=10, timeout=5)
    caplog.set_level(logging.WARNING, logger="prospecta.db.pool")
    held = []
    cms = [pool.dedicated() for _ in range(9)]
    try:
        for i, cm in enumerate(cms):
            held.append(cm.__enter__())
            n = [r for r in caplog.records if r.levelno == logging.WARNING]
            if i + 1 < 7:
                assert not n
        assert len(n) == 1                                   # 7th borrow warned; 8th, 9th did not
        assert "pool size 10" in n[0].getMessage() and "7 in use" in n[0].getMessage()
    finally:
        for cm in reversed(cms):
            cm.__exit__(None, None, None)
        pool.close()


def test_waiters_queue_then_time_out(bare_db):  # noqa: F811
    pool = ConnectionPool(bare_db, max_size=1, timeout=0.3)
    with pool.connection():
        with pytest.raises(PoolTimeout):
            with pool.dedicated():
                pass
    pool2 = ConnectionPool(bare_db, max_size=1, timeout=10)
    order = []

    def borrower(n):
        with pool2.connection() as c:
            order.append(n)
            time.sleep(0.1)
            c.execute("SELECT 1")
    ts = [threading.Thread(target=borrower, args=(i,)) for i in range(6)]
    [x.start() for x in ts]
    [x.join(20) for x in ts]
    assert sorted(order) == list(range(6)) and pool2.open_connections <= 1
    pool.close()
    pool2.close()


def test_idle_connection_is_closed_after_timeout(bare_db):  # noqa: F811
    pool = ConnectionPool(bare_db, max_size=3, idle_seconds=0.3, timeout=5)
    with pool.connection() as a, pool.dedicated() as b:
        a.execute("SELECT 1")
        b.execute("SELECT 1")
    assert pool.open_connections == 2
    deadline = time.time() + 5
    while pool.open_connections and time.time() < deadline:
        time.sleep(0.05)
    assert pool.open_connections == 0 and a.closed and b.closed
    with pool.connection() as c:          # and it opens again on demand
        assert c.execute("SELECT 1").fetchone() == (1,)
    pool.close()


def test_connection_returns_after_error_and_uncommitted_work_is_rolled_back(bare_db):  # noqa: F811
    pool = ConnectionPool(bare_db, max_size=1, timeout=1)
    for _ in range(3):
        with pytest.raises(ValueError):
            with pool.connection() as c:
                c.execute("SELECT 1")
                raise ValueError("boom")
    with pool.connection() as c:          # sole connection is usable and clean
        assert c.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
    assert pool.in_use == 0
    pool.close()
