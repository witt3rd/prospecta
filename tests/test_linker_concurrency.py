"""F9: the Linker works on its own connection, orders its writes, retries
deadlocks, records failures loudly and link_pending picks them up."""
from __future__ import annotations

import threading

import psycopg
import psycopg.errors
import pytest

from prospecta._linker import Linker, pending_documents
from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM, stub_embed
from tests.test_graph_links import EntityLLM, conn_of, docs


@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=None, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    yield m
    m.close()


def _seed(mem, n=12):
    for i in range(n):
        mem.retain(f"Kelly met Forge at Acme, note {i}", source=f"n{i}", index_text=f"q {i}",
                   metadata={"created": "2024-01-01", "person": "kelly"})


def test_link_runs_on_a_dedicated_connection(mem):
    _seed(mem, 2)
    seen = []

    class Spy(Linker):
        def link_document(self, conn, bank_id, document_id, calls=None):
            seen.append(conn)
            return super().link_document(conn, bank_id, document_id, calls)

    mem._linker = Spy(llm=EntityLLM())
    d = next(iter(docs(mem).values()))
    mem.link_document(d)
    assert seen[0] is not mem._pool._conn and seen[0].closed


def test_concurrent_linkers_all_link_without_errors(fresh_db):
    ms = []
    m0 = Memory(database_url=fresh_db, bank_id="b", llm=None, embed=stub_embed)
    m0.create_bank("b", embedding_dim=EMBED_DIM)
    _seed(m0, 16)
    for _ in range(4):
        m = Memory(database_url=fresh_db, bank_id="b", llm=None, embed=stub_embed,
                   linker=Linker(llm=EntityLLM(), asynchronous=False))
        ms.append(m)
    ids = list(docs(m0).values())
    errs = []

    def work(m, part):
        try:
            for d in part:
                m.link_document(d)
        except Exception as exc:  # pragma: no cover
            errs.append(exc)

    # every linker walks the whole set, so they contend on the same entity and link rows
    ts = [threading.Thread(target=work, args=(m, ids if i % 2 else ids[::-1]))
          for i, m in enumerate(ms)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs
    with conn_of(m0) as c:
        assert c.execute("SELECT count(*) FROM memory_link_state WHERE status = 'error'"
                         ).fetchone()[0] == 0
        assert c.execute("SELECT count(*) FROM memory_link_state").fetchone()[0] == 16
    for m in ms:
        m.close()
    m0.close()


def test_deadlock_is_retried_with_backoff(mem, monkeypatch):
    _seed(mem, 2)
    monkeypatch.setattr("prospecta._linker.time.sleep", lambda s: None)
    calls = {"n": 0}

    class Flaky(Linker):
        def _temporal(self, conn, *a):
            calls["n"] += 1
            if calls["n"] < 3:
                raise psycopg.errors.DeadlockDetected("deadlock detected")
            return super()._temporal(conn, *a)

    mem._linker = Flaky()
    stats = mem.link_document(next(iter(docs(mem).values())))
    assert calls["n"] == 3 and stats["errors"] == []


def test_failed_step_is_loud_recorded_and_picked_up_by_link_pending(mem, monkeypatch, caplog):
    _seed(mem, 2)
    monkeypatch.setattr("prospecta._linker.time.sleep", lambda s: None)

    class Broken(Linker):
        def _temporal(self, conn, *a):
            raise psycopg.errors.DeadlockDetected("deadlock detected")

    mem._linker = Broken()
    with caplog.at_level("ERROR"):
        stats = mem.link_document(next(iter(docs(mem).values())))
    assert "temporal" in stats["errors"][0] and "deadlock" in stats["errors"][0]
    assert any(r.levelname == "ERROR" and "linker temporal failed" in r.getMessage()
               for r in caplog.records)
    with conn_of(mem) as c:
        assert c.execute("SELECT count(*) FROM memory_link_state WHERE status='error'"
                         ).fetchone()[0] == 1
        pend = pending_documents(c, "b", 10)
    assert len(pend) == 2   # the errored document is pending again, after the fresh one
    mem._linker = Linker()
    assert mem.link_pending() == 2
    with conn_of(mem) as c:
        assert c.execute("SELECT count(*) FROM memory_link_state WHERE status='error'"
                         ).fetchone()[0] == 0
