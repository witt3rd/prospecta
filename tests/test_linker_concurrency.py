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


# ----------------------------------------------------------- multi-writer (defect 4)

def _clustered(n=24):
    return [(f"c{i}", f"Kelly and Forge at Acme: topic {i % 4} cluster note {i}",
             {"created": f"2024-01-{1 + i:02d}", "person": "kelly" if i % 3 else "forge"})
            for i in range(n)]


def _link_set(m, bank):
    with conn_of(m) as c:
        return set(c.execute(
            "SELECT s.source, d.source, l.link_type, l.subtype FROM memory_links l "
            "JOIN memory_items a ON a.id = l.src JOIN documents s ON s.id = a.document_id "
            "JOIN memory_items b ON b.id = l.dst JOIN documents d ON d.id = b.document_id "
            "WHERE l.bank_id = %s", (bank,)).fetchall())


def _bank_docs(m, bank):
    with conn_of(m) as c:
        return [str(r[0]) for r in c.execute(
            "SELECT id FROM documents WHERE bank_id = %s ORDER BY source", (bank,))]


def _run_threads(fn, parts):
    errs = []

    def wrap(p):
        try:
            fn(p)
        except Exception as exc:
            errs.append(exc)
    ts = [threading.Thread(target=wrap, args=(p,)) for p in parts]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return errs


def _new(url, bank_id="b", **kw):
    return Memory(database_url=url, bank_id=bank_id, llm=None, embed=stub_embed, **kw)


def _state_counts(m, bank):
    with conn_of(m) as c:
        return c.execute(
            "SELECT count(*) FILTER (WHERE status = 'error'), count(*) "
            "FROM memory_link_state WHERE bank_id = %s", (bank,)).fetchone()


def _mk(url, bank, linker=True):
    return _new(url, bank_id=bank,
                linker=Linker(llm=EntityLLM(), asynchronous=False, entity_hub=1000) if linker else None)


def test_eight_writers_link_the_same_set_as_a_serial_run(fresh_db):
    corpus = _clustered()
    # serial reference (bank s): retain all, then the eight walks in turn (a single walk links
    # each pair from the later note only, so the comparison needs the same walks)
    ref = _mk(fresh_db, "s", linker=False)
    ref.create_bank("s", embedding_dim=EMBED_DIM)
    for src, text, md in corpus:
        ref.retain(text, source=src, index_text=f"q {src}", metadata=md)
    ref._linker = Linker(llm=EntityLLM(), entity_hub=1000)   # a hub cut depends on arrival order
    sids = _bank_docs(ref, "s")
    for i in range(8):   # the same eight walks the concurrent linkers make, one after another
        for d in sids[i:] + sids[:i]:
            ref.link_document(d, "s")
    for d in sids:   # a final pass: every note then sees every other note's entities
        ref.link_document(d, "s")
    serial = _link_set(ref, "s")
    assert serial
    # 8 writers (own Memory each) retain concurrently into bank b ...
    writers = [_mk(fresh_db, "b", linker=False) for _ in range(8)]
    writers[0].create_bank("b", embedding_dim=EMBED_DIM)

    def retain(i):
        for src, text, md in corpus[i::8]:
            writers[i].retain(text, source=src, index_text=f"q {src}", metadata=md)
    assert _run_threads(retain, range(8)) == []
    # ... then 8 linkers walk the whole set in their own order: contention on every document
    linkers = [_mk(fresh_db, "b") for _ in range(8)]
    ids = _bank_docs(writers[0], "b")
    assert _run_threads(lambda i: [linkers[i].link_document(d, "b") for d in
                                   (ids[i:] + ids[:i])], range(8)) == []
    for d in ids:   # the same final pass (a concurrent walk may link a pair from one side only)
        linkers[0].link_document(d, "b")
    assert _state_counts(ref, "b") == (0, len(corpus))
    assert _link_set(ref, "b") == serial
    [m.close() for m in writers + linkers + [ref]]


def test_eight_writers_retaining_with_inline_linking_fail_nothing(fresh_db):
    corpus = _clustered()
    writers = [_mk(fresh_db, "i") for _ in range(8)]
    writers[0].create_bank("i", embedding_dim=EMBED_DIM)

    def retain(i):
        for src, text, md in corpus[i::8]:
            writers[i].retain(text, source=src, index_text=f"q {src}", metadata=md)
    assert _run_threads(retain, range(8)) == []
    assert _state_counts(writers[0], "i") == (0, len(corpus))
    [w.close() for w in writers]


def test_same_document_is_linked_once_under_the_advisory_lock(mem):
    _seed(mem, 3)
    d = next(iter(docs(mem).values()))
    inside, peak, lock = [0], [0], threading.Lock()

    class Probe(Linker):
        def _temporal(self, conn, *a):
            with lock:
                inside[0] += 1
                peak[0] = max(peak[0], inside[0])
            import time
            time.sleep(0.1)
            try:
                return super()._temporal(conn, *a)
            finally:
                with lock:
                    inside[0] -= 1
    mem._linker = Probe()
    assert _run_threads(lambda _: mem.link_document(d), range(4)) == []
    assert peak[0] == 1


def test_one_memory_shared_by_threads_uses_a_connection_per_thread(mem):
    seen = []

    def grab(_):
        with mem._pool.connection() as c:
            seen.append(id(c))
            c.execute("SELECT 1")
    assert _run_threads(grab, range(4)) == []
    assert len(set(seen)) == 4


def test_linking_a_deleted_document_is_skipped_not_half_linked(mem):
    _seed(mem, 2)
    mem._linker = Linker()
    stats = mem.link_document("00000000-0000-0000-0000-000000000000")
    assert stats["skipped"] == "document gone"
    with conn_of(mem) as c:
        assert c.execute("SELECT count(*) FROM memory_link_state").fetchone()[0] == 0
