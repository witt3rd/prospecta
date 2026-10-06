"""Bank embedding migration (design 8.4): backfill, resume, shadow reads,
and a dump -> restore-into-scratch -> migrate rehearsal. All with stub
embedders; never a live database."""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest

from prospecta._embed_migrate import migrate_bank, retrying_embed
from prospecta.memory import BankConfigConflict, Memory
from tests._stub_embedder import stub_embed

NEW_DIM = 64


def embed64(texts):
    """Deterministic 64-dim stub (the 'new model'): 32-d stub, twice."""
    return [v + [-x for x in v] for v in stub_embed(texts)]


def _paras(n, size, tag=""):
    return "\n\n".join(f"P{i}{tag} " + ("word " * size).strip() for i in range(n))


@pytest.fixture
def old_bank(memory_with_bank):
    """'test' 32-d bank: 5 docs (3 multi-chunk), 2 questions each."""
    m = memory_with_bank
    bodies = [_paras(1, 10, "a"), _paras(12, 30, "b"), _paras(20, 30, "c"),
              _paras(2, 20, "d"), _paras(15, 30, "e")]
    for i, body in enumerate(bodies):
        m.retain(body, index_text=[f"What is doc {i}?", f"Who wrote doc {i}?"],
                 source=f"src{i}")
    return m


def _snapshot(url, bank):
    with psycopg.connect(url) as c:
        return c.execute(
            "SELECT content, original_chunk, kind, embedding::text FROM memory_items "
            "WHERE bank_id=%s ORDER BY content, kind", (bank,)).fetchall()


def _counts(url, bank):
    with psycopg.connect(url) as c:
        d = c.execute("SELECT count(*) FROM documents WHERE bank_id=%s", (bank,)).fetchone()[0]
        k = dict(c.execute(
            "SELECT kind, count(*) FROM memory_items WHERE bank_id=%s GROUP BY kind",
            (bank,)).fetchall())
    return d, k


def test_migrate_creates_v2_bank_and_backfills(old_bank, fresh_db):
    before = _snapshot(fresh_db, "test")
    r = migrate_bank(old_bank, "test", embed=embed64, embedding_dim=NEW_DIM,
                     embedding_model_id="openai/text-embedding-3-large@1536")
    assert r.target_bank == "test-v2" and r.remaining == 0
    assert r.documents_migrated == 5 and r.questions_inserted == 10
    with psycopg.connect(fresh_db) as c:
        dim, model = c.execute(
            "SELECT embedding_dim, embedding_model_id FROM banks WHERE bank_id='test-v2'"
        ).fetchone()
        assert (dim, model) == (NEW_DIM, "openai/text-embedding-3-large@1536")
        # chunks tie back to the new parent doc, offsets index original_text
        rows = c.execute(
            "SELECT i.original_chunk, i.char_start, i.char_end, d.original_text, "
            "d.document_metadata ->> 'migrated_from_document_id', vector_dims(i.embedding) "
            "FROM memory_items i JOIN documents d ON d.id=i.document_id "
            "WHERE i.bank_id='test-v2' AND i.kind='chunk'").fetchall()
        assert rows and len(rows) == r.chunks_inserted
        for chunk, cs, ce, text, origin, vd in rows:
            assert text[cs:ce] == chunk and len(chunk) <= 1000 and origin and vd == NEW_DIM
        qrows = c.execute(
            "SELECT content, llm_generated, vector_dims(embedding) FROM memory_items "
            "WHERE bank_id='test-v2' AND kind='question'").fetchall()
        assert len(qrows) == 10 and all(q[2] == NEW_DIM for q in qrows)
    # the old bank is untouched, row for row
    assert _snapshot(fresh_db, "test") == before


def test_migrate_resumes_by_document_id(old_bank, fresh_db):
    r1 = migrate_bank(old_bank, "test", embed=embed64, embedding_dim=NEW_DIM, max_documents=2)
    assert r1.documents_migrated == 2 and r1.remaining == 3
    r2 = migrate_bank(old_bank, "test", embed=embed64, embedding_dim=NEW_DIM)
    assert r2.documents_already_done == 2 and r2.documents_migrated == 3 and r2.remaining == 0
    r3 = migrate_bank(old_bank, "test", embed=embed64, embedding_dim=NEW_DIM)
    assert r3.documents_migrated == 0 and r3.chunks_inserted == 0

    # identical to a clean single run
    old_bank.create_bank("clean", embedding_dim=NEW_DIM)
    migrate_bank(old_bank, "test", target_bank="clean", embed=embed64, embedding_dim=NEW_DIM)
    assert _counts(fresh_db, "test-v2") == _counts(fresh_db, "clean")
    assert _snapshot(fresh_db, "test-v2") == _snapshot(fresh_db, "clean")


def test_migrate_crash_midway_then_resume_loses_and_duplicates_nothing(old_bank, fresh_db):
    calls = {"n": 0}

    def flaky(texts):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("provider down")
        return embed64(texts)

    with pytest.raises(RuntimeError, match="provider down"):
        migrate_bank(old_bank, "test", embed=flaky, embedding_dim=NEW_DIM,
                     batch_size=8, retries=0)
    partial, _ = _counts(fresh_db, "test-v2")
    assert 0 <= partial < 5
    r = migrate_bank(old_bank, "test", embed=embed64, embedding_dim=NEW_DIM, batch_size=8)
    assert r.remaining == 0
    old_bank.create_bank("clean", embedding_dim=NEW_DIM)
    migrate_bank(old_bank, "test", target_bank="clean", embed=embed64, embedding_dim=NEW_DIM)
    assert _snapshot(fresh_db, "test-v2") == _snapshot(fresh_db, "clean")


@pytest.mark.parametrize("batch_size", [8, 128])
def test_embed_batches_never_exceed_batch_size(old_bank, batch_size):
    sizes = []

    def rec(texts):
        sizes.append(len(texts))
        return embed64(texts)

    r = migrate_bank(old_bank, "test", embed=rec, embedding_dim=NEW_DIM,
                     batch_size=batch_size)
    assert max(sizes) <= batch_size and r.embed_batches == len(sizes)
    assert sum(sizes) == r.chunks_inserted + r.questions_inserted
    assert batch_size == 128 or any(s == batch_size for s in sizes)  # full batches used


def test_retrying_embed_backs_off_then_succeeds():
    delays, calls = [], {"n": 0}

    def f(texts):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("429")
        return [[0.0]] * len(texts)

    out = retrying_embed(f, base_delay=1.0, sleep=delays.append)(["a"])
    assert out == [[0.0]] and delays == [1.0, 2.0]


def test_dimension_mismatch_and_conflicts_are_refused(old_bank):
    with pytest.raises(RuntimeError, match="dim"):
        migrate_bank(old_bank, "test", embed=stub_embed, embedding_dim=NEW_DIM, retries=0)
    with pytest.raises(ValueError):
        migrate_bank(old_bank, "test", target_bank="test", embed=embed64,
                     embedding_dim=NEW_DIM)
    with pytest.raises(ValueError, match="does not exist"):
        migrate_bank(old_bank, "nope", embed=embed64, embedding_dim=NEW_DIM)
    old_bank.create_bank("other-v2", embedding_dim=16)
    with pytest.raises(BankConfigConflict):
        migrate_bank(old_bank, "test", target_bank="other-v2", embed=embed64,
                     embedding_dim=NEW_DIM)


def test_shadow_recall_stores_old_and_new_events(old_bank, fresh_db):
    migrate_bank(old_bank, "test", embed=embed64, embedding_dim=NEW_DIM)
    m = Memory(database_url=fresh_db, bank_id="test", embed=stub_embed,
               shadow_bank_id="test-v2", shadow_embed=embed64)
    try:
        res = m.recall(["What is doc 1?"], limit=5, mode="semantic")
    finally:
        m.close()
    assert res and all(r.bank_id == "test" for r in res)  # caller sees old only
    with psycopg.connect(fresh_db) as c:
        rows = c.execute(
            "SELECT bank_id, n_results, trace -> 'shadow' FROM recall_events "
            "ORDER BY bank_id").fetchall()
    assert [r[0] for r in rows] == ["test", "test-v2"]
    old, new = rows[0][2], rows[1][2]
    assert old["id"] == new["id"] and old["role"] == "old" and new["role"] == "new"
    assert old["peer_bank"] == "test-v2" and new["peer_bank"] == "test"
    assert new["error"] is None and rows[1][1] > 0
    assert 0.0 < old["jaccard"] <= 1.0  # origin ids line up across banks


def test_shadow_failure_never_breaks_recall(old_bank, fresh_db):
    migrate_bank(old_bank, "test", embed=embed64, embedding_dim=NEW_DIM)

    def boom(texts):
        raise RuntimeError("shadow embedder down")

    m = Memory(database_url=fresh_db, bank_id="test", embed=stub_embed,
               shadow_bank_id="test-v2", shadow_embed=boom)
    try:
        res = m.recall(["What is doc 1?"], limit=5, mode="semantic")
    finally:
        m.close()
    assert res
    with psycopg.connect(fresh_db) as c:
        err = c.execute(
            "SELECT trace -> 'shadow' ->> 'error' FROM recall_events "
            "WHERE bank_id='test-v2'").fetchone()[0]
    assert "shadow embedder down" in err


def test_no_shadow_by_default_writes_one_event(old_bank, fresh_db):
    old_bank.recall(["What is doc 1?"], limit=3, mode="semantic")
    with psycopg.connect(fresh_db) as c:
        assert c.execute("SELECT count(*) FROM recall_events").fetchone()[0] == 1


def test_shadow_config_validation(fresh_db):
    with pytest.raises(ValueError):
        Memory(database_url=fresh_db, bank_id="a", shadow_bank_id="b")
    with pytest.raises(ValueError):
        Memory(database_url=fresh_db, bank_id="a", shadow_bank_id="a", shadow_embed=embed64)


@pytest.mark.skipif(not shutil.which("pg_dump") and not os.environ.get("PROSPECTA_PG_BIN"),
                    reason="pg_dump/pg_restore not available")
def test_rehearsal_dump_restore_scratch_then_migrate(old_bank, fresh_db, pg_container):
    """The runbook, end to end: dump, restore into a scratch DB, compare
    counts, migrate on the scratch copy only; the source DB is untouched."""
    bindir = os.environ.get("PROSPECTA_PG_BIN")
    pg_dump = os.path.join(bindir, "pg_dump") if bindir else shutil.which("pg_dump")
    pg_restore = os.path.join(bindir, "pg_restore") if bindir else shutil.which("pg_restore")
    dump = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"q3-{time.time_ns()}.dump")
    scratch = f"scratch_{time.time_ns()}"
    base = fresh_db.rsplit("/", 1)[0] if "host=" not in fresh_db else None
    parsed = urlparse(fresh_db)
    scratch_url = urlunparse(parsed._replace(path=f"/{scratch}"))
    admin_url = urlunparse(parsed._replace(path="/postgres"))
    try:
        subprocess.run([pg_dump, "-Fc", "-f", dump, fresh_db], check=True)
        subprocess.run([pg_restore, "--list", dump], check=True, capture_output=True)
        with psycopg.connect(admin_url, autocommit=True) as c:
            c.execute(f'CREATE DATABASE "{scratch}"')
        with psycopg.connect(scratch_url, autocommit=True) as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        subprocess.run([pg_restore, "--no-owner", "-d", scratch_url, dump],
                       check=True, capture_output=True)
        assert _counts(scratch_url, "test") == _counts(fresh_db, "test")

        mem = Memory(database_url=scratch_url, bank_id="test", embed=stub_embed)
        try:
            r = migrate_bank(mem, "test", embed=embed64, embedding_dim=NEW_DIM)
        finally:
            mem.close()
        assert r.remaining == 0 and r.documents_migrated == 5
        assert _counts(scratch_url, "test") == _counts(fresh_db, "test")
        # the database we dumped from never got a v2 bank
        with psycopg.connect(fresh_db) as c:
            assert c.execute("SELECT count(*) FROM banks WHERE bank_id='test-v2'"
                             ).fetchone()[0] == 0
    finally:
        if os.path.exists(dump):
            os.remove(dump)
        with psycopg.connect(admin_url, autocommit=True) as c:
            c.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s",
                (scratch,))
            c.execute(f'DROP DATABASE IF EXISTS "{scratch}"')
