"""Tests for migration runner + 0001_initial.sql."""
import multiprocessing as mp
import time
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest

from prospecta.db.migrate import get_schema_version, run_migrations


@pytest.fixture
def fresh_db(pg_container):
    """Spin up a fresh database within the session container per test."""
    base_url = pg_container.get_connection_url()
    if "+psycopg2" in base_url:
        base_url = base_url.replace("+psycopg2", "")
    if "+psycopg" in base_url:
        base_url = base_url.replace("+psycopg", "")

    db_name = f"test_migrate_{int(time.time() * 1000)}"
    with psycopg.connect(base_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{db_name}"')

    parsed = urlparse(base_url)
    new_url = urlunparse(parsed._replace(path=f"/{db_name}"))

    # The new DB doesn't have the vector extension by default; install it.
    with psycopg.connect(new_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    yield new_url

    with psycopg.connect(base_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (db_name,),
            )
            cur.execute(f'DROP DATABASE "{db_name}"')


def test_run_migrations_creates_version_table(fresh_db):
    result = run_migrations(fresh_db)
    assert 1 in result["applied"]
    assert get_schema_version(fresh_db) == 1


def test_run_migrations_idempotent(fresh_db):
    run_migrations(fresh_db)
    result = run_migrations(fresh_db)
    assert result["applied"] == []
    assert 1 in result["skipped"]
    assert get_schema_version(fresh_db) == 1


def test_all_required_tables_exist(fresh_db):
    run_migrations(fresh_db)
    expected = {
        "prospecta_schema_version",
        "banks",
        "documents",
        "memory_items",
        "retain_events",
        "recall_events",
        "formulate_events",
        "llm_calls",
        "sweep_passes",
        "sweeper_state",
    }
    with psycopg.connect(fresh_db) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            actual = {row[0] for row in cur.fetchall()}
    missing = expected - actual
    assert not missing, f"Missing tables: {missing}"


def test_dim_check_trigger_fires(fresh_db):
    """Trigger should raise when embedding dim != bank's embedding_dim."""
    run_migrations(fresh_db)
    with psycopg.connect(fresh_db, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO banks (bank_id, embedding_dim) VALUES ('test', 384)"
            )
            cur.execute(
                "INSERT INTO documents (bank_id, original_text, content_hash) "
                "VALUES ('test', 'hello', 'h1') RETURNING id"
            )
            doc_id = cur.fetchone()[0]
            wrong_vec = "[" + ",".join(["0.1"] * 100) + "]"  # 100-dim
            with pytest.raises(psycopg.errors.RaiseException):
                cur.execute(
                    "INSERT INTO memory_items "
                    "(bank_id, document_id, content, original_chunk, embedding, llm_generated) "
                    "VALUES ('test', %s, 'c', 'o', %s::vector, FALSE)",
                    (doc_id, wrong_vec),
                )


def test_dim_check_trigger_allows_correct_dim(fresh_db):
    run_migrations(fresh_db)
    with psycopg.connect(fresh_db, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO banks (bank_id, embedding_dim) VALUES ('test', 384)"
            )
            cur.execute(
                "INSERT INTO documents (bank_id, original_text, content_hash) "
                "VALUES ('test', 'hello', 'h1') RETURNING id"
            )
            doc_id = cur.fetchone()[0]
            correct_vec = "[" + ",".join(["0.1"] * 384) + "]"
            cur.execute(
                "INSERT INTO memory_items "
                "(bank_id, document_id, content, original_chunk, embedding, llm_generated) "
                "VALUES ('test', %s, 'c', 'o', %s::vector, FALSE)",
                (doc_id, correct_vec),
            )
            cur.execute("SELECT COUNT(*) FROM memory_items WHERE bank_id = 'test'")
            assert cur.fetchone()[0] == 1


def test_pgvector_extension_loaded(fresh_db):
    run_migrations(fresh_db)
    with psycopg.connect(fresh_db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            assert cur.fetchone() is not None


def _worker_run_migrations(database_url, results_queue, idx):
    """Worker function for advisory-lock concurrency test."""
    try:
        result = run_migrations(database_url)
        results_queue.put((idx, "ok", result))
    except Exception as e:
        results_queue.put((idx, "error", str(e)))


def test_advisory_lock_serializes_concurrent_migrations(fresh_db):
    """4 workers calling run_migrations() against fresh DB; exactly one runs
    the migration; others observe version row already present."""
    ctx = mp.get_context("spawn")
    results_queue = ctx.Queue()
    workers = [
        ctx.Process(target=_worker_run_migrations, args=(fresh_db, results_queue, i))
        for i in range(4)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=60)
        assert not w.is_alive(), "Worker hung"

    outcomes = []
    while not results_queue.empty():
        outcomes.append(results_queue.get())

    assert len(outcomes) == 4
    error_outcomes = [o for o in outcomes if o[1] == "error"]
    assert not error_outcomes, f"Workers errored: {error_outcomes}"

    applied_count = sum(1 for _, _, r in outcomes if 1 in r["applied"])
    skipped_count = sum(1 for _, _, r in outcomes if 1 in r["skipped"])
    assert applied_count == 1, f"Expected exactly 1 worker to apply; got {applied_count}"
    assert skipped_count == 3, f"Expected 3 workers to skip; got {skipped_count}"

    with psycopg.connect(fresh_db) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM prospecta_schema_version WHERE version = 1")
            assert cur.fetchone()[0] == 1
