"""The stored exact vector (migration 0020): memory_items.exact_tsv.

The Stage 1 exact retriever matches and ranks the 'simple' word vector of
original_chunk. Computing that vector per matching row made a query of only very common
words cost a tokenization of every chunk that holds them; it is now a STORED generated
column with its own GIN index, defined once (prospecta.db.keyword_index.indexed_expression,
the rule migration 0018 indexes). These tests pin: the column and index exist and follow
the one rule, the migration upgrades and re-runs and rolls back safely, the index build
never waits on a lock (the 0004/0018 deadlock caveat), the Stage 1 plan reads the column
and no longer tokenizes, and the results equal the old expression form row for row."""
from __future__ import annotations

import threading
import time

import psycopg
import pytest

from prospecta.db import migrate
from prospecta.db.exact_tsv import (
    COLUMN, INDEX_NAME, LOCK_KEY, ensure_exact_tsv_index, has_exact_tsv_index,
)
from prospecta.db.keyword_index import indexed_expression
from prospecta.db.migrate import get_schema_version, run_migrations
from prospecta.db.queries import _stage1_statement, stage1_search
from tests._stub_embedder import EMBED_DIM
from tests._synthetic_bank import load_bank, random_query_vector, vocabulary

TEXTS = [
    "Deploy v1.2.3 from src/main.rs using config.yaml today.",
    r"Windows path C:\tools\bin\run.exe failed with error 0x80070005.",
    "The night crew called the bridge lamps electric dreams.",
    "ELECTRIC DREAMS flickered over the harbour.",
    "Env PROSPECTA_DB_POOL_MAX caps connections; ticket ABC-123.",
    "Nothing relevant, just cats.",
    "the and a of",
    "",
]


def put(conn, bank: str, texts, dim: int = EMBED_DIM) -> None:
    if not conn.execute("SELECT 1 FROM banks WHERE bank_id = %s", (bank,)).fetchone():
        conn.execute("INSERT INTO banks (bank_id, embedding_dim) VALUES (%s, %s)", (bank, dim))
    for i, text in enumerate(texts):
        (doc,) = conn.execute(
            "INSERT INTO documents (bank_id, source, original_text, content_hash, created_on) "
            "VALUES (%s, %s, '', %s, DATE '2024-01-01' + %s) RETURNING id",
            (bank, f"doc-{bank}-{i}", f"h{bank}{i}", i)).fetchone()
        conn.execute(
            "INSERT INTO memory_items (bank_id, document_id, content, original_chunk, embedding, kind, ordinal) "
            "VALUES (%s, %s, %s, %s, array_fill(0.1::real, ARRAY[%s])::vector, 'chunk', 0)",
            (bank, doc, text, text, dim))


def column_info(conn):
    return conn.execute(
        "SELECT a.attgenerated, format_type(a.atttypid, a.atttypmod) FROM pg_attribute a "
        "WHERE a.attrelid = 'memory_items'::regclass AND a.attname = %s AND NOT a.attisdropped",
        (COLUMN,)).fetchone()


def roll_back_0020(url):
    """The rollback of the operator runbook: drop the index, the column, the version row."""
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(f'DROP INDEX IF EXISTS "{INDEX_NAME}"')
        conn.execute(f"ALTER TABLE memory_items DROP COLUMN IF EXISTS {COLUMN}")
        conn.execute("DELETE FROM prospecta_schema_version WHERE version = 20")


# --------------------------------------------------------------------------
# the column and its index
# --------------------------------------------------------------------------

def test_migration_adds_a_stored_generated_tsvector_column_and_a_valid_gin_index(fresh_db):
    assert get_schema_version(fresh_db) >= 20
    with psycopg.connect(fresh_db) as conn:
        assert column_info(conn) == ("s", "tsvector")          # 's' = STORED generated
        assert has_exact_tsv_index(conn)
        (indexdef,) = conn.execute("SELECT indexdef FROM pg_indexes WHERE indexname = %s",
                                   (INDEX_NAME,)).fetchone()
        assert "USING gin" in indexdef and f"({COLUMN})" in indexdef


def test_the_column_is_the_one_word_rule_and_follows_every_write(fresh_db):
    with psycopg.connect(fresh_db, autocommit=True) as conn:
        put(conn, "b", TEXTS)
        expr = indexed_expression().replace("original_chunk", "m.original_chunk")
        n, wrong = conn.execute(
            f"SELECT count(*), count(*) FILTER (WHERE m.{COLUMN} IS DISTINCT FROM {expr}) "
            "FROM memory_items m").fetchone()
        assert (n, wrong) == (len(TEXTS), 0)
        # an update is followed too, with no application code involved
        conn.execute("UPDATE memory_items SET original_chunk = 'rewritten zq-77.4' "
                     "WHERE original_chunk LIKE 'Nothing%'")
        assert conn.execute(
            f"SELECT count(*) FROM memory_items WHERE {COLUMN} @@ websearch_to_tsquery('simple', 'zq-77.4')"
        ).fetchone()[0] == 1
        # a path is matched whole and by its pieces (the 0018 rule), a quoted phrase by adjacency
        assert conn.execute(
            f"SELECT count(*) FROM memory_items WHERE {COLUMN} @@ websearch_to_tsquery('simple', 'main.rs')"
        ).fetchone()[0] == 1
        assert conn.execute(
            f"SELECT count(*) FROM memory_items WHERE {COLUMN} @@ websearch_to_tsquery('simple', '\"electric dreams\"')"
        ).fetchone()[0] == 2


def test_the_migration_file_holds_no_copy_of_the_rule():
    """One definition: the SQL file carries a placeholder the runner fills from
    indexed_expression(), so migration, index and query cannot drift."""
    path = next(p for v, p in migrate._list_migrations() if v == 20)
    text = path.read_text()
    assert "to_tsvector" not in text and "translate(" not in text
    rendered = migrate._render(text)
    assert "{{" not in rendered and indexed_expression() in rendered
    assert "STORED" in rendered and COLUMN in rendered


# --------------------------------------------------------------------------
# upgrade, re-run, rollback
# --------------------------------------------------------------------------

def test_upgrade_from_the_previous_release_fills_existing_rows_and_builds_the_index(fresh_db):
    roll_back_0020(fresh_db)
    with psycopg.connect(fresh_db, autocommit=True) as conn:
        assert column_info(conn) is None and not has_exact_tsv_index(conn)
        put(conn, "b", TEXTS)                  # rows written before the upgrade
    result = run_migrations(fresh_db)
    assert result["applied"] == [20]
    with psycopg.connect(fresh_db) as conn:
        assert column_info(conn) == ("s", "tsvector") and has_exact_tsv_index(conn)
        assert conn.execute(
            f"SELECT count(*) FROM memory_items WHERE {COLUMN} @@ websearch_to_tsquery('simple', 'v1.2.3')"
        ).fetchone()[0] == 1
        assert conn.execute(f"SELECT count(*) FROM memory_items WHERE {COLUMN} IS NULL").fetchone()[0] == 0


def test_rerun_changes_nothing_and_repairs_an_interrupted_index_build(fresh_db):
    assert run_migrations(fresh_db)["applied"] == []
    assert ensure_exact_tsv_index(fresh_db)["created"] is False
    with psycopg.connect(fresh_db, autocommit=True) as conn:
        conn.execute("UPDATE pg_index SET indisvalid = false WHERE indexrelid = "
                     f"'{INDEX_NAME}'::regclass")                  # what an interrupted build leaves
        assert not has_exact_tsv_index(conn)
    assert run_migrations(fresh_db)["applied"] == []               # `prospecta migrate` repairs it
    with psycopg.connect(fresh_db) as conn:
        assert has_exact_tsv_index(conn)
    assert ensure_exact_tsv_index(fresh_db, rebuild=True)["rebuilt"] is True
    with psycopg.connect(fresh_db) as conn:
        assert has_exact_tsv_index(conn)


def test_rollback_then_migrate_again_round_trips(fresh_db):
    with psycopg.connect(fresh_db, autocommit=True) as conn:
        put(conn, "b", TEXTS)
    roll_back_0020(fresh_db)
    assert get_schema_version(fresh_db) == 19
    with psycopg.connect(fresh_db, autocommit=True) as conn:
        # the previous release's expression form still answers on the rolled-back schema
        expr = indexed_expression().replace("original_chunk", "m.original_chunk")
        assert conn.execute(
            f"SELECT count(*) FROM memory_items m WHERE {expr} @@ websearch_to_tsquery('simple', 'main.rs')"
        ).fetchone()[0] == 1
    assert run_migrations(fresh_db)["applied"] == [20]
    assert get_schema_version(fresh_db) == 20


def test_building_the_index_polls_for_its_lock_and_never_blocks_on_it(fresh_db):
    """CREATE INDEX CONCURRENTLY waits for every open transaction, so a peer blocked in
    pg_advisory_lock would deadlock it (the 0004/0018 caveat): the build must poll with
    pg_try_advisory_lock. While the lock is held elsewhere no backend may be waiting on an
    advisory lock, and the build finishes once it is released."""
    with psycopg.connect(fresh_db, autocommit=True) as holder, \
            psycopg.connect(fresh_db, autocommit=True) as watcher:
        holder.execute("SELECT pg_advisory_lock(%s)", (LOCK_KEY,))
        done = []
        t = threading.Thread(target=lambda: done.append(ensure_exact_tsv_index(fresh_db, rebuild=True)))
        t.start()
        time.sleep(0.5)
        assert t.is_alive()                                        # waiting its turn ...
        waiting = watcher.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' "
            "AND wait_event = 'advisory'").fetchone()[0]
        assert waiting == 0                                        # ... by polling, not by blocking
        holder.execute("SELECT pg_advisory_unlock(%s)", (LOCK_KEY,))
        t.join(timeout=30)
        assert not t.is_alive() and done[0]["rebuilt"] is True


# --------------------------------------------------------------------------
# Stage 1 reads the column: plan, and results equal to the expression form
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def bank_url(pg_container):
    """One 10,000-chunk synthetic bank (function words in most chunks) for the plan and
    equality tests; read-only, so shared."""
    from tests.conftest import _normalize_url
    base = _normalize_url(pg_container.get_connection_url())
    name = f"test_exact_tsv_{int(time.time() * 1000)}"
    with psycopg.connect(base, autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE "{name}"')
    from urllib.parse import urlparse, urlunparse
    url = urlunparse(urlparse(base)._replace(path=f"/{name}"))
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    run_migrations(url)
    load_bank(url, "eq", n_chunks=10_000, dim=32, function_words=0.6)
    with psycopg.connect(url, autocommit=True) as conn:
        put(conn, "eq", TEXTS, dim=32)
        conn.execute("ANALYZE memory_items")
    yield url
    with psycopg.connect(base, autocommit=True) as conn:
        conn.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = %s "
                     "AND pid <> pg_backend_pid()", (name,))
        conn.execute(f'DROP DATABASE "{name}"')


V = vocabulary()
EQUALITY_QUERIES = [
    '"electric dreams"', "electric dreams", "0x80070005", "v1.14", "main.rs", "src/main.rs",
    "harbour OR crew -lamps", "what did the crew say about the harbour lamps",
    "the", "the and a of", "to in", V[0], V[20], V[99], f"{V[30]} {V[50]}", f"{V[1500]} OR {V[1501]}",
    f'"{V[0]} {V[1]}"', "zzzunseen", "ABC-123", "PROSPECTA_DB_POOL_MAX", "",
]


def _old_form(sql: str) -> str:
    """The statement as it was before 0020: the exact retriever over the expression."""
    assert f"m.{COLUMN}" in sql
    return sql.replace(f"m.{COLUMN}", indexed_expression().replace("original_chunk", "m.original_chunk"))


def _run(conn, sql, params):
    with conn.transaction():
        conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
        conn.execute("SET LOCAL hnsw.ef_search = 100")
        cur = conn.execute(sql, params, prepare=False)
        cols = [c.name for c in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


@pytest.mark.parametrize("query", EQUALITY_QUERIES)
def test_results_equal_the_expression_form_row_for_row(bank_url, query):
    sql, params = _stage1_statement(bank_id="eq", query_text=query,
                                    query_embedding=random_query_vector(32, 3))
    with psycopg.connect(bank_url) as conn:
        new = _run(conn, sql, params)
        old = _run(conn, _old_form(sql), params)
    assert new == old        # same chunks, same order, same ranks, same scores and snippets


def test_the_statement_has_no_copy_of_the_word_rule():
    sql, _ = _stage1_statement(bank_id="b", query_text="x", query_embedding=[0.1] * 4)
    assert "translate(" not in sql
    assert sql.count(f"m.{COLUMN}") == 3         # the rank window, the match, the order


@pytest.mark.parametrize("query", ["the", "the and a of", V[99], "0x80070005"])
def test_stage1_plan_reads_the_stored_vector_and_never_tokenizes(bank_url, query):
    sql, params = _stage1_statement(bank_id="eq", query_text=query,
                                    query_embedding=random_query_vector(32, 3))
    with psycopg.connect(bank_url) as conn:
        plan = "\n".join(r[0] for r in conn.execute("EXPLAIN (VERBOSE) " + sql, params, prepare=False))
    assert "to_tsvector" not in plan and "translate" not in plan, plan   # nothing re-tokenizes a chunk
    assert COLUMN in plan


def test_stage1_still_finds_the_planted_needles(bank_url):
    with psycopg.connect(bank_url) as conn:
        rows = stage1_search(conn, bank_id="eq", query_text='"electric dreams"',
                             query_embedding=random_query_vector(32, 3))
    assert sum(r["exact_rank"] is not None for r in rows) >= 2
