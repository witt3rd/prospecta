"""Stored keyword index: same results with and without it, planner uses it, rebuild is safe."""
from __future__ import annotations

import psycopg
import pytest

from prospecta.channels import QueryPlan, RecallState
from prospecta.channels.keyword import KeywordChunks
from prospecta.db.keyword_index import INDEX_NAME, ensure_keyword_index, has_keyword_index, indexed_expression
from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM, stub_embed, stub_llm

TEXTS = {
    "a.md": "Deploy v1.2.3 from src/main.rs using config.yaml today.",
    "b.md": "Env PROSPECTA_DB_POOL_MAX caps connections; ticket ABC-123.",
    "c.md": "Commit 3f9a1c7e2b fixed it.",
    "d.md": "Nothing relevant, just cats.",
}
QUERIES = ["v1.2.3", "main.rs", "src/main.rs", "config.yaml", "PROSPECTA_DB_POOL_MAX",
           "ABC-123", "3f9a1c", "cats and dogs", "zzzunseen",
           '"v1.2.3 from src/main.rs"', '"from src"', 'using "config.yaml" today']


@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=stub_llm, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    for src, t in TEXTS.items():
        m.retain(t, source=src, index_text="zzz " + src)
    yield m
    m.close()


def _run(url, q):
    with psycopg.connect(url) as conn:
        ch = KeywordChunks()
        try:
            out = ch.retrieve(QueryPlan(text=q), RecallState(conn=conn, bank_id="b"))
        except ValueError:
            out = []
        return [(c.source, round(c.score, 6)) for c in out], ch.used_index


def test_migration_builds_a_valid_index(fresh_db):
    with psycopg.connect(fresh_db) as conn:
        assert has_keyword_index(conn)


def test_identifier_queries_find_same_passages_with_and_without_index(mem):
    url = mem.database_url
    with_idx = {q: _run(url, q) for q in QUERIES}
    assert all(used for _, used in with_idx.values())
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(f'DROP INDEX "{INDEX_NAME}"')
    without = {q: _run(url, q) for q in QUERIES}
    assert not any(used for _, used in without.values())
    assert {q: r for q, (r, _) in with_idx.items()} == {q: r for q, (r, _) in without.items()}
    assert [s for s, _ in with_idx["v1.2.3"][0]] == ["a.md"]
    assert [s for s, _ in with_idx["3f9a1c"][0]] == ["c.md"]


def test_index_is_kept_current_by_the_database(mem):
    mem.retain("Brand new id zq-77.4 appears here.", source="e.md", index_text="zzz e")
    got, used = _run(mem.database_url, "zq-77.4")
    assert used and [s for s, _ in got] == ["e.md"]


def test_ensure_is_idempotent_repairs_invalid_and_rebuild_keeps_serving(mem):
    url = mem.database_url
    assert ensure_keyword_index(url)["created"] is False
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("UPDATE pg_index SET indisvalid = false WHERE indexrelid = "
                     f"'{INDEX_NAME}'::regclass")
        assert not has_keyword_index(conn)
    assert _run(url, "v1.2.3")[1] is False            # fallback scan still answers
    assert ensure_keyword_index(url)["created"] is True
    assert ensure_keyword_index(url, rebuild=True)["rebuilt"] is True
    got, used = _run(url, "v1.2.3")
    assert used and [s for s, _ in got] == ["a.md"]


def test_stemmed_query_falls_back_to_scan(mem):
    with psycopg.connect(mem.database_url) as conn:
        ch = KeywordChunks(stemming=True)
        out = ch.retrieve(QueryPlan(text="cat"), RecallState(conn=conn, bank_id="b"))
    assert [c.source for c in out] == ["d.md"] and ch.used_index is False


def test_explain_uses_index_on_50000_documents(mem):
    url = mem.database_url
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute("""
            WITH d AS (
              INSERT INTO documents (bank_id, source, original_text, content_hash)
              SELECT 'b', 'bulk' || g, 'filler text number ' || g || ' lorem ipsum dolor',
                     'h' || g FROM generate_series(1, 50000) g RETURNING id, original_text)
            INSERT INTO memory_items (bank_id, document_id, content, original_chunk, embedding)
            SELECT 'b', id, 'q', original_text, array_fill(0.1::real, ARRAY[%s])::vector FROM d
        """, (EMBED_DIM,))
        conn.execute("ANALYZE memory_items")
        expr = indexed_expression()
        plan = "\n".join(r[0] for r in conn.execute(
            f"EXPLAIN SELECT id FROM memory_items WHERE bank_id = 'b' AND {expr} @@ "
            "to_tsquery('simple', $$'v1.2.3'$$)").fetchall())
    assert INDEX_NAME in plan, plan
    # and the channel itself, on the big bank
    got, used = _run(url, "v1.2.3")
    assert used and [s for s, _ in got] == ["a.md"]
