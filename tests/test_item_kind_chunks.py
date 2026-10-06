"""0004_item_kind_parent: migration on a populated table, per-kind HNSW
indexes, the chunker, and the opt-in child-chunk retain path."""
from __future__ import annotations

import psycopg
import pytest

from prospecta._chunker import chunk_paragraphs
from prospecta.db import migrate
from prospecta.db.queries import hnsw_kind_index_name
from tests._stub_embedder import EMBED_DIM, stub_embed


# --------------------------------------------------------------------------
# chunker
# --------------------------------------------------------------------------

def _paras(n, size):
    return "\n\n".join(f"P{i} " + ("word " * size).strip() for i in range(n))


def test_chunker_small_text_single_chunk():
    cs = chunk_paragraphs("Hello there.\n\nSecond para.")
    assert len(cs) == 1
    assert cs[0].content == "Hello there.\n\nSecond para."
    assert (cs[0].ordinal, cs[0].char_start, cs[0].char_end) == (0, 0, len(cs[0].content))


def test_chunker_empty():
    assert chunk_paragraphs("") == []
    assert chunk_paragraphs("  \n\n ") == []


def test_chunker_bounds_offsets_and_paragraph_boundaries():
    text = _paras(12, 30)  # ~150 chars per paragraph
    cs = chunk_paragraphs(text, 1000, 100)
    assert len(cs) > 1
    for i, c in enumerate(cs):
        assert c.ordinal == i
        assert len(c.content) <= 1000
        assert text[c.char_start:c.char_end] == c.content
        assert text[c.char_start:].startswith("P")  # starts at a paragraph
        assert c.char_end == len(text) or text[c.char_end:].startswith("\n\n")
    # overlap: consecutive chunks share the trailing paragraph (<= 100 chars? 150 >
    # 100 here, so none shared) -> contiguous, covering everything
    assert cs[0].char_start == 0 and cs[-1].char_end == len(text)


def test_chunker_overlap_repeats_small_trailing_paragraph():
    text = "\n\n".join(["A" * 400, "B" * 400, "tail " * 4, "C" * 400, "D" * 400])
    cs = chunk_paragraphs(text, 1000, 100)
    assert len(cs) >= 2
    assert cs[0].content.rstrip().endswith("tail tail tail tail")
    assert cs[1].content.startswith("tail")  # the small paragraph is the overlap
    assert all(len(c.content) <= 1000 for c in cs)


def test_chunker_splits_long_paragraph_with_word_overlap():
    text = " ".join(f"w{i}" for i in range(1000))  # one 4k+ char paragraph
    cs = chunk_paragraphs(text, 1000, 100)
    assert len(cs) >= 5
    for a, b in zip(cs, cs[1:]):
        assert len(a.content) <= 1000
        assert b.char_start < a.char_end  # overlap
        assert a.char_end - b.char_start <= 100
        assert text[b.char_start - 1].isspace()  # starts at a word
    assert text[cs[-1].char_start:cs[-1].char_end] == cs[-1].content
    assert cs[-1].char_end == len(text)


# --------------------------------------------------------------------------
# retain
# --------------------------------------------------------------------------

def _items(url, bank="test"):
    with psycopg.connect(url) as conn:
        return conn.execute(
            "SELECT kind, ordinal, char_start, char_end, document_id, original_chunk "
            "FROM memory_items WHERE bank_id=%s ORDER BY kind, ordinal",
            (bank,),
        ).fetchall()


def test_retain_default_writes_only_question_items(memory_with_bank, fresh_db):
    memory_with_bank.retain("Body.\n\nMore body.", index_text=["What body?"], source="s1")
    rows = _items(fresh_db)
    assert [r[0] for r in rows] == ["question"]
    assert rows[0][1:4] == (None, None, None)
    assert rows[0][5] == "Body.\n\nMore body."


def test_retain_child_chunks_writes_chunks_with_parent(memory_with_bank, fresh_db):
    body = _paras(30, 30)
    doc = memory_with_bank.retain(
        body, index_text=["What is P1?", "What is P2?"], source="s2", child_chunks=True
    )
    rows = _items(fresh_db)
    chunks = [r for r in rows if r[0] == "chunk"]
    qs = [r for r in rows if r[0] == "question"]
    assert len(qs) == 2 and len(chunks) > 1
    for i, r in enumerate(chunks):
        assert r[1] == i and str(r[4]) == doc
        assert len(r[5]) <= 1000 and body[r[2]:r[3]] == r[5]
    assert all(str(r[4]) == doc and r[5] == body for r in qs)


def test_retain_child_chunks_replace_regenerates(memory_with_bank, fresh_db):
    memory_with_bank.retain("same", index_text=["q"], source="s3", child_chunks=True)
    memory_with_bank.retain("same", index_text=["q"], source="s3", child_chunks=True)
    assert [r[0] for r in _items(fresh_db)] == ["chunk", "question"]


# --------------------------------------------------------------------------
# migration on a populated table + indexes
# --------------------------------------------------------------------------

def _vec(i):
    return "[" + ",".join(str(float((i + k) % 7)) for k in range(EMBED_DIM)) + "]"


def _apply_upto(url, upto, start=1):
    with psycopg.connect(url) as conn:
        migrate._ensure_version_table(conn)
        for v, path in migrate._list_migrations():
            if start <= v <= upto:
                conn.execute(path.read_text())
                conn.execute(
                    "INSERT INTO prospecta_schema_version (version, description) "
                    "VALUES (%s, %s)", (v, path.stem))
        conn.commit()


@pytest.fixture
def populated_v3(pg_container):
    import time
    from urllib.parse import urlparse, urlunparse
    base = pg_container.get_connection_url().replace("+psycopg2", "").replace("+psycopg", "")
    name = f"test_k4_{int(time.time() * 1_000_000)}"
    with psycopg.connect(base, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{name}"')
    url = urlunparse(urlparse(base)._replace(path=f"/{name}"))
    with psycopg.connect(url, autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
    _apply_upto(url, 3)
    with psycopg.connect(url) as conn:
        conn.execute("INSERT INTO banks (bank_id, embedding_dim) VALUES ('live', %s)", (EMBED_DIM,))
        doc = conn.execute(
            "INSERT INTO documents (bank_id, source, original_text, content_hash) "
            "VALUES ('live','a.md','body','h') RETURNING id").fetchone()[0]
        for i in range(7):  # directory-index chunk rows (v0.1 metadata)
            conn.execute(
                "INSERT INTO memory_items (bank_id, document_id, content, original_chunk, embedding, metadata) "
                "VALUES ('live', %s, %s, %s, %s::vector, %s::jsonb)",
                (doc, f"c{i}", f"c{i}", _vec(i),
                 f'{{"chunk_index": {i}, "start_char": {i*10}, "end_char": {i*10+9}, "source_path": "a.md"}}'))
        for i in range(3):  # question rows
            conn.execute(
                "INSERT INTO memory_items (bank_id, document_id, content, original_chunk, embedding) "
                "VALUES ('live', %s, %s, 'body', %s::vector)", (doc, f"q{i}", _vec(i)))
        conn.commit()
    # the old per-bank index exists, as on a live bank
    from prospecta.db.queries import HNSW_INDEX_TEMPLATE, hnsw_index_name
    with psycopg.connect(url, autocommit=True) as c:
        c.execute(HNSW_INDEX_TEMPLATE.format(
            index_name=hnsw_index_name("live"), bank_id_literal="live", embedding_dim=EMBED_DIM))
    yield url
    with psycopg.connect(base, autocommit=True) as c:
        c.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                  "WHERE datname=%s AND pid<>pg_backend_pid()", (name,))
        c.execute(f'DROP DATABASE "{name}"')


def _kinds(url):
    with psycopg.connect(url) as c:
        return c.execute(
            "SELECT kind, count(*), min(ordinal), max(char_end) FROM memory_items "
            "GROUP BY kind ORDER BY kind").fetchall()


def test_migration_on_populated_table(populated_v3):
    url = populated_v3
    res = migrate.run_migrations(url)
    assert res["applied"] == [4, 10]
    assert _kinds(url) == [("chunk", 7, 0, 69), ("question", 3, None, None)]
    with psycopg.connect(url) as c:
        # column defaults are constants; existing row data untouched
        assert c.execute("SELECT count(*) FROM memory_items").fetchone()[0] == 10
        idx = {r[0]: r[1] for r in c.execute(
            "SELECT indexname, indexdef FROM pg_indexes WHERE tablename='memory_items'")}
        assert "memory_items_bank_kind_idx" in idx
        assert "memory_items_embedding_live_idx" in idx  # old index kept
        for kind in ("question", "chunk"):
            d = idx[hnsw_kind_index_name("live", kind)]
            assert "hnsw" in d and f"vector({EMBED_DIM})" in d
            assert "bank_id = 'live'" in d and f"kind = '{kind}'" in d
        valid = c.execute(
            "SELECT bool_and(indisvalid) FROM pg_index WHERE indrelid='memory_items'::regclass"
        ).fetchone()[0]
        assert valid
    # idempotent
    assert migrate.run_migrations(url)["applied"] == []
    assert migrate.finish_0004(url)["backfilled"] == 0


def test_backfill_is_batched_and_resumable(populated_v3):
    url = populated_v3
    migrate.run_migrations(url)
    with psycopg.connect(url) as c:
        c.execute("UPDATE memory_items SET kind='question', ordinal=NULL, "
                  "char_start=NULL, char_end=NULL")
        c.commit()
    assert migrate.finish_0004(url, batch_size=2)["backfilled"] == 7
    assert _kinds(url)[0][:2] == ("chunk", 7)


def test_finish_0004_runs_after_crash_post_commit(populated_v3):
    url = populated_v3
    _apply_upto(url, 4, start=4)
    assert _kinds(url) == [("question", 10, None, None)]
    res = migrate.run_migrations(url)
    assert res["applied"] == [10]
    assert _kinds(url) == [("chunk", 7, 0, 69), ("question", 3, None, None)]
    with psycopg.connect(url) as c:
        names = {r[0] for r in c.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename='memory_items'")}
    assert "memory_items_bank_kind_idx" in names
    assert hnsw_kind_index_name("live", "chunk") in names
    assert hnsw_kind_index_name("live", "question") in names


def test_invalid_index_is_rebuilt(populated_v3):
    url = populated_v3
    migrate.run_migrations(url)
    name = hnsw_kind_index_name("live", "chunk")
    with psycopg.connect(url, autocommit=True) as c:
        c.execute("UPDATE pg_index SET indisvalid = false WHERE indexrelid = %s::regclass", (name,))
    migrate.run_migrations(url)
    with psycopg.connect(url) as c:
        assert c.execute(
            "SELECT indisvalid FROM pg_index WHERE indexrelid = %s::regclass", (name,)
        ).fetchone()[0]


def test_kind_index_name_fits_and_is_unique_for_long_banks():
    a = hnsw_kind_index_name("x" * 63, "chunk")
    b = hnsw_kind_index_name("x" * 62 + "y", "chunk")
    assert len(a) <= 63 and len(b) <= 63 and a != b
    assert hnsw_kind_index_name("my-bank", "question") == "memory_items_embedding_my_bank_question_idx"


def test_create_bank_makes_per_kind_indexes(memory_with_bank, fresh_db):
    with psycopg.connect(fresh_db) as c:
        names = {r[0] for r in c.execute(
            "SELECT indexname FROM pg_indexes WHERE tablename='memory_items'")}
    assert hnsw_kind_index_name("test", "chunk") in names
    assert hnsw_kind_index_name("test", "question") in names


def test_chunk_search_uses_partial_index_without_starvation(memory_with_bank, fresh_db):
    # many question items, few chunk items: a kind filter must still return chunks
    for i in range(40):
        memory_with_bank.retain(f"note {i} alpha", index_text=[f"question {i}"], source=f"n{i}")
    memory_with_bank.retain("target chunk text", index_text=["tq"], source="t", child_chunks=True)
    q = "[" + ",".join(str(x) for x in stub_embed(["target chunk text"])[0]) + "]"
    with psycopg.connect(fresh_db) as c:
        c.execute("SET enable_seqscan = off")
        sql = (f"SELECT kind FROM memory_items WHERE bank_id='test' AND kind='chunk' "
               f"ORDER BY embedding::vector({EMBED_DIM}) <=> %s::vector LIMIT 5")
        rows = c.execute(sql, (q,)).fetchall()
        plan = "\n".join(r[0] for r in c.execute("EXPLAIN " + sql, (q,)).fetchall())
    assert rows == [("chunk",)]
    assert hnsw_kind_index_name("test", "chunk") in plan
