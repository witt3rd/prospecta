"""Per-bank partial GIN indexes (docs/bank-isolation.md): created with the bank, repaired by migrate,
used by Stage 1, and bank-scoped (another bank's rows never enter them)."""
import psycopg
import pytest

from prospecta.db import migrate
from prospecta.db.queries import (
    LEXICAL_INDEX_COLUMNS, hnsw_kind_index_name, lexical_kind_index_name,
)
from prospecta.memory import Memory

DIM = 32


def _lexical(url):
    with psycopg.connect(url) as c:
        return {r[0]: (r[1], r[2]) for r in c.execute(
            "SELECT c.relname, i.indisvalid AND i.indisready, pg_get_indexdef(i.indexrelid) "
            "FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid "
            "WHERE i.indrelid = 'memory_items'::regclass AND c.relname ~ '_tsv_.+_chunk_gin$' "
            "AND c.relname NOT IN ('memory_items_body_tsv_gin', 'memory_items_exact_tsv_gin')")}


def test_names_fit_and_are_unique_for_long_banks():
    a = lexical_kind_index_name("body_tsv", "x" * 63, "chunk")
    b = lexical_kind_index_name("body_tsv", "x" * 62 + "y", "chunk")
    assert len(a) <= 63 and len(b) <= 63 and a != b
    assert lexical_kind_index_name("exact_tsv", "my-bank", "chunk") == "memory_items_exact_tsv_my_bank_chunk_gin"
    with pytest.raises(ValueError):
        lexical_kind_index_name("body_tsv", "my-bank", "question")


def test_create_bank_builds_both_partial_indexes(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="alpha")
    mem.create_bank("alpha", embedding_dim=DIM)
    got = _lexical(fresh_db)
    assert set(got) == {lexical_kind_index_name(s, "alpha", "chunk") for s, _ in LEXICAL_INDEX_COLUMNS}
    for name, (valid, defn) in got.items():
        assert valid and "USING gin" in defn
        assert "bank_id = 'alpha'" in defn and "kind = 'chunk'" in defn
    mem.close()


def test_two_banks_each_have_their_own(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="alpha")
    mem.create_bank("alpha", embedding_dim=DIM)
    mem.create_bank("beta", embedding_dim=DIM)
    assert len(_lexical(fresh_db)) == 4
    mem.close()


def test_migrate_repairs_a_missing_and_an_invalid_index(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="alpha")
    mem.create_bank("alpha", embedding_dim=DIM)
    mem.close()
    body = lexical_kind_index_name("body_tsv", "alpha", "chunk")
    exact = lexical_kind_index_name("exact_tsv", "alpha", "chunk")
    with psycopg.connect(fresh_db, autocommit=True) as c:
        c.execute(f'DROP INDEX "{body}"')
        c.execute("UPDATE pg_index SET indisvalid = false WHERE indexrelid = %s::regclass", (exact,))
    assert migrate.ensure_bank_lexical_indexes(fresh_db)["built"] == 1
    got = _lexical(fresh_db)
    assert got[body][0] and got[exact][0]
    assert migrate.ensure_bank_lexical_indexes(fresh_db)["built"] == 0          # idempotent: nothing to build
    assert migrate.run_migrations(fresh_db)["applied"] == []


def test_a_bank_inserted_by_raw_sql_gets_them_on_migrate(fresh_db):
    with psycopg.connect(fresh_db, autocommit=True) as c:
        c.execute("INSERT INTO banks (bank_id, embedding_dim) VALUES ('raw', %s)", (DIM,))
    migrate.run_migrations(fresh_db)
    assert set(_lexical(fresh_db)) == {lexical_kind_index_name(s, "raw", "chunk") for s, _ in LEXICAL_INDEX_COLUMNS}


def test_stage1_lexical_retrievers_use_the_banks_own_index(memory_with_bank, fresh_db):
    memory_with_bank.create_bank("beta", embedding_dim=32)
    for i in range(200):
        memory_with_bank.retain(f"harbour note {i} " + ("lighthouse" if i == 7 else "tide"), source=f"a{i}")
    with psycopg.connect(fresh_db) as c:
        c.execute("ANALYZE memory_items")
        c.execute("SET enable_seqscan = off")
        for stem, column in LEXICAL_INDEX_COLUMNS:
            cfg = "english" if column == "body_tsv" else "simple"
            plan = "\n".join(r[0] for r in c.execute(
                f"EXPLAIN SELECT 1 FROM memory_items m WHERE m.bank_id = 'test' AND m.kind = 'chunk' "
                f"AND m.{column} @@ to_tsquery('{cfg}', 'lighthouse')").fetchall())
            assert lexical_kind_index_name(stem, "test", "chunk") in plan
            assert lexical_kind_index_name(stem, "beta", "chunk") not in plan
