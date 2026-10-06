"""Bm25Chunks: in-process BM25 channel, per-bank persisted/rebuilt index."""
from __future__ import annotations

import psycopg
import pytest

from prospecta.channels import (
    Bm25Chunks, Bm25Index, DEFAULT_CHANNEL_CONFIG, InProcessBm25, QueryPlan, RecallState,
    build_channels, validate_channel_config,
)
from prospecta.channels.bm25 import tokenize
from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM, stub_embed, stub_llm


# ------------------------------------------------------------ pure index

def test_tokenize_stems_and_drops_stop_words():
    assert tokenize("The cats were sitting on mats") == tokenize("cat sit mat")
    assert "the" not in tokenize("the of and")


def test_bm25_or_semantics_idf_and_length_normalisation():
    idx = Bm25Index.build([
        ("a", "revenue grew in europe"),
        ("b", "the cat sat on the mat"),
        ("c", "revenue revenue revenue cat " + "filler " * 40),
    ])
    # natural-language question: no document holds every term, OR still ranks
    hits = idx.search("how did revenue grow in europe and the cat", 10)
    assert [i for i, _ in hits][0] == "a"
    assert {i for i, _ in hits} == {"a", "b", "c"}
    assert all(s > 0 for _, s in hits)
    # rare term outweighs a common one
    idx2 = Bm25Index.build([("x", "common rare"), ("y", "common"), ("z", "common")])
    assert idx2.search("common rare", 3)[0][0] == "x"
    assert idx.search("zzzz unknown", 5) == []


def test_index_json_roundtrip_scores_identically():
    idx = Bm25Index.build([("a", "alpha beta"), ("b", "beta gamma gamma")])
    again = Bm25Index.from_json(__import__("json").loads(__import__("json").dumps(idx.to_json())))
    assert again.search("beta gamma", 5) == idx.search("beta gamma", 5)


def test_registry_knows_bm25_and_default_config_unchanged():
    validate_channel_config([{"name": "bm25", "weight": 1}])
    assert [c.name for c in build_channels([{"name": "bm25"}])] == ["bm25"]
    assert Bm25Chunks.kind == "recall"
    assert "bm25" not in [e["name"] for e in DEFAULT_CHANNEL_CONFIG]


# ---------------------------------------------------------- db integration

def _chunk(conn, bank, source, text, metadata="{}"):
    conn.execute(
        "INSERT INTO memory_items (bank_id, document_id, content, original_chunk, embedding,"
        " kind, ordinal, char_start, char_end, metadata) "
        "SELECT %s, d.id, %s, %s, %s::vector, 'chunk', 0, 0, %s, %s::jsonb FROM documents d "
        "WHERE d.source = %s",
        (bank, text, text, "[" + ",".join(map(str, stub_embed([text])[0])) + "]",
         len(text), metadata, source))


@pytest.fixture
def mem(fresh_db, tmp_path, monkeypatch):
    monkeypatch.setenv("PROSPECTA_BM25_DIR", str(tmp_path / "bm25"))
    m = Memory(database_url=fresh_db, bank_id="b", llm=stub_llm, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    for src, text in (("cat.md", "The cat sat on the warm mat by the window."),
                      ("rev.md", "Quarterly revenue grew by twelve percent in Europe.")):
        m.retain(text, source=src, index_text="q " + src)
    with psycopg.connect(fresh_db) as conn:
        _chunk(conn, "b", "cat.md", "The cat sat on the warm mat by the window.")
        _chunk(conn, "b", "cat.md", "Cats nap on windowsills all afternoon.")
        _chunk(conn, "b", "rev.md", "Quarterly revenue grew by twelve percent in Europe.")
        conn.commit()
    yield m
    m.close()


def test_channel_retrieves_by_or_query_document_level(mem):
    with psycopg.connect(mem.database_url) as conn:
        out = Bm25Chunks().retrieve(
            QueryPlan(text="how did the revenue grow in Europe, and the cat?"),
            RecallState(conn=conn, bank_id="b"), 10)
    assert [c.source for c in out][0] == "rev.md"
    assert len({c.document_id for c in out}) == len(out) == 2  # one candidate per note
    assert [c.rank for c in out] == [1, 2]
    assert out[0].channel == "bm25" and out[0].score > out[1].score > 0
    assert out[0].evidence.startswith("Quarterly revenue") and out[0].item_id


def test_bm25_in_a_recall_with_trace(mem):
    mem.set_channel_config(DEFAULT_CHANNEL_CONFIG + [{"name": "bm25", "weight": 1}])
    res = mem.search("revenue europe twelve percent", limit=5)
    assert res[0].source == "rev.md" and res[0].scores["bm25"] > 0
    mem.recall(["revenue europe"], limit=5)
    with psycopg.connect(mem.database_url) as conn:
        channels = conn.execute("SELECT channels FROM recall_events WHERE bank_id='b' "
                                "ORDER BY id DESC LIMIT 1").fetchone()[0]
    b = next(c for c in channels if c["name"] == "bm25")
    assert b["error"] is None and b["n"] >= 1


def test_metadata_filter_applies(mem):
    with psycopg.connect(mem.database_url) as conn:
        conn.execute("UPDATE memory_items SET metadata = '{\"person\": \"ann\"}'::jsonb "
                     "WHERE original_chunk LIKE 'Cats nap%'")
        conn.commit()
        out = Bm25Chunks().retrieve(
            QueryPlan(text="cat nap windowsill revenue"),
            RecallState(conn=conn, bank_id="b", metadata_filter={"person": "ann"}), 10)
    assert [c.source for c in out] == ["cat.md"]
    assert out[0].evidence.startswith("Cats nap")


def test_index_persisted_reused_and_rebuilt_on_change(mem, tmp_path):
    d = tmp_path / "idx"
    be = InProcessBm25(d)
    with psycopg.connect(mem.database_url) as conn:
        r1 = be.search(conn, "b", "revenue europe", 5)
        assert be.builds == 1 and (d / "b.bm25.json.gz").exists()
        be.search(conn, "b", "cat", 5)
        assert be.builds == 1  # served from memory
        fresh = InProcessBm25(d)  # a new process: loaded from disk, not rebuilt
        assert fresh.search(conn, "b", "revenue europe", 5) == r1
        assert fresh.builds == 0
        # new chunk -> fingerprint changes -> rebuild, and it is searchable
        _chunk(conn, "b", "rev.md", "Zanzibar logistics expanded dramatically.")
        conn.commit()
        hits = be.search(conn, "b", "zanzibar", 5)
        assert be.builds == 2 and len(hits) == 1
        # delete -> rebuild again
        conn.execute("DELETE FROM memory_items WHERE original_chunk LIKE 'Zanzibar%'")
        conn.commit()
        assert be.search(conn, "b", "zanzibar", 5) == [] and be.builds == 3
    # corrupt file is ignored, never fatal
    (d / "b.bm25.json.gz").write_bytes(b"garbage")
    with psycopg.connect(mem.database_url) as conn:
        assert InProcessBm25(d).search(conn, "b", "revenue", 5)


def test_banks_are_isolated(mem, tmp_path):
    mem.create_bank("other", embedding_dim=EMBED_DIM)
    with psycopg.connect(mem.database_url) as conn:
        assert InProcessBm25(tmp_path / "i").search(conn, "other", "revenue", 5) == []


def test_pg_search_backend_is_only_a_seam():
    class Fake:
        def search(self, conn, bank_id, query, n):
            return []
    ch = Bm25Chunks()
    ch.backend = Fake()  # any object with search() plugs in
    assert ch.retrieve(QueryPlan(text="x"), RecallState(conn=None, bank_id="b"), 5) == []


# ------------------------------------------------------- pg_search seam

class _Stub:
    def __init__(self, tag, avail=True):
        self.tag, self.avail, self.calls = tag, avail, 0

    def available(self, conn):
        return self.avail

    def search(self, conn, bank_id, query, n):
        self.calls += 1
        return [(self.tag, 1.0)]


def test_auto_backend_selection_with_stub():
    from prospecta.channels import AutoBm25
    pg, fb = _Stub("pg"), _Stub("fb")
    assert AutoBm25(pg, fb).search(None, "b", "q", 5) == [("pg", 1.0)]
    pg.avail = False
    assert AutoBm25(pg, fb).search(None, "b", "q", 5) == [("fb", 1.0)]

    class Boom(_Stub):
        def available(self, conn):
            raise RuntimeError("x")
    assert AutoBm25(Boom("pg"), fb).search(None, "b", "q", 5) == [("fb", 1.0)]


def test_migration_0012_is_noop_without_extension(mem):
    from prospecta.channels import PgSearchBm25
    from prospecta.channels.bm25 import PG_SEARCH_INDEX
    with psycopg.connect(mem.database_url) as conn:
        assert conn.execute("SELECT 1 FROM pg_available_extensions WHERE name='pg_search'"
                            ).fetchone() is None
        assert conn.execute("SELECT 1 FROM pg_indexes WHERE indexname=%s",
                            (PG_SEARCH_INDEX,)).fetchone() is None
        assert conn.execute("SELECT 1 FROM prospecta_schema_version WHERE version=12"
                            ).fetchone()
        assert PgSearchBm25().available(conn) is False
        out = Bm25Chunks().retrieve(QueryPlan(text="revenue europe"),
                                    RecallState(conn=conn, bank_id="b"), 5)
    assert out and out[0].source == "rev.md"  # default AutoBm25 fell back to bm25s
