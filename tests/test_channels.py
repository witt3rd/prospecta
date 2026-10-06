"""Channel interface, registry, weighted RRF fusion, per-channel trace
(migrations 0007 + 0008)."""
from __future__ import annotations

import psycopg
import pytest

from prospecta.channels import (
    DEFAULT_CHANNEL_CONFIG,
    Candidate,
    build_channels,
    fuse,
    validate_channel_config,
)
from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM, stub_embed, stub_llm


def C(doc, source, channel, rank, score=0.0, item=None):
    return Candidate(document_id=doc, item_id=item, source=source, channel=channel,
                     rank=rank, score=score, evidence=f"{doc}-{channel}")


# ---------------------------------------------------------------- fusion

def test_fuse_weighted_rrf_math_and_order():
    lists = {
        "a": [C("d1", "s1", "a", 1), C("d2", "s2", "a", 2)],
        "b": [C("d2", "s2", "b", 1), C("d3", "s3", "b", 2)],
    }
    out = fuse(lists, {"a": 4, "b": 1}, k=60)
    got = {f.document_id: f.score for f in out}
    assert got["d1"] == pytest.approx(4 / 61)
    assert got["d2"] == pytest.approx(4 / 62 + 1 / 61)
    assert got["d3"] == pytest.approx(1 / 62)
    assert [f.document_id for f in out] == ["d2", "d1", "d3"]
    d2 = out[0]
    assert d2.ranks == {"a": 2, "b": 1}
    assert d2.best.channel == "a"  # evidence from the heaviest channel


def test_fuse_ties_break_by_source_name_deterministically():
    lists = {"a": [C("dz", "zeta", "a", 1)], "b": [C("dy", "alpha", "b", 1)]}
    out = fuse(lists, {"a": 1, "b": 1})
    assert [f.source for f in out] == ["alpha", "zeta"]
    # input order of channels does not matter
    out2 = fuse(dict(reversed(list(lists.items()))), {"a": 1, "b": 1})
    assert [f.source for f in out2] == ["alpha", "zeta"]


def test_fuse_zero_weight_contributes_nothing_and_pool_caps():
    lists = {"a": [C(f"d{i}", f"s{i:02d}", "a", i + 1) for i in range(5)],
             "dark": [C("x", "x", "dark", 1)]}
    out = fuse(lists, {"a": 1, "dark": 0}, pool=3)
    assert [f.document_id for f in out] == ["d0", "d1", "d2"]


def test_fuse_document_counts_once_per_channel_at_best_rank():
    lists = {"a": [C("d1", "s", "a", 1, item="i1"), C("d1", "s", "a", 2, item="i2")]}
    out = fuse(lists, {"a": 1})
    assert out[0].score == pytest.approx(1 / 61)


# -------------------------------------------------------------- registry

def test_registry_builds_enabled_in_order_and_validates():
    cfg = [{"name": "question", "enabled": True, "weight": 1},
           {"name": "dense_chunk", "enabled": False, "weight": 4}]
    built = build_channels(cfg)
    assert [c.name for c in built] == ["question"]
    assert [c.name for c in build_channels(DEFAULT_CHANNEL_CONFIG)] == ["dense_chunk", "question", "meta"]
    with pytest.raises(ValueError):
        validate_channel_config([{"name": "nope"}])
    with pytest.raises(ValueError):
        validate_channel_config([{"name": "bm25_none", "weight": 1}])
    with pytest.raises(ValueError):
        validate_channel_config([{"name": "question", "weight": -1}])


# --------------------------------------------------------- db integration

@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=stub_llm, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    m.retain("The cat sat on the warm mat by the window.", source="cat.md",
             index_text="where did the cat sit")
    m.retain("Quarterly revenue grew by twelve percent in Europe.", source="rev.md",
             index_text="how did revenue grow")
    # a chunk item for each, as the directory-index path writes them
    with psycopg.connect(fresh_db) as conn:
        for src, text in (("cat.md", "cat sat warm mat window"),
                          ("rev.md", "revenue grew twelve percent europe")):
            conn.execute(
                "INSERT INTO memory_items (bank_id, document_id, content, original_chunk,"
                " embedding, kind, ordinal, char_start, char_end) "
                "SELECT 'b', d.id, %s, %s, %s::vector, 'chunk', 0, 0, %s FROM documents d "
                "WHERE d.source = %s",
                (text, text, "[" + ",".join(map(str, stub_embed([text])[0])) + "]",
                 len(text), src))
        conn.commit()
    yield m
    m.close()


def test_default_config_is_empty_and_recall_stays_legacy(mem):
    with psycopg.connect(mem.database_url) as conn:
        assert conn.execute("SELECT channel_config FROM banks WHERE bank_id='b'").fetchone()[0] == []
    res = mem.search("where did the cat sit")
    assert res and set(res[0].scores) == {"semantic", "lexical", "lexical_body", "rrf"}


def test_channel_recall_fuses_documents_and_scores_per_channel(mem):
    mem.set_channel_config(DEFAULT_CHANNEL_CONFIG)
    res = mem.search("cat sat warm mat window", limit=5)
    assert [r.source for r in res][0] == "cat.md"
    assert len({r.document_id for r in res}) == len(res)  # document level
    top = res[0]
    assert top.scores["dense_chunk"] > 0
    assert set(top.scores) >= {"semantic", "lexical", "lexical_body", "rrf",
                               "dense_chunk", "question"}
    assert top.original_chunk == "cat sat warm mat window"  # best chunk evidence


def test_set_channel_config_validates_and_resets(mem):
    with pytest.raises(ValueError):
        mem.set_channel_config([{"name": "bogus"}])
    mem.set_channel_config(DEFAULT_CHANNEL_CONFIG)
    mem.set_channel_config([])
    assert set(mem.search("cat")[0].scores) == {"semantic", "lexical", "lexical_body", "rrf"}


def test_recall_trace_persists_plan_channels_fusion_and_candidates(mem):
    mem.set_channel_config(DEFAULT_CHANNEL_CONFIG)
    mem.recall(["cat sat warm mat window"], limit=5)
    with psycopg.connect(mem.database_url) as conn:
        ev_id, plan, channels, fusion = conn.execute(
            "SELECT id, plan, channels, fusion FROM recall_events "
            "WHERE bank_id='b' ORDER BY id DESC LIMIT 1").fetchone()
        cands = conn.execute(
            "SELECT channel, rank, score, document_id, item_id FROM recall_event_candidates "
            "WHERE recall_event_id=%s ORDER BY channel, rank", (ev_id,)).fetchall()
    assert plan["queries"][0]["text"] == "cat sat warm mat window"
    assert [c["name"] for c in channels] == ["dense_chunk", "question", "meta"]
    assert all(c["error"] is None and "latency_ms" in c for c in channels)
    assert [c["n"] >= 1 for c in channels] == [True, True, False]  # no filter extracted
    assert fusion["method"] == "weighted_rrf" and fusion["k"] == 60
    assert fusion["weights"] == {"dense_chunk": 4.0, "question": 1.0, "meta": 3.0}
    assert len(fusion["pool"]) == 2
    by_channel = {c[0] for c in cands}
    assert by_channel == {"dense_chunk", "question"}  # meta had no filter
    dense = [c for c in cands if c[0] == "dense_chunk"]
    assert [c[1] for c in dense] == [1, 2]
    assert dense[0][2] >= dense[1][2]
    assert all(c[3] and c[4] for c in cands)


def test_legacy_recall_leaves_trace_columns_null(mem):
    mem.recall(["cat sat"], limit=3)
    with psycopg.connect(mem.database_url) as conn:
        row = conn.execute(
            "SELECT plan, channels, fusion, id FROM recall_events "
            "WHERE bank_id='b' ORDER BY id DESC LIMIT 1").fetchone()
        n = conn.execute("SELECT count(*) FROM recall_event_candidates "
                         "WHERE recall_event_id=%s", (row[3],)).fetchone()[0]
    assert row[:3] == (None, None, None) and n == 0


def test_failing_channel_is_recorded_and_others_still_fuse(mem, monkeypatch):
    from prospecta.channels import semantic
    mem.set_channel_config(DEFAULT_CHANNEL_CONFIG)
    monkeypatch.setattr(semantic.AnticipatedQuestions, "_sql", "SELECT broken FROM nowhere")
    mem.recall(["cat sat warm mat window"], limit=5)
    with psycopg.connect(mem.database_url) as conn:
        channels = conn.execute(
            "SELECT channels FROM recall_events WHERE bank_id='b' "
            "ORDER BY id DESC LIMIT 1").fetchone()[0]
    q = next(c for c in channels if c["name"] == "question")
    assert q["error"] and q["n"] == 0
    assert next(c for c in channels if c["name"] == "dense_chunk")["n"] >= 1
