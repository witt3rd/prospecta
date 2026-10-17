"""The channel interface, the registry (`hybrid_chunk` is the one channel), weighted RRF fusion and
the per-channel trace (migrations 0007 + 0008)."""
from __future__ import annotations

import psycopg
import pytest

from prospecta.channels import (
    Candidate,
    build_channels,
    effective_channel_config,
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

def test_registry_builds_the_enabled_channel_and_validates():
    cfg = [{"name": "hybrid_chunk", "enabled": True, "weight": 1}]
    assert [c.name for c in build_channels(cfg)] == ["hybrid_chunk"]
    assert build_channels([{"name": "hybrid_chunk", "enabled": False}]) == []
    assert build_channels([{"name": "dense_chunk", "weight": 4}, {"name": "meta"}]) == []   # removed: skipped
    validate_channel_config([])
    validate_channel_config(cfg)
    for bad in ([{"name": "nope"}], [{"name": "dense_chunk", "weight": 1}], [{"name": "bm25", "weight": 1}],
                [{"name": "hybrid_chunk", "weight": -1}], [{"name": "hybrid_chunk"}, {"name": "hybrid_chunk"}],
                {"name": "hybrid_chunk"}):
        with pytest.raises(ValueError):
            validate_channel_config(bad)


def test_the_effective_config_adds_the_default_when_no_channel_is_named():
    default = effective_channel_config([])
    assert default == [{"name": "hybrid_chunk", "enabled": True, "weight": 1, "params": {}}]
    mine = [{"name": "hybrid_chunk", "weight": 2, "params": {"fused_limit": 7}}]
    assert effective_channel_config(mine) == mine
    gone = [{"name": "dense_chunk", "weight": 4}, {"name": "meta", "weight": 3}]
    assert effective_channel_config(gone) == default + gone       # the default runs; the rest is ignored
    off = [{"name": "hybrid_chunk", "enabled": False}]
    assert effective_channel_config(off) == off                    # an explicit choice stays
    assert effective_channel_config(None) == default


# --------------------------------------------------------- db integration

@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=stub_llm, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    m.set_recall_config({})
    m.retain("The cat sat on the warm mat by the window.", source="cat.md")
    m.retain("Quarterly revenue grew by twelve percent in Europe.", source="rev.md")
    yield m
    m.close()


def test_a_new_bank_stores_hybrid_chunk_and_an_empty_config_runs_it_too(mem):
    with psycopg.connect(mem.database_url) as conn:
        assert [e["name"] for e in conn.execute(
            "SELECT channel_config FROM banks WHERE bank_id='b'").fetchone()[0]] == ["hybrid_chunk"]
        conn.execute("UPDATE banks SET channel_config = '[]'::jsonb WHERE bank_id = 'b'")
        conn.commit()
    res = mem.search("where did the cat sit")
    assert res and set(res[0].scores) == {"hybrid_chunk", "rrf"}


def test_channel_recall_scores_per_channel_at_document_level(mem):
    res = mem.search("cat sat warm mat window", limit=5)
    assert [r.source for r in res][0] == "cat.md"
    assert len({r.document_id for r in res}) == len(res)  # document level
    top = res[0]
    assert top.scores["hybrid_chunk"] > 0
    assert set(top.scores) == {"hybrid_chunk", "rrf"}
    assert top.original_chunk == "The cat sat on the warm mat by the window."  # the chunk is the evidence


def test_set_channel_config_validates_and_resets(mem):
    with pytest.raises(ValueError):
        mem.set_channel_config([{"name": "bogus"}])
    mem.set_channel_config([{"name": "hybrid_chunk", "weight": 1}])
    mem.set_channel_config([])
    assert set(mem.search("cat")[0].scores) == {"hybrid_chunk", "rrf"}


def test_recall_trace_persists_plan_channels_fusion_and_candidates(mem):
    mem.recall(["cat sat warm mat window"], limit=5)
    with psycopg.connect(mem.database_url) as conn:
        ev_id, plan, channels, fusion = conn.execute(
            "SELECT id, plan, channels, fusion FROM recall_events "
            "WHERE bank_id='b' ORDER BY id DESC LIMIT 1").fetchone()
        cands = conn.execute(
            "SELECT channel, rank, score, document_id, item_id FROM recall_event_candidates "
            "WHERE recall_event_id=%s ORDER BY channel, rank", (ev_id,)).fetchall()
    assert plan["queries"][0]["text"] == "cat sat warm mat window"
    assert [c["name"] for c in channels] == ["hybrid_chunk"]
    assert all(c["error"] is None and "latency_ms" in c and c["n"] >= 1 for c in channels)
    assert fusion["method"] == "weighted_rrf" and fusion["k"] == 60
    assert fusion["weights"] == {"hybrid_chunk": 1.0}
    assert len(fusion["pool"]) == 2
    assert {c[0] for c in cands} == {"hybrid_chunk"}
    assert [c[1] for c in cands] == [1, 2]
    assert cands[0][2] >= cands[1][2]
    assert all(c[3] and c[4] for c in cands)


def test_a_failing_channel_is_recorded_and_the_recall_returns_nothing_without_raising(mem, monkeypatch):
    from prospecta.channels import hybrid
    monkeypatch.setattr(hybrid, "stage1_search", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    res = mem.recall(["cat sat warm mat window"], limit=5)
    with psycopg.connect(mem.database_url) as conn:
        channels = conn.execute(
            "SELECT channels FROM recall_events WHERE bank_id='b' "
            "ORDER BY id DESC LIMIT 1").fetchone()[0]
    assert res == []
    assert channels[0]["name"] == "hybrid_chunk" and "boom" in channels[0]["error"] and channels[0]["n"] == 0
