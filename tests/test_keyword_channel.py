"""KeywordChunks: raw-text keyword channel, stemming off, never silently empty."""
from __future__ import annotations

import psycopg
import pytest

from prospecta.channels import QueryPlan, RecallState, REGISTRY, build_channels
from prospecta.channels.keyword import KeywordChunks, keyword_terms
from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM, stub_embed, stub_llm


def test_terms_are_raw_lowercase_unstemmed_and_keep_stop_words(mem):
    with psycopg.connect(mem.database_url) as conn:
        assert keyword_terms(conn, "The Cats were sitting, on mats!") == [
            "the", "cats", "were", "sitting", "on", "mats"]
        assert keyword_terms(conn, "  ?! ") == []


def test_registered_off_by_default_and_params_validate():
    assert "keyword" in REGISTRY
    ch = build_channels([{"name": "keyword", "weight": 1}])[0].channel
    assert ch.stemming is False
    assert build_channels([{"name": "keyword", "params": {"stemming": True}}])[0].channel.stemming is True
    with pytest.raises(ValueError):
        KeywordChunks(stemming="yes")


@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=stub_llm, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    # index_text deliberately unrelated to the body: the question-only bank case
    m.retain("The cat sat on the warm mat by the window.", source="cat.md", index_text="zzz one")
    m.retain("Quarterly revenue grew by twelve percent.", source="rev.md", index_text="zzz two")
    yield m
    m.close()


def _run(mem, text, **params):
    with psycopg.connect(mem.database_url) as conn:
        ch = KeywordChunks(**params)
        out = ch.retrieve(QueryPlan(text=text), RecallState(conn=conn, bank_id="b"))
    return ch, out


def test_finds_raw_words_in_the_stored_passage_text(mem):
    _, out = _run(mem, "what did the cat sit on?")      # "sit" is not in the text unstemmed
    assert [c.source for c in out] == ["cat.md"]        # matched on cat/the/on, not on 'sit'
    assert out[0].channel == "keyword" and out[0].rank == 1 and out[0].score > 0
    assert "warm mat" in out[0].evidence


def test_stemming_is_off_unless_asked(mem):
    _, off = _run(mem, "mats")      # the text says "mat"
    assert off == []
    _, on = _run(mem, "mats", stemming=True)
    assert [c.source for c in on] == ["cat.md"]


def test_stop_word_only_question_still_matches(mem):
    _, out = _run(mem, "the on")
    assert [c.source for c in out] == ["cat.md"]


def test_or_semantics_one_shared_word_is_enough(mem):
    _, out = _run(mem, "revenue of a zebra herd")
    assert [c.source for c in out] == ["rev.md"]


def test_no_overlap_is_reported_not_silent(mem):
    ch, out = _run(mem, "zebra herd")
    assert out == []
    assert ch.last_note and "no passage shares a word" in ch.last_note


def test_unusable_query_with_passages_raises(mem):
    with pytest.raises(ValueError, match="no usable words"):
        _run(mem, "?! ...")


def test_empty_bank_is_empty_without_error(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="e", llm=stub_llm, embed=stub_embed)
    m.create_bank("e", embedding_dim=EMBED_DIM)
    with psycopg.connect(fresh_db) as conn:
        out = KeywordChunks().retrieve(QueryPlan(text="anything"), RecallState(conn=conn, bank_id="e"))
    assert out == []
    m.close()


def test_keyword_and_floor_union_wired_into_a_recall(mem):
    from prospecta.channels import DEFAULT_CHANNEL_CONFIG
    mem.set_channel_config(DEFAULT_CHANNEL_CONFIG + [{"name": "keyword", "weight": 1}])
    mem.set_recall_config({"rerank": {"enabled": False}, "fusion": {"method": "floor_union", "floor": 3}})
    mem.recall(["warm mat"], limit=5)
    with psycopg.connect(mem.database_url) as conn:
        channels, fusion = conn.execute(
            "SELECT channels, fusion FROM recall_events WHERE bank_id='b' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    kw = next(c for c in channels if c["name"] == "keyword")
    assert kw["error"] is None and kw["n"] >= 1
    assert fusion["method"] == "floor_union" and fusion["floor"] == 3


@pytest.mark.parametrize("query", [
    "v1.2.3", "src/main.rs", "main.rs", "3fa9c2d41b7e", "3fa9c2", "config.yaml",
    "PROSPECTA_DB_POOL_MAX", "max_connections", "ECONNRESET", "ABC-123", "PR-66",
    "0123456789ab"])
def test_identifier_shapes_are_findable(fresh_db, query):
    m = Memory(database_url=fresh_db, bank_id="b", llm=stub_llm, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    m.retain("Released v1.2.3 from src/main.rs; see config.yaml, id 3fa9c2d41b7e, "
             "PROSPECTA_DB_POOL_MAX max_connections ECONNRESET ABC-123 PR-66 0123456789ab.",
             source="ids.md", index_text="zzz")
    m.retain("Unrelated gardening note about tomatoes.", source="other.md", index_text="zzz")
    try:
        _, out = _run(m, query)
    finally:
        m.close()
    assert [c.source for c in out] == ["ids.md"]
