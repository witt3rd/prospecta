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


# --- phrase matching and the adjacency boost -------------------------------------

PHRASE_TEXTS = {
    "adj.md": "The quick brown fox jumps over the lazy dog.",
    "scat.md": "A brown dog, very quick, watched the fox.",
    "rel.md": "Deploy v1.2.3 from src/main.rs now.",
    "apart.md": "Version v1.2.3 shipped; src is old and main.rs is elsewhere.",
}


@pytest.fixture
def pmem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=stub_llm, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    for src, t in PHRASE_TEXTS.items():
        m.retain(t, source=src, index_text="zzz " + src)
    yield m
    m.close()


def _sources(mem, text, **params):
    return [c.source for c in _run(mem, text, **params)[1]]


def test_quoted_phrase_finds_only_adjacent_in_order(pmem):
    assert _sources(pmem, '"quick brown"') == ["adj.md"]
    assert _sources(pmem, '"brown quick"') == []          # order matters
    assert _sources(pmem, '"quick fox"') == []            # a word between


def test_quoted_phrase_with_version_and_path(pmem):
    assert _sources(pmem, '"deploy v1.2.3"') == ["rel.md"]
    assert _sources(pmem, '"v1.2.3 from src/main.rs now"') == ["rel.md"]
    assert _sources(pmem, '"src/main.rs"') == ["rel.md"]  # not src ... main.rs apart


def test_unquoted_query_keeps_every_any_word_hit_and_ranks_adjacent_first(pmem):
    got = _sources(pmem, "quick brown")
    assert got == ["adj.md", "scat.md"]
    assert set(_sources(pmem, "quick brown deploy")) == {"adj.md", "scat.md", "rel.md"}
    assert _sources(pmem, "quick brown deploy")[0] == "adj.md"


def test_adjacent_ranks_above_scattered_for_paths_too(pmem):
    assert _sources(pmem, "src main.rs") == ["rel.md", "apart.md"]


def test_quoted_phrase_mixed_with_loose_words(pmem):
    assert _sources(pmem, 'dog "quick brown"') == ["adj.md"]     # phrase is required


def test_quote_edge_cases(pmem):
    # adj.md and scat.md tie on score; the order between ties is by random id, so the
    # test fixes it: compare sorted
    assert sorted(_sources(pmem, '"quick brown')) == ["adj.md", "scat.md"]  # unbalanced: plain words
    assert sorted(_sources(pmem, '"" quick')) == ["adj.md", "scat.md"]      # empty phrase ignored


def test_quoted_phrase_without_index_matches_with_index(pmem):
    import psycopg as pg
    from prospecta.db.keyword_index import INDEX_NAME
    qs = ['"quick brown"', '"src/main.rs"', '"v1.2.3 from src/main.rs now"',
          "quick brown", "src main.rs", 'dog "quick brown"']
    with_idx = {q: _sources(pmem, q) for q in qs}
    with pg.connect(pmem.database_url, autocommit=True) as c:
        c.execute(f'DROP INDEX "{INDEX_NAME}"')
    with pg.connect(pmem.database_url) as c:
        ch = KeywordChunks()
        ch.retrieve(QueryPlan(text='"quick brown"'), RecallState(conn=c, bank_id="b"))
        assert ch.used_index is False
    assert {q: _sources(pmem, q) for q in qs} == with_idx
    assert with_idx['"quick brown"'] == ["adj.md"]


# --- the adjacency boost is bounded to the top N rows -----------------------------

def test_boost_top_validates_and_defaults():
    assert KeywordChunks().boost_top == 300
    assert build_channels([{"name": "keyword", "params": {"boost_top": 5}}])[0].channel.boost_top == 5
    for bad in (-1, "5", True, 1.5):
        with pytest.raises(ValueError):
            KeywordChunks(boost_top=bad)


@pytest.fixture
def many(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=stub_llm, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    for i in range(12):
        m.retain(f"alpha filler{i} beta", source=f"s{i:02d}.md", index_text=f"zzz {i}")
    m.retain("alpha beta", source="adj.md", index_text="zzz adj")
    yield m
    m.close()


def _counted_run(mem, text, **params):
    """Run the channel with every pair check counting its evaluations in a sequence."""
    import prospecta.channels.keyword as kw
    with psycopg.connect(mem.database_url, autocommit=True) as c:
        c.execute("DROP SEQUENCE IF EXISTS evals")
        c.execute("CREATE SEQUENCE evals")
    orig = kw._pair_adjacent
    kw._pair_adjacent = lambda cfg, param: f"(nextval('evals') > 0 AND {orig(cfg, param)})"
    try:
        out = _run(mem, text, **params)[1]
    finally:
        kw._pair_adjacent = orig
    with psycopg.connect(mem.database_url) as c:
        n = c.execute("SELECT last_value, is_called FROM evals").fetchone()
    return out, (n[0] if n[1] else 0)


def test_boost_is_evaluated_for_at_most_top_n_rows_and_keeps_every_hit(many):
    out, evals = _counted_run(many, "alpha beta", boost_top=5)
    assert evals == 5                      # one pair, five rows
    assert len(out) == 13                  # every any-word hit stays
    out_all, evals_all = _counted_run(many, "alpha beta", boost_top=1000)
    assert evals_all == 13
    assert {c.source for c in out} == {c.source for c in out_all}


def test_rows_outside_the_top_n_keep_base_order_after_the_boosted_rows(many):
    out = _run(many, "alpha beta", boost_top=5)[1]
    assert len(out) == 13
    base = [c.source for c in _run(many, "alpha beta", boost_top=0)[1]]
    got = [c.source for c in out]
    assert set(got[:5]) == set(base[:5])             # the top 5 by base score, reordered
    assert got[5:] == base[5:]                       # the rest after them, base order
    out0, evals0 = _counted_run(many, "alpha beta", boost_top=0)
    assert evals0 == 0 and len(out0) == 13
