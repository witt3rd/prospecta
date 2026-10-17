"""Hybrid plan slice 2: native Postgres Stage 1 (docs/postgres-hybrid.md section 6).

The reserved-slot hybrid query over kind='chunk' items: a semantic, a stemmed
lexical and an unstemmed exact/phrase retriever, filters pushed into all three,
fused by SQL with an rrf_score and an exact_snippet. Real Postgres, invented text,
the stub embedder (so the semantic retriever's order is not under test, its slots
and filters are)."""
from __future__ import annotations

import psycopg
import pytest

from prospecta.channels import (
    DEFAULT_CHANNEL_CONFIG, QueryPlan, RecallState, build_channels, run_channels,
    validate_channel_config,
)
from prospecta.db.queries import _stage1_statement, stage1_search
from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM, stub_embed
from tests._synthetic_bank import load_bank, random_query_vector


def search(mem, text, bank="test", **kw):
    with psycopg.connect(mem.database_url) as conn:
        return stage1_search(conn, bank_id=bank, query_text=text,
                             query_embedding=stub_embed([text])[0], **kw)


def by_source(rows):
    return {r["source"]: r for r in rows}


def matched(rows, column):
    """Sources a retriever matched (its rank column is not NULL)."""
    return {r["source"] for r in rows if r[column] is not None}


def retain_all(mem, notes):
    for source, text in notes.items():
        mem.retain(text, source=source)


PHRASE_NOTES = {
    "adjacent.md": "The night crew called the bridge lamps electric dreams.",
    "upper.md": "ELECTRIC DREAMS flickered over the harbour.",
    "scattered.md": "Electric lamps burn late; the crew keeps its dreams for the morning.",
    "singular.md": "One electric dream was enough for the crew.",
    "other.md": "Orchids need humid air and a shaded window.",
}


# --------------------------------------------------------------------------
# the three retrievers
# --------------------------------------------------------------------------

def test_quoted_phrase_is_adjacent_case_insensitive_unstemmed_and_beats_scattered(memory_with_bank):
    retain_all(memory_with_bank, PHRASE_NOTES)
    rows = search(memory_with_bank, '"electric dreams"')
    # exact: adjacency in either case, no stemming ('dream' is not 'dreams')
    assert matched(rows, "exact_rank") == {"adjacent.md", "upper.md"}
    # stemmed: adjacency too, but stemmed ('electric dream' matches)
    assert matched(rows, "stem_rank") == {"adjacent.md", "upper.md", "singular.md"}
    order = [r["source"] for r in rows]
    for hit in ("adjacent.md", "upper.md"):
        assert order.index(hit) < order.index("scattered.md")
    assert by_source(rows)["scattered.md"]["exact_snippet"] is None


def test_unquoted_words_match_anywhere_in_the_chunk(memory_with_bank):
    retain_all(memory_with_bank, PHRASE_NOTES)
    rows = search(memory_with_bank, "electric dreams")
    assert matched(rows, "exact_rank") == {"adjacent.md", "upper.md", "scattered.md"}


IDENTIFIER_NOTES = {
    "update.md": "The updater stopped with error 0x80070005 after the v1.14 rollout.",
    "next.md": "Release v1.15 notes mention error 0x80070006 only.",
    "guide.md": "Version 1.14 of the guide covers the install steps.",
}


@pytest.mark.parametrize("query, expected", [
    ("v1.14", {"update.md"}),
    ("0x80070005", {"update.md"}),
    ("v1.15", {"next.md"}),
    ("0x80070006 v1.15", {"next.md"}),
    ("v1.14 0x80070006", set()),          # words are ANDed: no note has both
])
def test_identifiers_match_exactly_through_the_simple_retriever(memory_with_bank, query, expected):
    retain_all(memory_with_bank, IDENTIFIER_NOTES)
    assert matched(search(memory_with_bank, query), "exact_rank") == expected


BOOLEAN_NOTES = {
    "augur.md": "augur wrote the memory index",
    "forge.md": "forge wrote the memory index",
    "shadow.md": "shadow copies of memory were kept",
    "both.md": "the shadow of augur followed the memory",
    "plain.md": "gardens need rain",
}


def test_boolean_or_matches_either_word(memory_with_bank):
    retain_all(memory_with_bank, BOOLEAN_NOTES)
    rows = search(memory_with_bank, "augur OR forge")
    assert matched(rows, "exact_rank") == matched(rows, "stem_rank") == {"augur.md", "forge.md", "both.md"}


def test_minus_excludes_notes_with_the_word(memory_with_bank):
    retain_all(memory_with_bank, BOOLEAN_NOTES)
    rows = search(memory_with_bank, "memory -shadow")
    assert matched(rows, "exact_rank") == matched(rows, "stem_rank") == {"augur.md", "forge.md"}


def test_only_chunk_items_are_candidates(memory_with_bank):
    memory_with_bank.retain("Gardens need rain.", source="q.md",
                            index_text=["lighthouse lighthouse lighthouse"])
    memory_with_bank.retain("The lighthouse stands on the cliff.", source="c.md")
    rows = search(memory_with_bank, "lighthouse")
    assert {r["source"] for r in rows} == {"c.md"}


# --------------------------------------------------------------------------
# exact_snippet
# --------------------------------------------------------------------------

def test_exact_snippet_is_the_verbatim_match_in_context(memory_with_bank):
    retain_all(memory_with_bank, PHRASE_NOTES)
    rows = by_source(search(memory_with_bank, '"electric dreams"'))
    # ts_headline trims stop words at the edges; the words keep their own case
    assert "lamps <<electric>> <<dreams>>" in rows["adjacent.md"]["exact_snippet"]
    assert rows["upper.md"]["exact_snippet"].startswith("<<ELECTRIC>> <<DREAMS>> flickered")


def test_stemmed_only_match_gets_the_english_snippet_and_a_semantic_only_hit_none(memory_with_bank):
    retain_all(memory_with_bank, {"runs.md": "She runs every morning along the river.",
                                  "gardens.md": "Gardens need rain."})
    rows = by_source(search(memory_with_bank, "running"))
    assert rows["runs.md"]["exact_rank"] is None and rows["runs.md"]["stem_rank"] == 1
    assert "<<runs>> every morning" in rows["runs.md"]["exact_snippet"]
    assert rows["gardens.md"]["stem_rank"] is None and rows["gardens.md"]["exact_snippet"] is None


def test_exact_snippet_is_computed_for_the_winners_only(memory_with_bank):
    retain_all(memory_with_bank, {f"n{i}.md": f"lighthouse number {i}" for i in range(6)})
    rows = search(memory_with_bank, "lighthouse", fused_limit=2)
    assert len(rows) == 2 and all("<<lighthouse>>" in r["exact_snippet"] for r in rows)


# --------------------------------------------------------------------------
# fusion
# --------------------------------------------------------------------------

def test_fused_rows_carry_ranks_rrf_score_and_citation_fields(memory_with_bank):
    memory_with_bank.retain("---\ncreated: 2024-03-05\n---\nThe lighthouse keeper wrote a note.",
                            source="keeper.md", metadata={"project": "p1"})
    [row] = search(memory_with_bank, "lighthouse keeper")
    assert (row["sem_rank"], row["stem_rank"], row["exact_rank"]) == (1, 1, 1)
    assert float(row["rrf_score"]) == pytest.approx(0.45 / 61 + 0.25 / 61 + 0.30 / 61)
    assert str(row["created_on"]) == "2024-03-05" and row["source"] == "keeper.md"
    assert row["content"] == row["original_chunk"] == "The lighthouse keeper wrote a note."
    assert row["metadata"]["project"] == "p1"


def test_an_exact_hit_is_a_candidate_whatever_the_semantic_slots_hold(memory_with_bank):
    retain_all(memory_with_bank, {f"filler{i}.md": f"gardens and orchids number {i}" for i in range(8)})
    retain_all(memory_with_bank, {"needle.md": "the zzyzx beacon"})
    needle = by_source(search(memory_with_bank, "zzyzx", semantic_limit=1))["needle.md"]
    assert needle["exact_rank"] == needle["stem_rank"] == 1


def test_slot_limits_and_the_fused_cut_are_configurable(memory_with_bank):
    retain_all(memory_with_bank, {f"n{i}.md": f"lighthouse number {i}" for i in range(8)})
    default = search(memory_with_bank, "lighthouse")
    assert len(default) == 8
    one = search(memory_with_bank, "lighthouse", semantic_limit=1, stemmed_limit=1, exact_limit=1)
    assert 1 <= len(one) <= 3
    assert len(search(memory_with_bank, "lighthouse", fused_limit=3)) == 3


# --------------------------------------------------------------------------
# filters are pushed into all three retrievers, before their slots are cut
# --------------------------------------------------------------------------

@pytest.fixture
def two_notes(memory_with_bank):
    """old.md is the best hit of every retriever; new.md is the runner-up."""
    mem = memory_with_bank
    mem.retain("lighthouse lighthouse lighthouse", source="old.md",
               metadata={"created_on": "2020-01-01", "project": "a", "audience_key": "team-a"})
    mem.retain("the lighthouse keeper", source="new.md",
               metadata={"created_on": "2024-05-05", "project": "b", "audience_key": "team-b"})
    return mem


ONE_SLOT = {"semantic_limit": 1, "stemmed_limit": 1, "exact_limit": 1}


def test_without_a_filter_the_best_hit_takes_every_single_slot(two_notes):
    assert {r["source"] for r in search(two_notes, "lighthouse", **ONE_SLOT)} == {"old.md"}


@pytest.mark.parametrize("filters", [
    {"date_from": "2024-01-01"},
    {"excluded_sources": ["old.md"]},
    {"metadata_filter": {"project": "b"}},
    {"metadata_filter": {"_allow": {"project": ["b"]}}},
    {"metadata_filter": {"_allow": {"audience_key": ["team-b"]}}},
])
def test_each_filter_applies_inside_every_retriever(two_notes, filters):
    [row] = search(two_notes, "lighthouse", **ONE_SLOT, **filters)
    # new.md took the single slot of the semantic, the stemmed AND the exact retriever
    assert row["source"] == "new.md"
    assert (row["sem_rank"], row["stem_rank"], row["exact_rank"]) == (1, 1, 1)


def test_date_window_is_inclusive_and_bounds_the_note_date(two_notes):
    only = lambda **kw: {r["source"] for r in search(two_notes, "lighthouse", **kw)}   # noqa: E731
    assert only(date_from="2020-01-01", date_to="2020-01-01") == {"old.md"}
    assert only(date_from="2020-01-02", date_to="2024-05-04") == set()
    assert only(date_from="2024-05-05") == {"new.md"}
    assert only(date_to="2024-05-05") == {"old.md", "new.md"}


def test_a_note_without_created_on_is_dated_by_created_at(two_notes):
    with psycopg.connect(two_notes.database_url) as conn:
        conn.execute("UPDATE documents SET created_on = NULL, created_at = '2022-06-15T12:00:00Z' "
                     "WHERE source = 'old.md'")
        conn.commit()
    rows = search(two_notes, "lighthouse", date_from="2022-06-15", date_to="2022-06-15")
    assert {r["source"] for r in rows} == {"old.md"}


def test_excluded_sources_may_name_several_and_none_means_no_exclusion(two_notes):
    assert {r["source"] for r in search(two_notes, "lighthouse", excluded_sources=["old.md", "new.md"])} == set()
    assert {r["source"] for r in search(two_notes, "lighthouse", excluded_sources=[])} == {"old.md", "new.md"}
    assert {r["source"] for r in search(two_notes, "lighthouse", excluded_sources=None)} == {"old.md", "new.md"}


def test_other_banks_never_appear(memory_with_bank, fresh_db):
    other = Memory(database_url=fresh_db, bank_id="other", embed=stub_embed)
    other.create_bank("other", embedding_dim=EMBED_DIM)
    try:
        other.retain("the lighthouse of the other bank", source="o.md")
        memory_with_bank.retain("the lighthouse of this bank", source="t.md")
        assert {r["source"] for r in search(memory_with_bank, "lighthouse")} == {"t.md"}
        assert {r["source"] for r in search(memory_with_bank, "lighthouse", bank="other")} == {"o.md"}
    finally:
        other.close()


# --------------------------------------------------------------------------
# the query uses the indexes, and sets the pgvector scan mode
# --------------------------------------------------------------------------

def test_the_scan_settings_are_set_for_the_query_transaction_only(memory_with_bank):
    retain_all(memory_with_bank, {"a.md": "lighthouse"})
    query = {"bank_id": "test", "query_text": "lighthouse",
             "query_embedding": stub_embed(["lighthouse"])[0]}
    with psycopg.connect(memory_with_bank.database_url) as conn:
        conn.execute("SELECT 1")                  # an open transaction, as inside a recall
        stage1_search(conn, **query)
        assert conn.execute("SHOW hnsw.iterative_scan").fetchone()[0] == "relaxed_order"
        assert conn.execute("SHOW hnsw.ef_search").fetchone()[0] == "100"
        conn.rollback()
        assert conn.execute("SHOW hnsw.iterative_scan").fetchone()[0] == "off"   # never leaks out
        assert conn.execute("SHOW hnsw.ef_search").fetchone()[0] == "40"
        stage1_search(conn, hnsw_ef_search=64, **query)
        assert conn.execute("SHOW hnsw.ef_search").fetchone()[0] == "64"


def test_the_plan_uses_the_hnsw_and_both_gin_indexes(fresh_db):
    load_bank(fresh_db, "plan", n_chunks=10_000, dim=32)   # big enough for the planner to prefer them
    sql, params = _stage1_statement(
        bank_id="plan", query_text="0x80070005", query_embedding=random_query_vector(32, 1))
    with psycopg.connect(fresh_db) as conn:
        plan = "\n".join(r[0] for r in conn.execute("EXPLAIN " + sql, params, prepare=False))
    assert "Index Scan using memory_items_embedding_plan" in plan   # on embedding::vector(N), the indexed expression
    assert "Bitmap Index Scan on memory_items_body_tsv_gin" in plan
    assert "Bitmap Index Scan on memory_items_exact_tsv_gin" in plan   # the stored exact vector (0020)
    assert "memory_items_keyword_gin" not in plan                      # the 0018 index: no code reads it any more


# --------------------------------------------------------------------------
# the hybrid_chunk channel
# --------------------------------------------------------------------------

def state_for(conn, **kw):
    return RecallState(conn=conn, bank_id="test", embed=stub_embed, **kw)


def test_channel_is_registered_and_opt_in():
    validate_channel_config([{"name": "hybrid_chunk", "weight": 1}])
    assert [c.name for c in build_channels([{"name": "hybrid_chunk"}])] == ["hybrid_chunk"]
    assert "hybrid_chunk" not in [e["name"] for e in DEFAULT_CHANNEL_CONFIG]


@pytest.mark.parametrize("params", [{"semantic_limit": 0}, {"exact_limit": -1}, {"fused_limit": 2.5},
                                    {"hnsw_ef_search": True}, {"stemmed_limit": "20"}])
def test_channel_params_are_validated(params):
    with pytest.raises(ValueError, match="hybrid_chunk"):
        build_channels([{"name": "hybrid_chunk", "params": params}])


def test_channel_returns_document_level_candidates_with_evidence_and_snippet(memory_with_bank):
    mem = memory_with_bank
    retain_all(mem, PHRASE_NOTES)
    mem.retain("\n\n".join(f"Paragraph {i} about the harbour. " + "The crew mends nets. " * 40
                           for i in range(4)), source="long.md")      # several chunks, one note
    with psycopg.connect(mem.database_url) as conn:
        fused, trace = run_channels(
            state_for(conn), QueryPlan(text='"electric dreams"'), [{"name": "hybrid_chunk", "weight": 1}])
    cands = [c for c in trace["candidates"] if c["channel"] == "hybrid_chunk"]
    assert len({c["document_id"] for c in cands}) == len(cands)           # one candidate per note
    assert [c["rank"] for c in cands] == list(range(1, len(cands) + 1))
    assert trace["channels"][0]["error"] is None
    top = fused[0]
    assert top.source in {"adjacent.md", "upper.md"}
    assert top.best.evidence in {PHRASE_NOTES["adjacent.md"], PHRASE_NOTES["upper.md"]}
    assert top.best.detail["exact_snippet"].count("<<") == 2 and top.best.detail["exact_rank"] == 1


def test_channel_params_reach_the_query(memory_with_bank):
    retain_all(memory_with_bank, {f"n{i}.md": f"lighthouse number {i}" for i in range(8)})
    with psycopg.connect(memory_with_bank.database_url) as conn:
        _, trace = run_channels(
            state_for(conn), QueryPlan(text="lighthouse"),
            [{"name": "hybrid_chunk", "params": {"fused_limit": 3}}])
    assert trace["channels"][0]["n"] == 3


def test_channel_reads_the_per_call_filters_from_the_recall_state(two_notes):
    with psycopg.connect(two_notes.database_url) as conn:
        def docs(**kw):
            fused, _ = run_channels(state_for(conn, **kw), QueryPlan(text="lighthouse"),
                                    [{"name": "hybrid_chunk"}])
            return {f.source for f in fused}
        assert docs() == {"old.md", "new.md"}
        assert docs(excluded_sources=["old.md"]) == {"new.md"}
        assert docs(date_from="2024-01-01") == {"new.md"}
        assert docs(date_to="2021-01-01") == {"old.md"}
        assert docs(metadata_filter={"project": "a"}) == {"old.md"}


def test_search_through_the_bank_config_finds_the_phrase_first(memory_with_bank):
    mem = memory_with_bank
    retain_all(mem, PHRASE_NOTES)
    mem.set_channel_config([{"name": "hybrid_chunk", "weight": 1}])
    hits = mem.search('"electric dreams"', limit=2)
    assert {h.source for h in hits} == {"adjacent.md", "upper.md"}
    assert hits[0].original_chunk in {PHRASE_NOTES["adjacent.md"], PHRASE_NOTES["upper.md"]}
