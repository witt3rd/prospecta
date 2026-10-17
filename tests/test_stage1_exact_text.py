"""Stage 1 `exact_text`: what the exact retriever reads INSTEAD of the query.

Query expansion (docs/postgres-hybrid.md, Mode 1) gives every search a `query` (a description, for
the semantic and stemmed retrievers and the reranker) and `terms` (names, identifiers and
"quoted phrases" copied verbatim from the prompt, for the exact retriever). `stage1_search`
takes the terms as `exact_text`; None leaves the exact retriever reading `query_text` as it
always did. Real Postgres, invented text."""
from __future__ import annotations

import psycopg
import pytest

from prospecta.channels import QueryPlan, RecallState
from prospecta.channels.hybrid import HybridChunks
from prospecta.db.queries import _stage1_statement, stage1_search
from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM, stub_embed

NOTES = {
    "update.md": "The updater stopped with error 0x80070005 right after the v1.14 rollout.",
    "dreams.md": "The night crew called the bridge lamps electric dreams.",
    "scatter.md": "Electric lamps burn late and the crew keeps its dreams for the morning shift.",
    "pool.md": "PROSPECTA_DB_POOL_MAX caps the database connections of the gateway.",
    "dan.md": "Dan Maycock said the pontoon survey is due in May.",
    "plain.md": "Orchids need humid air and a shaded window.",
}


@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="test", embed=stub_embed)
    m.create_bank("test", embedding_dim=EMBED_DIM)
    for source, text in NOTES.items():
        m.retain(text, source=source, metadata={"created_on": "2024-03-05"})
    yield m
    m.close()


def search(mem, text, **kw):
    with psycopg.connect(mem.database_url) as conn:
        return stage1_search(conn, bank_id="test", query_text=text,
                             query_embedding=stub_embed([text])[0], **kw)


def matched(rows, column):
    return {r["source"] for r in rows if r[column] is not None}


def test_without_exact_text_the_exact_retriever_reads_the_query(mem):
    for q in ('"electric dreams"', "electric dreams", "v1.14", "capacity planning", "the"):
        assert search(mem, q) == search(mem, q, exact_text=None) == search(mem, q, exact_text=q), q


def test_the_exact_retriever_reads_the_exact_text_and_the_others_the_query(mem):
    rows = search(mem, "capacity planning", exact_text="PROSPECTA_DB_POOL_MAX")
    assert matched(rows, "exact_rank") == {"pool.md"}                   # found by words the query lacks
    pool = next(r for r in rows if r["source"] == "pool.md")
    assert "<<PROSPECTA>>" in pool["exact_snippet"] and "<<MAX>>" in pool["exact_snippet"]   # its parts, marked
    assert matched(rows, "stem_rank") == set()                           # the query matched no word
    rows = search(mem, "pontoon survey", exact_text="PROSPECTA_DB_POOL_MAX")
    assert matched(rows, "stem_rank") == {"dan.md"}                      # the query still feeds the stemmed one
    assert matched(rows, "exact_rank") == {"pool.md"}                    # and the terms, not the query, the exact one


def test_several_terms_are_an_or_and_a_quoted_phrase_stays_adjacent(mem):
    rows = search(mem, "background", exact_text='"electric dreams" OR v1.14 OR "Dan Maycock"')
    assert matched(rows, "exact_rank") == {"dreams.md", "update.md", "dan.md"}
    assert "scatter.md" not in matched(rows, "exact_rank")               # the words apart are not the phrase
    assert matched(search(mem, "background", exact_text='"dreams electric"'), "exact_rank") == set()
    first = next(r for r in rows if r["source"] == "update.md")
    assert "<<v1.14>>" in first["exact_snippet"]


def test_exact_text_that_holds_words_is_something_to_ask_even_when_the_query_has_none(mem):
    for q in ("the and a of", "", "   "):
        rows = search(mem, q, exact_text="PROSPECTA_DB_POOL_MAX")
        assert matched(rows, "exact_rank") == {"pool.md"}, q
        rows = search(mem, q, exact_text="PROSPECTA_DB_POOL_MAX", date_from="2024-01-01", date_to="2024-12-31")
        assert matched(rows, "exact_rank") == {"pool.md"} and not any(r["date_rank"] for r in rows), q
    # no exact text and no words: the date-only listing and the empty answer are unchanged
    assert search(mem, "the and a of", date_from="2024-01-01")[0]["date_rank"] == 1
    assert search(mem, "") == []


def test_the_filters_apply_to_the_exact_text_retriever_too(mem):
    assert search(mem, "capacity", exact_text="v1.14", date_from="2025-01-01") == []
    assert matched(search(mem, "capacity", exact_text="v1.14", excluded_sources=["update.md"]),
                   "exact_rank") == set()
    assert matched(search(mem, "capacity", exact_text="v1.14", date_to="2024-12-31"), "exact_rank") == {"update.md"}


def test_the_statement_carries_the_exact_text_as_its_own_parameter():
    sql, params = _stage1_statement(bank_id="b", query_text="a question", query_embedding=[0.1] * 4)
    assert params["qx"] == "a question"
    sql, params = _stage1_statement(bank_id="b", query_text="a question", query_embedding=[0.1] * 4,
                                    exact_text='v1.14 OR "Dan Maycock"')
    assert params["qt"] == "a question" and params["qx"] == 'v1.14 OR "Dan Maycock"'
    assert "websearch_to_tsquery('simple', %(qx)s)" in sql and "websearch_to_tsquery('simple', %(qt)s)" not in sql


def test_the_channel_passes_the_plans_exact_text(mem):
    with psycopg.connect(mem.database_url) as conn:
        state = RecallState(conn=conn, bank_id="test", embed=stub_embed)
        plan = QueryPlan(text="capacity planning", exact_text="PROSPECTA_DB_POOL_MAX")
        got = HybridChunks().retrieve(plan, state)
        assert [c.source for c in got if c.detail["exact_rank"] is not None] == ["pool.md"]
        assert plan.to_json()["exact_text"] == "PROSPECTA_DB_POOL_MAX"
        assert "exact_text" not in QueryPlan(text="q").to_json()
