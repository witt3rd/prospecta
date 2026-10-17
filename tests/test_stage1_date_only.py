"""Stage 1 with a date window and no searchable words (hybrid plan, slice 5 finding B).

"What did we do in March?" has no words worth matching: the stemmed ('english') reading
of the question is empty (blank, punctuation, or only stop words) and the date window is
the whole question. Stage 1 then lists the chunks inside the window, newest note first,
as `date` candidates for the reranker; no word retriever and no embedding is involved.
Every other filter still applies. A question with words keeps the three retrievers, and
no words and no window is empty. Real Postgres, invented text, the stub embedder."""
from __future__ import annotations

import psycopg
import pytest

from prospecta._types import RecalledMemory
from prospecta.channels import QueryPlan, RecallState, run_channels
from prospecta.channels.hybrid import HybridChunks
from prospecta.cited import format_cited_rag_block, match_kind
from prospecta.db.queries import stage1_search
from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM, stub_embed

NOTES = {   # source -> (text, created_on, project, audience)
    "jan.md": ("The harbour lamps were repainted in January.", "2024-01-10", "a", "team-a"),
    "feb.md": ("Orchids need humid air and a shaded window.", "2024-02-14", "b", "team-b"),
    "mar1.md": ("We decided the archive stays append-only.", "2024-03-05", "a", "team-a"),
    "mar2.md": ("The tree view loads folders lazily, and the crew liked it.", "2024-03-20", "b", "team-b"),
    "apr.md": ("The lighthouse keeper wrote a note about the lamps.", "2024-04-02", "a", "team-b"),
}
NO_WORDS = ["", "   ", "!!!", "the", "the and a of", "what is it to be", '"the"']


@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="test", embed=stub_embed)
    m.create_bank("test", embedding_dim=EMBED_DIM)
    for source, (text, day, project, audience) in NOTES.items():
        m.retain(text, source=source, metadata={"created_on": day, "project": project,
                                                "audience_key": audience})
    yield m
    m.close()


def search(mem, text, bank="test", **kw):
    with psycopg.connect(mem.database_url) as conn:
        return stage1_search(conn, bank_id=bank, query_text=text,
                             query_embedding=stub_embed([text])[0], **kw)


def sources(rows):
    return [r["source"] for r in rows]


MARCH = {"date_from": "2024-03-01", "date_to": "2024-03-31"}


# --------------------------------------------------------------------------
# the window, newest note first
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", NO_WORDS[:-1])
def test_a_wordless_question_lists_the_window_newest_note_first(mem, text):
    rows = search(mem, text, date_from="2024-02-01", date_to="2024-04-30")
    assert sources(rows) == ["apr.md", "mar2.md", "mar1.md", "feb.md"]
    assert [str(r["created_on"]) for r in rows] == ["2024-04-02", "2024-03-20", "2024-03-05", "2024-02-14"]


def test_date_rows_are_marked_and_carry_no_word_evidence(mem):
    rows = search(mem, "the and a of", **MARCH)
    assert [r["date_rank"] for r in rows] == [1, 2]
    for r in rows:
        assert (r["sem_rank"], r["stem_rank"], r["exact_rank"], r["exact_snippet"]) == (None,) * 4
        assert r["content"] == r["original_chunk"] and r["metadata"]["project"] in ("a", "b")
        assert r["document_id"] and r["id"]
    first, second = (float(r["rrf_score"]) for r in rows)
    assert first > second > 0                                  # newest first is also the fallback order


def test_a_window_open_on_one_side_is_a_window(mem):
    assert sources(search(mem, "", date_from="2024-03-10")) == ["apr.md", "mar2.md"]
    assert sources(search(mem, "", date_to="2024-02-28")) == ["feb.md", "jan.md"]


def test_the_window_is_inclusive_and_a_note_without_created_on_is_dated_by_created_at(mem):
    assert sources(search(mem, "", date_from="2024-03-05", date_to="2024-03-05")) == ["mar1.md"]
    with psycopg.connect(mem.database_url) as conn:
        conn.execute("UPDATE documents SET created_on = NULL, created_at = '2024-03-12T09:00:00Z' "
                     "WHERE source = 'jan.md'")
        conn.commit()
    assert sources(search(mem, "", **MARCH)) == ["mar2.md", "jan.md", "mar1.md"]


def test_an_empty_window_is_empty(mem):
    assert search(mem, "", date_from="1999-01-01", date_to="1999-12-31") == []
    assert search(mem, "the", date_from="2024-03-21", date_to="2024-04-01") == []


def test_the_list_is_cut_at_fused_limit(mem):
    rows = search(mem, "", date_from="2024-01-01", date_to="2024-12-31", fused_limit=3)
    assert sources(rows) == ["apr.md", "mar2.md", "mar1.md"]


def test_notes_of_one_day_are_listed_newest_created_first_and_a_note_chunks_in_order(mem):
    mem.retain("A second note of the same day.", source="mar1b.md", metadata={"created_on": "2024-03-05"})
    assert sources(search(mem, "", date_from="2024-03-05", date_to="2024-03-05")) == ["mar1b.md", "mar1.md"]
    long_note = " ".join(f"Sentence number {i} about nothing in particular." for i in range(120))
    mem.retain(long_note, source="long.md", metadata={"created_on": "2024-05-01"})
    rows = search(mem, "", date_from="2024-05-01", date_to="2024-05-01")
    assert len(rows) > 1 and set(sources(rows)) == {"long.md"}
    assert [r["original_chunk"] for r in rows] == sorted(
        (r["original_chunk"] for r in rows), key=long_note.index)       # in the note's own order


# --------------------------------------------------------------------------
# every other pushed-down filter still applies
# --------------------------------------------------------------------------

def test_excluded_sources_apply(mem):
    assert sources(search(mem, "", **MARCH, excluded_sources=["mar2.md"])) == ["mar1.md"]
    assert search(mem, "", **MARCH, excluded_sources=["mar1.md", "mar2.md"]) == []
    assert sources(search(mem, "", **MARCH, excluded_sources=[])) == ["mar2.md", "mar1.md"]


@pytest.mark.parametrize("meta, expected", [
    ({"project": "a"}, ["mar1.md"]),
    ({"_allow": {"project": ["b"]}}, ["mar2.md"]),
    ({"_allow": {"audience_key": ["team-a"]}}, ["mar1.md"]),
    ({"project": "a", "_allow": {"audience_key": ["team-b"]}}, []),
])
def test_metadata_and_audience_filters_apply(mem, meta, expected):
    assert sources(search(mem, "", **MARCH, metadata_filter=meta)) == expected


def test_other_banks_and_question_items_never_appear(mem, fresh_db):
    other = Memory(database_url=fresh_db, bank_id="other", embed=stub_embed)
    other.create_bank("other", embedding_dim=EMBED_DIM)
    try:
        other.retain("A note of another bank.", source="o.md", metadata={"created_on": "2024-03-06"})
        mem.retain("Nothing about questions lives here.", source="q.md", metadata={"created_on": "2024-03-07"})
        with psycopg.connect(mem.database_url) as conn:
            conn.execute(          # an old bank's question item (written raw: nothing writes them now)
                "UPDATE memory_items SET kind = 'question' WHERE document_id IN "
                "(SELECT id FROM documents WHERE source = 'q.md')")
            conn.commit()
            kinds = {k for (k,) in conn.execute(
                "SELECT m.kind FROM memory_items m JOIN documents d ON d.id = m.document_id "
                "WHERE d.source = 'q.md'")}
        assert kinds == {"question"}          # the bank holds it as a question item only
        assert sources(search(mem, "", **MARCH)) == ["mar2.md", "mar1.md"]
        assert sources(search(mem, "", **MARCH, bank="other")) == ["o.md"]
    finally:
        other.close()


# --------------------------------------------------------------------------
# what stays as it was
# --------------------------------------------------------------------------

def test_a_question_with_words_and_a_window_keeps_the_three_retrievers(mem):
    rows = search(mem, "lamps", date_from="2024-01-01", date_to="2024-12-31")
    assert {r["source"] for r in rows} == set(NOTES)           # the semantic slots fill up as before
    assert all(r["date_rank"] is None for r in rows)
    lexical = {r["source"] for r in rows if r["exact_rank"] is not None}
    assert lexical == {"jan.md", "apr.md"}
    only_march = search(mem, "lamps", **MARCH)
    assert {r["source"] for r in only_march} == {"mar1.md", "mar2.md"}   # semantic only: no lexical hit
    assert all(r["date_rank"] is None and r["exact_rank"] is None for r in only_march)


def test_no_window_and_no_words_is_empty_and_a_stop_word_question_still_runs_the_retrievers(mem):
    for blank in ("", "   ", "!!!", "-", '""'):
        assert search(mem, blank) == []
    rows = search(mem, "the")                  # no window: nothing to list; the exact retriever answers
    assert {r["source"] for r in rows if r["exact_rank"] is not None} == {
        "jan.md", "mar1.md", "mar2.md", "apr.md"}
    assert all(r["date_rank"] is None for r in rows)


def test_every_row_has_a_date_rank_column(mem):
    assert all("date_rank" in r for r in search(mem, "lamps"))
    assert all("date_rank" in r for r in search(mem, "", **MARCH))


# --------------------------------------------------------------------------
# the channel, the contract, the block
# --------------------------------------------------------------------------

def counting_embed():
    calls = []

    def embed(texts):
        calls.append(list(texts))
        return stub_embed(texts)
    return embed, calls


def test_the_channel_lists_the_window_without_embedding_the_question(mem):
    embed, calls = counting_embed()
    with psycopg.connect(mem.database_url) as conn:
        state = RecallState(conn=conn, bank_id="test", embed=embed,
                            date_from="2024-02-01", date_to="2024-04-30")
        got = HybridChunks().retrieve(QueryPlan(text="the and a of"), state)
        assert [c.source for c in got] == ["apr.md", "mar2.md", "mar1.md", "feb.md"]
        assert [c.rank for c in got] == [1, 2, 3, 4]
        assert all(c.detail["date_rank"] == c.rank and c.detail["sem_rank"] is None for c in got)
        assert all(c.score > 0 for c in got) and got[0].evidence == NOTES["apr.md"][0]
        assert calls == []                                      # no embedding call for a wordless question
        HybridChunks().retrieve(QueryPlan(text="lamps"), state)
        assert calls == [["lamps"]]                             # a question with words embeds as before


def test_search_hits_say_match_date_and_come_newest_first(mem):
    mem.set_channel_config([{"name": "hybrid_chunk", "weight": 1}])
    hits = mem.search("the and a of", limit=10, date_from="2024-02-01", date_to="2024-03-31")
    assert [h.source for h in hits] == ["mar2.md", "mar1.md", "feb.md"]
    assert {h.match_kind for h in hits} == {"date"} and [h.created_on for h in hits] == [
        "2024-03-20", "2024-03-05", "2024-02-14"]
    assert all(h.exact_snippet is None for h in hits)
    words = mem.search("lamps", limit=10, date_from="2024-01-01", date_to="2024-12-31")
    assert "date" not in {h.match_kind for h in words}


def test_match_kind_date_is_the_last_resort_and_renders_in_the_block():
    assert match_kind(exact=False, lexical=False, semantic=False, date=True) == "date"
    assert match_kind(exact=False, lexical=False, semantic=True, date=True) == "semantic"
    assert match_kind(exact=True, lexical=False, semantic=False, date=True) == "exact"
    assert match_kind(exact=False, lexical=False, semantic=False) is None
    item = RecalledMemory(content="Note text.", original_chunk="Note text.", source="mar1.md", score=0.1,
                          scores={}, metadata={}, bank_id="b", document_id="d", created_on="2024-03-05",
                          kind="note", match_kind="date")
    assert format_cited_rag_block([item], heading=None) == (
        "[1] source=mar1.md | date=2024-03-05 | kind=note | match=date\n    Note text.")


def test_run_channels_runs_the_date_listing(mem):
    mem.set_channel_config([{"name": "hybrid_chunk", "weight": 1}])
    with psycopg.connect(mem.database_url) as conn:
        state = RecallState(conn=conn, bank_id="test", embed=stub_embed, **MARCH)
        fused, trace = run_channels(state, QueryPlan(text="the"), [{"name": "hybrid_chunk", "weight": 1}])
        assert [f.source for f in fused] == ["mar2.md", "mar1.md"]
        assert trace["channels"][0]["n"] == 2 and trace["channels"][0]["error"] is None
