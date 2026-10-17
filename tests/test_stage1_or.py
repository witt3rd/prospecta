"""Stage 1: the stemmed retriever reads a plain question as an OR of its words.

websearch_to_tsquery ANDs every word, so a natural-language question (Mode 1A recall,
Mode 1B task facets, ask()) matched the stemmed retriever only when ONE chunk held every
word of it. A question with no search operator (no straight double quote, no standalone
`OR`, no word led by `-`) now matches the OR of its stemmed words (stop words drop), ranked
by ts_rank_cd, so coverage orders the candidates and the reranker decides. The exact /
phrase retriever keeps websearch semantics for every query, and a query WITH an operator
keeps websearch semantics on both retrievers: the search-box syntax of
docs/postgres-hybrid.md is exactly as documented. Real Postgres, invented text."""
from __future__ import annotations

import psycopg
import pytest

from prospecta.db import queries
from prospecta.db.queries import _stage1_statement, has_websearch_operators, stage1_search
from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM, stub_embed

NOTES = {   # the question's words: harbour lamp(s) crew repaint winter storm (the rest are stop words)
    "all.md": "Which lamps did the harbour crew repaint before the winter storm? The old lamps.",
    "five.md": "The harbour crew repainted the lamps before the storm arrived.",
    "two.md": "The crew repainted the fence before dinner.",
    "one.md": "Orchids need humid air and a shaded window in winter.",
    "other.md": "The archive stays append-only and deletes are tombstones.",
    "paint.md": "Painting the shed is on the list for spring.",
}
QUESTION = "which harbour lamps did the crew repaint before the winter storm"


@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="test", embed=stub_embed)
    m.create_bank("test", embedding_dim=EMBED_DIM)
    for source, text in NOTES.items():
        m.retain(text, source=source)
    yield m
    m.close()


def search(mem, text, **kw):
    with psycopg.connect(mem.database_url) as conn:
        return stage1_search(conn, bank_id="test", query_text=text,
                             query_embedding=stub_embed([text])[0], **kw)


def matched(rows, column):
    return {r["source"] for r in rows if r[column] is not None}


def stem_order(rows):
    return [r["source"] for r in sorted((r for r in rows if r["stem_rank"] is not None),
                                        key=lambda r: r["stem_rank"])]


# --------------------------------------------------------------------------
# the rule
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text, operators", [
    (QUESTION, False),
    ("what did we decide, and why? (see notes) 2024-05-05", False),
    ("a well-known plan - and its cost -- ok", False),    # a hyphen inside a word, alone, or doubled
    ("ORDER BY the harbour", False),                        # OR inside a word is not the operator
    ("harbour or crew", False),                             # lower case `or` is a word, not the documented operator
    ("", False),
    ('"electric dreams"', True),
    ('what is "electric dreams" about', True),
    ("harbour OR crew", True),
    ("OR crew", True),
    ("harbour OR", True),
    ("memory -shadow", True),
    ("-shadow memory", True),
    ("plan -budget -scope", True),
])
def test_which_questions_carry_search_operators(text, operators):
    assert has_websearch_operators(text) is operators


def test_a_natural_question_finds_a_note_sharing_two_of_its_words_and_the_and_reading_did_not(mem):
    rows = search(mem, QUESTION)
    assert matched(rows, "stem_rank") == {"all.md", "five.md", "two.md", "one.md"}   # any word, stemmed
    with psycopg.connect(mem.database_url) as conn:
        and_hits = {s for (s,) in conn.execute(
            "SELECT d.source FROM memory_items m JOIN documents d ON d.id = m.document_id "
            "WHERE m.body_tsv @@ websearch_to_tsquery('english', %s)", (QUESTION,))}
    assert and_hits == {"all.md"}      # what the stemmed retriever returned before: the one note with every word
    two = next(r for r in rows if r["source"] == "two.md")
    assert two["exact_rank"] is None and "<<crew>>" in two["exact_snippet"]    # the stemmed snippet marks its words


def test_more_of_the_words_ranks_higher(mem):
    assert stem_order(search(mem, QUESTION)) == ["all.md", "five.md", "two.md", "one.md"]


def test_stop_words_drop_and_a_stop_word_question_matches_nothing_in_the_stemmed_retriever(mem):
    assert matched(search(mem, "the and a of to in"), "stem_rank") == set()
    assert matched(search(mem, "what did the"), "stem_rank") == set()


def test_the_exact_retriever_keeps_websearch_semantics_for_a_natural_question(mem):
    rows = search(mem, QUESTION)
    assert matched(rows, "exact_rank") == {"all.md"}      # AND: only the chunk holding every word as written
    assert matched(search(mem, "harbour crew lamps"), "exact_rank") == {"five.md", "all.md"}
    assert matched(search(mem, "harbour crew orchids"), "exact_rank") == set()   # AND, not OR


def test_the_stemmed_or_stems_its_words(mem):
    assert matched(search(mem, "repainting lamp"), "stem_rank") == {"all.md", "five.md", "two.md"}
    # stems: 'repainting' reaches 'repaint' and 'repainted', 'lamp' reaches 'lamps' ('paint.md' stays out);
    # the exact retriever matches words as written
    assert matched(search(mem, "repainting lamp"), "exact_rank") == set()


def test_a_one_word_question_is_unchanged(mem):
    assert matched(search(mem, "lamps"), "stem_rank") == {"five.md", "all.md"}


# --------------------------------------------------------------------------
# a query WITH an operator keeps websearch semantics on both retrievers
# --------------------------------------------------------------------------

def test_a_quoted_phrase_keeps_adjacency_and_and(mem):
    rows = search(mem, '"harbour crew" storm')
    assert matched(rows, "stem_rank") == {"five.md", "all.md"}      # phrase AND word, not an OR
    assert matched(rows, "exact_rank") == {"five.md", "all.md"}
    assert matched(search(mem, '"crew harbour"'), "stem_rank") == set()          # adjacency, in order


def test_minus_keeps_excluding_and_the_rest_is_and(mem):
    assert matched(search(mem, "lamps -storm"), "stem_rank") == set()
    assert matched(search(mem, "lamps -orchids"), "stem_rank") == {"five.md", "all.md"}
    assert matched(search(mem, "harbour winter -orchids"), "stem_rank") == {"all.md"}   # AND of the two words


def test_the_or_keyword_keeps_websearch_or(mem):
    rows = search(mem, "orchids OR archive")
    assert matched(rows, "stem_rank") == matched(rows, "exact_rank") == {"one.md", "other.md"}
    rows = search(mem, "harbour lamps OR orchids")        # (harbour AND lamps) OR orchids
    assert matched(rows, "stem_rank") == {"five.md", "all.md", "one.md"}
    assert matched(search(mem, "harbour winter OR archive"), "stem_rank") == {"all.md", "other.md"}


def test_identifiers_still_match_whole(mem):
    mem.retain("The updater stopped with error 0x80070005 after the v1.14 rollout.", source="update.md")
    for q in ("0x80070005", "v1.14", "0x80070005 v1.14"):
        assert matched(search(mem, q), "exact_rank") == {"update.md"}, q


# --------------------------------------------------------------------------
# the OR is built from parsed words, never from the text
# --------------------------------------------------------------------------

NASTY = [
    "it's the crew's plan & lamps | storm ! (harbour) :* <-> \\ ' ;--",
    "lamps' & 'storm", "lamps\\", "lamps:*", "lamps:A", "!lamps", "lamps <2> storm", "lamps | | storm",
    "a@b.example lamps http://x.example/p?q=1&r=2 /usr/bin/env C:\\tools\\run.exe",
    "'; DROP TABLE memory_items; --", "lamps ) ( storm", "\x01lamps\x1f storm", "lamps\tstorm\nharbour",
    "lamps\u00a0storm \u201cquoted\u201d", "!!! ??? ...", "&", "|", "'", "\\", "( )",
]


@pytest.mark.parametrize("text", NASTY)
def test_punctuation_and_operator_like_text_cannot_break_or_inject(mem, text):
    assert has_websearch_operators(text) is False
    rows = search(mem, text)                     # must not raise
    assert isinstance(rows, list)
    with psycopg.connect(mem.database_url) as conn:      # and nothing was dropped
        assert conn.execute("SELECT count(*) FROM memory_items").fetchone()[0] == len(NOTES)


def test_or_reading_of_punctuated_words_matches_the_words(mem):
    rows = search(mem, "the crew's plan & lamps | storm ! (harbour) :* <-> \\ '")
    assert matched(rows, "stem_rank") == {"all.md", "five.md", "two.md"}      # crew, lamp, storm, harbour


def test_the_statement_uses_the_or_form_only_without_operators():
    plain, _ = _stage1_statement(bank_id="b", query_text="which lamps", query_embedding=[0.1] * 4)
    quoted, _ = _stage1_statement(bank_id="b", query_text='"which lamps"', query_embedding=[0.1] * 4)
    assert "unnest(to_tsvector('english'" in plain and "websearch_to_tsquery('english'" not in plain
    assert "websearch_to_tsquery('english'" in quoted and "unnest(" not in quoted
    assert "websearch_to_tsquery('simple'" in plain and "websearch_to_tsquery('simple'" in quoted


# --------------------------------------------------------------------------
# a long question uses every one of its words (no cap, nothing dropped silently)
# --------------------------------------------------------------------------

def test_every_word_of_a_long_question_is_used(mem):
    """There is no cap on the words of the OR: a word is never dropped for being short
    (the old rule kept the 8 longest and ignored the rest without a word)."""
    words = ["qw" * (n // 2) + "q" * (n % 2) for n in range(3, 21)]      # 18 words of 3..20 letters (under the 20 stemmed slots)
    for n, w in enumerate(words):
        mem.retain(f"Note holding {w} only.", source=f"w{n + 3}.md")
    assert matched(search(mem, " ".join(words)), "stem_rank") == {f"w{n}.md" for n in range(3, 21)}


def test_there_is_no_word_cap_in_the_statement_or_the_api(mem):
    assert not hasattr(queries, "STAGE1_STEM_OR_TERMS")
    sql, params = _stage1_statement(bank_id="b", query_text="a long plain question here",
                                    query_embedding=[0.1] * 4)
    assert "stem_or_terms" not in params and "stem_or_terms" not in sql
    with pytest.raises(TypeError):
        _stage1_statement(bank_id="b", query_text="x", query_embedding=[0.1] * 4, stem_or_terms=2)
