"""Mode 1 of docs/postgres-hybrid.md, the writer's terms: only proper names and identifiers.

A skill-wake turn (a cron-started turn whose prompt is a skill invocation: a marker line, then the
skill's instruction text) made the query writer copy the INSTRUCTION VOCABULARY into `terms`
(session, cron job, SILENT, DELIVERY, IMPORTANT, instructions ...), and those words matched half the
bank. The writer's prompt now says terms are proper names, identifiers, versions, paths, error codes
and "quoted phrases" only, and a deterministic filter after parsing drops a term that is a plain
English word unless the prompt quoted it. Everything here is invented text of the same shape."""
from __future__ import annotations

import json
from datetime import date

import pytest

from prospecta._common_words import COMMON_WORDS
from prospecta._expand import (
    guard_terms, is_plain_word, keep_term, parse_writer_reply, render_writer_prompt, verbatim_terms,
)
from prospecta._types import SearchSpec
from tests.test_expand import FakeRerank, FakeWriter, S, build, sources

TODAY = date(2024, 6, 1)

# The shape of the real bad prompt, invented: the skill marker, the skill's instruction text with its
# all-caps rules, a delivery rule, a path, one name. None of its words is a thing to look up.
SKILL_PROMPT = (
    '[IMPORTANT: The user has invoked the "harbour-watch" skill, indicating they want you to follow '
    'its instructions. The full skill content is loaded below.]\n\n'
    "# Harbour watch\n\n"
    "This is a scheduled cron job. Each session you read the notes list and write a short report for "
    "the user. IMPORTANT: do NOT message anyone unless something changed; if there is nothing new, "
    "reply with exactly [SILENT]. DELIVERY: the report goes to the crew channel only. Follow these "
    "instructions in order, and never touch /srv/atlas/harbour/notes. Access questions go to Dan Maycock.")

INSTRUCTION_WORDS = ["session", "cron job", "SILENT", "DELIVERY", "IMPORTANT", "instructions",
                     "IMPORTANT, instructions", "list", "user", "report", "NOT", "cron", "job"]


@pytest.fixture
def made(fresh_db):
    mems = []

    def make(**kw):
        m = build(fresh_db, **kw)
        mems.append(m)
        return m
    yield make
    for m in mems:
        m.close()


def terms_of(result):
    return [t for s in result.searches for t in s.terms]


# --------------------------------------------------------------------------
# the plain-word test
# --------------------------------------------------------------------------

def test_the_embedded_word_list_is_small_lowercase_and_covers_the_instruction_vocabulary():
    assert 3_000 < len(COMMON_WORDS) < 15_000
    assert all(w.isascii() and w.isalpha() and w == w.lower() for w in COMMON_WORDS)
    for w in ("session", "cron", "job", "list", "user", "report", "important", "not", "silent",
              "delivery", "instructions", "the", "and", "message"):
        assert is_plain_word(w), w
        assert is_plain_word(w.upper()), w                    # IMPORTANT, SILENT, NOT


def test_inflections_of_a_plain_word_are_plain():
    for w in ("sessions", "reports", "jobs", "listed", "listing", "delivered", "quickly", "queries", "users"):
        assert is_plain_word(w), w


@pytest.mark.parametrize("w", ["Maycock", "Hawthorne", "prospecta", "rclone", "OneDrive", "kubelet", "NASA",
                               "v1.14", "0x80070005", "a/b", "session_id", "HybridChunks", "", "42", "ab1"])
def test_names_identifiers_and_unknown_words_are_not_plain(w):
    assert not is_plain_word(w)


def test_a_markdown_heading_or_bullet_starts_a_sentence_so_its_first_word_is_no_name():
    text = "# Harbour watch\n\n- Check the notes\n1. Report it\n> Quote this\n**Important** is bold.\nThen go."
    assert verbatim_terms(text) == []
    assert verbatim_terms("# Harbour Watch Rota\n\nthen ask Dan Maycock.") == ["Harbour Watch Rota", "Dan Maycock"]


def test_a_capitalised_plain_word_the_prompt_also_writes_in_lower_case_is_no_name():
    text = "Each session you read the notes. Use the Session view and the Report list, then write the report."
    assert verbatim_terms(text) == []
    assert kept(text, "Session", "Report", "session") == []
    # a plain word that only ever appears capitalised, in the middle of a sentence, is still a name
    assert verbatim_terms("Please ask Rose about the Library again.") == ["Rose", "Library"]
    assert verbatim_terms("Please ask Dan Maycock and the Harbour Office; the office is closed.") == [
        "Dan Maycock", "Harbour Office"]                                    # one word of the run is not plain


def test_contractions_of_plain_words_are_plain():
    for w in ("don't", "it's", "can't", "I'm", "we'll", "they've", "it\u2019s"):
        assert is_plain_word(w), w
    assert not is_plain_word("Dan's") and not is_plain_word("o'clock-ish")


# --------------------------------------------------------------------------
# the filter: what survives, what goes
# --------------------------------------------------------------------------

def reply(*searches):
    return json.dumps({"searches": list(searches)})


def kept(prompt, *terms):
    parsed = parse_writer_reply(reply(S("something to find", terms)), TODAY, prompt)
    return list(parsed.searches[0].terms)


def test_a_skill_style_instruction_prompt_yields_no_terms_from_its_instruction_words():
    got = kept(SKILL_PROMPT, *INSTRUCTION_WORDS)
    assert got == []
    parsed = parse_writer_reply(reply(S("something to find", INSTRUCTION_WORDS)), TODAY, SKILL_PROMPT)
    assert len(parsed.dropped_terms) == len(INSTRUCTION_WORDS)           # every one counted, none lost silently
    assert parsed.dropped_searches == 0                                  # the search has a query: it stays


def test_the_guard_does_not_put_back_instruction_words_of_a_skill_prompt():
    [spec], back = guard_terms([SearchSpec("something to find")], SKILL_PROMPT)
    assert "SILENT" not in back and "DELIVERY" not in back and "IMPORTANT" not in back
    assert set(back) == {'"harbour-watch"', "/srv/atlas/harbour/notes", "Dan Maycock"}
    assert verbatim_terms("DELIVERY: reply SILENT, NOT NOW, in UTC; but ask NASA.") == ["NASA"]


def test_all_caps_words_are_kept_only_when_the_prompt_quotes_them():
    prompt = 'When idle reply "SILENT" and no more; IMPORTANT: stay SILENT otherwise. DELIVERY is by mail.'
    assert verbatim_terms(prompt) == ['"SILENT"']
    assert kept(prompt, "SILENT", '"SILENT"', "IMPORTANT", "DELIVERY") == ["SILENT"]   # quoted in the prompt: kept, once


def test_proper_names_versions_paths_codes_and_quoted_phrases_survive():
    prompt = ('Dan Maycock hit error 0x80070005 after v1.14 in /a/b/c and src/main.rs; set '
              'PROSPECTA_DB_POOL_MAX, see HybridChunks, run --force, text-embedding-3-large, ABC-123, '
              'and "electric dreams" please. Greg, then, asked about Hawthorne Lane.')
    terms = ["Dan Maycock", "0x80070005", "v1.14", "/a/b/c", "src/main.rs", "PROSPECTA_DB_POOL_MAX",
             "HybridChunks", "--force", "text-embedding-3-large", "ABC-123", '"electric dreams"',
             "Hawthorne Lane", "Greg", "config.yaml", "2024"]
    assert kept(prompt, *terms) == terms


def test_a_name_capitalised_mid_sentence_survives_and_the_same_word_in_lower_case_does_not():
    assert kept("Please ask Rose about the Library again.", "Rose", "Library") == ["Rose", "Library"]
    assert kept("Please ask rose about the library again.", "rose", "library") == []
    assert kept("The notes were short.", "The", "notes", "short") == []      # capital by position only
    assert kept("Remind Maycock about it.", "Maycock") == ["Maycock"]        # not a word at all: a name


def test_a_quoted_phrase_of_plain_words_survives_only_when_the_prompt_quoted_it():
    assert kept('He said "the night crew" twice.', '"the night crew"', "the night crew") == ['"the night crew"']
    assert kept("He said the night crew twice.", '"the night crew"', "night crew") == []


def test_a_multi_word_term_stays_when_one_word_of_it_is_a_name_or_identifier():
    assert kept("Look at the Maycock survey.", "the Maycock survey") == ["the Maycock survey"]
    assert kept("Look at the survey list.", "survey list", "cron job", "list of users") == []


def test_numbers_ordinals_and_times_are_not_terms_but_codes_and_versions_are():
    assert kept("On the 3rd at 12pm, code 404 in 2024.", "3rd", "12pm", "7", "404", "2024") == ["404", "2024"]


def test_case_and_quote_duplicates_collapse():
    prompt = 'Ask Dan Maycock about "Dan Maycock" and v1.14.'
    assert kept(prompt, "Dan Maycock", "dan maycock", '"Dan Maycock"', "V1.14", "v1.14") == ["Dan Maycock", "V1.14"]


def test_the_counts_of_a_mixed_reply():
    prompt = SKILL_PROMPT + " Version v2.3 broke it."
    raw = reply(S("crew notes", ["Dan Maycock", "session", "v2.3"]),
                S("delivery rules", ["DELIVERY", "SILENT"]),
                {"query": "", "terms": ["cron job", "instructions"]},          # nothing left of it
                {"terms": ["Dan Maycock"]})                                    # terms only: the query is its terms
    parsed = parse_writer_reply(raw, TODAY, prompt)
    assert [(s.query, s.terms) for s in parsed.searches] == [
        ("crew notes", ("Dan Maycock", "v2.3")), ("delivery rules", ()), ("Dan Maycock", ("Dan Maycock",))]
    assert parsed.dropped_terms == ["session", "DELIVERY", "SILENT", "cron job", "instructions"]
    assert parsed.dropped_searches == 1 and parsed.why is None


def test_without_a_prompt_the_reply_is_parsed_unfiltered_as_before():
    parsed = parse_writer_reply(reply(S("q", ["session", "v1.14"])), TODAY, None)
    assert parsed.searches[0].terms == ("session", "v1.14") and parsed.dropped_terms == []


def test_keep_term_needs_no_search():
    assert keep_term("v1.14", "") and not keep_term("session", "") and not keep_term("  ", "")


# --------------------------------------------------------------------------
# the writer's prompt
# --------------------------------------------------------------------------

def test_the_writer_prompt_says_terms_are_only_names_and_identifiers_never_common_words():
    p = render_writer_prompt("what did Dan decide?", kind="chat", agent_last=None, max_searches=20, today=TODAY)
    terms_rule = next(line for line in p.splitlines() if line.startswith("- `terms`"))
    assert "ONLY" in terms_rule and "proper names" in terms_rule and "identifiers" in terms_rule
    assert "NEVER common words" in terms_rule
    for example in ("session", "report", "IMPORTANT", "cron", "instructions"):
        assert example in terms_rule                                         # the bad cases, named
    assert "VERBATIM" in terms_rule and "[]" in terms_rule


# --------------------------------------------------------------------------
# the entry point
# --------------------------------------------------------------------------

def test_a_skill_style_prompt_through_the_entry_point_carries_none_of_the_instruction_words(made):
    writer = FakeWriter([S("crew channel notes", INSTRUCTION_WORDS[:4] + ["Dan Maycock"]),
                         S("harbour notes path", ["/srv/atlas/harbour/notes", "instructions"]),
                         S("skill instructions", INSTRUCTION_WORDS[4:])])
    mem = made(llm=writer, rerank=FakeRerank())
    got = mem.expand_and_recall(SKILL_PROMPT, gate=False)
    assert got.error is None
    assert [s.terms for s in got.searches] == [("Dan Maycock", '"harbour-watch"'),    # the guard put the skill's name back
                                               ("/srv/atlas/harbour/notes",), ()]
    flat = terms_of(got)
    assert set(flat) == {"Dan Maycock", "/srv/atlas/harbour/notes", '"harbour-watch"'}
    assert len(flat) == 3                                                    # counts: three terms of the thirteen
    assert any("common words" in f and "session" in f for f in got.fallbacks)
    assert not any(w in t for t in flat for w in ("session", "cron", "SILENT", "DELIVERY", "IMPORTANT", "instruction"))


def test_a_search_with_neither_query_nor_terms_left_is_dropped_and_the_rest_run(made):
    writer = FakeWriter([{"query": "", "terms": ["session", "cron job"]}, S("pontoon survey", ["Dan Maycock"]),
                         {"terms": ["SILENT"]}])
    mem = made(llm=writer, rerank=FakeRerank())
    got = mem.expand_and_recall("Did Dan Maycock move the pontoon survey?", gate=False)
    assert [s.query for s in got.searches] == ["pontoon survey"] and [r.index for r in got.runs] == [1]
    assert "survey.md" in sources(got)
    assert any("2 searches" in f and "common words" in f for f in got.fallbacks)


def test_when_every_search_is_dropped_nothing_is_asked_no_fallback_on_the_prompt(made):
    rerank = FakeRerank()
    writer = FakeWriter([{"query": "", "terms": ["session"]}, {"terms": ["cron job", "SILENT"]}])
    mem = made(llm=writer, rerank=rerank)
    got = mem.expand_and_recall(SKILL_PROMPT, gate=False)
    assert got.block == "" and got.items == [] and got.searches == [] and got.runs == []
    assert got.skipped and "common words" in got.skipped_reason and got.error is None
    assert rerank.requests == []                                              # no Stage 1, no Stage 2
    assert not any("one search with the prompt itself" in f for f in got.fallbacks)


def test_a_writer_that_fails_on_a_normal_short_message_still_falls_back_to_one_search(made):
    mem = made(llm=FakeWriter(raw="sorry, no"), rerank=FakeRerank())
    msg = "what did we decide about the harbour lamps?"
    got = mem.expand_and_recall(msg, gate=False)
    assert got.searches == [SearchSpec(query=msg)] and not got.skipped and got.items
    assert got.fallbacks[0].startswith("writer:")


def test_the_fallback_search_carries_only_names_and_identifiers_of_the_prompt(made):
    mem = made(llm=FakeWriter(raw="sorry, no"), rerank=FakeRerank())
    got = mem.expand_and_recall(SKILL_PROMPT, gate=False)
    assert got.searches[0].terms == ('"harbour-watch"', "/srv/atlas/harbour/notes", "Dan Maycock")


def test_a_normal_message_keeps_its_names_and_loses_its_common_words(made):
    writer = FakeWriter([S("pontoon survey due date", ["Dan Maycock", "survey", "due", "May"]),
                         S("harbour lamp flicker", ["relay", "crew"])])
    mem = made(llm=writer, rerank=FakeRerank())
    got = mem.expand_and_recall("Did Dan Maycock say the survey is due in May, or are the lamps and "
                                "the relay the issue?", gate=False)
    assert [s.terms for s in got.searches] == [("Dan Maycock",), ("relay",)]
    assert "survey.md" in sources(got)
