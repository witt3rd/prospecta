"""Grounded, cited synthesis (opt-in recall_synth mode, migration 0011).
The LLM is a stub; evidence is checked in the prompt it receives."""
from __future__ import annotations

import json

import psycopg
import pytest

from prospecta._synth import NoteEvidence, extract_citations, pick_chunks
from prospecta.channels import DEFAULT_CHANNEL_CONFIG
from prospecta.memory import Memory
from prospecta.stages import LLMResult
from tests._stub_embedder import EMBED_DIM, stub_embed

LONG = [f"alpha part {i} cat sat on the mat number {i}" for i in range(5)]
NOTES = {"alpha.md": LONG, "bravo.md": ["bravo cat sat near the window"],
         "charlie.md": ["charlie cat sat under the table"],
         "delta.md": ["delta dog barked at the moon"]}


def vec(text):
    return "[" + ",".join(map(str, stub_embed([text])[0])) + "]"


class SynthLLM:
    def __init__(self, answer="Cats sat [alpha.md] and [bravo.md]."):
        self.answer, self.prompts = answer, []

    def __call__(self, messages, *, json_mode=False):
        text = messages[-1]["content"]
        if json_mode:
            return json.dumps({"queries": [{"text": "cat sat"}]})
        self.prompts.append(text)
        return LLMResult(self.answer, model="stub-sonnet", tokens_in=50, tokens_out=9,
                         cost_usd=0.001)


@pytest.fixture
def mem(fresh_db):
    llm = SynthLLM()
    m = Memory(database_url=fresh_db, bank_id="b", llm=llm, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    m.set_channel_config(DEFAULT_CHANNEL_CONFIG)
    for src, chunks in NOTES.items():
        m.retain("\n".join(chunks), source=src, index_text="\n".join(chunks))
    with psycopg.connect(fresh_db) as conn:
        for src, chunks in NOTES.items():
            for i, text in enumerate(chunks):
                conn.execute(
                    "INSERT INTO memory_items (bank_id, document_id, content, original_chunk,"
                    " embedding, kind, ordinal, char_start, char_end) "
                    "SELECT 'b', d.id, %s, %s, %s::vector, 'chunk', %s, 0, %s FROM documents d "
                    "WHERE d.source = %s", (text, text, vec(text), i, len(text), src))
        conn.commit()
    m.llm = llm
    yield m
    m.close()


def test_pick_chunks_window_and_fallbacks():
    rows = [(f"c{i}", i) for i in range(6)]
    assert pick_chunks(rows, "c3", None) == ["c2", "c3", "c4"]
    assert pick_chunks(rows, "c0", None) == ["c0", "c1", "c2"]
    assert pick_chunks(rows, "c5", None) == ["c3", "c4", "c5"]
    assert pick_chunks([], "only", None) == ["only"]
    assert pick_chunks(rows, "question text", None) == ["question text", "c0", "c1"]
    assert pick_chunks(rows, "x", 2, n=1) == ["c2"]


def test_extract_citations_known_unknown_and_dedupe():
    notes = [NoteEvidence("alpha.md", "d1", ["x"]), NoteEvidence("trip 2024.md", "d2", ["y"])]
    got = extract_citations("A [alpha.md]; B [Trip 2024] [alpha.md] [ghost.md] [1]", notes)
    assert [(c["note"], c["known"]) for c in got] == [
        ("alpha.md", True), ("trip 2024.md", True), ("ghost.md", False)]
    assert got[0]["document_id"] == "d1" and got[2]["document_id"] is None


def test_default_recall_synth_unchanged(mem):
    res = mem.recall_synth("cat sat")
    assert res.citations == [] and res.evidence == []
    assert "Retrieved chunks" in mem.llm.prompts[-1]


def test_grounded_top_notes_cited_and_stored(mem):
    res = mem.recall_synth("cat sat", grounded=True, evidence_top=4)
    prompt = mem.llm.prompts[-1]
    assert "not in memory" in prompt and "### [alpha.md]" in prompt
    assert len(res.evidence) == 4
    assert all(len(e["chunks"]) <= 3 for e in res.evidence)
    assert res.synthesis == "Cats sat [alpha.md] and [bravo.md]."
    assert [c["note"] for c in res.citations] == ["alpha.md", "bravo.md"]
    assert all(c["known"] for c in res.citations)
    with psycopg.connect(mem.database_url) as conn:
        syn, cit = conn.execute(
            "SELECT synthesis, citations FROM recall_events WHERE bank_id='b' "
            "ORDER BY id DESC LIMIT 1").fetchone()
    assert syn == res.synthesis
    assert [c["note"] for c in cit] == ["alpha.md", "bravo.md"]


def test_grounded_alpha_gets_neighbour_chunks_at_most_three(mem):
    res = mem.recall_synth("alpha part 2 cat", grounded=True, evidence_top=6)
    alpha = next(e for e in res.evidence if e["note"] == "alpha.md")
    assert len(alpha["chunks"]) == 3
    assert alpha["chunks"] == sorted(alpha["chunks"])  # document order


def test_grounded_scope_set_gives_whole_set(mem):
    res = mem.recall_synth("cat sat", grounded=True, evidence_top=1,
                           scope=["delta.md", "charlie.md"])
    assert sorted(e["note"] for e in res.evidence) == ["charlie.md", "delta.md"]
    assert "whole set of notes in scope" in mem.llm.prompts[-1]


def test_grounded_scope_too_large_falls_back_to_top(mem):
    res = mem.recall_synth("cat sat", grounded=True, evidence_top=2, scope=["nope.md"])
    assert len(res.evidence) == 2   # empty scope match: top notes
    assert "whole set of notes in scope" not in mem.llm.prompts[-1]


def test_grounded_not_in_memory_has_no_citations(mem):
    mem.llm.answer = "not in memory"
    res = mem.recall_synth("cat sat", grounded=True)
    assert res.synthesis == "not in memory" and res.citations == []
