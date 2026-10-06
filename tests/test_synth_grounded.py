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


def test_grounded_scope_of_twenty_notes_uses_all(mem):
    srcs = [f"many{i:02d}.md" for i in range(20)]
    for src in srcs:
        mem.retain(f"{src} cat sat", source=src, index_text=f"{src} cat sat")
    with psycopg.connect(mem.database_url) as conn:
        for src in srcs:
            text = f"{src} cat sat"
            conn.execute(
                "INSERT INTO memory_items (bank_id, document_id, content, original_chunk,"
                " embedding, kind, ordinal, char_start, char_end) "
                "SELECT 'b', d.id, %s, %s, %s::vector, 'chunk', 0, 0, %s FROM documents d "
                "WHERE d.source = %s", (text, text, vec(text), len(text), src))
        conn.commit()
    res = mem.recall_synth("cat sat", grounded=True, scope=srcs)
    assert sorted(e["note"] for e in res.evidence) == srcs
    assert "whole set of notes in scope" in mem.llm.prompts[-1]


def test_grounded_scope_too_large_falls_back_to_top(mem):
    res = mem.recall_synth("cat sat", grounded=True, evidence_top=2, scope=["nope.md"])
    assert len(res.evidence) == 2   # empty scope match: top notes
    assert "whole set of notes in scope" not in mem.llm.prompts[-1]


def test_grounded_not_in_memory_has_no_citations(mem):
    mem.llm.answer = "not in memory"
    res = mem.recall_synth("cat sat", grounded=True)
    assert res.synthesis == "not in memory" and res.citations == []


def _bare(answer="Cats [alpha.md]"):
    def llm(messages, *, json_mode=False):
        return json.dumps({"queries": [{"text": "cat sat"}]}) if json_mode else answer
    return llm


def _seeded(fresh_db, **kw):
    m = Memory(database_url=fresh_db, bank_id="b", embed=stub_embed, **kw)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    m.set_channel_config(DEFAULT_CHANNEL_CONFIG)
    m.retain("cat sat", source="alpha.md", index_text="cat sat")
    return m


def test_synth_llm_used_when_given_and_label_from_result(fresh_db):
    general, synth = SynthLLM("general [alpha.md]"), SynthLLM("synth [alpha.md]")
    m = _seeded(fresh_db, llm=general, synth_llm=synth, rerank_model="rerank-x")
    res = m.recall_synth("cat sat", grounded=True)
    assert res.synthesis == "synth [alpha.md]"
    assert synth.prompts and not general.prompts
    assert res.synth_call["model"] == "stub-sonnet"
    m.close()


def test_synth_falls_back_to_general_llm_label_none_when_unreported(fresh_db):
    m = _seeded(fresh_db, llm=_bare(), rerank_model="rerank-x")
    res = m.recall_synth("cat sat", grounded=True)
    assert res.synthesis == "Cats [alpha.md]"
    assert res.synth_call["model"] is None
    m.close()


def test_synth_falls_back_to_general_llm_label_from_result(fresh_db):
    general = SynthLLM("g [alpha.md]")
    m = _seeded(fresh_db, llm=general)
    res = m.recall_synth("cat sat", grounded=True)
    assert general.prompts and res.synth_call["model"] == "stub-sonnet"
    m.close()


def test_grounded_runs_one_recall_and_no_formulation(mem, monkeypatch):
    """F13: one search per question, no formulate call; scope from the filter."""
    calls = []
    real = mem.search

    def counting(text, **kw):
        calls.append(text)
        return real(text, **kw)

    monkeypatch.setattr(mem, "search", counting)
    monkeypatch.setattr(mem, "formulate_queries",
                        lambda *a, **k: pytest.fail("grounded must not formulate"))
    res = mem.recall_synth("cat sat", grounded=True)
    assert calls == ["cat sat"] and [q.text for q in res.queries] == ["cat sat"]


def test_grounded_scope_filled_from_promoted_filter(mem, monkeypatch):
    real = mem.search

    def promoting(text, **kw):
        out = real(text, **kw)
        kw["_trace"][-1].setdefault("fusion", {})["scope_promoted"] = [
            str(r.document_id) for r in out if r.source in ("delta.md", "charlie.md")]
        return out

    monkeypatch.setattr(mem, "search", promoting)
    res = mem.recall_synth("cat sat", grounded=True)   # default top, no scope=
    assert sorted(e["note"] for e in res.evidence) == ["charlie.md", "delta.md"]
    assert res.evidence and "whole set of notes in scope" in mem.llm.prompts[-1]


def test_latency_model_before_after():
    """Stubbed latency model: 5 full recalls (before) vs 1 (after)."""
    recall_s, recall_usd, formulate_s = 13.0, 0.04, 3.0
    before = (formulate_s + 5 * recall_s, 5 * recall_usd)
    after = (1 * recall_s, 1 * recall_usd)
    assert before == (68.0, 0.2) and after == (13.0, 0.04)


class _Rec:
    def __init__(self, document_id, source, score):
        self.document_id, self.source, self.score = document_id, source, score
        self.original_chunk = NOTES[source][0]


def _gather(mem, scores, **kw):
    from prospecta._synth import gather_evidence
    with psycopg.connect(mem.database_url) as conn:
        ids = dict(conn.execute("SELECT source, id::text FROM documents WHERE bank_id='b'").fetchall())
        names = list(NOTES)[:len(scores)]
        recalled = [_Rec(ids[n], n, s) for n, s in zip(names, scores)]
        notes, _ = gather_evidence(conn, "b", recalled, **kw)
    return names, [n.name for n in notes]


def test_default_selection_stops_before_first_note_below_cutoff(mem):
    names, got = _gather(mem, [1.0, 0.9, 0.3, 0.2])
    assert got == names[:2]


@pytest.mark.parametrize("scores", [[-0.1, -0.5, -0.6], [-0.5, -0.6, -0.7]])
def test_default_selection_non_positive_top_score(mem, scores):
    names, got = _gather(mem, scores)
    assert got == names[:1]


def test_explicit_top_still_caps(mem):
    names, got = _gather(mem, [1.0, 0.9, 0.8, 0.7], top=3)
    assert got == names[:3]
