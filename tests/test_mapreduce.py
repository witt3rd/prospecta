"""Map-reduce recall for set questions. The LLM is a stub that reads the
notes in each prompt; a synthetic person has 9 gold notes among 60."""
from __future__ import annotations

import json
import re

import psycopg
import pytest

from prospecta.memory import Memory
from prospecta.stages import LLMResult
from tests._stub_embedder import EMBED_DIM, stub_embed

NAMES = ["Pip", "Skipper", "Quill", "The Captain", "Inky"]
# 9 gold notes: name -> notes that state it (Pip and Quill appear in several)
GOLD = {"g1": "Pip", "g2": "Pip", "g3": "Skipper", "g4": "Quill", "g5": "Quill",
        "g6": "The Captain", "g7": "Inky", "g8": "Pip", "g9": "Skipper"}


class MapReduceLLM:
    def __init__(self, bad_reduce=False, fail_batch=None, fail_times=99):
        self.map_prompts, self.reduce_prompts = [], []
        self.bad_reduce, self.fail_batch, self.fail_times = bad_reduce, fail_batch, fail_times

    def __call__(self, messages, *, json_mode=False):
        text = messages[-1]["content"]
        if "## Extracted facts" in text:
            self.reduce_prompts.append(text)
            if self.bad_reduce:
                return "not json at all"
            facts: dict[str, list[str]] = {}
            for fact, note in re.findall(r"^- (.*) \[(.*)\]$", text, re.M):
                facts.setdefault(fact.casefold().strip(), []).append(note)
            items = [{"fact": k, "notes": v} for k, v in facts.items()]
            return LLMResult(json.dumps({"items": items}), model="stub-sonnet",
                             tokens_in=10, tokens_out=5, cost_usd=0.002)
        self.map_prompts.append(text)
        if self.fail_batch and len(self.map_prompts) <= self.fail_times * self.fail_batch \
                and len(self.map_prompts) > (self.fail_batch - 1) * self.fail_times:
            raise RuntimeError("boom")
        facts = []
        for src, body in re.findall(r"### \[(.*?)\]\n(.*?)(?=\n### \[|\Z)", text.split("## Notes")[1], re.S):
            m = re.search(r"also called (.+?)\.", body)
            if m:
                facts.append({"fact": f"nickname {m.group(1)}", "note": src})
        return LLMResult(json.dumps({"facts": facts}), model="stub-sonnet",
                         tokens_in=100, tokens_out=20, cost_usd=0.01)


def _build(fresh_db, llm, *, via="person"):
    m = Memory(database_url=fresh_db, bank_id="b", llm=llm, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    for i in range(60):
        gold = GOLD.get(f"g{i + 1}") if i < 9 else None
        body = f"Pat did ordinary thing number {i}."
        if gold:
            body = f"Pat, also called {gold}. {body}"
        name = f"g{i + 1}.md" if gold else f"n{i}.md"
        front = f"---\nperson: Pat\n---\n" if via == "person" else ""
        m.retain(front + body, source=name, index_text=body)
    for i in range(20):
        m.retain(f"---\nperson: Other\n---\nOther, also called Decoy{i}.", source=f"o{i}.md",
                 index_text=f"decoy {i}")
    return m


def test_mapreduce_finds_all_five_names_with_citations(fresh_db):
    llm = MapReduceLLM()
    m = _build(fresh_db, llm)
    res = m.recall_mapreduce("what are Pat's nicknames?", entity="Pat", batch_size=10)
    for n in NAMES:
        assert n.casefold() in res.synthesis.casefold()
    got = {e["fact"].replace("nickname ", ""): {c["note"] for c in e["citations"]}
           for e in res.evidence}
    assert len(got) == 5 and all(c["known"] for e in res.evidence for c in e["citations"])
    assert got["pip"] == {"g1.md", "g2.md", "g8.md"}
    assert {c["note"] for c in res.citations} == {f"g{i}.md" for i in range(1, 10)}
    assert len(llm.map_prompts) == 6 and not any("Decoy" in p for p in llm.map_prompts)
    m.close()


def test_mapreduce_trace_in_recall_events(fresh_db):
    m = _build(fresh_db, MapReduceLLM())
    res = m.recall_mapreduce("what are Pat's nicknames?", entity="Pat", batch_size=25)
    assert res.synth_call["notes_visited"] == 60 and res.synth_call["facts_found"] == 5
    with psycopg.connect(fresh_db) as conn:
        mode, plan, cost, n_calls, cites = conn.execute(
            "SELECT mode, plan, cost_usd, n_llm_calls, citations FROM recall_events "
            "WHERE mode = 'mapreduce'").fetchone()
    assert plan["n_batches"] == 3 and [p["notes_visited"] for p in plan["progress"]] == [25, 50, 60]
    assert plan["progress"][-1]["facts_total"] == 9
    assert n_calls == 4 and float(cost) == pytest.approx(0.032)
    assert len(cites) == 9
    m.close()


def test_mapreduce_by_entity_table_and_aliases(fresh_db):
    m = _build(fresh_db, MapReduceLLM(), via="none")   # no documents.person for Pat
    with psycopg.connect(fresh_db) as conn:
        ids = {s: i for i, s in conn.execute("SELECT id, source FROM documents WHERE bank_id='b'")}
        assert not conn.execute("SELECT 1 FROM documents WHERE person = 'Pat'").fetchone()
        gold_item = conn.execute("SELECT id FROM memory_items WHERE document_id = %s LIMIT 1",
                                 (ids["g1.md"],)).fetchone()[0]
        eid = conn.execute("INSERT INTO memory_entities (bank_id, name, norm, etype) "
                           "VALUES ('b', 'Patricia', 'patricia', 'person') RETURNING id").fetchone()[0]
        conn.execute("INSERT INTO memory_item_entities (item_id, entity_id) VALUES (%s, %s)",
                     (gold_item, eid))
        conn.commit()
    res = m.recall_mapreduce("nicknames?", entity="Nobody", aliases=["Patricia"])
    assert res.synth_call["notes_visited"] == 1 and "pip" in res.synthesis.casefold()
    m.close()


def test_mapreduce_reduce_failure_falls_back_to_deterministic_merge(fresh_db):
    m = _build(fresh_db, MapReduceLLM(bad_reduce=True))
    res = m.recall_mapreduce("nicknames?", entity="pat")
    assert len(res.evidence) == 5
    m.close()


def test_mapreduce_failed_batch_is_surfaced_not_silent(fresh_db):
    llm = MapReduceLLM(fail_batch=1, fail_times=2)   # batch 1 fails both attempts
    m = _build(fresh_db, llm)
    seen = []
    res = m.recall_mapreduce("nicknames?", entity="Pat", batch_size=30, on_progress=seen.append)
    assert seen[0]["error"].startswith("RuntimeError") and seen[1]["error"] is None
    assert "WARNING: incomplete" in res.synthesis and "g1.md" in res.synthesis
    assert len(llm.map_prompts) == 3
    m.close()


def test_mapreduce_transient_batch_failure_is_retried(fresh_db):
    llm = MapReduceLLM(fail_batch=1, fail_times=1)   # first attempt fails, retry succeeds
    m = _build(fresh_db, llm)
    res = m.recall_mapreduce("nicknames?", entity="Pat", batch_size=30)
    assert "WARNING" not in res.synthesis and len(res.evidence) == 5
    m.close()


def test_mapreduce_empty_entity_set_and_bad_batch(fresh_db):
    m = _build(fresh_db, MapReduceLLM())
    res = m.recall_mapreduce("nicknames?", entity="Nobody")
    assert res.synthesis == "not in memory" and res.citations == []
    with pytest.raises(ValueError):
        m.recall_mapreduce("x", entity="Pat", batch_size=0)
    m.close()


def test_excerpt_long_note_keeps_entity_paragraphs():
    from prospecta._mapreduce import excerpt
    text = "intro\n\n" + "filler\n\n" * 50 + "Pat is also called Pip.\n\n" + "tail\n\n" * 50
    out = excerpt(text, ["Pat"], max_chars=100)
    assert "Pip" in out and "filler" not in out and out.startswith("intro")
