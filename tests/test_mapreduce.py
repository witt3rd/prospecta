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
            groups: dict[str, list[int]] = {}
            for n, fact in re.findall(r"^(\d+)\. (.*) \[.*\]$", text, re.M):
                groups.setdefault(fact.casefold().strip(), []).append(int(n))
            items = [{"fact": k, "ids": v} for k, v in groups.items()]
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


# ---------------------------------------------------------------- regression: Greg's names

NICKS = ["Wizard of Oz", "Gregsy"]
PENS = ["Ann Archer", "Jay Fenwick", "Mara Quill"]
ALL5 = NICKS + PENS


class RealisticLLM:
    """Behaves like the real replies: the map step extracts names of any kind only
    when the prompt tells it to (otherwise only the word the question used);
    some replies hold the JSON twice around a prose line; the reduce step
    leaves singletons out of its groups."""

    def __init__(self):
        self.n = 0

    def __call__(self, messages, *, json_mode=False):
        text = messages[-1]["content"]
        self.n += 1
        if "## Extracted facts" in text:
            groups: dict[str, list[int]] = {}
            for n, fact in re.findall(r"^(\d+)\. (.*) \[.*\]$", text, re.M):
                groups.setdefault(fact.casefold().strip(), []).append(int(n))
            multi = [{"fact": k, "ids": v} for k, v in groups.items() if len(v) > 1]
            body = json.dumps({"items": multi})   # singletons are left out
        else:
            every = "extract EVERY name" in text
            facts = []
            for src, nt in re.findall(r"### \[(.*?)\]\n(.*?)(?=\n### \[|\Z)",
                                      text.split("## Notes")[1], re.S):
                for kind, name in re.findall(r"(nickname|pen name) (.+?)[.;]", nt):
                    if every or kind == "nickname":
                        facts.append({"fact": f"name {name}", "note": src})
            body = json.dumps({"facts": facts})
            if self.n % 2 == 0:
                body = f"{body}\nHere is the JSON again:\n{body}"
        return LLMResult(body, model="stub-sonnet", tokens_in=10, tokens_out=5, cost_usd=0.001)


def test_mapreduce_greg_all_five_names_among_200_notes(fresh_db):
    llm = RealisticLLM()
    m = Memory(database_url=fresh_db, bank_id="b", llm=llm, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    # 9 gold notes: 3 by documents.person, 2 by an entity row, 4 only by alias text
    gold = {
        "g1.md": ("Greg", "Greg, nickname Wizard of Oz."),
        "g2.md": ("Greg", "Greg's pen name Ann Archer; nickname Gregsy."),
        "g3.md": ("Greg", "Wrote as pen name Jay Fenwick."),
        "g4.md": ("", "Entity-linked note: nickname Wizard of Oz."),
        "g5.md": ("", "Entity-linked note: nickname Gregsy."),
        "g6.md": ("", "Ann Archer wrote this; pen name Mara Quill."),     # alias text only
        "g7.md": ("", "A cast list. Ann Archer appears; nickname Wizard of Oz."),
        "g8.md": ("", "Reading night: Mara Quill. pen name Jay Fenwick."),
        "g9.md": ("", "Another day with the Gregsy: nickname Gregsy."),
    }
    for src, (person, body) in gold.items():
        front = f"---\nperson: {person}\n---\n" if person else ""
        m.retain(front + body, source=src, index_text=body)
    for i in range(191):
        m.retain(f"Plain note {i} about nothing in particular.", source=f"n{i}.md",
                 index_text=f"plain {i}")
    with psycopg.connect(fresh_db) as conn:
        ids = {s_: i for i, s_ in conn.execute("SELECT id, source FROM documents WHERE bank_id='b'")}
        eid = conn.execute("INSERT INTO memory_entities (bank_id, name, norm, etype) "
                           "VALUES ('b', 'Greg', 'greg', 'person') RETURNING id").fetchone()[0]
        for src in ("g4.md", "g5.md"):
            item = conn.execute("SELECT id FROM memory_items WHERE document_id = %s LIMIT 1",
                                (ids[src],)).fetchone()[0]
            conn.execute("INSERT INTO memory_item_entities (item_id, entity_id) VALUES (%s, %s)",
                         (item, eid))
        for alias in ("Ann Archer", "Mara Quill", "Jay Fenwick", "Gregsy"):   # no row for Wizard of Oz
            conn.execute("INSERT INTO memory_entity_aliases (bank_id, norm, alias, entity_id) "
                         "VALUES ('b', %s, %s, %s)", (alias.lower(), alias, eid))
        conn.commit()
    assert len(ids) == 200
    res = m.recall_mapreduce("what are Greg's nicknames?", entity="Greg", batch_size=4)
    assert {e["fact"].casefold() for e in res.evidence} == {f"name {n}".casefold() for n in ALL5}
    assert {c["note"] for c in res.citations} == set(gold)   # all nine gold notes read and cited
    assert res.synth_call["notes_visited"] == 9
    for n in ALL5:
        assert n.casefold() in res.synthesis.casefold()
    m.close()


def test_mapreduce_depth_defaults_deep_and_is_recorded(fresh_db):
    m = _build(fresh_db, MapReduceLLM())
    m.recall_mapreduce("nicknames?", entity="Pat")
    m.recall_mapreduce("nicknames?", entity="Pat", depth="standard")
    with psycopg.connect(fresh_db) as conn:
        rows = conn.execute("SELECT depth FROM recall_events WHERE mode = 'mapreduce' "
                            "ORDER BY id").fetchall()
    assert rows == [("deep",), ("standard",)]
    with pytest.raises(ValueError):
        m.recall_mapreduce("nicknames?", entity="Pat", depth="shallow")


def test_mapreduce_candidate_set_is_precise_with_common_alias_word(fresh_db):
    """9 gold notes among 1,500; 1,300 merely mention the alias word 'Wizard' once."""
    m = Memory(database_url=fresh_db, bank_id="b", llm=RealisticLLM(), embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    gold = {
        "g1.md": ("Greg", "Greg, nickname Wizard of Oz."),
        "g2.md": ("Greg", "Greg's pen name Ann Archer; nickname Gregsy."),
        "g3.md": ("Greg", "Wrote as pen name Jay Fenwick."),
        "g4.md": ("Greg", "Nickname Wizard of Oz."),
        "g5.md": ("Greg", "Nickname Gregsy."),
        "g6.md": ("", "Wizard of Oz wrote this; Wizard of Oz again; pen name Mara Quill."),
        "g7.md": ("", "Wizard of Oz, Wizard of Oz: nickname Wizard of Oz."),
        "g8.md": ("", "Wizard of Oz. Wizard of Oz. pen name Jay Fenwick."),
        "g9.md": ("", "Wizard of Oz Wizard of Oz Wizard of Oz; nickname Gregsy."),
    }
    for src, (person, body) in gold.items():
        front = f"---\nperson: {person}\n---\n" if person else ""
        m.retain(front + body, source=src, index_text=body)
    for i in range(1300):
        body = (f"Note {i}: we watched the Wizard of Oz in passing. " + "Filler text. " * 20)
        m.retain(body, source=f"w{i}.md", index_text=f"w {i}")
    for i in range(191):
        m.retain(f"Plain note {i}.", source=f"n{i}.md", index_text=f"plain {i}")
    with psycopg.connect(fresh_db) as conn:
        eid = conn.execute("INSERT INTO memory_entities (bank_id, name, norm, etype) "
                           "VALUES ('b', 'Greg', 'greg', 'person') RETURNING id").fetchone()[0]
        conn.execute("INSERT INTO memory_entity_aliases (bank_id, norm, alias, entity_id) "
                     "VALUES ('b', 'wizard of oz', 'Wizard of Oz', %s)", (eid,))
        conn.commit()
        assert conn.execute("SELECT count(*) FROM documents").fetchone()[0] == 1500
        from prospecta._mapreduce import fetch_entity_notes
        before = fetch_entity_notes(conn, "b", ["Greg"], scan_names=["Wizard of Oz"],
                                    scan_relevance=0.0)
        after = fetch_entity_notes(conn, "b", ["Greg"], scan_names=["Wizard of Oz"])
    print(f"candidate set before={len(before)} after={len(after)}")
    assert len(before) > 1300
    srcs = {s for _, s, _ in after}
    assert set(gold) <= srcs and len(srcs - set(gold)) < 100
    res = m.recall_mapreduce("what are Greg's nicknames?", entity="Greg", batch_size=4)
    assert {e["fact"].casefold() for e in res.evidence} == {f"name {n}".casefold() for n in ALL5}
    assert res.synth_call["notes_visited"] == len(after)
    m.close()


def test_mapreduce_keeps_every_entity_linked_note_despite_weak_text_hits(fresh_db):
    """Notes tied to the person ONLY by entity link / alias row (no person
    front-matter, name absent or weakly present in text) are never cut by the
    scan relevance; only unlinked text-scan hits are subject to it."""
    m = Memory(database_url=fresh_db, bank_id="b", llm=MapReduceLLM(), embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    linked = []
    for i in range(30):   # linked only by entity: text never says the name (or says it once, buried)
        body = f"Quiet note {i}, also called Nick{i}. " + ("filler " * 200 if i % 2 else "")
        if i % 3 == 0:
            body += " Patricia once."
        src = f"linked{i}.md"
        m.retain(body, source=src, index_text=body)
        linked.append(src)
    for i in range(5):   # strong text-scan hits (raise the best scan score)
        body = f"strong {i} " + "Patricia " * 50
        m.retain(body, source=f"strong{i}.md", index_text=body)
    with psycopg.connect(fresh_db) as conn:
        eid = conn.execute("INSERT INTO memory_entities (bank_id, name, norm, etype) "
                           "VALUES ('b', 'Pat', 'pat', 'person') RETURNING id").fetchone()[0]
        conn.execute("INSERT INTO memory_entity_aliases (bank_id, entity_id, alias, norm) "
                     "VALUES ('b', %s, 'Patricia', 'patricia')", (eid,))
        for src in linked:
            item = conn.execute(
                "SELECT m.id FROM memory_items m JOIN documents d ON d.id = m.document_id "
                "WHERE d.source = %s LIMIT 1", (src,)).fetchone()[0]
            conn.execute("INSERT INTO memory_item_entities (item_id, entity_id) VALUES (%s, %s)",
                         (item, eid))
        for i, person in enumerate(["Pat  Smith", "Pat\tSmith", " pat smith "]):
            m.retain(f"Unrelated body {i}.", source=f"ws{i}.md", index_text=f"ws {i}")
            conn.execute("UPDATE documents SET person = %s WHERE source = %s", (person, f"ws{i}.md"))
        conn.commit()
        from prospecta import _mapreduce
        names, _, _ = _mapreduce.resolve_names(conn, "b", ["Pat", "Pat Smith"])
        got = {s for _, s, _ in _mapreduce.fetch_entity_notes(
            conn, "b", names, scan_names=["Patricia"], scan_relevance=0.25)}
        got_alias = {s for _, s, _ in _mapreduce.fetch_entity_notes(
            conn, "b", ["Patricia"], scan_names=["Patricia"], scan_relevance=0.99)}
    assert set(linked) <= got and set(linked) <= got_alias
    assert {"ws0.md", "ws1.md", "ws2.md"} <= got
    m.close()


def test_mapreduce_citations_resolve_through_shared_fold(fresh_db):
    class Sloppy(MapReduceLLM):
        def __call__(self, messages, *, json_mode=False):
            r = super().__call__(messages, json_mode=json_mode)
            if isinstance(r, LLMResult):
                r = LLMResult(r.text.replace("People/Mr. Nelson.md", "people\\\\mr.  NELSON.MD"),
                              model=r.model, tokens_in=r.tokens_in, tokens_out=r.tokens_out,
                              cost_usd=r.cost_usd)
            return r

    m = Memory(database_url=fresh_db, bank_id="b", llm=Sloppy(), embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    body = "Pat, also called Pip."
    m.retain("---\nperson: Pat\n---\n" + body, source="People/Mr. Nelson.md", index_text=body)
    res = m.recall_mapreduce("nicknames?", entity="Pat")
    cites = [c for e in res.evidence for c in e["citations"]]
    assert cites and all(c["known"] and c["note"] == "People/Mr. Nelson.md" for c in cites)
    m.close()
