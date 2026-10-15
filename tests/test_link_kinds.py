"""Per-bank link kinds. Semantic, temporal and entity neighbours are answered by
search (vector, date range, term), so by default only the relations a search
cannot answer are stored: CAUSAL (model-judged) and NEXT (chunk order).
`link_kinds` picks the kinds; `extract_entities` is a separate switch."""
from __future__ import annotations

import hashlib
import json
import random
import re

import psycopg
import pytest

from prospecta._linker import Linker, RelationHit
from prospecta._test_helpers import no_cut
from prospecta.channels import DEFAULT_CHANNEL_CONFIG, Candidate, GraphExpand, QueryPlan, RecallState
from prospecta.memory import Memory
from prospecta.stages import LLMResult
from tests._stub_embedder import EMBED_DIM, stub_embed

ALL_KINDS = ["next", "causal", "semantic", "temporal", "entity", "related_to"]


class EntityLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, messages, *, json_mode=False):
        self.calls += 1
        text = messages[-1]["content"].split("## Note")[-1]
        ents = [{"name": k, "type": "person"} for k in ("Kelly", "Ann") if k in text]
        return LLMResult(json.dumps({"entities": ents}), model="stub", tokens_in=1,
                         tokens_out=1, cost_usd=0.0)


class Judge:
    """Says every candidate is both semantically related and a cause."""

    def __init__(self):
        self.pairs = 0

    def judge(self, source_text, candidates, calls):
        self.pairs += len(candidates)
        out = []
        for i in range(len(candidates)):
            out += [RelationHit(i, "SEMANTIC", "RELATED_TO", 0.9),
                    RelationHit(i, "CAUSAL", "LEADS_TO", 0.9)]
        return out


@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=None, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    yield m
    m.close()


def q(m, sql):
    with psycopg.connect(m.database_url) as c:
        return c.execute(sql).fetchone()[0]


def kinds_written(m):
    with psycopg.connect(m.database_url) as c:
        return {(a, b) for a, b in c.execute("SELECT link_type, subtype FROM memory_links")}


def link_all(m, linker):
    m._linker = linker
    with psycopg.connect(m.database_url) as c:
        ids = [str(r[0]) for r in c.execute("SELECT id FROM documents ORDER BY created_at, id")]
    for d in ids:
        m.link_document(d)


LONG = ("First paragraph here. " * 40 + "\n\n" + "Second paragraph here. " * 40 + "\n\n"
        + "Third paragraph here. " * 40)


def fill(m):
    m.retain(LONG, source="long", index_text="long q", child_chunks=True,
             metadata={"created": "2024-01-10", "person": "p"})
    m.retain("Kelly met Ann on monday", source="a", index_text="kelly ann monday",
             metadata={"created": "2024-01-10", "person": "p"})
    m.retain("Kelly wrote again", source="b", index_text="kelly ann wrote",
             metadata={"created": "2024-01-11", "person": "p"})


# (1) link_kinds: setting, default, filtering

def test_default_link_kinds_are_next_and_causal(mem):
    assert mem.get_link_config()["link_kinds"] == ["next", "causal"]


def test_default_builds_only_next_and_causal(mem):
    fill(mem)
    link_all(mem, Linker(judge=Judge(), asynchronous=False))
    assert kinds_written(mem) == {("TEMPORAL", "NEXT"), ("CAUSAL", "LEADS_TO")}


def test_each_kind_is_built_only_when_listed(mem):
    fill(mem)
    mem.set_link_config({"link_kinds": ["next"]})
    link_all(mem, Linker(judge=Judge(), asynchronous=False))
    assert kinds_written(mem) == {("TEMPORAL", "NEXT")}
    mem.set_link_config({"link_kinds": []})
    mem.drop_links(yes=True)
    with psycopg.connect(mem.database_url) as c:
        c.execute("DELETE FROM memory_link_state")
    link_all(mem, Linker(judge=Judge(), asynchronous=False))
    assert q(mem, "SELECT count(*) FROM memory_links") == 0


def test_all_kinds_still_available_on_request(mem):
    fill(mem)
    mem.set_link_config({"link_kinds": ALL_KINDS, "extract_entities": True})
    link_all(mem, Linker(llm=EntityLLM(), judge=Judge(), asynchronous=False))
    got = {st for _, st in kinds_written(mem)}
    assert {"NEXT", "LEADS_TO", "RELATED_TO", "PRECEDES", "SHARED_ENTITY"} <= got


def test_related_to_fallback_only_when_listed(mem):
    fill(mem)
    link_all(mem, Linker(asynchronous=False, vector_floor=-1.0))   # no judge: pgvector fallback
    assert ("SEMANTIC", "RELATED_TO") not in kinds_written(mem)
    mem.set_link_config({"link_kinds": ["related_to"]})
    with psycopg.connect(mem.database_url) as c:
        c.execute("DELETE FROM memory_link_state")
    link_all(mem, Linker(asynchronous=False, vector_floor=-1.0))
    assert kinds_written(mem) == {("SEMANTIC", "RELATED_TO")}


def test_link_kinds_validates(mem):
    for bad in ({"link_kinds": ["bogus"]}, {"link_kinds": "next"}, {"link_kinds": [1]},
                {"extract_entities": "yes"}, {"links": True}):
        with pytest.raises(ValueError):
            mem.set_link_config(bad)


# (2) extract_entities: its own switch, default off

def test_entities_off_by_default_and_tables_untouched(mem):
    fill(mem)
    llm = EntityLLM()
    link_all(mem, Linker(llm=llm, judge=Judge(), asynchronous=False))
    assert llm.calls == 0
    for t in ("memory_entities", "memory_item_entities", "memory_entity_aliases",
              "memory_alias_state"):
        assert q(mem, f"SELECT count(*) FROM {t}") == 0, t
    assert mem.get_link_config()["extract_entities"] is False


def test_entities_on_independent_of_link_kinds(mem):
    fill(mem)
    mem.set_link_config({"extract_entities": True, "link_kinds": []})
    link_all(mem, Linker(llm=EntityLLM(), asynchronous=False))
    assert q(mem, "SELECT count(*) FROM memory_entities") >= 2
    assert q(mem, "SELECT count(*) FROM memory_item_entities") >= 2
    assert q(mem, "SELECT count(*) FROM memory_links") == 0


def test_entity_kind_without_extraction_writes_nothing(mem):
    fill(mem)
    mem.set_link_config({"link_kinds": ["entity"], "extract_entities": False})
    link_all(mem, Linker(llm=EntityLLM(), asynchronous=False))
    assert q(mem, "SELECT count(*) FROM memory_links") == 0


# (3) graph defaults

def test_graph_defaults_causal_and_next_only_and_enabled():
    g = {c["name"]: c for c in DEFAULT_CHANNEL_CONFIG}["graph"]
    assert g["enabled"] and g["weight"] > 0
    tw = g["params"]["type_weights"]
    assert tw["CAUSAL"] == 0.8 and tw["NEXT"] == 0.5
    assert all(v == 0 for k, v in tw.items() if k not in ("CAUSAL", "NEXT"))
    from prospecta.channels.graph import DEFAULT_TYPE_WEIGHTS
    assert DEFAULT_TYPE_WEIGHTS["CAUSAL"] == 0.8 and DEFAULT_TYPE_WEIGHTS["NEXT"] == 0.5
    assert all(v == 0 for k, v in DEFAULT_TYPE_WEIGHTS.items() if k not in ("CAUSAL", "NEXT"))


def _items(m, *sources):
    with psycopg.connect(m.database_url) as c:
        return [c.execute("SELECT i.id FROM memory_items i JOIN documents d ON d.id=i.document_id "
                          "WHERE d.source=%s ORDER BY i.id", (s,)).fetchone()[0] for s in sources]


def _expand(m, seed, **params):
    with psycopg.connect(m.database_url) as c:
        with c.cursor() as cur:
            cur.execute("SELECT id FROM documents WHERE source=%s", (seed,))
            doc = str(cur.fetchone()[0])
        st = RecallState(conn=c, bank_id="b", pool=[Candidate(
            document_id=doc, item_id=None, source=seed, channel="dense_chunk",
            rank=1, score=1.0, evidence=seed)])
        return {x.source: x for x in GraphExpand(**params).retrieve(QueryPlan(text="q"), st)}


def _insert(m, rows):
    with psycopg.connect(m.database_url) as c:
        for a, b, lt, st in rows:
            c.execute("INSERT INTO memory_links (bank_id, src, dst, link_type, subtype) "
                      "VALUES ('b', %s, %s, %s, %s)", (a, b, lt, st))


def test_graph_expands_causal_and_next_with_zero_semantic_rows(mem):
    for s in "ABC":
        mem.retain(f"note {s}", source=s, index_text=f"q {s}")
    a, b, c = (x[0] for x in _items(mem, "A", "B", "C"))
    _insert(mem, [(a, b, "CAUSAL", "LEADS_TO"), (b, c, "TEMPORAL", "NEXT")])
    assert q(mem, "SELECT count(*) FROM memory_links WHERE link_type='SEMANTIC'") == 0
    from prospecta.channels.graph import DEFAULT_TYPE_WEIGHTS
    out = _expand(mem, "A", type_weights=DEFAULT_TYPE_WEIGHTS, node_min_rel=0)
    assert out["B"].score == pytest.approx(0.5 * 0.8)
    assert out["C"].score == pytest.approx(0.5 * 0.8 * 0.5 * 0.5)   # a NEXT hop weighs 0.5


# (4) existing rows: kept, weight 0 unless the bank says otherwise

def test_old_semantic_temporal_rows_weigh_zero_unless_bank_config_says(mem):
    for s in "ABCD":
        mem.retain(f"note {s}", source=s, index_text=f"q {s}")
    a, b, c, d = (x[0] for x in _items(mem, "A", "B", "C", "D"))
    _insert(mem, [(a, b, "SEMANTIC", "RELATED_TO"), (a, c, "TEMPORAL", "PRECEDES"),
                  (a, d, "ENTITY", "SHARED_ENTITY")])
    n = q(mem, "SELECT count(*) FROM memory_links")
    from prospecta.channels.graph import DEFAULT_TYPE_WEIGHTS
    assert _expand(mem, "A", type_weights=DEFAULT_TYPE_WEIGHTS) == {}
    on = _expand(mem, "A", type_weights={**DEFAULT_TYPE_WEIGHTS, "SEMANTIC": 1.0})
    assert set(on) == {"B"}
    assert q(mem, "SELECT count(*) FROM memory_links") == n   # nothing was touched


def test_bank_channel_config_can_raise_a_kind_weight(mem):
    for s in "AB":
        mem.retain(f"note {s}", source=s, index_text=f"q {s}")
    a, b = (x[0] for x in _items(mem, "A", "B"))
    _insert(mem, [(a, b, "SEMANTIC", "RELATED_TO")])
    cfg = no_cut(DEFAULT_CHANNEL_CONFIG)
    mem.set_channel_config(cfg)
    base = {r.source: r.scores.get("graph", 0) for r in mem.search("note A", limit=10)}
    assert not base.get("B")
    cfg = [{**c, "params": {**c["params"], "type_weights": {"SEMANTIC": 1.0}}}
           if c["name"] == "graph" else c for c in cfg]
    mem.set_channel_config(cfg)
    assert {r.source: r.scores.get("graph", 0) for r in mem.search("note A", limit=10)}["B"] > 0


# (5) growth: a 2,000-note bank, only the listed kinds, rows linear in notes

CLUSTER = 10


def _vec(text):
    m = re.search(r"\bc(\d+)\b", text)
    if not m:
        return stub_embed([text])[0]
    base = random.Random(int(m.group(1))).gauss
    v = [base(0, 1) for _ in range(EMBED_DIM)]
    noise = random.Random(hashlib.sha256(text.encode()).digest())
    v = [x + 0.05 * noise.gauss(0, 1) for x in v]
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def test_2000_note_bank_writes_only_listed_kinds_and_rows_grow_linearly(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=None,
               embed=lambda ts: [_vec(t) for t in ts])
    m.create_bank("b", embedding_dim=EMBED_DIM)
    m.set_link_config({"link_kinds": ["next", "causal"]})
    N = 2000
    for i in range(N):
        m.retain(f"note {i} on topic c{i // CLUSTER}", source=f"n{i}",
                 index_text=f"topic c{i // CLUSTER} note {i}")
    with psycopg.connect(fresh_db) as c:
        ids = [str(r[0]) for r in c.execute(
            "SELECT d.id FROM documents d ORDER BY (regexp_replace(d.source,'n','')::int)")]
    m._linker = Linker(judge=(judge := Judge()), asynchronous=False)
    per_note = {}
    for i, d in enumerate(ids, 1):
        m.link_document(d)
        if i in (500, 1000, 2000):
            per_note[i] = q(m, "SELECT count(*) FROM memory_links") / i
    assert kinds_written(m) <= {("TEMPORAL", "NEXT"), ("CAUSAL", "LEADS_TO")}
    assert q(m, "SELECT count(*) FROM memory_links WHERE link_type='CAUSAL'") > 0
    assert per_note[500] == per_note[1000] == per_note[2000] > 0
    # the candidates are the vector neighbours of a note, never the whole bank
    assert judge.pairs / N <= CLUSTER
    m.close()
