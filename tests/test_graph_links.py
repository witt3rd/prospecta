"""Jev-Mem techniques as Postgres-native channels (design 8.7): migration 0006,
the Linker (typed links, temporal links in SQL, entities), GraphExpand and the
JevReader. LLM and Jev are stubs; nothing leaves the process."""
from __future__ import annotations

import json
import re

import psycopg
import pytest

from prospecta._linker import JevRelationJudge, Linker, parse_entities
from prospecta.channels import (
    DEFAULT_CHANNEL_CONFIG, Candidate, GraphExpand, QueryPlan, RecallState, build_channels,
    run_channels,
)
from prospecta.memory import Memory
from prospecta.stages import (
    JevReader, JevScore, Item, LLMResult, StageDeps, run_stages, validate_recall_config,
)
from tests._stub_embedder import EMBED_DIM, stub_embed


class EntityLLM:
    """Stub Sonnet: names every Capitalised word that is in `known`."""

    def __init__(self, known=("Kelly", "Forge", "Acme")):
        self.known, self.prompts = known, []

    def __call__(self, messages, *, json_mode=False):
        assert json_mode
        text = messages[-1]["content"].split("## Note")[-1]
        self.prompts.append(text)
        ents = [{"name": k, "type": "org" if k == "Acme" else "person"}
                for k in self.known if k in text]
        return LLMResult(json.dumps({"entities": ents}), model="stub-sonnet",
                         tokens_in=50, tokens_out=5, cost_usd=0.001)


@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=None, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    yield m
    m.close()


def conn_of(m):
    return psycopg.connect(m.database_url)


def docs(m):
    with conn_of(m) as c:
        return {s: str(i) for i, s in c.execute("SELECT id, source FROM documents")}


def links(m, **where):
    q = ("SELECT s.source, d.source, l.link_type, l.subtype, l.origin, l.confidence "
         "FROM memory_links l JOIN memory_items a ON a.id=l.src JOIN documents s ON s.id=a.document_id "
         "JOIN memory_items b ON b.id=l.dst JOIN documents d ON d.id=b.document_id")
    with conn_of(m) as c:
        return [tuple(r) for r in c.execute(q + " ORDER BY 1,2,4")]


def link_all(m, linker):
    m._linker = linker
    for d in docs(m).values():
        m.link_document(d)


# ------------------------------------------------------------------ migration

def test_migration_0006_is_additive_and_cascades(mem):
    mem.retain("alpha", source="a", index_text="alpha")
    mem.retain("beta", source="b", index_text="beta")
    with conn_of(mem) as c:
        for t in ("memory_entities", "memory_item_entities", "memory_links", "memory_link_state"):
            assert c.execute(f"SELECT count(*) FROM {t}").fetchone()[0] == 0
        assert c.execute("SELECT max(version) >= 6 FROM prospecta_schema_version").fetchone()[0]
        a, b = [r[0] for r in c.execute("SELECT id FROM memory_items ORDER BY id")]
        c.execute("INSERT INTO memory_links (bank_id, src, dst, link_type, subtype) "
                  "VALUES ('b', %s, %s, 'SEMANTIC', 'RELATED_TO')", (a, b))
        c.commit()
        c.execute("DELETE FROM memory_items WHERE id = %s", (a,))
        c.commit()
        assert c.execute("SELECT count(*) FROM memory_links").fetchone()[0] == 0


def test_doc_date_function(mem):
    with conn_of(mem) as c:
        f = lambda meta: c.execute(
            "SELECT prospecta_doc_date(%s::jsonb, '2030-01-02T12:00:00Z')", (json.dumps(meta),)
        ).fetchone()[0].isoformat()
        assert f({"created": "2024-03-04"}) == "2024-03-04"
        assert f({"created": "2024-03-04T10:00:00"}) == "2024-03-04"
        assert f({"created": "2024-02-31"}) == "2030-01-02"   # invalid date falls back
        assert f({"created": "yesterday"}) == "2030-01-02"
        assert f({}) == "2030-01-02"


# ---------------------------------------------------------------- GraphExpand

def seed_chain(mem):
    """A -SEMANTIC-> B <-CAUSAL- C (stored C->B), B -ENTITY- D, E isolated."""
    for s in "ABCDE":
        mem.retain(f"note {s} text", source=s, index_text=f"q {s}")
    ids = docs(mem)
    with conn_of(mem) as c:
        item = {s: c.execute("SELECT id FROM memory_items WHERE document_id=%s",
                             (ids[s],)).fetchone()[0] for s in ids}
        for a, b, lt, st in (("A", "B", "SEMANTIC", "RELATED_TO"),
                             ("C", "B", "CAUSAL", "LEADS_TO"),
                             ("B", "D", "ENTITY", "SHARED_ENTITY")):
            c.execute("INSERT INTO memory_links (bank_id, src, dst, link_type, subtype) "
                      "VALUES ('b', %s, %s, %s, %s)", (item[a], item[b], lt, st))
        c.commit()
    return ids


def pool_of(ids, *names):
    return [Candidate(document_id=ids[n], item_id=None, source=n, channel="dense_chunk",
                      rank=i + 1, score=1.0, evidence=n) for i, n in enumerate(names)]


def expand(mem, ids, seeds, limit=50, **params):
    with conn_of(mem) as c:
        st = RecallState(conn=c, bank_id="b", pool=pool_of(ids, *seeds))
        return GraphExpand(**params).retrieve(QueryPlan(text="q"), st, limit)


def test_graph_one_hop_by_join_both_directions(mem):
    ids = seed_chain(mem)
    out = expand(mem, ids, ["A"], max_hops=1)
    assert [c.source for c in out] == ["B"]
    assert out[0].detail["hops"] == 1 and out[0].detail["link_types"] == ["SEMANTIC"]
    # B seeds: A (src side), C (dst side: the link is stored C->B), D
    assert {c.source for c in expand(mem, ids, ["B"], max_hops=1)} == {"A", "C", "D"}


def test_graph_two_hops_decay_and_type_weights(mem):
    ids = seed_chain(mem)
    out = {c.source: c for c in expand(mem, ids, ["A"])}
    assert set(out) == {"B", "C", "D"} and out["C"].detail["hops"] == 2
    # A->B semantic (1.0) then B->C causal (0.8), D entity (0.5), decay 0.5 per hop
    assert out["B"].score == pytest.approx(0.5)
    assert out["C"].score == pytest.approx(0.5 * 0.5 * 0.8)
    assert out["D"].score == pytest.approx(0.5 * 0.5 * 0.5)
    assert [c.source for c in expand(mem, ids, ["A"])] == ["B", "C", "D"]
    assert [c.rank for c in expand(mem, ids, ["A"])] == [1, 2, 3]
    # configurable: weights and decay
    tw = {c.source: c.score for c in expand(
        mem, ids, ["A"], type_weights={"CAUSAL": 0.0}, decay=1.0)}
    assert "C" not in tw and tw["B"] == pytest.approx(1.0)


def test_graph_node_cap_and_seed_exclusion(mem):
    ids = seed_chain(mem)
    assert [c.source for c in expand(mem, ids, ["A"], node_cap=1)] == ["B"]
    assert expand(mem, ids, ["E"]) == []                   # isolated seed
    assert expand(mem, ids, []) == []                      # empty pool
    # a seed's own items are never its expansion
    assert "A" not in {c.source for c in expand(mem, ids, ["A"])}


def test_graph_default_config_enabled_and_configurable():
    names = {c["name"]: c for c in DEFAULT_CHANNEL_CONFIG}
    assert names["graph"]["enabled"] and names["graph"]["weight"] > 0
    built = {c.name: c for c in build_channels(DEFAULT_CHANNEL_CONFIG)}
    assert built["graph"].channel.kind == "expand"
    assert list(built) == ["dense_chunk", "question", "meta", "graph"]   # filter then expand last
    cfg = [{"name": "graph", "weight": 2.5, "params": {"node_cap": 7, "decay": 0.3}}]
    g = build_channels(cfg)[0]
    assert g.weight == 2.5 and g.channel.params["node_cap"] == 7


def test_graph_channel_joins_the_default_blend(mem):
    mem.set_channel_config(DEFAULT_CHANNEL_CONFIG)
    mem.retain("cats purr softly", source="seed", index_text="cats purr softly")
    mem.retain("quantum widgets fail", source="far", index_text="quantum widgets fail")
    ids = docs(mem)
    with conn_of(mem) as c:
        a, b = [c.execute("SELECT id FROM memory_items WHERE document_id=%s", (ids[s],))
                .fetchone()[0] for s in ("seed", "far")]
        c.execute("INSERT INTO memory_links (bank_id, src, dst, link_type, subtype) "
                  "VALUES ('b', %s, %s, 'CAUSAL', 'LEADS_TO')", (a, b))
        c.commit()
    traces: list = []
    res = mem.search("cats purr softly", limit=10, _trace=traces)
    by_src = {r.source: r for r in res}
    # the stub embedder returns every item, so both are in the pool; each is the
    # other's graph neighbour and gets a graph score
    assert by_src["far"].scores["graph"] > 0 and by_src["seed"].scores["graph"] > 0
    ch = {c["name"]: c for c in traces[0]["channels"]}
    assert ch["graph"]["kind"] == "expand" and ch["graph"]["n"] >= 1 and not ch["graph"]["error"]
    # weight 0: the channel still runs but adds nothing to the fused score
    mem.set_channel_config([c if c["name"] != "graph" else {**c, "weight": 0}
                            for c in DEFAULT_CHANNEL_CONFIG])
    off = {r.source: r.score for r in mem.search("cats purr softly", limit=10)}
    assert off["far"] < by_src["far"].score


# --------------------------------------------------------------------- Linker

def test_linker_temporal_next_and_person(mem):
    mem.retain("jan note", source="jan", index_text="x1", metadata={"created": "2024-01-10", "person": "ann"})
    mem.retain("jan two", source="jan2", index_text="x2", metadata={"created": "2024-01-11", "person": "ann"})
    mem.retain("june note", source="jun", index_text="x3", metadata={"created": "2024-06-01", "person": "ann"})
    mem.retain("bob note", source="bob", index_text="x4", metadata={"created": "2024-01-10", "person": "bob"})
    link_all(mem, Linker())
    ls = links(mem)
    subs = {(a, b, st) for a, b, lt, st, o, c in ls if lt == "TEMPORAL"}
    assert ("jan", "jan2", "PRECEDES") in subs and ("jan2", "jan", "SUCCEEDS") in subs
    assert ("jan", "jan2", "TEMPORALLY_CLOSE") in subs or ("jan2", "jan", "TEMPORALLY_CLOSE") in subs
    assert ("jan2", "jun", "PRECEDES") in subs
    assert not any("jun" in (a, b) and st == "TEMPORALLY_CLOSE" for a, b, st in subs)
    assert not any("bob" in (a, b) for a, b, *_ in ls)      # another person: no temporal link
    assert {o for a, b, lt, st, o, c in ls if lt == "TEMPORAL"} == {"sql"}


def test_linker_next_between_chunks_of_one_note(mem):
    text = ("First paragraph. " * 40 + "\n\n" + "Second paragraph. " * 40 + "\n\n"
            + "Third paragraph. " * 40)
    mem.retain(text, source="long", index_text="q", child_chunks=True)
    link_all(mem, Linker())
    with conn_of(mem) as c:
        n_chunks = c.execute("SELECT count(*) FROM memory_items WHERE kind='chunk'").fetchone()[0]
        nxt = c.execute("SELECT count(*) FROM memory_links WHERE subtype='NEXT'").fetchone()[0]
        assert n_chunks >= 2 and nxt == n_chunks - 1


def test_linker_entities_and_shared_entity_links(mem):
    mem.retain("Kelly went to Acme on Monday.", source="a", index_text="qa")
    mem.retain("Acme hired Kelly last year.", source="b", index_text="qb")
    mem.retain("Nothing named here at all.", source="c", index_text="qc")
    llm = EntityLLM()
    link_all(mem, Linker(llm=llm))
    with conn_of(mem) as c:
        ents = sorted(r[:2] for r in c.execute("SELECT name, etype FROM memory_entities"))
        assert ents == [("Acme", "org"), ("Kelly", "person")]
        assert c.execute("SELECT count(*) FROM memory_item_entities").fetchone()[0] == 4
    ent = [(a, b) for a, b, lt, st, o, cf in links(mem) if lt == "ENTITY"]
    assert ("a", "b") in ent or ("b", "a") in ent
    assert not any("c" in pair for pair in ent)
    assert len(llm.prompts) == 3


def test_parse_entities_filters_noise():
    raw = '{"entities":[{"name":" Kelly ","type":"Person"},{"name":"kelly","type":"person"},' \
          '{"name":"X","type":"weird"},"Plain",{"nope":1}]}'
    assert parse_entities(raw) == [("Kelly", "person"), ("X", "other"), ("Plain", "other")]


def test_linker_pgvector_fallback_without_a_judge(mem):
    mem.retain("cats purr softly on warm mats", source="a", index_text="cats purr softly on warm mats")
    mem.retain("cats purr softly on warm mats today", source="b",
               index_text="cats purr softly on warm mats today")
    mem.retain("quarterly revenue grew twelve percent", source="c",
               index_text="quarterly revenue grew twelve percent")
    link_all(mem, Linker(vector_floor=0.5))
    sem = [(a, b, o) for a, b, lt, st, o, cf in links(mem) if lt == "SEMANTIC"]
    assert ("a", "b", "pgvector") in sem and ("b", "a", "pgvector") in sem
    assert not any("c" in (a, b) for a, b, o in sem)


class JevStub:
    """System One stub: scores by a rule on (question title, candidate text)."""

    def __init__(self, rule, fail=False):
        self.rule, self.fail, self.requests = rule, fail, []

    def __call__(self, request, timeout):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("jev down")
        answers = {k: {"score": self.rule(q["instructions"]["title"], q["instructions"]["text"],
                                          request["state"]["query"])}
                   for k, q in request["questions"].items()}
        return {"model": "typesafe/jev-1.13", "answers": answers,
                "usage": {"input_tokens": 100, "output_tokens": 3, "cost": 0.0004}}


def test_linker_jev_typed_links_with_threshold(mem):
    mem.retain("the pipe burst in the kitchen", source="cause", index_text="q1")
    mem.retain("the kitchen floor was flooded", source="effect", index_text="q2")
    mem.retain("an unrelated note about tax", source="other", index_text="q3")

    def rule(rel, cand, query):
        if "burst" in query and "flooded" in cand:
            return {"semantic": 3, "causes": 3, "caused_by": 0}[rel]
        if "flooded" in query and "burst" in cand:
            return {"semantic": 3, "causes": 0, "caused_by": 3}[rel]
        return 1   # 0.33 < 0.6: below Jev-Mem's threshold

    jev = JevStub(rule)
    link_all(mem, Linker(judge=JevRelationJudge(jev)))
    ls = [(a, b, lt, st, o) for a, b, lt, st, o, c in links(mem) if o == "jev"]
    assert ("cause", "effect", "CAUSAL", "LEADS_TO", "jev") in ls
    assert ("effect", "cause", "SEMANTIC", "RELATED_TO", "jev") in ls
    assert not any("other" in (a, b) for a, b, *_ in ls)
    # caused_by is stored as the reverse CAUSAL edge: the cause leads to the effect, once
    assert sum(1 for l in ls if l[:2] == ("cause", "effect") and l[2] == "CAUSAL") == 1
    # one question per (candidate, relation), at most 16 per request, wire shape intact
    for r in jev.requests:
        assert set(r) == {"model", "state", "questions"} and len(r["questions"]) <= 16
        assert all(q["type"] == "score" and len(q["criteria"]) == 4 for q in r["questions"].values())


def test_linker_jev_failure_falls_back_to_pgvector_and_records(mem):
    mem.retain("cats purr softly on warm mats", source="a", index_text="cats purr softly on warm mats")
    mem.retain("cats purr softly on warm mats today", source="b",
               index_text="cats purr softly on warm mats today")
    m = mem
    m._linker = Linker(judge=JevRelationJudge(JevStub(None, fail=True)), vector_floor=0.5)
    stats = m.link_document(docs(m)["a"])
    assert any(e.startswith("jev:") for e in stats["errors"]) and stats["semantic"] >= 1
    assert {o for a, b, lt, st, o, c in links(m) if lt == "SEMANTIC"} == {"pgvector"}
    with conn_of(m) as c:
        status, err = c.execute("SELECT status, error FROM memory_link_state "
                                "WHERE document_id=%s", (docs(m)["a"],)).fetchone()
    assert status == "error" and "jev" in err


def test_state_pending_replace_and_idempotence(mem):
    mem.retain("alpha body", source="a", index_text="alpha")
    mem.retain("beta body", source="b", index_text="beta")
    mem._linker = Linker(asynchronous=False)
    assert mem.link_pending() == 2 and mem.link_pending() == 0
    with conn_of(mem) as c:
        assert c.execute("SELECT count(*) FROM memory_link_state WHERE status='linked'").fetchone()[0] == 2
    n = len(links(mem))
    mem.link_document(docs(mem)["a"])           # re-linking adds nothing
    assert len(links(mem)) == n
    mem.retain("alpha body", source="a", index_text="alpha again")   # replace: links reset
    with conn_of(mem) as c:   # asynchronous=False: the replace relinked inside retain
        assert c.execute("SELECT count(*) FROM memory_link_state").fetchone()[0] == 2
    mem._linker = None
    mem.retain("alpha body", source="a", index_text="alpha third")   # no linker: state reset
    mem._linker = Linker(asynchronous=False)
    assert mem.link_pending() == 1


def test_async_worker_links_after_retain_with_accounting(fresh_db):
    llm, traces = EntityLLM(), []
    m = Memory(database_url=fresh_db, bank_id="b", llm=None, embed=stub_embed,
               linker=Linker(llm=llm),
               tracer=lambda kind, payload: traces.append((kind, payload)))
    m.create_bank("b", embedding_dim=EMBED_DIM)
    m.retain("Kelly met Acme.", source="a", index_text="qa")
    m.retain("Acme met Kelly.", source="b", index_text="qb")
    m.wait_for_links()
    with conn_of(m) as c:
        assert c.execute("SELECT count(*) FROM memory_link_state").fetchone()[0] == 2
        assert c.execute("SELECT count(*) FROM memory_links WHERE link_type='ENTITY'").fetchone()[0] >= 1
    calls = [p for k, p in traces if k == "llm_call" and p.get("purpose") == "extract_entities"]
    assert len(calls) == 2 and calls[0]["cost_usd"] == 0.001
    m.close()


# --------------------------------------------------------------------- Reader

def _items(n=3):
    from prospecta.channels.fusion import FusedDoc
    out = []
    for i in range(n):
        c = Candidate(document_id=f"d{i}", item_id=None, source=f"s{i}", channel="x",
                      rank=i + 1, score=1.0, evidence=f"evidence {i}")
        out.append(Item(doc=FusedDoc(f"d{i}", f"s{i}", 1.0, c), header=f"s{i}", evidence=f"evidence {i}"))
    return out


def stop_stub(sufficient, cont=0, missing=0, bad=False):
    def t(req, timeout):
        vals = {"evidence_sufficient": sufficient, "continue_useful": cont, "missing_evidence": missing}
        ans = {k: {"score": 9 if bad else vals[q["instructions"]["title"]]}
               for k, q in req["questions"].items()}
        return {"answers": ans, "usage": {"input_tokens": 10, "output_tokens": 1, "cost": 0.0001}}
    return t


def test_jev_reader_verdicts():
    calls: list = []
    assert JevReader(stop_stub(3)).read("q", _items(), 2, calls) == (True, [])
    assert JevReader(stop_stub(1, cont=3)).read("q", _items(), 2, calls) == (False, ["evidence 0"])
    assert JevReader(stop_stub(1, missing=2)).read("q", _items(), 2, calls)[0] is False
    assert JevReader(stop_stub(1)).read("q", _items(), 2, calls) == (True, [])
    assert len(calls) == 4 and calls[0]["cost_usd"] == 0.0001 and calls[0]["purpose"] == "jev_reader"
    with pytest.raises(ValueError):
        JevReader(stop_stub(1, bad=True)).read("q", _items(), 2, calls)
    assert calls[-1]["error"]


def test_reader_type_config_and_hop_with_jev(mem):
    with pytest.raises(ValueError):
        validate_recall_config({"reader": {"enabled": True, "type": "nope"}})
    validate_recall_config({"reader": {"enabled": True, "type": "jev"}})
    mem.set_channel_config(DEFAULT_CHANNEL_CONFIG)
    mem.retain("cats purr softly", source="n1", index_text="cats purr softly")
    mem.retain("bridge fact about whiskers", source="n2", index_text="bridge fact about whiskers")
    mem.set_recall_config({"rerank": {"enabled": False},
                           "reader": {"enabled": True, "type": "jev", "top": 2}})
    mem._jev = JevScore(stop_stub(1, cont=3))
    traces: list = []
    mem.search("cats purr softly", limit=5, _trace=traces)
    hops = traces[0]["hops"]
    assert hops["reader"] == "jev_reader" and hops["verdict"] == "follow_up"
    assert hops["follow_ups"] and not hops["error"]


def test_hub_entity_reached_by_entity_join_not_capped_links(mem):
    """A hub entity (more holders than ENTITY_HUB) gets no per-anchor links, and
    GraphExpand still enumerates every holder through memory_item_entities."""
    n = 14
    for i in range(n):
        mem.retain(f"Kelly note {i}", source=f"h{i}", index_text=f"qh{i}")
    link_all(mem, Linker(llm=EntityLLM(("Kelly",)), entity_hub=5))
    with conn_of(mem) as c:
        # once the entity passed the threshold (6th holder) no anchor writes links any more
        assert c.execute(
            "SELECT count(*) FROM memory_links l JOIN memory_items a ON a.id = l.src "
            "JOIN documents d ON d.id = a.document_id WHERE l.link_type='ENTITY' "
            "AND d.source NOT IN ('h0','h1','h2','h3','h4')").fetchone()[0] == 0
    ids = docs(mem)
    out = expand(mem, ids, ["h0"], max_hops=1, hub=5, node_cap=100)
    assert {c.source for c in out} == {f"h{i}" for i in range(1, n)}      # all 13, not 10
    # below the hub threshold the entity is linked directly and lossless
    with conn_of(mem) as c:
        c.execute("DELETE FROM memory_link_state")
        c.commit()
    link_all(mem, Linker(llm=EntityLLM(("Kelly",)), entity_hub=50))
    with conn_of(mem) as c:
        assert c.execute("SELECT count(*) FROM memory_links WHERE link_type='ENTITY'"
                         ).fetchone()[0] == n * (n - 1)
