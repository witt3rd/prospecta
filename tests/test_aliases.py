"""Alias / also-known-as extraction (migration 0014): nicknames and pen names
resolve to the person entity so a query for the person reaches notes that use
any alias. Stub LLM; a synthetic 9-note set, aliases spread over notes."""
from __future__ import annotations

import json

import psycopg
import pytest

from prospecta._linker import Linker, parse_aliases
from prospecta.channels import GraphExpand, QueryPlan, RecallState
from prospecta.memory import Memory
from prospecta.stages import LLMResult
from tests._stub_embedder import EMBED_DIM, stub_embed
from tests.test_graph_links import conn_of, docs, pool_of

ALIASES = {"Gman": "nickname", "The Ghost": "nickname", "J. Quill": "pen name"}

NOTES = {   # source -> text; processed in this order, definitions come late
    "n1": "Gman fixed the boiler again.",
    "n2": "J. Quill published a new chapter.",
    "n3": "Greg went for a walk in the garden.",          # the seed: names Greg
    "n4": "The Ghost left the party early.",
    "n5": "Unrelated note about Kelly and tea.",
    "n6": "Gman and J. Quill are the same person: Greg, also known as Gman, "
          "The Ghost and J. Quill.",
    "n7": "J. Quill signed the book.",
    "n8": "The Ghost again, on the roof.",
    "n9": "Gman won the chess club.",
}


class AliasLLM:
    """Stub Sonnet: names Greg-family persons; Greg lists his aliases only in n6."""

    def __init__(self, broken=()):
        self.calls, self.broken = 0, set(broken)

    def __call__(self, messages, *, json_mode=False):
        text = messages[-1]["content"].split("## Note")[-1]
        self.calls += 1
        if any(t in text for t in self.broken):
            raise RuntimeError("boom")
        ents = []
        if "also known as" in text:
            ents.append({"name": "Greg", "type": "person", "aliases": list(ALIASES)})
        elif "Greg" in text:
            ents.append({"name": "Greg", "type": "person"})
        for a in ALIASES:
            if a in text:
                ents.append({"name": a, "type": "person"})
        if "Kelly" in text:
            ents.append({"name": "Kelly", "type": "person"})
        return LLMResult(json.dumps({"entities": ents}), model="stub-sonnet",
                         tokens_in=5, tokens_out=5, cost_usd=0.0)


@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=None, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    for s, t in NOTES.items():
        m.retain(t, source=s, index_text=f"q {s}")
    yield m
    m.close()


def link(mem, linker, order=None):
    mem._linker = linker
    ids = docs(mem)
    for s in order or NOTES:
        mem.link_document(ids[s])


def reached(mem, seed):
    ids = docs(mem)
    with conn_of(mem) as c:
        st = RecallState(conn=c, bank_id="b", pool=pool_of(ids, seed))
        return {x.source for x in GraphExpand(max_hops=1, type_weights={"TEMPORAL": 0.0}).retrieve(QueryPlan(text="q"), st, 50)}


def test_parse_aliases_tolerant():
    raw = ('{"entities": [{"name": "Greg", "type": "person", "aliases": ["Gman", " ", 3]},'
           ' {"name": "Acme", "type": "org", "aliases": ["X"]}, {"name": "Bo", "type": "person",'
           ' "aliases": "Bobby"}, "junk"]}')
    assert parse_aliases(raw) == [("Greg", ["Gman"]), ("Bo", ["Bobby"])]
    assert parse_aliases('{"entities": []}') == []


def test_query_for_person_reaches_every_alias_note(mem):
    link(mem, Linker(llm=AliasLLM(), asynchronous=False))
    assert reached(mem, "n3") == {f"n{i}" for i in range(1, 10)} - {"n3", "n5"}
    with conn_of(mem) as c:   # aliases resolve to the one person entity
        rows = c.execute("SELECT a.alias, e.name FROM memory_entity_aliases a "
                         "JOIN memory_entities e ON e.id = a.entity_id ORDER BY 1").fetchall()
        assert [tuple(r) for r in rows] == [("Gman", "Greg"), ("J. Quill", "Greg"),
                                            ("The Ghost", "Greg")]
    # and a query for an alias reaches the person's other notes
    assert {"n3", "n1", "n6"} <= reached(mem, "n9")


@pytest.mark.parametrize("order", [list(NOTES), list(reversed(list(NOTES)))])
def test_order_independent(mem, order):
    link(mem, Linker(llm=AliasLLM(), asynchronous=False), order)
    assert reached(mem, "n3") == {f"n{i}" for i in range(1, 10)} - {"n3", "n5"}


def test_backfill_is_resumable_by_document_id(mem):
    linker = Linker(llm=AliasLLM(broken={"chess"}), asynchronous=False)
    with psycopg.connect(mem.database_url) as c:
        n, last = linker.backfill_aliases(c, "b", limit=4)
        assert n == 4
        n2, last2 = linker.backfill_aliases(c, "b", limit=100, after=last)
        assert n2 == 5 and last2 > last
        done = {r[0]: r[1] for r in c.execute("SELECT d.source, s.status FROM memory_alias_state s "
                                              "JOIN documents d ON d.id = s.document_id")}
        assert done["n9"] == "error" and sum(v == "done" for v in done.values()) == 8
        # rerun: only the errored document is retried, now healthy
        linker.llm.broken.clear()
        n3, _ = linker.backfill_aliases(c, "b")
        assert n3 == 1
        assert linker.backfill_aliases(c, "b")[0] == 0
    assert reached(mem, "n3") == {f"n{i}" for i in range(1, 10)} - {"n3", "n5"}
