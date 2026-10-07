"""One entity-name resolver: matching by tokens, not exact normalised equality."""
from __future__ import annotations

import psycopg
import pytest

from prospecta._entities import matches, resolve_entities, tokens
from prospecta._mapreduce import fetch_entity_notes
from prospecta.channels import MetadataScope, QueryPlan, RecallState
from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM, stub_embed
from tests.test_graph_links import conn_of


@pytest.mark.parametrize("q,c", [
    ("Nelson", "Mr. Nelson"), ("Mr. Nelson", "Nelson"), ("Dr Nelson", "Mr. Nelson"),
    ("Nelson, J.", "Nelson"), ("Nelson, J.", "J. Nelson"), ("Jose", "José"),
    ("JOSE  garcia", "José García"), ("Garcia", "José García"), ("Prof. Kelly", "kelly"),
    ("Velveteen Rabbit", "The Velveteen Rabbit (product)"), ("Greg", "Greg Ratajik"),
])
def test_matches(q, c):
    assert matches(q, c)


@pytest.mark.parametrize("q,c", [("Nelson", "Nilsson"), ("Greg", "Gertrude"), ("Mr.", "")])
def test_no_match(q, c):
    assert not matches(q, c)


def test_tokens_drop_honorifics():
    assert tokens("Dame Judi Dench") == ["judi", "dench"]


@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=None, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    for s in ["n1", "n2", "n3", "n4", "n5"]:
        fm = "---\nperson: Mr. Nelson\n---\n" if s == "n4" else ""
        m.retain(f"{fm}text of {s}", source=s, index_text=f"q {s}")
    yield m
    m.close()


def _link(mem, source, entity_name, etype="person", alias=None):
    with conn_of(mem) as c:
        with c.cursor() as cur:
            cur.execute("SELECT m.id FROM memory_items m JOIN documents d ON d.id = m.document_id "
                        "WHERE d.source = %s LIMIT 1", (source,))
            item = cur.fetchone()[0]
            norm = " ".join(entity_name.lower().split())
            cur.execute("INSERT INTO memory_entities (bank_id, name, norm, etype) VALUES ('b', %s, %s, %s) "
                        "ON CONFLICT (bank_id, norm, etype) DO UPDATE SET name = EXCLUDED.name RETURNING id",
                        (entity_name, norm, etype))
            eid = cur.fetchone()[0]
            cur.execute("INSERT INTO memory_item_entities (item_id, entity_id, n) VALUES (%s, %s, 1)", (item, eid))
            if alias:
                cur.execute("INSERT INTO memory_entity_aliases (bank_id, norm, alias, entity_id) "
                            "VALUES ('b', %s, %s, %s)", (alias.lower(), alias, eid))
        c.commit()
    return eid


def _sources(mem, names):
    with conn_of(mem) as c:
        return {s for _, s, _ in fetch_entity_notes(c, "b", names)}


def test_bare_name_reaches_titled_entity(mem):
    _link(mem, "n1", "Mr. Nelson")
    assert _sources(mem, ["Nelson"]) == {"n1", "n4"}   # n4: documents.person 'Mr. Nelson'


def test_all_variants_and_two_nelsons_union(mem):
    _link(mem, "n1", "Mr. Nelson")
    _link(mem, "n2", "Dr. Nelson")
    _link(mem, "n3", "Nelson", etype="other")
    _link(mem, "n5", "Nilsson")
    with conn_of(mem) as c:
        assert len(resolve_entities(c, "b", ["Nelson"]).ids) == 3
    for q in ["Nelson", "Mr Nelson", "Dr. Nelson", "Nelson, J.", "NELSON"]:
        assert {"n1", "n2", "n3"} <= _sources(mem, [q]), q
    assert "n5" not in _sources(mem, ["Nelson"])


def test_diacritics(mem):
    _link(mem, "n1", "José García")
    assert _sources(mem, ["Jose"]) == {"n1"}
    assert _sources(mem, ["jose garcia"]) == {"n1"}
    _link(mem, "n2", "Jose Perez")
    assert _sources(mem, ["José"]) == {"n1", "n2"}


def test_alias_row_resolves(mem):
    _link(mem, "n1", "Gregory Smith", alias="The Ghost")
    assert _sources(mem, ["The Ghost"]) == {"n1"}
    assert _sources(mem, ["Ghost"]) == {"n1"}


def test_filter_people_resolves(mem):
    from prospecta.channels.base import Filters
    with conn_of(mem) as c:
        st = RecallState(conn=c, bank_id="b", embed=lambda ts: [stub_embed(t) if False else stub_embed([t])[0] for t in ts])
        out = MetadataScope().retrieve(QueryPlan(text="q", filters=Filters(people=["Nelson"], hard=True)), st)
    assert {x.source for x in out} == {"n4"}
