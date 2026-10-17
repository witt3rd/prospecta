"""The meta channel's person-name matching: a name in a question reaches the `documents.person` that is that
person, by tokens, not by exact equality ("Nelson" reaches "Mr. Nelson"). (It once also widened through the
entity and alias tables of the linker; those were removed with it, docs/legacy.md.)"""
from __future__ import annotations

import pytest

from prospecta.channels import MetadataScope, QueryPlan, RecallState
from prospecta.channels.base import Filters
from prospecta.channels.meta import _tokens, name_matches, resolve_people
from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM, stub_embed


@pytest.mark.parametrize("q,c", [
    ("Nelson", "Mr. Nelson"), ("Mr. Nelson", "Nelson"), ("Dr Nelson", "Mr. Nelson"),
    ("Nelson, J.", "Nelson"), ("Nelson, J.", "J. Nelson"), ("Jose", "José"),
    ("JOSE  garcia", "José García"), ("Garcia", "José García"), ("Prof. Kelly", "kelly"),
    ("Velveteen Rabbit", "The Velveteen Rabbit (product)"), ("Greg", "Greg Ratajik"),
])
def test_matches(q, c):
    assert name_matches(q, c)


@pytest.mark.parametrize("q,c", [("Nelson", "Nilsson"), ("Greg", "Gertrude"), ("Mr.", "")])
def test_no_match(q, c):
    assert not name_matches(q, c)


def test_tokens_drop_honorifics():
    assert _tokens("Dame Judi Dench") == ["judi", "dench"]


@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=None, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    for source, person in [("n1", "Mr. Nelson"), ("n2", "Dr. Nelson"), ("n3", "José García"),
                           ("n4", "Nilsson"), ("n5", None)]:
        fm = f"---\nperson: {person}\n---\n" if person else ""
        m.retain(f"{fm}text of {source}", source=source)
    yield m
    m.close()


def test_a_bare_name_reaches_every_titled_person(mem):
    with mem._pool.connection() as conn:
        assert resolve_people(conn, "b", ["Nelson"]) == ["Dr. Nelson", "Mr. Nelson"]
        assert resolve_people(conn, "b", ["Jose"]) == ["José García"]
        assert resolve_people(conn, "b", ["Nilsson"]) == ["Nilsson"]
        assert resolve_people(conn, "b", ["  "]) == []


def test_the_filter_channel_resolves_people_by_name(mem):
    with mem._pool.connection() as conn:
        st = RecallState(conn=conn, bank_id="b", embed=stub_embed)
        out = MetadataScope().retrieve(QueryPlan(text="q", filters=Filters(people=["Nelson"], hard=True)), st)
    assert {x.source for x in out} == {"n1", "n2"}
