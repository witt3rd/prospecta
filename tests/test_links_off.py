"""Linking is off by default (per bank setting turns it on); entities keep their own
switch; the graph channel is silent by default; `drop-links` removes link rows
only on request and never locks the table."""
from __future__ import annotations

import json

import psycopg
import pytest

from prospecta._linker import Linker
from prospecta._test_helpers import no_cut
from prospecta.channels import DEFAULT_CHANNEL_CONFIG
from prospecta.memory import Memory
from prospecta.stages import LLMResult
from tests._stub_embedder import EMBED_DIM, stub_embed


class EntityLLM:
    def __call__(self, messages, *, json_mode=False):
        text = messages[-1]["content"].split("## Note")[-1]
        ents = [{"name": k, "type": "person"} for k in ("Kelly", "Ann") if k in text]
        return LLMResult(json.dumps({"entities": ents}), model="stub", tokens_in=1,
                         tokens_out=1, cost_usd=0.0)


@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="b", llm=None, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    yield m
    m.close()


def q(m, sql):
    with psycopg.connect(m.database_url) as c:
        return c.execute(sql).fetchone()[0]


def fill(m):
    m.retain("Kelly met Ann on monday", source="a", index_text="kelly ann monday",
             metadata={"created": "2024-01-10", "person": "p"})
    m.retain("Kelly wrote again", source="b", index_text="kelly wrote",
             metadata={"created": "2024-01-11", "person": "p"})
    m._linker = Linker(llm=EntityLLM(), asynchronous=False)
    with psycopg.connect(m.database_url) as c:
        ids = [str(r[0]) for r in c.execute("SELECT id FROM documents")]
    for d in ids:
        m.link_document(d)


def test_default_writes_no_link_rows_but_keeps_entities(mem):
    fill(mem)
    assert q(mem, "SELECT count(*) FROM memory_links") == 0
    assert q(mem, "SELECT count(*) FROM memory_entities") >= 2
    assert q(mem, "SELECT count(*) FROM memory_item_entities") >= 2


def test_default_without_a_linker_step_for_retain_path(mem):
    mem._linker = Linker(llm=EntityLLM(), asynchronous=False)
    mem.retain("Kelly met Ann", source="a", index_text="kelly ann")
    assert q(mem, "SELECT count(*) FROM memory_links") == 0


def test_bank_setting_turns_links_on(mem):
    mem.set_link_config({"links": True})
    fill(mem)
    assert q(mem, "SELECT count(*) FROM memory_links") > 0
    assert mem.get_link_config() == {"links": True, "entities": True}


def test_entities_have_their_own_switch(mem):
    mem.set_link_config({"links": False, "entities": False})
    fill(mem)
    assert q(mem, "SELECT count(*) FROM memory_entities") == 0
    assert q(mem, "SELECT count(*) FROM memory_links") == 0


def test_link_config_validates(mem):
    with pytest.raises(ValueError):
        mem.set_link_config({"links": "yes"})
    with pytest.raises(ValueError):
        mem.set_link_config({"bogus": True})


def test_link_pending_does_not_loop_when_off(mem):
    mem.set_link_config({"links": False, "entities": False})
    mem.retain("x", source="a", index_text="x")
    mem._linker = Linker(asynchronous=False)
    assert mem.link_pending() == 0


def test_graph_default_weight_is_zero():
    g = {c["name"]: c for c in DEFAULT_CHANNEL_CONFIG}["graph"]
    assert g["weight"] == 0 and g["enabled"]


def test_recall_same_with_and_without_link_rows(mem):
    mem.set_channel_config(no_cut(DEFAULT_CHANNEL_CONFIG))
    mem.set_link_config({"links": True})
    fill(mem)
    assert q(mem, "SELECT count(*) FROM memory_links") > 0
    with_rows = [(r.source, r.scores.get("dense_chunk"), r.scores.get("question"),
                  r.scores.get("lexical")) for r in mem.search("kelly ann", limit=10)]
    mem.drop_links(yes=True)
    assert q(mem, "SELECT count(*) FROM memory_links") == 0
    none = [(r.source, r.scores.get("dense_chunk"), r.scores.get("question"),
             r.scores.get("lexical")) for r in mem.search("kelly ann", limit=10)]
    assert with_rows and with_rows == none


def test_drop_links_refuses_without_yes_and_deletes_nothing(mem):
    mem.set_link_config({"links": True})
    fill(mem)
    n = q(mem, "SELECT count(*) FROM memory_links")
    with pytest.raises(PermissionError):
        mem.drop_links()
    assert q(mem, "SELECT count(*) FROM memory_links") == n


def test_drop_links_removes_only_this_banks_links_and_reports(mem):
    mem.set_link_config({"links": True})
    fill(mem)
    n = q(mem, "SELECT count(*) FROM memory_links")
    ents = q(mem, "SELECT count(*) FROM memory_item_entities")
    out = mem.drop_links(yes=True, batch=1)
    assert out["rows"] == n and out["bytes"] > 0
    assert q(mem, "SELECT count(*) FROM memory_links") == 0
    assert q(mem, "SELECT count(*) FROM memory_item_entities") == ents
    assert q(mem, "SELECT count(*) FROM memory_items") > 0


def test_cli_drop_links_needs_yes(fresh_db, monkeypatch, capsys):
    from prospecta.cli import __main__ as cli, _common
    m = Memory(database_url=fresh_db, bank_id="b", llm=None, embed=stub_embed)
    m.create_bank("b", embedding_dim=EMBED_DIM)
    m.set_link_config({"links": True})
    fill(m)
    n = q(m, "SELECT count(*) FROM memory_links")
    monkeypatch.setattr(_common, "make_memory", lambda a: m)
    assert cli.main(["--database-url", fresh_db, "--bank", "b", "drop-links"]) == 2
    assert q(m, "SELECT count(*) FROM memory_links") == n
    assert str(n) in capsys.readouterr().out   # the preview says how many
    assert cli.main(["--database-url", fresh_db, "--bank", "b", "drop-links", "--yes"]) == 0
    assert q(m, "SELECT count(*) FROM memory_links") == 0
    m.close()
