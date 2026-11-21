"""POST /retain and the per-request `bank` of the HTTP service, over a real socket and a real
Postgres (stub embedder, no model). Oracles are rows read straight from the database and the
bytes the test itself sent, never the service's own answer."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import psycopg
import pytest

from prospecta.memory import Memory
from prospecta.service import MAX_BODY, Server, Service, make_handler
from tests._stub_embedder import EMBED_DIM, stub_embed

TOKEN = "t-secret"


@pytest.fixture
def served(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="main", embed=stub_embed)
    mem.create_bank("main", embedding_dim=EMBED_DIM)
    srv = Server(("127.0.0.1", 0), make_handler(Service(mem, None), TOKEN))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield fresh_db, base
    srv.shutdown()
    srv.server_close()
    mem.close()


def call(base, path, body=None, *, token=TOKEN):
    req = urllib.request.Request(
        base + path, None if body is None else json.dumps(body).encode(),
        method="GET" if body is None else "POST",
        headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def rows(url, sql, *args):
    with psycopg.connect(url) as c:
        return c.execute(sql, args or None).fetchall()


def test_retain_is_keyed_by_source(served):
    url, base = served
    note = {"content": "The harbor crane was repainted orange.", "source": "k1"}
    st, a = call(base, "/retain", note)
    assert (st, a["status"]) == (200, "created")
    st, b = call(base, "/retain", note)
    assert (st, b["status"], b["document_id"]) == (200, "unchanged", a["document_id"])
    st, c = call(base, "/retain", {**note, "content": "The harbor crane was repainted green."})
    assert (st, c["status"]) == (200, "replaced")
    # one note under the key, and it is the new text
    assert rows(url, "SELECT original_text FROM documents WHERE bank_id='main' AND source='k1'") \
        == [("The harbor crane was repainted green.",)]
    assert rows(url, "SELECT count(*) FROM memory_items WHERE bank_id='main' "
                     "AND content LIKE '%orange%'") == [(0,)]


def test_same_text_under_another_source_is_409(served):
    _, base = served
    body = {"content": "Tide tables are posted at the pier.", "source": "a"}
    assert call(base, "/retain", body)[0] == 200
    st, err = call(base, "/retain", {**body, "source": "b"})
    assert st == 409 and "'a'" in err["error"]


def test_banks_are_isolated_and_must_exist_unless_created(served):
    url, base = served
    note = {"content": "The lighthouse lamp burns paraffin.", "source": "s"}
    st, err = call(base, "/retain", {**note, "bank": "other"})
    assert st == 404 and "other" in err["error"]
    assert rows(url, "SELECT count(*) FROM banks WHERE bank_id='other'") == [(0,)]
    assert call(base, "/retain", {**note, "bank": "other", "create": True})[0] == 200
    q = {"questions": ["lighthouse lamp paraffin"]}
    assert call(base, "/lookup", {**q, "bank": "other"})[1]["results"][0]["results"]
    assert call(base, "/lookup", q)[1]["results"][0]["results"] == []     # default bank: untouched
    assert call(base, "/lookup", {**q, "bank": "nope"})[0] == 404
    assert call(base, "/lookup", {**q, "bank": "bad bank!"})[0] == 400
    assert call(base, "/coverage?bank=nope")[0] == 404     # a GET names its bank in the query


def test_a_bank_of_another_embedding_dimension_is_409(served):
    url, base = served
    other = Memory(database_url=url, bank_id="narrow", embed=stub_embed)
    other.create_bank("narrow", embedding_dim=EMBED_DIM // 2)
    other.close()
    st, _ = call(base, "/retain", {"content": "x y z", "source": "s", "bank": "narrow"})
    assert st == 409


def test_a_huge_note_is_kept_whole(served):
    url, base = served
    words = (f"w{i:06d} " for i in range(MAX_BODY // 4))        # ~ 2 * MAX_BODY bytes
    text = "".join(words) + "ENDMARKERZQX"
    assert len(text.encode()) > 2 * MAX_BODY
    st, out = call(base, "/retain", {"content": text, "source": "big"})
    assert (st, out["status"]) == (200, "created")
    assert rows(url, "SELECT original_text FROM documents WHERE source='big'") == [(text,)]
    st, got = call(base, "/lookup", {"questions": ["ENDMARKERZQX"]})
    assert "ENDMARKERZQX" in got["results"][0]["results"][0]["content"]


def test_questions_mode_returns_a_standalone_note(served):
    _, base = served
    call(base, "/retain", {"content": "Quartz clocks drift about a second a month.",
                           "source": "clocks", "metadata": {"project": "p1", "kind": "note",
                                                            "created_on": "2024-05-06"}})
    st, out = call(base, "/lookup", {"questions": ["quartz clocks drift"], "project": "p1"})
    (hit,) = out["results"][0]["results"]
    assert hit["kind"] == "chunk" and hit["source"] == "clocks"
    assert hit["created_on"] == "2024-05-06" and "Quartz clocks" in hit["content"]
    st, task = call(base, "/lookup", {"task": {"title": "clocks", "ask": "quartz clocks drift"},
                                      "excluded_sources": ["clocks"]})
    assert st == 200 and task["task"]["results"] == []          # anti-echo list is honoured

