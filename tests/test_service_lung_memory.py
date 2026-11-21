"""POST /lung-memory/1: the adapter that speaks lung's `lung-memory/1` wire, over a real socket and a
real Postgres (stub embedder, no model).

The oracle is the Lean spec itself: `lung/spec/LungSpec/MemoryWire.lean`, PART B. The functions
`lean_*` below are a line-by-line transcription of its decoder (`decode`, `decodeOutcome`,
`readRecords`, `Record.ofJson`, `decodeRetain`) and of `Request.ofJson`, and the key sets are those of
`Record.toJson`, `Outcome.fields`, `RecallReply.toJson` and `RetainReply.toJson`. Row values (ids,
sources, dates) are read straight from the database, never from the service's own answer. Invented
notes only."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import psycopg
import pytest

from prospecta.memory import Memory
from prospecta.service import Server, Service, make_handler, zdr_attested_from_env
from tests._stub_embedder import EMBED_DIM, stub_embed

TOKEN = "t-secret"
PATH = "/lung-memory/1"

# ---- the Lean decoder, transcribed (MemoryWire.lean, PART B) -----------------------------------

REASON_WORDS = ["transport", "timeout", "malformed", "scope", "cost", "provider", "retention"]  # Reason.words


def lean_str(j, k):                       # def str: a string field, anything else is none
    v = j.get(k) if isinstance(j, dict) else None
    return v if isinstance(v, str) else None


def lean_record(j):                       # Record.ofJson: five strings, four of them non-empty
    f = [lean_str(j, k) for k in ("bank", "id", "text", "source", "date")]
    if any(x is None for x in f):
        return None
    bank, id_, text, source, date = f
    return (bank, id_, text, source, date) if bank != "" and id_ != "" and source != "" and date != "" else None


def lean_outcome(j):                      # decodeOutcome + readRecords
    s = lean_str(j, "status")
    if s is None:
        return ("unavailable", "malformed")
    if s == "ok":
        a = j.get("records")
        if not isinstance(a, list):
            return ("unavailable", "malformed")
        rs = [lean_record(x) for x in a]
        if any(r is None for r in rs):
            return ("unavailable", "malformed")
        if not rs:
            return ("empty",)
        return ("found", rs) if len({r[1] for r in rs}) == len(rs) else ("unavailable", "malformed")
    if s == "unavailable":
        r = lean_str(j, "reason")
        return ("unavailable", r) if r in REASON_WORDS else ("unavailable", "malformed")
    return ("unavailable", "malformed")


def lean_decode(j):                       # decode: the retention word is required
    a = lean_str(j, "retention")
    if a not in ("attested", "unattested"):
        return ("unavailable", "malformed"), "unattested"
    return lean_outcome(j), a


def lean_decode_retain(j):                # decodeRetain
    a = lean_str(j, "retention")
    if a not in ("attested", "unattested"):
        return ("unavailable", "malformed"), "unattested"
    s = lean_str(j, "status")
    if s is None:
        return ("unavailable", "malformed"), a
    if s == "retained":
        i = lean_str(j, "id")
        return (("retained", i) if i else ("unavailable", "malformed")), a
    if s == "unavailable":
        r = lean_str(j, "reason")
        return (("unavailable", r) if r in REASON_WORDS else ("unavailable", "malformed")), a
    return ("unavailable", "malformed"), a


def lean_request_ok(j):                   # Request.ofJson: does the body parse as one of the three ops?
    def strs(v):
        return isinstance(v, list) and all(isinstance(x, str) for x in v)
    op = lean_str(j, "op")
    if op == "recall":
        return lean_str(j, "task") is not None and strs(j.get("banks"))
    if op == "search":
        return lean_str(j, "query") is not None and strs(j.get("banks"))
    if op == "retain":
        b, t, s = (lean_str(j, k) for k in ("bank", "text", "source"))
        return b is not None and t is not None and s is not None and b != "" and s != ""   # Request.ok
    return False


# ---- fixtures ----------------------------------------------------------------------------------

@pytest.fixture
def served(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="main", embed=stub_embed)
    mem.create_bank("main", embedding_dim=EMBED_DIM)
    servers = []

    def serve(**kw):
        srv = Server(("127.0.0.1", 0), make_handler(Service(mem, None, **kw), TOKEN))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield fresh_db, mem, serve
    for s in servers:
        s.shutdown()
        s.server_close()
    mem.close()


def post(base, body, *, token=TOKEN, raw=None):
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(base + PATH, data, method="POST",
                                 headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def rows(url, sql, *args):
    with psycopg.connect(url) as c:
        return c.execute(sql, args or None).fetchall()


def seed(mem, bank, notes):
    """Notes into `bank` (made if missing): {source: (text, created_on)}."""
    if not rows(mem._database_url, "SELECT 1 FROM banks WHERE bank_id=%s", bank):
        mem.create_bank(bank, embedding_dim=EMBED_DIM)
    view = Memory(database_url=mem._database_url, bank_id=bank, embed=stub_embed)
    ids = {src: view.retain(text, source=src, metadata={"created_on": day})
           for src, (text, day) in notes.items()}
    view.close()
    return ids


def check_recall_shape(reply, banks):
    """The reply is exactly what `RecallReply.toJson` writes and `decode` reads back, in scope."""
    assert set(reply) == {"status", "records", "retention"} or set(reply) == {"status", "reason", "retention"}
    outcome, att = lean_decode(reply)
    assert att == reply["retention"]
    if outcome[0] == "found":
        assert all(set(r) == {"bank", "id", "text", "source", "date"} for r in reply["records"])
        assert all(r[0] in banks for r in outcome[1])           # checkScope
    return outcome


NOTES = {
    "tides/1": ("The harbor crane was repainted orange before the regatta.", "2024-05-06"),
    "tides/2": ("Tide tables are posted at the north pier every morning.", "2023-11-30"),
}


# ---- tests -------------------------------------------------------------------------------------

def test_search_and_recall_return_records_the_lean_decoder_reads(served):
    url, mem, serve = served
    base = serve()
    ids = seed(mem, "lung-class-a", NOTES)
    for req in ({"op": "search", "query": "harbor crane repainted orange", "banks": ["lung-class-a"]},
                {"op": "recall", "task": "harbor crane repainted orange", "banks": ["lung-class-a"]}):
        assert lean_request_ok(req)
        st, reply = post(base, req)
        assert st == 200
        outcome = check_recall_shape(reply, ["lung-class-a"])
        assert outcome[0] == "found"
        rec = next(r for r in outcome[1] if r[1] == ids["tides/1"])
        # expected record, from the database rows and the text the test wrote
        (src, day) = rows(url, "SELECT source, created_on::text FROM documents WHERE id=%s",
                          ids["tides/1"])[0]
        assert rec == ("lung-class-a", ids["tides/1"], NOTES["tides/1"][0], src, day)
        assert (src, day) == ("tides/1", "2024-05-06")
        assert reply["retention"] == "unattested"


def test_retain_creates_the_bank_and_is_idempotent_by_source(served):
    url, _, serve = served
    base = serve()
    req = {"op": "retain", "bank": "lung-domain-x", "text": "Quartz clocks drift a second a month.",
           "source": "lung/session-1"}
    assert lean_request_ok(req)
    st, a = post(base, req)
    assert st == 200 and set(a) == {"status", "id", "retention"}
    (kind, doc_id), att = lean_decode_retain(a)
    assert kind == "retained" and att == "unattested"
    assert rows(url, "SELECT id::text, source FROM documents WHERE bank_id='lung-domain-x'") \
        == [(doc_id, "lung/session-1")]
    st, b = post(base, req)                                   # replay: the same key, the same note
    assert lean_decode_retain(b)[0] == ("retained", doc_id)
    st, c = post(base, {**req, "text": "Quartz clocks drift two seconds a month."})
    # new text under the key replaces the old: one note, and the id reported is the stored one
    (kind, new_id), _ = lean_decode_retain(c)
    assert kind == "retained"
    assert rows(url, "SELECT id::text, original_text FROM documents WHERE bank_id='lung-domain-x' "
                     "AND source='lung/session-1'") == [(new_id, "Quartz clocks drift two seconds a month.")]


def test_recall_merges_banks_and_never_names_a_bank_outside_the_request(served):
    _, mem, serve = served
    base = serve()
    a = seed(mem, "lung-class-a", {"a/1": ("Orange crane paint is marine grade.", "2024-01-02")})
    b = seed(mem, "lung-domain-b", {"b/1": ("The crane operator logs orange paint stock.", "2024-02-03")})
    seed(mem, "lung-other", {"o/1": ("An orange crane stands in the other bank.", "2024-03-04")})
    for op, key in (("search", "query"), ("recall", "task")):
        st, reply = post(base, {"op": op, key: "orange crane paint",
                                "banks": ["lung-class-a", "lung-domain-b", "lung-domain-b"]})
        outcome = check_recall_shape(reply, ["lung-class-a", "lung-domain-b"])
        assert st == 200 and outcome[0] == "found"
        assert {(r[0], r[1]) for r in outcome[1]} == {("lung-class-a", a["a/1"]), ("lung-domain-b", b["b/1"])}


def test_nothing_found_is_ok_with_no_records(served):
    url, mem, serve = served
    base = serve()
    mem.create_bank("lung-empty", embedding_dim=EMBED_DIM)         # a bank with no notes in it
    for banks in (["lung-empty"], []):
        for req in ({"op": "search", "query": "zzyzx qwertyuiop", "banks": banks},
                    {"op": "recall", "task": "zzyzx qwertyuiop", "banks": banks}):
            st, reply = post(base, req)
            assert st == 200 and reply == {"status": "ok", "records": [], "retention": "unattested"}
            assert lean_decode(reply)[0] == ("empty",)


def unknown(name):
    return {"status": "unavailable", "reason": "scope", "detail": f"unknown bank: {name}",
            "retention": "unattested"}


def test_an_unknown_bank_is_unavailable_scope_and_is_not_made(served):
    url, mem, serve = served
    base = serve()
    seed(mem, "lung-class-a", NOTES)
    for banks in (["lung-nope"], ["lung-class-a", "lung-nope"], ["lung-nope", "bad name!"]):
        for req in ({"op": "search", "query": "harbor crane", "banks": banks},
                    {"op": "recall", "task": "harbor crane", "banks": banks}):
            st, reply = post(base, req)
            assert (st, reply) == (200, unknown("lung-nope"))     # the whole request fails, found notes too
            assert lean_decode(reply) == (("unavailable", "scope"), "unattested")
    st, reply = post(base, {"op": "search", "query": "x", "banks": ["bad name!"]})
    assert (st, reply) == (200, unknown("bad name!"))
    assert rows(url, "SELECT count(*) FROM banks WHERE bank_id IN ('lung-nope', 'bad name!')") == [(0,)]


SCRATCH_IDX = "SELECT indexname FROM pg_indexes WHERE indexname LIKE %s"


def test_delete_bank_removes_a_scratch_bank_with_its_documents_and_indexes(served):
    url, mem, serve = served
    base = serve()
    bank = "lung-scratch-r1"
    for i, text in enumerate(("Kelp farms grow along the north reef.", "Reef divers count kelp each May.")):
        st, a = post(base, {"op": "retain", "bank": bank, "text": text, "source": f"s/{i}"})
        assert lean_decode_retain(a)[0][0] == "retained"
    keep = seed(mem, "lung-class-keep", {"k/1": ("Kelp is kept elsewhere.", "2024-01-01")})
    safe = bank.replace("-", "_")
    # oracle: the database's own catalog, read before and after
    assert len(rows(url, SCRATCH_IDX, f"%{safe}%")) == 3
    assert rows(url, "SELECT count(*) FROM documents WHERE bank_id=%s", bank) == [(2,)]
    items_before = rows(url, "SELECT count(*) FROM memory_items WHERE bank_id=%s", bank)[0][0]
    assert items_before >= 2
    st, reply = post(base, {"op": "delete_bank", "bank": bank})
    assert (st, reply) == (200, {"status": "deleted", "bank": bank, "documents": 2, "retention": "unattested"})
    assert rows(url, "SELECT count(*) FROM banks WHERE bank_id=%s", bank) == [(0,)]
    assert rows(url, "SELECT count(*) FROM documents WHERE bank_id=%s", bank) == [(0,)]
    assert rows(url, "SELECT count(*) FROM memory_items WHERE bank_id=%s", bank) == [(0,)]
    assert rows(url, SCRATCH_IDX, f"%{safe}%") == []
    # another bank is untouched, indexes included
    assert rows(url, "SELECT count(*) FROM documents WHERE bank_id='lung-class-keep'") == [(1,)]
    assert len(rows(url, SCRATCH_IDX, "%lung_class_keep%")) == 3
    st, again = post(base, {"op": "delete_bank", "bank": bank})   # now it is unknown
    assert (st, again) == (200, unknown(bank))
    st, got = post(base, {"op": "search", "query": "kelp", "banks": ["lung-class-keep"]})
    assert [r[1] for r in check_recall_shape(got, ["lung-class-keep"])[1]] == [keep["k/1"]]


@pytest.mark.parametrize("name", ["lung-class-keep", "lung-scratch", "xlung-scratch-a", "LUNG-SCRATCH-a",
                                  "lung-scratch-nope"])
def test_delete_bank_refuses_a_name_that_is_not_scratch_and_touches_nothing(served, name):
    url, mem, serve = served
    base = serve()
    seed(mem, "lung-class-keep", {"k/1": ("Kelp is kept elsewhere.", "2024-01-01")})
    seed(mem, "lung-scratch", {"k/1": ("Kelp in a bank without the dash suffix.", "2024-01-01")})
    seed(mem, "xlung-scratch-a", {"k/1": ("Kelp in a bank with a prefix.", "2024-01-01")})
    before = rows(url, "SELECT bank_id, count(*) FROM documents GROUP BY 1 ORDER BY 1")
    idx_before = rows(url, "SELECT indexname FROM pg_indexes ORDER BY 1")
    st, reply = post(base, {"op": "delete_bank", "bank": name})
    if name == "lung-scratch-nope":                      # scratch name, no such bank
        assert (st, reply) == (200, unknown(name))
    else:
        assert (st, reply) == (200, {"status": "unavailable", "reason": "scope", "detail":
                                     "delete_bank is only allowed for lung-scratch-* banks",
                                     "retention": "unattested"})
    assert lean_decode(reply) == (("unavailable", "scope"), "unattested")
    assert rows(url, "SELECT bank_id, count(*) FROM documents GROUP BY 1 ORDER BY 1") == before
    assert rows(url, "SELECT indexname FROM pg_indexes ORDER BY 1") == idx_before


def test_a_hit_with_no_source_is_dropped_not_defaulted(served):
    url, mem, serve = served
    base = serve()
    ids = seed(mem, "lung-class-a", {
        "keep": ("Lamp oil is stored in the blue shed.", "2024-06-01"),
        "nosrc": ("Lamp oil is also stored in the red shed.", "2024-06-02"),
        "emptysrc": ("Lamp oil is never stored in the green shed.", "2024-06-03")})
    with psycopg.connect(url) as c:
        c.execute("UPDATE documents SET source=NULL WHERE id=%s", (ids["nosrc"],))
        c.execute("UPDATE documents SET source='' WHERE id=%s", (ids["emptysrc"],))
        c.commit()
    st, reply = post(base, {"op": "search", "query": "lamp oil stored shed", "banks": ["lung-class-a"]})
    outcome = check_recall_shape(reply, ["lung-class-a"])
    assert outcome[0] == "found" and [r[1] for r in outcome[1]] == [ids["keep"]]


def test_a_bank_the_service_cannot_read_is_unavailable_provider(served):
    url, mem, serve = served
    base = serve()
    other = Memory(database_url=url, bank_id="narrow", embed=stub_embed)
    other.create_bank("narrow", embedding_dim=EMBED_DIM // 2)
    other.close()
    for req in ({"op": "search", "query": "x y z", "banks": ["narrow"]},
                {"op": "recall", "task": "x y z", "banks": ["narrow"]}):
        st, reply = post(base, req)
        assert (st, reply) == (200, {"status": "unavailable", "reason": "provider", "retention": "unattested"})
        assert lean_decode(reply) == (("unavailable", "provider"), "unattested")
    st, reply = post(base, {"op": "retain", "bank": "narrow", "text": "x y z", "source": "s"})
    assert (st, reply) == (200, {"status": "unavailable", "reason": "provider", "retention": "unattested"})
    assert lean_decode_retain(reply) == (("unavailable", "provider"), "unattested")


@pytest.mark.parametrize("body", [
    {"op": "recall", "task": "t"},                                  # no banks
    {"op": "recall", "task": 7, "banks": []},                       # task not a string
    {"op": "search", "query": "q", "banks": ["a", 3]},              # a bank not a string
    {"op": "search", "banks": []},                                  # no query
    {"op": "retain", "bank": "", "text": "t", "source": "s"},       # Request.ok: bank named
    {"op": "retain", "bank": "b", "text": "t", "source": ""},       # Request.ok: source named
    {"op": "retain", "bank": "b", "source": "s"},                   # no text
    {"op": "forget", "bank": "b"},                                  # not one of the ops
    {"op": "delete_bank"},                                          # no bank
    {"op": "delete_bank", "bank": ""},                              # empty bank
    {"banks": []},                                                  # no op
])
def test_a_request_that_does_not_parse_is_400_with_the_malformed_reply(served, body):
    _, _, serve = served
    assert not lean_request_ok(body)
    st, reply = post(serve(), body)
    assert (st, reply) == (400, {"status": "unavailable", "reason": "malformed", "retention": "unattested"})
    assert lean_decode(reply) == (("unavailable", "malformed"), "unattested")
    assert lean_decode_retain(reply) == (("unavailable", "malformed"), "unattested")


def test_a_body_that_is_not_json_or_not_an_object_is_400_malformed(served):
    _, _, serve = served
    for raw in (b"not json", b"[1, 2]", b'"recall"'):
        st, reply = post(serve(), None, raw=raw)
        assert (st, reply["status"], reply["reason"]) == (400, "unavailable", "malformed")


def test_retention_is_attested_only_when_the_deployment_declares_it(served):
    _, mem, serve = served
    seed(mem, "lung-class-a", NOTES)
    yes, no = serve(zdr_attested=True), serve(zdr_attested=False)
    req = {"op": "search", "query": "harbor crane", "banks": ["lung-class-a"]}
    assert lean_decode(post(yes, req)[1])[1] == "attested"
    assert lean_decode(post(no, req)[1])[1] == "unattested"
    assert lean_decode(post(yes, {"op": "search", "query": "zzyzx", "banks": []})[1]) == (("empty",), "attested")
    assert lean_decode(post(yes, {"op": "x"})[1])[1] == "attested"      # also on a 400
    ret = {"op": "retain", "bank": "lung-class-a", "text": "A new note about moorings.", "source": "m"}
    assert lean_decode_retain(post(yes, ret)[1])[1] == "attested"
    assert lean_decode_retain(post(no, ret)[1])[1] == "unattested"


@pytest.mark.parametrize("value,expected", [("1", True), ("0", False), ("", False), ("true", False),
                                            ("yes", False), (None, False)])
def test_the_env_flag_is_exactly_one(monkeypatch, value, expected):
    monkeypatch.delenv("PROSPECTA_ZDR_ATTESTED", raising=False)
    if value is not None:
        monkeypatch.setenv("PROSPECTA_ZDR_ATTESTED", value)
    assert zdr_attested_from_env() is expected


def test_the_endpoint_needs_the_bearer_token_and_is_post_only(served):
    _, _, serve = served
    base = serve()
    req = {"op": "search", "query": "q", "banks": []}
    st, _ = post(base, req, token=None)
    assert st == 401
    st, _ = post(base, req, token="wrong")
    assert st == 401
    get = urllib.request.Request(base + PATH, headers={"Authorization": f"Bearer {TOKEN}"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(get, timeout=30)
    assert e.value.code == 404
