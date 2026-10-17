"""In-place source views: indexing, retirement, coverage, service."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import psycopg
import pytest

from prospecta.db.migrate import run_migrations
from prospecta.memory import Memory
from prospecta.service import Service, make_handler
from prospecta.source import SourceConfig, SourceContractError, SourceIndexer
from tests._stub_embedder import EMBED_DIM, stub_embed
from tests.test_shared_schema import bare_db  # noqa: F401  (fixture)

HOST_DDL = """
CREATE SCHEMA host_app;
CREATE TABLE host_app.notes (
    id int PRIMARY KEY, rev int NOT NULL DEFAULT 1, title text, body text,
    who text, proj text, day date, aud text, gone timestamptz);
CREATE VIEW host_app.records AS
  SELECT 'note'::text AS kind, id AS record_id, rev AS revision, title, body AS text,
         who AS person, proj AS project, day AS created_date, aud AS audience_key,
         gone AS deleted_at
  FROM host_app.notes;
"""


def _q(url, sql, *params):
    with psycopg.connect(url, autocommit=True) as c:
        return c.execute(sql, params).fetchall() if sql.lstrip().upper().startswith("SELECT") \
            else c.execute(sql, params)


@pytest.fixture
def env(bare_db, monkeypatch):  # noqa: F811
    # the bm25 channel's in-process index is an explicit opt-in, as in a bank that wants it
    monkeypatch.setattr(Bm25Chunks, "backend", AutoBm25(fallback=InProcessBm25()))
    with psycopg.connect(bare_db, autocommit=True) as c:
        c.execute(HOST_DDL)
    run_migrations(bare_db, schema="prospecta_shared")
    mem = Memory(database_url=bare_db, bank_id="b", embed=stub_embed,
                 llm=lambda messages, **k: "What is the first thing?\nWhy does the second thing matter?",
                 schema="prospecta_shared")
    mem.create_bank("b", embedding_dim=EMBED_DIM)
    ix = SourceIndexer(mem, SourceConfig("host_app", ("records",)), generate_questions=True)
    yield bare_db, mem, ix
    mem.close()


def _note(url, id, rev=1, body="Alpha paragraph about gardens.\n\nBeta paragraph about tea.",
          proj="p1", aud="team-a", gone=None, person="ann"):
    _q(url, "INSERT INTO host_app.notes VALUES (%s,%s,'T',%s,%s,%s,'2026-01-02',%s,%s) "
            "ON CONFLICT (id) DO UPDATE SET rev=EXCLUDED.rev, body=EXCLUDED.body, "
            "proj=EXCLUDED.proj, aud=EXCLUDED.aud, gone=EXCLUDED.gone",
       id, rev, body, person, proj, aud, gone)


def _items(url, rid):
    return _q(url, "SELECT kind, metadata->>'revision' FROM prospecta_shared.memory_items "
                   "WHERE metadata->>'record_id' = %s ORDER BY kind", rid)


def test_contract_check(env):
    url, mem, ix = env
    ix.check_contract()
    with pytest.raises(SourceContractError, match="does not exist"):
        SourceIndexer(mem, SourceConfig("host_app", ("nope",))).check_contract()
    _q(url, "CREATE VIEW host_app.bad AS SELECT 'x'::text AS kind")
    with pytest.raises(SourceContractError, match="lacks columns"):
        SourceIndexer(mem, SourceConfig("host_app", ("bad",))).check_contract()


def test_index_is_idempotent_and_keeps_no_text(env):
    url, mem, ix = env
    _note(url, 1)
    r = ix.index_record("note", "1")
    assert (r.status, r.revision) == ("indexed", "1") and r.items >= 3
    assert ix.index_record("note", "1").status == "unchanged"
    kinds = [k for k, _ in _items(url, "1")]
    assert kinds.count("question") == 2 and kinds.count("chunk") >= 1
    # no second store of record: the anchor document holds no text
    assert _q(url, "SELECT original_text FROM prospecta_shared.documents") == [("",)]
    assert _q(url, "SELECT count(*) FROM prospecta_shared.documents")[0][0] == 1
    # nothing written into the host's schema
    assert _q(url, "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
                   "WHERE n.nspname='host_app'")[0][0] == 3   # notes, its pkey index, records view


def test_index_makes_no_model_call_unless_questions_are_asked_for(env):
    url, mem, asking = env
    calls = []
    mem._llm = lambda messages, **k: calls.append(messages) or "What is the first thing?"
    plain = SourceIndexer(mem, SourceConfig("host_app", ("records",)))
    _note(url, 1)
    assert plain.index_record("note", "1").items >= 1
    assert calls == [] and {k for k, _ in _items(url, "1")} == {"chunk"}
    _note(url, 2)
    asking.index_record("note", "2")
    kinds = [k for k, _ in _items(url, "2")]
    assert len(calls) == 1 and kinds.count("question") == 1 and kinds.count("chunk") >= 1
    mem._llm = None
    with pytest.raises(ValueError, match="llm"):
        SourceIndexer(mem, SourceConfig("host_app", ("records",)), generate_questions=True)


def test_revision_change_retires_old_items(env):
    url, mem, ix = env
    _note(url, 1)
    ix.index_record("note", "1")
    _note(url, 1, rev=2, body="Completely new words about rockets.")
    r = ix.index_record("note", "1")
    assert r.status == "indexed" and r.revision == "2"
    assert {rev for _, rev in _items(url, "1")} == {"2"}
    assert _q(url, "SELECT count(*) FROM prospecta_shared.source_refs")[0][0] == 1
    assert _q(url, "SELECT count(*) FROM prospecta_shared.documents")[0][0] == 1


def test_deleted_marker_and_vanished_record_retire(env):
    url, mem, ix = env
    _note(url, 1); _note(url, 2)
    ix.index_record("note", "1"); ix.index_record("note", "2")
    _q(url, "UPDATE host_app.notes SET gone = now() WHERE id = 1")
    assert ix.index_record("note", "1").status == "removed"
    assert _items(url, "1") == []
    assert ix.index_record("note", "1").status == "absent"
    _q(url, "DELETE FROM host_app.notes WHERE id = 2")
    assert ix.index_record("note", "2").status == "removed"
    assert ix.index_record("note", "404").status == "absent"
    assert _q(url, "SELECT count(*) FROM prospecta_shared.memory_items")[0][0] == 0


def test_remove_record_explicit(env):
    url, mem, ix = env
    _note(url, 1)
    ix.index_record("note", "1")
    assert ix.remove_record("note", "1").status == "removed"
    assert ix.remove_record("note", "1").status == "absent"
    assert _items(url, "1") == []


def test_reindex_kind(env):
    url, mem, ix = env
    for i in (1, 2, 3):
        _note(url, i)
    assert ix.reindex_kind("note")["indexed"] == 3
    again = ix.reindex_kind("note")
    assert again["indexed"] == 0 and again["unchanged"] == 3
    _note(url, 2, rev=2); _note(url, 3, gone="2026-02-01")
    out = ix.reindex_kind("note")
    assert (out["indexed"], out["unchanged"], out["removed"]) == (1, 1, 1)


def test_coverage_names_missing_stale_orphaned(env):
    url, mem, ix = env
    _note(url, 1, proj="p1"); _note(url, 2, proj="p1"); _note(url, 3, proj="p2"); _note(url, 4, proj="p2")
    for i in (1, 2, 4):
        ix.index_record("note", str(i))
    _note(url, 2, rev=2, proj="p1")                       # stale (host moved on)
    _q(url, "UPDATE host_app.notes SET gone = now() WHERE id = 4")  # orphan: derived but deleted
    cov = {(c.kind, c.project): c for c in ix.coverage("note")}
    p1, p2 = cov[("note", "p1")], cov[("note", "p2")]
    assert (p1.live, p1.indexed) == (2, 1)
    assert p1.stale == [{"record_id": "2", "revision": "2", "indexed_revision": "1"}]
    assert p1.missing == []
    assert p2.missing == [{"record_id": "3", "revision": "1"}]
    assert p2.orphaned == [{"record_id": "4", "indexed_revision": "1"}]
    only = ix.coverage("note", project="p2")
    assert [c.project for c in only] == ["p2"]


def test_service_lookup_filters_and_shape(env):
    url, mem, ix = env
    _note(url, 1, body="Gardens need water and sun.", aud="team-a")
    _note(url, 2, body="Rockets need fuel and thrust.", aud="team-b")
    ix.reindex_kind("note")
    svc = Service(mem, ix)
    out = svc.lookup({"questions": ["what do gardens need", "how do rockets fly"], "limit": 5})
    assert [r["question"] for r in out["results"]] == ["what do gardens need", "how do rockets fly"]
    for r in out["results"]:
        assert {x["record_id"] for x in r["results"]} == {"1", "2"}
        assert all(set(x) == {"kind", "record_id", "revision", "score", "created_on", "match_kind",
                              "exact_snippet", "relevance_score"} for x in r["results"])
        assert r["results"] == sorted(r["results"], key=lambda x: -x["score"])
    only_b = svc.lookup({"questions": ["what do gardens need"], "audience_key": "team-b"})
    assert {x["record_id"] for x in only_b["results"][0]["results"]} == {"2"}
    from prospecta.service import ApiError
    for bad in ({}, {"questions": []}, {"questions": ["a"], "limit": 0}, {"questions": [3]}):
        with pytest.raises(ApiError):
            svc.lookup(bad)


def test_http_round_trip_and_auth(env):
    url, mem, ix = env
    _note(url, 1)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(Service(mem, ix), "tok"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"

    def call(method, path, body=None, token="tok"):
        req = urllib.request.Request(base + path, method=method,
                                     data=json.dumps(body).encode() if body is not None else None)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())
    try:
        assert call("GET", "/healthz", token=None)[0] == 200
        assert call("GET", "/coverage", token=None)[0] == 401
        assert call("GET", "/coverage?kind=note")[1]["coverage"][0]["missing"] == \
            [{"record_id": "1", "revision": "1"}]
        assert call("POST", "/index", {"kind": "note", "record_id": "1"})[1]["status"] == "indexed"
        assert call("POST", "/index", {"kind": "note", "record_id": "1"})[1]["status"] == "unchanged"
        st, out = call("POST", "/lookup", {"questions": ["gardens"]})
        assert st == 200 and out["results"][0]["results"][0]["record_id"] == "1"
        assert call("POST", "/remove", {"kind": "note", "record_id": "1"})[1]["status"] == "removed"
        assert call("GET", "/coverage?kind=note")[1]["coverage"][0]["indexed"] == 0
        assert call("POST", "/index", {"kind": "note"}, )[0] == 400
        assert call("GET", "/nope")[0] == 404
    finally:
        srv.shutdown(); srv.server_close()


# ---- allowed-audience filter -------------------------------------------------

from prospecta.channels.base import Filters, QueryPlan, RecallState  # noqa: E402
from prospecta.channels.bm25 import AutoBm25, Bm25Chunks, InProcessBm25  # noqa: E402
from prospecta.channels.graph import GraphExpand  # noqa: E402
from prospecta.channels.hybrid import HybridChunks  # noqa: E402
from prospecta.channels.meta import MetadataScope  # noqa: E402
from prospecta.channels.semantic import AnticipatedQuestions, DenseChunks  # noqa: E402

GARDEN = "Gardens need water and sun to grow."


def _ids(url, rid):
    return {str(r[0]) for r in _q(url, "SELECT document_id FROM prospecta_shared.source_refs "
                                       "WHERE record_id = %s", rid)}


def _channel_docs(mem, text, allow):
    """Document ids each channel offers as candidates for `text`."""
    flt = {"_allow": allow} if allow is not None else None
    plan = QueryPlan(text=text, filters=Filters(people=["ann"]))
    out = {}
    with mem._pool.connection() as conn:
        for ch in (DenseChunks(), AnticipatedQuestions(), Bm25Chunks(), HybridChunks(), MetadataScope()):
            st = RecallState(conn=conn, bank_id="b", embed=stub_embed, metadata_filter=flt)
            out[ch.name] = {c.document_id for c in ch.retrieve(plan, st)}
        st = RecallState(conn=conn, bank_id="b", embed=stub_embed, metadata_filter=flt)
        seed = DenseChunks().retrieve(plan, RecallState(conn=conn, bank_id="b", embed=stub_embed,
                                                        metadata_filter=flt))
        st.pool = seed
        out["graph"] = {c.document_id for c in GraphExpand().retrieve(plan, st)}
        conn.rollback()
    return out


def _link_all(url):
    _q(url, "INSERT INTO prospecta_shared.memory_links (bank_id, src, dst, link_type, subtype) "
            "SELECT 'b', a.id, c.id, 'SEMANTIC', 'RELATED_TO' "
            "FROM prospecta_shared.memory_items a, prospecta_shared.memory_items c "
            "WHERE a.document_id <> c.document_id AND a.kind = 'chunk' AND c.kind = 'chunk'")


def test_allowed_audience_excludes_record_from_every_channel(env):
    url, mem, ix = env
    _note(url, 1, body=GARDEN, aud="team-a")
    _note(url, 2, body=GARDEN + " Gardens also like tea.", aud="team-b")
    _note(url, 3, body=GARDEN + " Gardens and rain.", aud="")
    ix.reindex_kind("note")
    _link_all(url)
    d1, d2, d3 = _ids(url, "1"), _ids(url, "2"), _ids(url, "3")
    # Sanity: with no filter the other audiences are candidates, in every channel.
    base = _channel_docs(mem, GARDEN, None)
    for name in ("dense_chunk", "question", "bm25", "hybrid_chunk", "meta", "graph"):
        assert base[name], name
    assert (d2 | d3) & set().union(*base.values())
    # Restricted to team-a: nothing else, in any channel.
    got = _channel_docs(mem, GARDEN, {"audience_key": ["team-a"]})
    for name, docs in got.items():
        assert not (docs & (d2 | d3)), name
    assert got["dense_chunk"] == d1
    # The empty audience key is visible only when listed.
    got = _channel_docs(mem, GARDEN, {"audience_key": ["team-a", ""]})
    assert got["dense_chunk"] == d1 | d3
    assert not any(docs & d2 for docs in got.values())
    # An empty list allows nothing; projects filter the same way.
    assert all(not docs for docs in _channel_docs(mem, GARDEN, {"audience_key": []}).values())
    _note(url, 4, body=GARDEN + " Other project.", proj="p9", aud="team-a")
    ix.index_record("note", "4")
    got = _channel_docs(mem, GARDEN, {"audience_key": ["team-a"], "project": ["p1"]})
    assert not (got["dense_chunk"] & _ids(url, "4"))


def test_allowed_audience_never_reaches_a_result_or_follow_up(env):
    url, mem, ix = env
    _note(url, 1, body=GARDEN, aud="team-a")
    _note(url, 2, body=GARDEN + " Secret.", aud="team-b")
    ix.reindex_kind("note")
    svc = Service(mem, ix)
    for body in ({"allowed_audience_keys": ["team-a"]},
                 {"allowed_audience_keys": ["team-a"], "allowed_projects": ["p1"]}):
        out = svc.lookup({"questions": [GARDEN], **body})
        assert {x["record_id"] for x in out["results"][0]["results"]} == {"1"}
    none = svc.lookup({"questions": [GARDEN], "allowed_audience_keys": []})
    assert none["results"][0]["results"] == []
    from prospecta.service import ApiError
    with pytest.raises(ApiError):
        svc.lookup({"questions": ["a"], "allowed_audience_keys": "team-a"})
    # Library path, hybrid search through the bank's channels (incl. rerank inputs
    # and follow-ups, which run on the same filtered state).
    hits = mem.recall([GARDEN], metadata_filter={"_allow": {"audience_key": ["team-a"]}})
    assert hits and {h.metadata["record_id"] for h in hits} == {"1"}


def test_audience_change_takes_effect_after_reindex(env):
    url, mem, ix = env
    _note(url, 1, body=GARDEN, aud="team-a")
    ix.index_record("note", "1")
    svc = Service(mem, ix)

    def seen(key):
        out = svc.lookup({"questions": [GARDEN], "allowed_audience_keys": [key]})
        return {x["record_id"] for x in out["results"][0]["results"]}

    assert seen("team-a") == {"1"} and seen("team-b") == set()
    _note(url, 1, rev=2, body=GARDEN, aud="team-b")       # audience moves, revision moves
    assert seen("team-a") == {"1"}                         # not yet reindexed
    assert ix.index_record("note", "1").status == "indexed"
    assert seen("team-a") == set() and seen("team-b") == {"1"}
    assert {r[1] for r in _items(url, "1")} == {"2"}       # old items retired


def test_absent_allow_filter_changes_nothing(env):
    url, mem, ix = env
    _note(url, 1, body=GARDEN, aud="team-a")
    _note(url, 2, body=GARDEN + " More.", aud="team-b")
    _note(url, 3, body=GARDEN + " Again.", aud="")
    ix.reindex_kind("note")
    svc = Service(mem, ix)
    plain = svc.lookup({"questions": [GARDEN]})
    assert {x["record_id"] for x in plain["results"][0]["results"]} == {"1", "2", "3"}
    # Same results with no filter, an empty filter dict, or every audience listed.
    ref = [(h.document_id, h.score) for h in mem.recall([GARDEN])]
    for flt in (None, {}, {"person": "ann"}):
        got = [(h.document_id, h.score) for h in mem.recall([GARDEN], metadata_filter=flt)]
        assert got == ref
    allowed = [(h.document_id, h.score) for h in mem.recall(
        [GARDEN], metadata_filter={"_allow": {"audience_key": ["team-a", "team-b", ""]}})]
    assert allowed == ref
    assert _channel_docs(mem, GARDEN, None) == _channel_docs(mem, GARDEN, {})


def _rec(svc, refs):
    return svc.reconcile({"references": refs})


def test_reconcile_present_unchanged_absent_deleted_and_repeat(env):
    url, mem, ix = env
    svc = Service(mem, ix)
    _note(url, 1)
    _note(url, 2, gone="2026-02-01")
    r = _rec(svc, [{"kind": "note", "record_id": "1"}])
    assert (r["indexed"], r["unchanged"], r["removed"], r["unreadable"]) == (1, 0, 0, [])
    r = _rec(svc, [{"kind": "note", "record_id": "1"}])          # repeat: unchanged
    assert (r["indexed"], r["unchanged"], r["removed"]) == (0, 1, 0)
    _note(url, 1, rev=2, body="Changed text about rivers.")      # present and changed
    r = _rec(svc, [{"kind": "note", "record_id": "1"}])
    assert r["indexed"] == 1 and {rv for _, rv in _items(url, "1")} == {"2"}
    r = _rec(svc, [{"kind": "note", "record_id": "2"}, {"kind": "note", "record_id": "99"}])
    assert (r["indexed"], r["removed"], r["absent"]) == (0, 0, 2)  # deleted/absent, never indexed
    _q(url, "UPDATE host_app.notes SET gone = now() WHERE id = 1")   # deleted marker
    r = _rec(svc, [{"kind": "note", "record_id": "1"}])
    assert r["removed"] == 1 and _items(url, "1") == []
    assert _rec(svc, [{"kind": "note", "record_id": "1"}])["removed"] == 0   # repeat
    _note(url, 3)
    _rec(svc, [{"kind": "note", "record_id": "3"}])
    _q(url, "DELETE FROM host_app.notes WHERE id = 3")               # row absent
    assert _rec(svc, [{"kind": "note", "record_id": "3"}])["removed"] == 1
    assert _items(url, "3") == []


def test_reconcile_unreadable_and_validation(env):
    url, mem, ix = env
    svc = Service(mem, ix)
    r = _rec(svc, [{"kind": "note"}, {"kind": "note", "record_id": "7"}])
    assert len(r["unreadable"]) == 1 and r["absent"] == 1
    from prospecta.service import ApiError
    with pytest.raises(ApiError):
        svc.reconcile({"references": []})


def test_reconcile_accepts_thousands_of_references(env):
    url, mem, ix = env
    svc = Service(mem, ix)
    _note(url, 1)
    refs = [{"kind": "note", "record_id": "1"}] + [
        {"kind": "note", "record_id": str(1000 + i)} for i in range(1999)]
    r = _rec(svc, refs)
    total = r["indexed"] + r["unchanged"] + r["removed"] + r["absent"] + len(r["unreadable"])
    assert total == 2000 and r["indexed"] == 1 and r["absent"] == 1999


def test_reconcile_document_scope_replaces_pieces(env):
    url, mem, ix = env
    svc = Service(mem, ix)
    ref = {"kind": "note", "project": "p1", "title": "T"}
    _note(url, 1, body="Piece one about gardens.")
    _note(url, 2, body="Piece two about tea.")
    _note(url, 5, proj="other", body="Elsewhere.")
    r = _rec(svc, [ref])
    assert r["indexed"] == 1
    assert _q(url, "SELECT count(*) FROM prospecta_shared.source_refs")[0][0] == 1
    assert _rec(svc, [ref])["unchanged"] == 1                      # repeat
    rid = _q(url, "SELECT record_id FROM prospecta_shared.source_refs")[0][0]
    def texts():
        return " ".join(c for (c,) in _q(url, "SELECT content FROM prospecta_shared.memory_items "
                                              "WHERE kind='chunk' AND metadata->>'record_id' = %s", rid))
    assert "gardens" in texts() and "tea" in texts() and "Elsewhere" not in texts()
    _note(url, 2, rev=2, body="Piece two now about rivers.")        # one piece changes
    assert _rec(svc, [ref])["indexed"] == 1
    assert "rivers" in texts() and "tea" not in texts() and "gardens" in texts()
    _q(url, "UPDATE host_app.notes SET gone = now() WHERE id = 1")  # a piece is deleted
    assert _rec(svc, [ref])["indexed"] == 1
    assert "gardens" not in texts() and "rivers" in texts()
    _q(url, "UPDATE host_app.notes SET gone = now() WHERE id = 2")  # none left
    assert _rec(svc, [ref])["removed"] == 1
    assert _q(url, "SELECT count(*) FROM prospecta_shared.source_refs")[0][0] == 0
    _note(url, 5, proj="p1", aud="team-b")                          # mixed audiences
    _note(url, 6, proj="p1", aud="team-a")
    _q(url, "UPDATE host_app.notes SET title='M' WHERE id IN (5,6)")
    r = _rec(svc, [{"kind": "note", "project": "p1", "title": "M"}])
    assert r["indexed"] == 0 and len(r["unreadable"]) == 1


# -- concurrent indexing ----------------------------------------------------
import time  # noqa: E402

N_FAKE = 300
LLM_DELAY = 0.02   # stands in for the anticipated-question model call


def _slow_llm(messages, **k):
    text = json.dumps(messages)
    if "EXPLODE" in text:
        raise RuntimeError("model failed")
    time.sleep(LLM_DELAY)
    return "What is the first thing?\nWhy does the second thing matter?"


def _fake_records(url, n=N_FAKE, bad=()):
    with psycopg.connect(url, autocommit=True) as c:
        for i in range(n):
            body = f"Record {i} about topic {i % 7}.\n\nSecond paragraph {i}."
            if i in bad:
                body += " EXPLODE"
            c.execute("INSERT INTO host_app.notes VALUES (%s,1,'T',%s,'ann','p1','2026-01-02','team-a',%s)",
                      (i, body, "2026-02-01" if i % 50 == 49 else None))


def _indexer(url, bank, llm=_slow_llm, concurrency=1):
    mem = Memory(database_url=url, bank_id=bank, embed=stub_embed, llm=llm,
                 schema="prospecta_shared")
    mem.create_bank(bank, embedding_dim=EMBED_DIM)
    return mem, SourceIndexer(mem, SourceConfig("host_app", ("records",)), concurrency,
                              generate_questions=True)


def _snapshot(url, bank):
    return _q(url, "SELECT metadata->>'record_id', kind, content, ordinal FROM "
                   "prospecta_shared.memory_items WHERE bank_id = %s ORDER BY 1,2,3,4", bank)


def _refs(n=N_FAKE, extra=()):
    return [{"kind": "note", "record_id": str(i)} for i in range(n)] + list(extra)


def test_concurrent_reconcile_matches_sequential_and_is_faster(env):
    url, mem, ix = env
    _fake_records(url, bad={13, 200})
    refs = _refs(extra=[{"kind": "note"}, {"kind": "note", "record_id": "9999"}])
    mem1, ix1 = _indexer(url, "b1")
    mem2, ix2 = _indexer(url, "b8")
    try:
        t = time.perf_counter(); seq = Service(mem1, ix1).reconcile({"references": refs, "concurrency": 1})
        t_seq = time.perf_counter() - t
        t = time.perf_counter(); par = Service(mem2, ix2).reconcile({"references": refs, "concurrency": 8})
        t_par = time.perf_counter() - t
        print(f"\nreconcile {N_FAKE} records: concurrency 1 {t_seq:.2f}s, 8 {t_par:.2f}s, "
              f"speedup {t_seq / t_par:.1f}x")
        assert par == seq
        assert seq["indexed"] == N_FAKE - 2 - 6 and seq["absent"] == 7 and len(seq["unreadable"]) == 3
        assert [u["error"] for u in seq["unreadable"][:2]] == ["model failed"] * 2
        snap = _snapshot(url, "b1")
        assert snap and snap == _snapshot(url, "b8")
        assert t_par < t_seq
    finally:
        mem1.close()
        mem2.close()


def test_overlapping_requests_for_same_records(env):
    url, mem, ix = env
    _fake_records(url, n=60)
    mem2, ix2 = _indexer(url, "b")       # same bank, own Memory: a second "request"
    try:
        refs = _refs(60)
        out = {}

        def go(name, svc):
            out[name] = svc.reconcile({"references": refs, "concurrency": 8})
        ts = [threading.Thread(target=go, args=("a", Service(mem, ix))),
              threading.Thread(target=go, args=("b", Service(mem2, ix2)))]
        [t.start() for t in ts]; [t.join(60) for t in ts]
        assert not any(t.is_alive() for t in ts), "deadlock"
        assert out["a"]["unreadable"] == [] and out["b"]["unreadable"] == []
        assert out["a"]["indexed"] + out["b"]["indexed"] == 59   # record 49 is deleted in the view
        assert out["a"]["unchanged"] + out["b"]["unchanged"] == 59
        assert _q(url, "SELECT count(*) FROM prospecta_shared.documents")[0][0] == 59
        assert _q(url, "SELECT count(*) FROM prospecta_shared.source_refs")[0][0] == 59
        per = _q(url, "SELECT count(*) FROM prospecta_shared.memory_items GROUP BY "
                      "metadata->>'record_id'")
        assert len(per) == 59 and len({p[0] for p in per}) == 1
    finally:
        mem2.close()


def test_one_failing_record_does_not_stop_others(env):
    url, mem, ix = env
    _fake_records(url, n=40, bad={5})
    mem, ix = _indexer(url, "bf")
    r = Service(mem, ix).reconcile({"references": _refs(40), "concurrency": 8})
    assert r["indexed"] == 39 and len(r["unreadable"]) == 1
    assert r["unreadable"][0]["reference"] == {"kind": "note", "record_id": "5"}
    assert _q(url, "SELECT count(*) FROM prospecta_shared.source_refs")[0][0] == 39


def test_concurrency_setting_and_limits(env, monkeypatch):
    url, mem, ix = env
    svc = Service(mem, ix)
    for bad in (0, 33, "8", True, 1.5):
        with pytest.raises(Exception, match="concurrency must be"):
            svc.reconcile({"references": [{"kind": "note", "record_id": "1"}], "concurrency": bad})
    from prospecta.source import default_concurrency
    monkeypatch.delenv("PROSPECTA_INDEX_CONCURRENCY", raising=False)
    assert default_concurrency() == 8
    monkeypatch.setenv("PROSPECTA_INDEX_CONCURRENCY", "99")
    assert default_concurrency() == 32
    monkeypatch.setenv("PROSPECTA_INDEX_CONCURRENCY", "3")
    assert default_concurrency() == 3


def test_concurrent_reindex_matches_sequential(env):
    url, mem, ix = env
    _fake_records(url, n=60)
    mem1, ix = _indexer(url, "b1")
    mem2, ix2 = _indexer(url, "b8")
    try:
        a = ix.reindex_kind("note", 1)
        b = ix2.reindex_kind("note", 8)
        assert a == b and a["indexed"] == 59 and a["absent"] == 1
        assert _snapshot(url, "b1") == _snapshot(url, "b8")
    finally:
        mem1.close()
        mem2.close()


# ------------------------------------------------------------------ explain

def _channel_env(env):
    from prospecta.channels.registry import DEFAULT_CHANNEL_CONFIG
    url, mem, ix = env
    cfg = [dict(c) for c in DEFAULT_CHANNEL_CONFIG if c["name"] in ("dense_chunk", "question")]
    mem.set_channel_config(cfg)
    _note(url, 1, body="Gardens need water and sun.")
    ix.reindex_kind("note")
    return Service(mem, ix)


def test_explain_absent_response_is_byte_identical(env):
    svc = _channel_env(env)
    req = {"questions": ["what do gardens need"], "limit": 5}
    plain = json.dumps(svc.lookup(req), sort_keys=True)
    assert json.dumps(svc.lookup({**req, "explain": False}), sort_keys=True) == plain
    out = svc.lookup({**req, "explain": True})["results"][0]
    stripped = {"question": out["question"], "results": [
        {k: v for k, v in r.items() if k != "channels"} for r in out["results"]]}
    assert json.dumps({"results": [stripped]}, sort_keys=True) == plain
    assert "channels_run" not in json.loads(plain)["results"][0]


def test_explain_hit_found_by_two_channels_shows_both(env):
    svc = _channel_env(env)
    out = svc.lookup({"questions": ["what do gardens need"], "explain": True})["results"][0]
    assert sorted(out["channels_run"]) == ["dense_chunk", "question"]
    assert out["channels_failed"] == []
    hit = out["results"][0]
    assert {c["channel"] for c in hit["channels"]} == {"dense_chunk", "question"}
    w = {c["channel"]: c["weight"] for c in hit["channels"]}
    assert w == {"dense_chunk": 4.0, "question": 1.0}
    assert all(c["rank"] == 1 and isinstance(c["score"], float) for c in hit["channels"])


def test_explain_failed_and_skipped_channels_listed(env, monkeypatch):
    from prospecta.channels import semantic
    from prospecta.channels.registry import DEFAULT_CHANNEL_CONFIG
    svc = _channel_env(env)
    cfg = [dict(c) for c in DEFAULT_CHANNEL_CONFIG if c["name"] in ("dense_chunk", "question")]
    cfg.append({"name": "bm25", "enabled": False, "weight": 1})
    env[1].set_channel_config(cfg)
    monkeypatch.setattr(semantic.AnticipatedQuestions, "_sql", "SELECT broken FROM nowhere")
    out = svc.lookup({"questions": ["what do gardens need"], "explain": True})["results"][0]
    assert out["channels_run"] == ["dense_chunk"]
    failed = {f["channel"]: f["reason"] for f in out["channels_failed"]}
    assert failed["question"].startswith("failed:")
    assert failed["bm25"].startswith("skipped:") and "disabled" in failed["bm25"]
    assert [c["channel"] for c in out["results"][0]["channels"]] == ["dense_chunk"]


def test_explain_must_be_boolean(env):
    from prospecta.service import ApiError
    svc = _channel_env(env)
    with pytest.raises(ApiError):
        svc.lookup({"questions": ["a"], "explain": "yes"})


def test_coverage_document_scope_covered_stale_orphaned(env):
    url, mem, ix = env
    svc = Service(mem, ix)
    ref = {"kind": "note", "project": "p1", "title": "T"}
    _note(url, 1, body="Piece one about gardens.")
    _note(url, 2, body="Piece two about tea.")
    _rec(svc, [ref])
    rid = _q(url, "SELECT record_id FROM prospecta_shared.source_refs")[0][0]
    assert rid.startswith("doc:")

    def p1():
        return {c.project: c for c in ix.coverage("note")}["p1"]
    c = p1()                                                   # covered, not orphaned
    assert c.orphaned == [] and c.stale == []
    assert c.indexed >= 1
    _note(url, 2, rev=2, body="Piece two now about rivers.")   # a piece moved on
    c = p1()
    assert c.orphaned == []
    assert [s["record_id"] for s in c.stale] == [rid]
    _rec(svc, [ref])                                           # reconcile heals it
    c = p1()
    assert c.orphaned == [] and c.stale == []
    _q(url, "UPDATE host_app.notes SET gone = now() WHERE id IN (1,2)")
    assert [o["record_id"] for o in p1().orphaned] == [rid]    # no live rows: orphaned
