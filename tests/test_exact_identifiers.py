"""An exact identifier in the question comes back first.

The acceptance test of a hybrid memory: ask for `deliverable-<id>` and the
record that carries that id is the first answer, not a topical neighbour. The
fixtures are invented but shaped like the records of a live shared-mode bank
(the Spire memory service): files named `deliverable-<hex>.pr`, commissions
whose record id (`commission:ask_<uuid>:0`) is in no text, project names that
sit inside file paths, and prose that mentions a script by its path.

The dense side is a topical embedder that cannot see identifiers (it hashes
only words of four letters or more, as a real embedder treats a hex string as
noise), and every identifier has many decoys that are closer to the question
than the real record is. The service path is the one Spire calls: `/lookup`
with an allowed-audience list.
"""
from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest

from prospecta.db.migrate import run_migrations
from prospecta.memory import Memory
from prospecta.service import Service
from prospecta.source import SourceConfig, SourceIndexer
from tests._stub_embedder import EMBED_DIM
from tests.conftest import _normalize_url

SCHEMA = "prospecta_shared"
AUD = "team-a"


def topic_embed(texts: list[str]) -> list[list[float]]:
    """Dense neighbours by topic only: words of four letters or more. An
    identifier (a hex id, a number, a short token) adds nothing to the vector."""
    out = []
    for text in texts:
        vec = [0.0] * EMBED_DIM
        for w in re.findall(r"[a-z]{4,}", text.lower()):
            h = hashlib.sha256(w.encode()).digest()
            vec[h[0] % EMBED_DIM] += 1.0 if h[1] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        out.append([v / norm for v in vec] if norm else [1.0] + [0.0] * (EMBED_DIM - 1))
    return out


def _hex(i: int, n: int = 12) -> str:
    return hashlib.sha1(f"decoy-{i}".encode()).hexdigest()[:n]


@dataclass(frozen=True)
class Rec:
    kind: str
    record_id: str
    title: str | None
    text: str
    aud: str = AUD


EXTRA = " This also records the sign-off evidence collected during the audit."
RECORDS: list[Rec] = []
TARGET: dict[str, str] = {}       # query -> the record id that must come first


def _add(kind, rid, title, text, aud=AUD):
    RECORDS.append(Rec(kind, rid, title, text, aud))


def _file(rid, filename, purpose=None):
    _add("file", rid, filename, "\n\n".join(x for x in (filename, purpose) if x))


# 1. A file named deliverable-<hex>.pr. The text holds the name as one
#    host-shaped token, so a full-text query for the id parts finds nothing.
for i in range(24):
    _file(f"fact_f{i}", f"deliverable-{_hex(i)}.pr")
_file("fact_target_file", "deliverable-7f3a91c04b2e.pr", "Closing notes for the quarterly venue review")
TARGET["deliverable-7f3a91c04b2e"] = "fact_target_file"
TARGET["7f3a91c04b2e"] = "fact_target_file"          # the bare hex id

# 2. A commission whose id is its record id and appears in no text.
for i in range(24):
    uid = f"{_hex(i, 8)}-{_hex(i + 50, 4)}-4{_hex(i + 60, 3)}-8{_hex(i + 70, 3)}-{_hex(i + 80, 12)}"
    _add("commission", f"commission:ask_{uid}:0", "Commission: Rework the intake review",
         "Review the intake flow.\n\nGoal: faster intake for every venue.")
ASK = "ask_0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"
_add("commission", f"commission:{ASK}:0", "Commission: Rework the intake review",
     "Review the intake flow.\n\nGoal: faster intake for every venue.\n\nScope: onboarding, "
     "documents, approvals, notifications and the weekly digest." + EXTRA)
TARGET[ASK] = f"commission:{ASK}:0"

# 3. A project name that sits inside file paths.
for i in range(24):
    _add("passage", f"pass_s{i}", f"docs/stack-{i}.md",
         f"Stack notes: the compose file lives in /srv/tower-{1000 + i}/compose.yml.")
_add("passage", "pass_target_stack", "docs/stack-x.md",
     "Stack notes: the compose file lives in /srv/tower-4242/compose.yml and the data volume "
     "is mounted read only." + EXTRA)
TARGET["tower-4242"] = "pass_target_stack"

# 4. A script mentioned by its path (a file-shaped token), among near names.
for i in range(24):
    _add("passage", f"pass_a{i}", f"docs/audit-{i}.md", f"Run scripts/audit-ledgers{i}.py before the job.")
_add("passage", "pass_target_audit", "docs/audit-x.md",
     "Run scripts/audit-ledger.py before the nightly job; it reconciles balances and prints "
     "a summary table." + EXTRA)
TARGET["audit-ledger.py"] = "pass_target_audit"

# 5. Plain hyphenated words, which the full-text leg does find.
for i in range(24):
    _add("note", f"note_m{i}", f"Sweeper {i}", f"Sweeper notes {i}: the cleanup jobs run nightly.")
_add("note", "note_target_sweeper", "Sweeper",
     "The mesh-docker-sweeper removes stale containers every night." + EXTRA)
TARGET["mesh-docker-sweeper"] = "note_target_sweeper"

HEADLINE = ["deliverable-7f3a91c04b2e", ASK, "tower-4242", "audit-ledger.py", "mesh-docker-sweeper"]

# Ten more shapes of rare token, each with six near-miss decoys that are denser
# in the question's topic than the real record. (query, text around the token,
# a decoy token for index i).
SHAPES: list[tuple[str, str, str, "callable"]] = [
    ("file+hex+ext", "review-bundle-3d9f02ae17c5", "Archive {t}.zip",
     lambda i: f"review-bundle-{_hex(i)}"),
    ("basename in a path", "loader.py", "Edit frameworks/core/{t} before release",
     lambda i: f"loader{i}.py"),
    ("snake_case", "test_stale_bridge_closes_nothing", "Failing test: {t}",
     lambda i: f"test_stale_bridge_closes_nothin{i}"),
    ("anchor inside a url", "pullrequestreview-5405358",
     "See https://forge.example.org/acme/app/pull/41#{t} for the thread",
     lambda i: f"pullrequestreview-54053{i}0"),
    ("ticket number", "ITEM-2001", "Tracked as {t} in the backlog", lambda i: f"ITEM-20{i}2"),
    ("dated file", "e3h-k1-20261003.md", "Notes live in logs/{t} for that run",
     lambda i: f"e3h-k1-2026110{i}.md"),
    ("whole path", "experiments/position-20261008.md", "Result written to {t}",
     lambda i: f"experiments/position-2026110{i}.md"),
    ("dotted name", "arm.prefill_parts", "Call {t}() once per batch", lambda i: f"arm.prefill_part{i}"),
    ("short hex", "415c8e0acd", "Fixed in commit {t} on main", lambda i: _hex(i, 10)),
    ("hyphen words", "mechanism-relocates-upward", "The {t} effect shows up late",
     lambda i: f"mechanism-relocates-upwar{i}"),
]
for n, (shape, token, frame, decoy) in enumerate(SHAPES):
    for i in range(6):
        _add("passage", f"pass_{n}_d{i}", f"docs/shape-{n}-{i}.md", frame.format(t=decoy(i)) + ".")
    _add("passage", f"pass_{n}_target", f"docs/shape-{n}-t.md", frame.format(t=token) + "." + EXTRA)
    TARGET[token] = f"pass_{n}_target"

# The identifier as the tail of a longer word is not the identifier.
_add("passage", "pass_reloader", "docs/reloader.md", "Edit frameworks/core/reloader.py before release.")

# A token that is in too many records to be an identifier of one of them.
for i in range(60):
    _add("note", f"note_c{i}", f"Nightly {i}", f"Nightly batch {i} wrote nightly-batch-77 to the log.")

# Two identifiers: one record holds one, another the other, a third both.
_add("passage", "pass_zone", "docs/zone.md", "The relay sits in zone-5150 and is checked weekly." + EXTRA)
_add("passage", "pass_port", "docs/port.md", "The relay listens on gate-8080 and is checked weekly." + EXTRA)
_add("passage", "pass_both", "docs/both.md", "Move the relay from zone-5150 to gate-8080." + EXTRA)
# An identifier in another audience.
_add("note", "note_other_team", "Plan", "The plan is kept as vault-plan-8812 for the other team.",
     aud="team-b")


def _hits(svc: Service, question: str, limit: int = 5, audience=(AUD,), explain=False) -> list[dict]:
    body = {"questions": [question], "limit": limit, "allowed_audience_keys": list(audience)}
    if explain:
        body["explain"] = True
    return svc.lookup(body)["results"][0]["results"]


def _ids(svc, question, **kw) -> list[str]:
    return [h["record_id"] for h in _hits(svc, question, **kw)]


def _rank(svc, question: str, record_id: str, limit: int = 100) -> int | None:
    ids = _ids(svc, question, limit=limit)
    return ids.index(record_id) + 1 if record_id in ids else None


def _gate_off(monkeypatch) -> None:
    """The lookup as it was before the exact-identifier gate."""
    try:
        import prospecta._exact as exact
    except ImportError:          # before the fix there is no gate to turn off
        return
    monkeypatch.setattr(exact, "find_exact", lambda *a, **k: [])


@pytest.fixture(scope="module")
def env(pg_container):
    base = _normalize_url(pg_container.get_connection_url())
    name = f"test_exact_{int(time.time() * 1_000_000)}"
    with psycopg.connect(base, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{name}"')
    url = urlunparse(urlparse(base)._replace(path=f"/{name}"))
    with psycopg.connect(url, autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        c.execute("CREATE SCHEMA host_app")
        c.execute("CREATE TABLE host_app.records (kind text, record_id text, revision text, "
                  "title text, text text, person text, project text, created_date date, "
                  "audience_key text, deleted_at timestamptz)")
        with c.cursor() as cur:
            cur.executemany(
                "INSERT INTO host_app.records VALUES (%s, %s, '1', %s, %s, NULL, 'p1', "
                "'2026-01-02', %s, NULL)",
                [(r.kind, r.record_id, r.title, r.text, r.aud) for r in RECORDS])
    run_migrations(url, schema=SCHEMA)
    mem = Memory(database_url=url, bank_id="b", embed=topic_embed, schema=SCHEMA,
                 llm=lambda messages, **k: "What is the first thing?\nWhy does the second thing matter?")
    mem.create_bank("b", embedding_dim=EMBED_DIM)
    ix = SourceIndexer(mem, SourceConfig("host_app", ("records",)))
    for kind in {r.kind for r in RECORDS}:
        ix.reindex_kind(kind)
    yield Service(mem, ix), mem
    mem.close()
    with psycopg.connect(base, autocommit=True) as c:
        c.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                  "WHERE datname = %s AND pid <> pg_backend_pid()", (name,))
        c.execute(f'DROP DATABASE "{name}"')


# ------------------------------------------------------------ the acceptance

@pytest.mark.parametrize("question", HEADLINE + ["7f3a91c04b2e"])
def test_the_headline_queries_find_their_record_first(env, question):
    svc, _ = env
    assert _ids(svc, question)[0] == TARGET[question], _hits(svc, question)


@pytest.mark.parametrize("shape,token", [(s[0], s[1]) for s in SHAPES])
def test_ten_more_shapes_of_rare_token_come_back_first(env, shape, token):
    svc, _ = env
    assert _ids(svc, token)[0] == TARGET[token], (shape, _hits(svc, token))


def test_the_fixtures_reproduce_the_misses_without_the_gate(env, monkeypatch):
    """The precondition: without the gate the first four are missed (not even in
    the answer) or buried; the plain hyphenated words are found, as they were."""
    _gate_off(monkeypatch)
    svc, _ = env
    for question in ("deliverable-7f3a91c04b2e", "7f3a91c04b2e", ASK, "tower-4242", "audit-ledger.py"):
        assert _ids(svc, question)[0] != TARGET[question], question
    assert _rank(svc, "mesh-docker-sweeper", TARGET["mesh-docker-sweeper"]) <= 2


def test_ranks_before_and_after(env, monkeypatch, capsys):
    """Not an assertion of the numbers (the gate-off ranks depend on the stub
    embedder): it prints the rank of each record, gate off and on (-s)."""
    svc, _ = env
    after = {q: _rank(svc, q, TARGET[q]) for q in TARGET}
    _gate_off(monkeypatch)
    before = {q: _rank(svc, q, TARGET[q]) for q in TARGET}
    with capsys.disabled():
        print("\nrank of the right record (limit 100): query | before | after")
        for q in TARGET:
            print(f"  {q} | {before[q] or 'not found'} | {after[q]}")
    assert all(r == 1 for r in after.values())


def test_the_answer_says_why_a_hit_is_first(env):
    svc, _ = env
    out = svc.lookup({"questions": ["deliverable-7f3a91c04b2e"], "limit": 3,
                      "allowed_audience_keys": [AUD], "explain": True})["results"][0]
    top = out["results"][0]
    assert top["record_id"] == "fact_target_file"
    assert any(c["channel"] == "exact" and c["score"] == 1.0 for c in top["channels"])
    assert "exact" in out["channels_run"]


# --------------------------------------------------- what the gate leaves alone

def test_a_question_without_an_identifier_is_unchanged(env, monkeypatch):
    svc, _ = env
    q = "what did the intake review change for every venue"
    after = _hits(svc, q, limit=10)
    _gate_off(monkeypatch)
    assert after == _hits(svc, q, limit=10)
    assert all("exact" not in h for h in after)


def test_a_token_in_too_many_records_is_not_an_identifier(env, monkeypatch):
    svc, _ = env
    from prospecta import _exact
    with psycopg.connect(svc.memory.database_url) as conn:
        conn.execute(f'SET search_path = "{SCHEMA}", public')
        assert _exact.find_exact(conn, "b", "where is nightly-batch-77") == []
        assert _exact.find_exact(conn, "b", "where is nightly-batch-79") == []   # in none


def test_two_identifiers_rank_the_record_with_both_first(env):
    svc, _ = env
    ids = _ids(svc, "move it from zone-5150 to gate-8080", limit=5)
    assert ids[0] == "pass_both" and set(ids[1:3]) == {"pass_zone", "pass_port"}
    assert _ids(svc, "zone-5150")[0] in {"pass_zone", "pass_both"}


def test_an_identifier_in_another_audience_is_never_returned(env):
    svc, _ = env
    assert "note_other_team" not in _ids(svc, "vault-plan-8812", limit=100)
    assert "note_other_team" not in _ids(svc, "vault-plan-8812", limit=100, audience=("team-a", "team-c"))
    assert _ids(svc, "vault-plan-8812", audience=("team-b",))[0] == "note_other_team"
    assert _ids(svc, "vault-plan-8812", audience=()) == []


def test_a_longer_word_is_not_the_identifier_and_case_does_not_matter(env):
    svc, _ = env
    from prospecta import _exact
    with psycopg.connect(svc.memory.database_url) as conn:
        conn.execute(f'SET search_path = "{SCHEMA}", public')
        loader = _exact.find_exact(conn, "b", "loader.py")
        assert [h.source for h in loader] == ["passage:pass_1_target"]       # not reloader.py
        assert [h.source for h in _exact.find_exact(conn, "b", "ITEM-2001")] == \
            [h.source for h in _exact.find_exact(conn, "b", "item-2001")] == ["passage:pass_4_target"]


def test_a_failing_lookup_never_sinks_the_recall(env, monkeypatch, caplog):
    svc, _ = env
    from prospecta import _exact
    real = _exact._find

    def boom(*a, **k):
        raise RuntimeError("no table")
    monkeypatch.setattr(_exact, "_find", boom)
    broken = _hits(svc, "tower-4242", limit=5)
    assert "exact-identifier lookup failed" in caplog.text
    monkeypatch.setattr(_exact, "_find", real)
    _gate_off(monkeypatch)
    assert broken == _hits(svc, "tower-4242", limit=5)           # the other legs answered as they do


def test_the_record_id_alone_is_enough(env):
    """A record id lives in documents.source, never in the text: it still counts."""
    svc, _ = env
    with psycopg.connect(svc.memory.database_url) as conn:
        conn.execute(f'SET search_path = "{SCHEMA}", public')
        n = conn.execute("SELECT count(*) FROM memory_items WHERE original_chunk ILIKE %s",
                         (f"%{ASK}%",)).fetchone()[0]
    assert n == 0
    assert _ids(svc, f"what is {ASK}?")[0] == f"commission:{ASK}:0"


# -------------------------------------------------------- the configured path

def test_a_bank_with_a_channel_config_gets_the_same_gate(env):
    svc, mem = env
    mem.set_channel_config([{"name": "dense_chunk", "weight": 4}, {"name": "question", "weight": 1}])
    try:
        for q in HEADLINE:
            assert _ids(svc, q)[0] == TARGET[q], q
        out = svc.lookup({"questions": ["tower-4242"], "limit": 3, "allowed_audience_keys": [AUD],
                          "explain": True})["results"][0]
        assert any(c["channel"] == "exact" for c in out["results"][0]["channels"])
        assert "exact" in out["channels_run"]
    finally:
        mem.set_channel_config([])


# ---------------------------------------------------------------- pure parts

def test_identifier_tokens():
    from prospecta._exact import MAX_TOKENS, identifier_tokens as ids
    assert ids("deliverable-7f3a91c04b2e") == ["deliverable-7f3a91c04b2e"]
    assert ids("what is (deliverable-7f3a91c04b2e)?") == ["deliverable-7f3a91c04b2e"]
    assert ids('"audit-ledger.py", tower-4242; 7f3a91c04b2e.') == ["audit-ledger.py", "tower-4242", "7f3a91c04b2e"]
    assert ids("ask_0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d") == ["ask_0a1b2c3d-4e5f-4a6b-8c7d-9e0f1a2b3c4d"]
    assert ids("see src/core/loader.py:16 and v2.14.1") == ["src/core/loader.py:16", "v2.14.1"]
    assert ids("Tower-4242 tower-4242 TOWER-4242") == ["Tower-4242"]            # once, any case
    assert ids("123456") == ["123456"]
    # prose and short or single-word tokens are not identifiers
    assert ids("what did the intake review change?") == []
    assert ids("e.g. a v2 plan, 2024 or 12345 items, a-b") == []
    assert ids("") == []
    assert len(ids(" ".join(f"item-{n}000" for n in range(10)))) == MAX_TOKENS


def test_sql_patterns_escape_like_and_regex():
    from prospecta._exact import like_pattern, regex_pattern
    assert like_pattern("ask_f6%") == r"%ask\_f6\%%"
    assert regex_pattern("a.b-c") == r"(^|[^[:alnum:]])a\.b\-c($|[^[:alnum:]])"


def test_promote_rows_puts_exact_documents_first_by_tokens_then_rank():
    from prospecta._exact import ExactHit, promote_rows

    def row(doc, rrf):
        return {"id": f"i{doc}", "document_id": doc, "content": "c", "original_chunk": "o",
                "source": f"s{doc}", "metadata": {}, "tags": [], "sem_score": 0.0,
                "lex_content_score": 0.0, "lex_body_score": 0.0, "rrf_score": rrf}

    def hit(doc, *tokens):
        return ExactHit(doc, f"x{doc}", f"s{doc}", "c", "o", {}, [], tuple(tokens))

    rows = [row("a", 0.03), row("b", 0.02), row("c", 0.01)]
    before = [dict(r) for r in rows]
    out = promote_rows(rows, [hit("c", "t1"), hit("z", "t1", "t2"), hit("b", "t1")])
    assert rows == before                                   # the input is not touched
    assert [r["document_id"] for r in out] == ["z", "b", "c", "a"]
    assert [r.get("exact") for r in out] == [2.0, 1.0, 1.0, None]
    assert out[0]["id"] == "xz" and out[0]["source"] == "sz" and out[0]["rrf_score"] == 0.0
    # a gated row is scored above every ungated one, whatever the scale of the scores
    gated = [r["rrf_score"] + r["gate_boost"] for r in out[:3]]
    assert gated == sorted(gated, reverse=True) and min(gated) > max(r["rrf_score"] for r in rows)
    assert promote_rows(rows, []) == rows
