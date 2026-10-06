"""Hindsight import against a synthetic DB with the real Hindsight layout."""
from __future__ import annotations

import json
import time
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest

from prospecta import _import_hindsight as ih
from prospecta.memory import Memory
from tests import _hindsight_synth as synth
from tests._stub_embedder import EMBED_DIM, stub_embed

SEVENTEEN = {"banks", "documents", "chunks", "memory_units", "entities", "unit_entities",
             "entity_cooccurrences", "memory_links", "observation_history",
             "mental_model_history", "mental_models", "knowledge_pages",
             "invalidated_memory_units", "directives", "file_storage", "audit_log",
             "llm_requests"}
INFRA = {"alembic_version", "async_operations", "bank_stats_cache",
         "graph_maintenance_queue", "webhooks"}


def embed384(texts):
    calls.append(len(texts))
    return [[0.1] * synth.DIM for _ in texts]


calls: list[int] = []


@pytest.fixture
def hs_url(pg_url):
    name = f"hs_{int(time.time() * 1_000_000)}"
    with psycopg.connect(pg_url, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{name}"')
    url = urlunparse(urlparse(pg_url)._replace(path=f"/{name}"))
    synth.build(url)
    calls.clear()
    yield url
    with psycopg.connect(pg_url, autocommit=True) as c:
        c.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname=%s", (name,))
        c.execute(f'DROP DATABASE "{name}"')


def _n(m, sql, *a):
    with m._pool.cursor() as cur:
        cur.execute(sql, a)
        return cur.fetchone()[0]


def test_every_hindsight_table_is_carried_or_explained():
    assert set(ih.CARRIED) | set(ih.EDGES) | {"memory_units"} == SEVENTEEN
    assert set(ih.NOT_CARRIED) == INFRA


def test_import_carries_embeddings_and_is_idempotent(fresh_db, hs_url):
    m = Memory(database_url=fresh_db, bank_id="alpha", embed=embed384, llm=None)
    r = ih.import_bank(m, hs_url, "alpha", batch=100)
    n_units, n_links = synth.BANKS["alpha"]
    assert r.ok, r.failures
    t = r.tables
    assert (t["memory_units"].total, t["memory_units"].imported) == (n_units, n_units)
    assert (t["memory_links"].total, t["memory_links"].imported) == (n_links, n_links)
    assert t["unit_entities"].imported == t["unit_entities"].total > 0
    assert t["entity_cooccurrences"].imported == 29
    assert t["documents"].imported == 6 and t["file_storage"].imported == 6
    # embeddings: 6 units (i % 25 == 0) had none -> re-embedded; the rest carried verbatim
    assert r.embeddings == {"carried": n_units - 6, "re_embedded": 6}
    assert sum(calls) == 6
    for tbl in SEVENTEEN | INFRA:
        c = t[tbl]
        assert c.total == c.imported + c.skipped_duplicate + c.failed + c.not_carried, tbl
    assert all(t[x].not_carried == t[x].total for x in INFRA)
    assert t["webhooks"].total == 1 and _n(m, "SELECT count(*) FROM imported_records WHERE data::text LIKE '%%s3cret%%'") == 0

    # duplicate text did not lose a unit; provenance + timestamps + entities preserved
    assert _n(m, "SELECT count(*) FROM memory_items WHERE bank_id='alpha'") == n_units
    with m._pool.cursor() as cur:
        cur.execute("SELECT tags, document_metadata, created_at FROM documents "
                    "WHERE source = %s", (f"hindsight:{synth._u('unit-alpha-5')}",))
        tags, meta, created = cur.fetchone()
    assert "hindsight" in tags and "fact_type:observation" in tags
    assert any(x.startswith("entity:entity-") for x in tags)
    assert meta["hindsight"]["document_id"] == "doc-alpha-5" and created.day == 6
    assert meta["hindsight_entities"] == ["Entity 5 alpha"]

    before = (_n(m, "SELECT count(*) FROM documents"), _n(m, "SELECT count(*) FROM imported_edges"),
              _n(m, "SELECT count(*) FROM imported_records"))
    r2 = ih.import_bank(m, hs_url, "alpha", batch=100)
    after = (_n(m, "SELECT count(*) FROM documents"), _n(m, "SELECT count(*) FROM imported_edges"),
             _n(m, "SELECT count(*) FROM imported_records"))
    assert before == after
    assert r2.tables["memory_units"].skipped_duplicate == n_units
    assert r2.tables["memory_links"].skipped_duplicate == n_links
    assert sum(calls) == 6 and r2.ok


def test_reembeds_when_bank_dim_differs(fresh_db, hs_url):
    m = Memory(database_url=fresh_db, bank_id="beta32", embed=stub_embed, llm=None)
    m.create_bank("beta32", embedding_dim=EMBED_DIM)
    r = ih.import_bank(m, hs_url, "beta", batch=16)
    n = synth.BANKS["beta"][0]
    assert r.ok and r.embeddings == {"carried": 0, "re_embedded": n}
    assert m.bank_stats("beta32").memory_items == n


def test_failures_are_reported_not_dropped(fresh_db, hs_url):
    def boom(texts):
        raise RuntimeError("embedder down")
    m = Memory(database_url=fresh_db, bank_id="beta", embed=boom, llm=None)
    r = ih.import_bank(m, hs_url, "beta", batch=16)
    u = r.tables["memory_units"]
    n = synth.BANKS["beta"][0]
    assert u.failed == 2 and u.imported == n - 2 and not r.ok  # i%25==0 -> 2 units
    assert len(r.failures) == 2 and "embedder down" in r.failures[0]["reason"]
    assert u.total == u.imported + u.skipped_duplicate + u.failed


def test_cli(fresh_db, hs_url, monkeypatch, capsys):
    from prospecta.cli import _common
    from prospecta.cli.__main__ import main
    monkeypatch.setattr(_common, "make_memory", lambda a: Memory(
        database_url=fresh_db, bank_id=a.bank, embed=embed384, llm=None))
    assert main(["import", "hindsight", hs_url]) == 1  # two banks, none chosen
    assert main(["--bank", "alpha", "import", "hindsight", hs_url, "--hindsight-bank", "alpha",
                 "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out[0]["tables"]["memory_units"]["imported"] == synth.BANKS["alpha"][0]
    assert main(["import", "hindsight", hs_url, "--all-banks"]) == 0
