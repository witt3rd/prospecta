"""A bank created fresh runs the hybrid architecture out of the box; an existing bank is left alone.

New banks (Memory.create_bank, hence `prospecta create-bank`, `prospecta init`, the service and
`migrate-bank`) store channel_config = [hybrid_chunk] (Stage 1) and recall_config = a cross_encoder
rerank (Stage 2): no manual configuration. link_config is not written (nothing reads it). Existing banks
keep exactly what they stored: the defaults live in the INSERT of create_bank, never in a migration and
never in a column DEFAULT. Invented text, stub models."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import psycopg
import pytest

from prospecta import bank_defaults
from prospecta.channels import validate_channel_config
from prospecta.cli import _common
from prospecta.db.migrate import run_migrations
from prospecta.memory import Memory
from prospecta.stages import (
    RERANKER_MODEL, CrossEncoderReranker, build_reranker, resolve_reranker_model, validate_recall_config,
)
from tests._stub_embedder import EMBED_DIM, stub_embed

MIGRATIONS = Path(__file__).resolve().parent.parent / "prospecta" / "db" / "migrations"
HYBRID = [{"name": "hybrid_chunk", "enabled": True, "weight": 1, "params": {}}]
CROSS_ENCODER = {"rerank": {"enabled": True, "stage": "cross_encoder"}}

# The four banks as they were stored: augur and forge hold the column defaults; augur-v2 and forge-v2
# were set by hand to hybrid_chunk + cross_encoder.
HAND_SET_RECALL = {"rerank": {"enabled": True, "stage": "cross_encoder", "model": RERANKER_MODEL,
                              "top_n": 10, "min_score": 0.15, "timeout_s": 2.5,
                              "deduplicate_near_copies": True}}
LIVE_BANKS = {
    "augur": ([], {}, {}),
    "forge": ([], {}, {}),
    "augur-v2": (HYBRID, HAND_SET_RECALL, {}),
    "forge-v2": (HYBRID, HAND_SET_RECALL, {}),
}


def config_of(url: str, bank: str) -> tuple:
    with psycopg.connect(url) as c:
        return c.execute("SELECT channel_config, recall_config, link_config FROM banks "
                         "WHERE bank_id = %s", (bank,)).fetchone()


def whole_row(url: str, bank: str) -> tuple:
    with psycopg.connect(url) as c:
        return c.execute("SELECT * FROM banks WHERE bank_id = %s", (bank,)).fetchone()


def insert_stored_bank(url: str, bank: str, channel, recall, link) -> None:
    """A bank as an older install stored it: a raw INSERT, create_bank not involved."""
    with psycopg.connect(url) as c:
        c.execute("INSERT INTO banks (bank_id, embedding_dim, channel_config, recall_config, "
                  "link_config) VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb)",
                  (bank, EMBED_DIM, json.dumps(channel), json.dumps(recall), json.dumps(link)))


@pytest.fixture
def mem(fresh_db):
    m = Memory(database_url=fresh_db, bank_id="fresh", embed=stub_embed)
    yield m
    m.close()


class FakeRerank:
    """A rerank transport: every document scores 0.9, in request order."""

    def __init__(self):
        self.requests = []

    def __call__(self, req, timeout):
        self.requests.append(req)
        return {"model": req["model"], "usage": {"total_tokens": 5, "cost": 0.0},
                "results": [{"index": i, "relevance_score": 0.9}
                            for i in range(len(req["documents"]))]}


# ---------------------------------------------------------------- the new defaults

def test_the_default_configs_are_valid():
    validate_channel_config(bank_defaults.NEW_BANK_CHANNEL_CONFIG)
    validate_recall_config(bank_defaults.NEW_BANK_RECALL_CONFIG)
    assert not hasattr(bank_defaults, "NEW_BANK_LINK_CONFIG")
    assert set(bank_defaults.new_bank_configs()) == {"channel_config", "recall_config"}


def test_a_fresh_bank_stores_hybrid_chunk_and_cross_encoder_and_no_link_config(mem, fresh_db):
    mem.create_bank("fresh", embedding_dim=EMBED_DIM)
    channel, recall, link = config_of(fresh_db, "fresh")
    assert channel == HYBRID
    assert recall == CROSS_ENCODER
    assert link == {}                 # the column default: no code reads or writes link_config


def test_the_stage_2_model_of_a_fresh_bank_is_the_default_cross_encoder(mem, fresh_db, monkeypatch):
    monkeypatch.delenv("PROSPECTA_RERANKER_MODEL", raising=False)
    mem.create_bank("fresh", embedding_dim=EMBED_DIM)
    recall = config_of(fresh_db, "fresh")[1]
    stage = build_reranker(recall, rerank_transport=FakeRerank())
    assert isinstance(stage, CrossEncoderReranker)
    assert stage.model == RERANKER_MODEL == "qwen/qwen3-reranker-8b"


def test_the_environment_still_chooses_the_reranker_of_a_fresh_bank(mem, fresh_db, monkeypatch):
    """The bank does not pin the model, so PROSPECTA_RERANKER_MODEL keeps working."""
    monkeypatch.setenv("PROSPECTA_RERANKER_MODEL", "acme/other-reranker")
    mem.create_bank("fresh", embedding_dim=EMBED_DIM)
    recall = config_of(fresh_db, "fresh")[1]
    stage = build_reranker(recall, rerank_transport=FakeRerank())
    assert stage.model == "acme/other-reranker"
    assert resolve_reranker_model(None, recall["rerank"].get("model")) == "acme/other-reranker"


def test_each_new_bank_gets_its_own_copy(mem, fresh_db):
    mem.create_bank("one", embedding_dim=EMBED_DIM)
    bank_defaults.NEW_BANK_CHANNEL_CONFIG[0]["weight"] = 99      # a careless caller
    try:
        mem.create_bank("two", embedding_dim=EMBED_DIM)
        assert config_of(fresh_db, "one")[0] == HYBRID
    finally:
        bank_defaults.NEW_BANK_CHANNEL_CONFIG[0]["weight"] = 1


def test_a_fresh_bank_answers_stage_1_filters_and_runs_stage_2_without_setup(fresh_db):
    rerank = FakeRerank()
    m = Memory(database_url=fresh_db, bank_id="fresh", embed=stub_embed, rerank_transport=rerank)
    try:
        m.create_bank("fresh", embedding_dim=EMBED_DIM)
        m.retain("The harbour lamps get a warm filter.", source="lamps.md",
                 metadata={"created_on": "2024-03-05"})
        m.retain("Orchids need humid air.", source="orchid.md", metadata={"created_on": "2024-01-02"})
        got = m.search("harbour lamps", limit=5, date_from="2024-03-01", date_to="2024-03-31",
                       excluded_sources=["orchid.md"])
        assert [r.source for r in got] == ["lamps.md"]
        assert got[0].match_kind is not None            # the hybrid_chunk channel found it
        assert rerank.requests and got[0].scores.get("rerank") == pytest.approx(0.9)
    finally:
        m.close()


def test_mode_1_works_on_a_fresh_bank(fresh_db):
    writer_calls = []

    def writer(messages, *, json_mode=False, model=None):
        writer_calls.append(model)
        return json.dumps({"searches": [{"query": "harbour lamps", "terms": [],
                                         "date_from": None, "date_to": None}]})

    m = Memory(database_url=fresh_db, bank_id="fresh", embed=stub_embed, llm=writer,
               rerank_transport=FakeRerank())
    try:
        m.create_bank("fresh", embedding_dim=EMBED_DIM)
        m.retain("The harbour lamps get a warm filter.", source="lamps.md")
        got = m.expand_and_recall("what about the harbour lamps?", gate=False)
        assert got.error is None, got.error
        assert [i.source for i in got.items] == ["lamps.md"]
    finally:
        m.close()


# ---------------------------------------------------------------- every path that creates a bank

def test_create_bank_cli_gives_the_new_defaults(fresh_db, capsys):
    from prospecta.cli.bank import run_create_bank
    args = argparse.Namespace(database_url=fresh_db, id="cli-bank", embedding_dim=EMBED_DIM,
                              embedding_model_id=None, mission=None, retain_mission=None)
    assert run_create_bank(args) == 0
    assert config_of(fresh_db, "cli-bank") == (HYBRID, CROSS_ENCODER, {})


def test_init_creates_the_default_bank_with_the_new_defaults(fresh_db):
    from prospecta.cli.init import cmd_init
    args = argparse.Namespace(database_url=fresh_db, no_substrate=True, embedding_dim=EMBED_DIM,
                              substrate_timeout=5)
    assert cmd_init(args) == 0
    assert config_of(fresh_db, "default") == (HYBRID, CROSS_ENCODER, {})


def test_the_service_creates_its_bank_with_the_new_defaults(fresh_db, monkeypatch):
    from prospecta import service
    monkeypatch.setenv("DATABASE_URL", fresh_db)
    monkeypatch.setenv("PROSPECTA_BANK", "svc")
    monkeypatch.setenv("PROSPECTA_EMBEDDING_DIM", str(EMBED_DIM))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(_common, "_resolve_embedder", lambda dim=None: stub_embed)
    svc, _ = service.build_service()
    try:
        assert config_of(fresh_db, "svc") == (HYBRID, CROSS_ENCODER, {})
    finally:
        svc.memory.close()


def test_the_migrate_bank_target_is_a_new_bank(fresh_db):
    from prospecta._embed_migrate import migrate_bank
    m = Memory(database_url=fresh_db, bank_id="old", embed=stub_embed)
    try:
        insert_stored_bank(fresh_db, "old", [], {}, {})
        migrate_bank(m, "old", target_bank="old-v2", embed=stub_embed, embedding_dim=EMBED_DIM)
        assert config_of(fresh_db, "old-v2") == (HYBRID, CROSS_ENCODER, {})
        assert config_of(fresh_db, "old") == ([], {}, {})        # the source is not touched
    finally:
        m.close()


# ---------------------------------------------------------------- existing banks are not changed

@pytest.mark.parametrize("bank", sorted(LIVE_BANKS))
def test_create_bank_on_an_existing_bank_changes_nothing(mem, fresh_db, bank):
    channel, recall, link = LIVE_BANKS[bank]
    insert_stored_bank(fresh_db, bank, channel, recall, link)
    before = whole_row(fresh_db, bank)
    mem.create_bank(bank, embedding_dim=EMBED_DIM)                  # idempotent: a no-op
    mem.create_bank(bank, embedding_dim=EMBED_DIM, mission="a mission")
    assert whole_row(fresh_db, bank) == before                      # every column, updated_at too
    assert config_of(fresh_db, bank) == (channel, recall, link)


@pytest.fixture
def bank_db_at_0019(pg_container, tmp_path, monkeypatch):
    """A database migrated to 0019 only (everything before the newest migration), plus a way to
    run the rest: an install about to be upgraded."""
    import time
    from urllib.parse import urlparse, urlunparse
    from prospecta.db import migrate

    base_url = pg_container.get_connection_url().replace("+psycopg2", "").replace("+psycopg", "")
    name = f"test_upg_{int(time.time() * 1_000_000)}"
    with psycopg.connect(base_url, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{name}"')
    url = urlunparse(urlparse(base_url)._replace(path=f"/{name}"))
    with psycopg.connect(url, autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
    old = tmp_path / "migrations"
    old.mkdir()
    for f in migrate.MIGRATIONS_DIR.glob("*.sql"):
        if int(f.name[:4]) <= 19:
            (old / f.name).symlink_to(f)
    full = migrate.MIGRATIONS_DIR
    monkeypatch.setattr(migrate, "MIGRATIONS_DIR", old)
    migrate.run_migrations(url)
    monkeypatch.setattr(migrate, "MIGRATIONS_DIR", full)
    yield url
    with psycopg.connect(base_url, autocommit=True) as c:
        c.execute("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                  "WHERE datname = %s AND pid <> pg_backend_pid()", (name,))
        c.execute(f'DROP DATABASE "{name}"')


@pytest.mark.parametrize("bank", sorted(LIVE_BANKS))
def test_upgrading_an_install_leaves_its_banks_unchanged(bank_db_at_0019, bank):
    """The newest migrations run over a database that already holds the live banks: every column of
    every bank row, config and updated_at included, is the same afterwards."""
    url = bank_db_at_0019
    channel, recall, link = LIVE_BANKS[bank]
    insert_stored_bank(url, bank, channel, recall, link)
    before = whole_row(url, bank)
    applied = run_migrations(url)["applied"]
    assert applied and min(applied) == 20                              # the upgrade really ran
    assert whole_row(url, bank) == before
    assert config_of(url, bank) == (channel, recall, link)


def test_no_migration_writes_bank_config_or_changes_its_column_defaults():
    """The guarantee, in the source: a migration may ADD the config columns (0008, 0009, 0019, with
    the empty constants as defaults) and nothing else: no UPDATE/DELETE/INSERT of `banks` and no
    SET DEFAULT on a column. The new-bank defaults live in create_bank's INSERT."""
    offenders = []
    for path in sorted(MIGRATIONS.glob("*.sql")):
        sql = re.sub(r"--[^\n]*", "", path.read_text())
        for stmt in re.split(r";\s*\n", sql):
            s = " ".join(stmt.split()).lower()
            if re.search(r"\b(update|delete from|insert into|truncate)\s+(only\s+)?(public\.)?banks\b", s):
                offenders.append((path.name, s[:80]))
            if "alter table banks" in s and re.search(r"\b(set default|drop default|alter column)\b", s):
                offenders.append((path.name, s[:80]))
    assert offenders == []


def test_the_column_defaults_stay_empty_and_an_empty_config_runs_the_hybrid_default(fresh_db):
    """A bank inserted by anything but create_bank (an older library, a restore, raw SQL) stores `[]` / `{}`
    and runs the hybrid_chunk defaults with Stage 1's order (no rerank stage is configured)."""
    with psycopg.connect(fresh_db) as c:
        c.execute("INSERT INTO banks (bank_id, embedding_dim) VALUES ('raw', %s)", (EMBED_DIM,))
    assert config_of(fresh_db, "raw") == ([], {}, {})
    m = Memory(database_url=fresh_db, bank_id="raw", embed=stub_embed)
    try:
        m.retain("The harbour lamps get a warm filter.", source="lamps.md")
        assert [r.source for r in m.search("harbour lamps", date_from="2000-01-01")] == ["lamps.md"]
    finally:
        m.close()


# ---------------------------------------------------------------- the live banks behave the same

@pytest.fixture
def live(fresh_db):
    """Memory on each of the four banks as stored, one note and one chunk each, reading its own bank. The
    first note is held as a question item only (an old bank's, written raw: nothing writes them now)."""
    mems = {}
    for bank, (channel, recall, link) in LIVE_BANKS.items():
        insert_stored_bank(fresh_db, bank, channel, recall, link)
        m = Memory(database_url=fresh_db, bank_id=bank, embed=stub_embed,
                   rerank_transport=FakeRerank())
        m.retain("Kelly met Ann on monday near the harbour lamps", source="note.md")
        with psycopg.connect(fresh_db) as c:
            c.execute("UPDATE memory_items SET kind = 'question', ordinal = NULL, char_start = NULL, "
                      "char_end = NULL WHERE bank_id = %s AND document_id IN "
                      "(SELECT id FROM documents WHERE bank_id = %s AND source = 'note.md')", (bank, bank))
            c.commit()
        m.retain("The harbour lamps get a warm filter.", source="chunk.md")   # a chunk item
        mems[bank] = m
    yield mems
    for m in mems.values():
        m.close()


def test_the_banks_with_empty_configs_run_hybrid_chunk(live):
    for bank in ("augur", "forge"):
        m = live[bank]
        got = m.search("harbour lamps", limit=3, date_from="2000-01-01")     # Stage 1 filters: no error now
        assert [r.source for r in got] == ["chunk.md"]          # chunks only: the question item is invisible
        assert m.expand_and_recall("harbour lamps", gate=False).error is None


def test_the_hand_set_banks_still_run_hybrid_chunk_and_cross_encoder(live):
    for bank in ("augur-v2", "forge-v2"):
        m = live[bank]
        got = m.search("harbour lamps", limit=3, date_from="2000-01-01")
        assert [r.source for r in got] == ["chunk.md"]        # chunks only: the question item is invisible
        assert got[0].scores.get("rerank") == pytest.approx(0.9)
        assert m._rerank_transport.requests


class RecordingLLM:
    def __init__(self):
        self.calls = 0

    def __call__(self, messages, *, json_mode=False):
        self.calls += 1
        return "{}"


def test_retain_extracts_no_entities_and_calls_no_model_on_any_bank(live, mem, fresh_db):
    """A bank that stored `{}` ran entity extraction on every write under the old library; the
    linker is gone, so no bank does, the four live ones included."""
    mem.create_bank("fresh", embedding_dim=EMBED_DIM)
    llm = RecordingLLM()
    for m in (*live.values(), mem):
        m._llm = llm
    mem.retain("Kelly met Ann on monday", source="fresh.md")
    for bank in LIVE_BANKS:
        live[bank].retain("Kelly wrote again on tuesday", source="again.md")
    assert llm.calls == 0
    with psycopg.connect(fresh_db) as c:
        assert c.execute("SELECT count(*) FROM memory_entities").fetchone()[0] == 0
        assert c.execute("SELECT count(*) FROM memory_links").fetchone()[0] == 0
    for bank, stored in LIVE_BANKS.items():
        assert config_of(fresh_db, bank) == stored               # and no bank was rewritten


def test_rerank_timeout_default_is_4s_and_a_stored_value_wins():
    from prospecta import stages
    assert stages.RERANK_TIMEOUT_S == 4.0
    # a fresh bank stores no timeout: it follows the code default
    assert "timeout_s" not in bank_defaults.NEW_BANK_RECALL_CONFIG["rerank"]
    t = lambda *a, **k: None
    assert build_reranker(CROSS_ENCODER, rerank_transport=t).timeout == 4.0
    assert build_reranker({"rerank": {"enabled": True, "stage": "cross_encoder", "timeout_s": 2.5}},
                          rerank_transport=t).timeout == 2.5
    import inspect
    from prospecta import _expand_run
    src = inspect.getsource(_expand_run)
    assert 'rr.get("timeout_s", RERANK_TIMEOUT_S)' in src
