"""The retired recall code is gone, a new bank can never build it, and a bank that still names it works.

The architecture is `hybrid_chunk` + the cross-encoder (docs/postgres-hybrid.md). What it no longer uses was
removed from the code: the linker, entity extraction and the graph channel; `recall_mapreduce`; the `bm25`,
`keyword` and `question` channels and write-time question generation; the Sonnet / Jev rerank stages, the Jev
gate, the reader and its hop, the rerank blend. Their migrations and tables stay (an applied migration is never
edited), existing banks keep their stored config, and a stored config that names a removed channel or stage is
skipped with one warning per process, never an error that fails a recall. See docs/legacy.md.

Real Postgres, invented text, the fixture banks of tests/_live_banks.py (the shape of the four live banks)."""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
from pathlib import Path

import psycopg
import pytest

from prospecta.channels import REGISTRY, validate_channel_config
from prospecta.cli.__main__ import _build_parser
from prospecta.db.migrate import run_migrations
from prospecta.memory import Memory
from tests._live_banks import (
    CORPUS, HAND_SET_RECALL, HYBRID, QUERIES, FakeRerank, ScriptedLLM, build_bank, embed, retired_row_counts,
    set_stored_config, stored_config,
)
from tests._stub_embedder import EMBED_DIM

ROOT = Path(__file__).resolve().parent.parent / "prospecta"
RETIRED_CHANNELS = ("bm25", "keyword", "question", "graph")


@pytest.fixture(autouse=True)
def _fresh_warnings():
    """Each test starts a 'process' that has warned about nothing."""
    try:
        from prospecta import _warn
    except ImportError:      # before the removal there is no such module
        yield
        return
    saved = set(_warn._seen)
    _warn._seen.clear()
    yield
    _warn._seen.clear()
    _warn._seen.update(saved)


def retired_warnings(caplog, name: str) -> list[str]:
    return [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and f"'{name}'" in r.getMessage() and "retired" in r.getMessage()]


def sources(results) -> list[str]:
    return [r.source for r in results]


# ------------------------------------------------------------- the code is gone

@pytest.mark.parametrize("module", [
    "prospecta._linker", "prospecta._entities", "prospecta._linkdrop", "prospecta._mapreduce",
    "prospecta._index_text", "prospecta.channels.graph", "prospecta.channels.bm25",
    "prospecta.channels.keyword", "prospecta.db.keyword_index", "prospecta.cli.link_pass",
    "prospecta.cli.drop_links"])
def test_a_retired_module_is_gone(module):
    assert importlib.util.find_spec(module) is None


@pytest.mark.parametrize("prompt", ["extract-entities", "mapreduce-map", "mapreduce-reduce", "generate-index-text",
                                    "rerank-listwise", "read-sufficiency"])
def test_a_retired_prompt_is_gone(prompt):
    assert not (ROOT / "prompts" / f"{prompt}.md").exists()


@pytest.mark.parametrize("prompt", ["ask-cited", "expand-searches", "formulate-queries", "rag-synthesize",
                                    "synthesize-grounded", "judge-answer"])
def test_the_prompts_the_live_paths_use_stay(prompt):
    assert (ROOT / "prompts" / f"{prompt}.md").exists()


def test_memory_has_no_linking_or_map_reduce_surface():
    for name in ("link_document", "link_pending", "link_pass", "drop_links", "wait_for_links",
                 "set_link_config", "get_link_config", "recall_mapreduce"):
        assert not hasattr(Memory, name), name
    params = inspect.signature(Memory.__init__).parameters
    for name in ("linker", "rerank_llm", "rerank_model"):
        assert name not in params, name


def test_memory_refuses_a_linker(fresh_db):
    with pytest.raises(TypeError):
        Memory(database_url=fresh_db, embed=embed, linker=object())


@pytest.mark.parametrize("command", ["link-pass", "drop-links", "keyword-index"])
def test_the_retired_cli_commands_are_gone(command):
    with pytest.raises(SystemExit) as e:
        _build_parser().parse_args([command])
    assert e.value.code == 2


def test_the_cli_never_builds_a_linker_whatever_the_environment_says(monkeypatch):
    from prospecta.cli import _common
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-a-secret")
    monkeypatch.setenv("PROSPECTA_LINKER", "on")
    kw = _common._stage_kwargs()
    assert "linker" not in kw and "rerank_llm" not in kw
    assert set(kw) <= {"synth_llm", "rerank_transport", "jev"}


def test_the_retired_environment_variables_are_not_read():
    text = "\n".join(p.read_text() for p in ROOT.rglob("*.py"))
    for name in ("PROSPECTA_LINKER", "PROSPECTA_LINK_COMPLETENESS", "PROSPECTA_JUDGE_FLOOR_FRAC", "PROSPECTA_BM25_DIR"):
        assert name not in text, name


# ------------------------------------------------ a new bank never links or extracts

def test_a_new_bank_never_links_extracts_entities_or_calls_a_model_on_write(fresh_db):
    llm = ScriptedLLM()
    mem = Memory(database_url=fresh_db, bank_id="fresh", embed=embed, llm=llm)
    try:
        mem.create_bank("fresh", embedding_dim=EMBED_DIM)
        for source, created_on, kind, person, text in CORPUS["augur"][:6]:
            mem.retain(text, source=source, metadata={"created_on": created_on, "person": person or "x"})
        mem.retain("a note with a question", source="q.md", index_text=["What is this note?"], child_chunks=True)
    finally:
        mem.close()
    assert llm.calls == []
    assert not any(retired_row_counts(fresh_db, "fresh").values())
    # kept on purpose: an older library reading this bank must still find linking and entities off
    assert stored_config(fresh_db, "fresh")[2] == {"links": False, "entities": False}


def test_retain_no_longer_generates_questions(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="fresh", embed=embed, llm=ScriptedLLM())
    try:
        mem.create_bank("fresh", embedding_dim=EMBED_DIM)
        with pytest.raises(TypeError):
            mem.retain("text", source="a.md", generate_questions=True)
        with pytest.raises(TypeError):
            mem.retain("text", source="a.md", index_text_prompt_override="x")
        mem.retain("text", source="b.md", index_text=["What is it?"], child_chunks=True)   # the caller's own: stays
        with psycopg.connect(fresh_db) as c:
            kinds = sorted(r[0] for r in c.execute("SELECT kind FROM memory_items WHERE bank_id = 'fresh'"))
        assert kinds == ["chunk", "question"]
    finally:
        mem.close()


def test_the_source_indexer_no_longer_generates_questions(fresh_db):
    from prospecta.source import SourceConfig, SourceIndexer
    mem = Memory(database_url=fresh_db, embed=embed, llm=ScriptedLLM())
    try:
        with pytest.raises(TypeError):
            SourceIndexer(mem, SourceConfig(schema="x", views={}), generate_questions=True)
    finally:
        mem.close()


# ---------------------------------------- migrations and tables stay, the index is not built

def test_the_migration_files_of_the_retired_code_stay():
    names = {p.name[:4] for p in (ROOT / "db" / "migrations").glob("*.sql")}
    assert {"0006", "0012", "0014", "0016", "0018", "0019"} <= names


def test_the_tables_and_columns_of_the_retired_code_stay_for_existing_banks(fresh_db):
    with psycopg.connect(fresh_db) as c:
        tables = {r[0] for r in c.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")}
        cols = {r[0] for r in c.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'banks'")}
    assert {"memory_entities", "memory_item_entities", "memory_links", "memory_link_state",
            "memory_entity_aliases", "memory_alias_state", "memory_link_pairs"} <= tables
    assert {"channel_config", "recall_config", "link_config"} <= cols


def test_a_new_database_does_not_build_the_keyword_index_and_keeps_the_exact_one(fresh_db):
    run_migrations(fresh_db)      # idempotent; the fixture already ran it
    with psycopg.connect(fresh_db) as c:
        indexes = {r[0] for r in c.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'memory_items'")}
    assert "memory_items_keyword_gin" not in indexes       # no code reads it any more
    assert "memory_items_exact_tsv_gin" in indexes         # Stage 1's exact retriever does


# ---------------------------------------------- a stored config that names a removed channel

STORED_WITH_RETIRED = [
    {"name": "dense_chunk", "enabled": True, "weight": 4, "params": {}},
    {"name": "question", "enabled": True, "weight": 1, "params": {}},
    {"name": "bm25", "enabled": True, "weight": 1, "params": {}},
    {"name": "keyword", "enabled": True, "weight": 1, "params": {}},
    {"name": "graph", "enabled": True, "weight": 0, "params": {"max_hops": 2, "decay": 0.5}},
    {"name": "meta", "enabled": True, "weight": 3, "params": {}},
]
LIVE_ONLY = [e for e in STORED_WITH_RETIRED if e["name"] in ("dense_chunk", "meta")]


@pytest.fixture
def chunk_bank(fresh_db):
    mem = build_bank(fresh_db, "augur-v2", "augur", legacy=False, llm=ScriptedLLM(), rerank_transport=FakeRerank())
    yield mem
    mem.close()


@pytest.mark.parametrize("name", RETIRED_CHANNELS)
def test_a_new_config_cannot_name_a_retired_channel(name, chunk_bank):
    with pytest.raises(ValueError, match="retired"):
        validate_channel_config([{"name": name, "weight": 1}])
    with pytest.raises(ValueError, match="retired"):
        chunk_bank.set_channel_config([{"name": name, "weight": 1}])
    assert name not in REGISTRY


def test_a_bank_that_still_names_retired_channels_searches_with_the_rest(chunk_bank, fresh_db, caplog):
    q = QUERIES["augur"][0]
    set_stored_config(fresh_db, "augur-v2", LIVE_ONLY, {}, {})
    expected = chunk_bank.search(q, limit=8)
    set_stored_config(fresh_db, "augur-v2", STORED_WITH_RETIRED, {}, {})
    trace: list = []
    with caplog.at_level(logging.WARNING):
        got = chunk_bank.search(q, limit=8, _trace=trace)
        chunk_bank.search(q, limit=8)      # a second recall in the same process
    assert expected and sources(got) == sources(expected)
    assert [c["name"] for c in trace[0]["channels"]] == ["dense_chunk", "meta"]
    assert {s["name"] for s in trace[0]["skipped"]} == set(RETIRED_CHANNELS)    # recorded, not run
    for name in RETIRED_CHANNELS:                       # one warning each, in the whole process
        assert len(retired_warnings(caplog, name)) == 1, name


def test_a_config_of_retired_channels_only_returns_nothing_and_does_not_raise(chunk_bank, fresh_db, caplog):
    set_stored_config(fresh_db, "augur-v2", [{"name": "bm25", "weight": 1}, {"name": "keyword", "weight": 1}], {}, {})
    with caplog.at_level(logging.WARNING):
        assert chunk_bank.search(QUERIES["augur"][0]) == []
    assert len(retired_warnings(caplog, "bm25")) == 1 and len(retired_warnings(caplog, "keyword")) == 1


def test_query_expansion_runs_on_hybrid_chunk_beside_a_retired_channel(chunk_bank, fresh_db, caplog):
    set_stored_config(fresh_db, "augur-v2", HYBRID + [{"name": "bm25", "enabled": True, "weight": 1}],
                      HAND_SET_RECALL, {})
    with caplog.at_level(logging.WARNING):
        res = chunk_bank.expand_and_recall("what did we decide about the lamps?", "chat", gate=False, record=False)
    assert res.error is None and res.items
    assert len(retired_warnings(caplog, "bm25")) <= 1


def test_the_legacy_lanes_of_the_default_channel_set_still_search(chunk_bank, fresh_db, caplog):
    """The measured default set (dense_chunk, question, graph, meta) of a bank that stored it."""
    lanes = [{"name": "dense_chunk", "enabled": True, "weight": 4, "params": {}},
             {"name": "question", "enabled": True, "weight": 1, "params": {}},
             {"name": "graph", "enabled": True, "weight": 0, "params": {}},
             {"name": "meta", "enabled": True, "weight": 3, "params": {}}]
    set_stored_config(fresh_db, "augur-v2", lanes, {}, {})
    with caplog.at_level(logging.WARNING):
        res = chunk_bank.search(QUERIES["augur"][0], limit=5)
    assert res
    assert retired_warnings(caplog, "question") and retired_warnings(caplog, "graph")


# ---------------------------------------------- a stored recall_config that names a removed stage

RETIRED_RECALL = {
    "rerank": {"enabled": True, "stage": "sonnet_listwise", "min_rel_score": 0.15,
               "blend": {"enabled": True, "weight_rerank": 0.7, "weight_fused": 0.3}},
    "gate": {"enabled": True, "threshold": 2.95},
    "reader": {"enabled": True, "min_rel_score": 0.6, "hop_min_rel_score": 0.4},
}


def test_a_bank_that_still_names_removed_stages_keeps_the_stage_1_order_and_calls_no_model(
        chunk_bank, fresh_db, caplog):
    q = QUERIES["augur"][0]
    set_stored_config(fresh_db, "augur-v2", HYBRID, {}, {})
    expected = chunk_bank.search(q, limit=8)                       # no stage at all: Stage 1's order
    llm = chunk_bank._llm
    llm.calls.clear()
    set_stored_config(fresh_db, "augur-v2", HYBRID, RETIRED_RECALL, {})
    trace: list = []
    with caplog.at_level(logging.WARNING):
        got = chunk_bank.search(q, limit=8, _trace=trace)
        chunk_bank.search(q, limit=8)
    assert expected and sources(got) == sources(expected)
    assert [round(r.score, 6) for r in got] == [round(r.score, 6) for r in expected]
    assert llm.calls == []                                         # no Sonnet rerank, no reader
    assert not trace[0].get("hops")
    for what in ("sonnet_listwise", "gate", "reader"):
        assert len([r for r in caplog.records if f"'{what}'" in r.getMessage()
                    and "retired" in r.getMessage()]) == 1, what


def test_a_rerank_block_with_no_stage_is_the_retired_default_and_is_skipped(chunk_bank, fresh_db, caplog):
    set_stored_config(fresh_db, "augur-v2", HYBRID, {"rerank": {"enabled": True}}, {})
    with caplog.at_level(logging.WARNING):
        res = chunk_bank.search(QUERIES["augur"][0], limit=5)
    assert res and all(r.relevance_score == r.score for r in res)


@pytest.mark.parametrize("config", [
    {"gate": {"enabled": True}},
    {"gate": {"enabled": False}},
    {"reader": {"enabled": True}},
    {"link_completeness": "connected"},
    {"rerank": {"enabled": True, "stage": "sonnet_listwise"}},
    {"rerank": {"enabled": True, "stage": "jev_score"}},
    {"rerank": {"enabled": True}},
    {"rerank": {"enabled": True, "stage": "cross_encoder", "blend": {"enabled": True}}},
])
def test_a_new_recall_config_cannot_name_a_removed_stage(config, chunk_bank):
    with pytest.raises(ValueError):
        chunk_bank.set_recall_config(config)


def test_the_stages_that_stay_are_configurable(chunk_bank):
    from prospecta import stages
    assert stages.STAGES == ("cross_encoder",)
    chunk_bank.set_recall_config({
        "rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0.2, "top_n": 5},
        "models": {"decision_model": "typesafe/jev-1.14"}, "stage3_sufficiency": {"enabled": True},
        "stage0_gate": {"enabled": True}, "expansion": {"max_searches": 5}, "depth": "deep"})
    chunk_bank.set_recall_config({})
    for gone in ("SonnetListwise", "SonnetReader", "JevGate", "JevReader", "BlendedReranker", "DEFAULT_RECALL_CONFIG",
                 "blend_order", "read_all", "SONNET_MODEL", "CHEAP_CHANNELS", "JEV_GATE_THRESHOLD"):
        assert not hasattr(stages, gone), gone
    for kept in ("CrossEncoderReranker", "JevScore", "JevSufficiency", "STAGE3_THRESHOLD", "JEV_MODEL", "JEV_URL",
                 "openrouter_jev_transport", "openrouter_rerank_transport", "read_recall_config", "run_stages"):
        assert hasattr(stages, kept), kept


# -------------------------------------------------------- no bank is rewritten

@pytest.mark.parametrize("bank,corpus,legacy", [("augur", "augur", True), ("forge", "forge", True),
                                                ("augur-v2", "augur", False), ("forge-v2", "forge", False)])
def test_using_a_live_shaped_bank_never_rewrites_its_stored_config(bank, corpus, legacy, fresh_db):
    mem = build_bank(fresh_db, bank, corpus, legacy=legacy, llm=ScriptedLLM(), rerank_transport=FakeRerank())
    try:
        def row():
            with psycopg.connect(fresh_db) as c:
                return c.execute("SELECT * FROM banks WHERE bank_id = %s", (bank,)).fetchone()
        before = row()
        q = QUERIES[corpus][0]
        mem.search(q)
        mem.recall(QUERIES[corpus][:2])
        mem.ask(q)
        mem.recall_synth(q)
        mem.retain("one more note about the lamps", source="new.md", metadata={"created_on": "2026-05-30"})
        mem.expand_and_recall("what did we decide?", "chat", gate=False, record=False)
        assert row() == before
        assert stored_config(fresh_db, bank) == ((HYBRID if not legacy else []), (HAND_SET_RECALL if not legacy else {}), {})
    finally:
        mem.close()


def test_a_legacy_bank_with_the_rows_of_a_linker_replaces_and_removes_documents(fresh_db):
    """The rows the linker wrote stay in an existing bank; writing to it must not trip over them."""
    mem = build_bank(fresh_db, "augur", "augur", legacy=True)
    try:
        before = retired_row_counts(fresh_db, "augur")
        assert all(n > 0 for n in before.values())
        source = CORPUS["augur"][0][0]
        mem.retain("The lamps are now cold white; the warm filter was dropped.", source=source,
                   metadata={"created_on": "2026-05-31"})      # same source, new text: the old items go
        assert mem.remove_documents([CORPUS["augur"][1][0]]) == 1
        assert mem.search("harbour lamps", limit=5)
        after = retired_row_counts(fresh_db, "augur")
        assert after["memory_entities"] == before["memory_entities"]   # the bank's entities are kept
        assert after["memory_links"] < before["memory_links"]          # links of removed items cascaded
    finally:
        mem.close()
