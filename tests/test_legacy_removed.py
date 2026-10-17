"""There is one design, and no legacy code behind it.

The architecture is `hybrid_chunk` + the cross-encoder (docs/postgres-hybrid.md): a note is written as chunks,
Stage 1 is one native Postgres query over the chunks, Stage 2 is the cross-encoder, Mode 1 expands a prompt
into searches, Mode 2B asks. Everything older was removed: the linker and its entities and links, the graph,
bm25, keyword and question channels, the Sonnet / Jev rerank, the gate and the reader, map-reduce, write-time
questions (`index_text`, `child_chunks`), the legacy three-channel read path (`channel_config = []`), the
`dense_chunk` and `meta` channels with their filter extraction, `recall_synth` and `formulate_queries`, the
`mode` of a search, the question copy of `migrate-bank`, the `link_config` stamp.

What stays is the data: every migration, every table and column, and every row of an existing bank. An old
bank (the `augur` and `forge` fixtures of tests/_live_banks.py: `channel_config = []`, question items beside
the chunks, the rows of a linker that once ran) opens and searches through `hybrid_chunk` without an error
and without a rewrite of its stored row; a stored config that names what was removed is ignored, silently.

Real Postgres, invented text, the fixture banks of tests/_live_banks.py."""
from __future__ import annotations

import importlib.util
import inspect
import logging
from pathlib import Path

import psycopg
import pytest

from prospecta.cli.__main__ import _build_parser
from prospecta.db.migrate import run_migrations
from prospecta.memory import Memory
from tests._live_banks import (
    CORPUS, HAND_SET_RECALL, HYBRID, QUERIES, FakeRerank, ScriptedLLM, build_bank, embed, retired_row_counts,
    set_stored_config, stored_config,
)
from tests._stub_embedder import EMBED_DIM

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "prospecta"


def sources(results) -> list[str]:
    return [r.source for r in results]


def stored_row(url: str, bank: str):
    with psycopg.connect(url) as c:
        return c.execute("SELECT * FROM banks WHERE bank_id = %s", (bank,)).fetchone()


def item_kinds(url: str, bank: str) -> dict:
    with psycopg.connect(url) as c:
        return dict(c.execute("SELECT kind, count(*) FROM memory_items WHERE bank_id = %s GROUP BY kind",
                              (bank,)).fetchall())


def kind_of(url: str, item_id: str) -> str:
    with psycopg.connect(url) as c:
        return c.execute("SELECT kind FROM memory_items WHERE id = %s", (item_id,)).fetchone()[0]


# ------------------------------------------------------------- the code is gone

GONE_MODULES = [
    # removed earlier (the linker, graph, bm25 / keyword / question channels, write-time questions)
    "prospecta._linker", "prospecta._entities", "prospecta._linkdrop", "prospecta._mapreduce",
    "prospecta._index_text", "prospecta.channels.graph", "prospecta.channels.bm25",
    "prospecta.channels.keyword", "prospecta.db.keyword_index", "prospecta.cli.link_pass",
    "prospecta.cli.drop_links",
    # removed now
    "prospecta._formulate", "prospecta._rag", "prospecta._synth", "prospecta._warn",
    "prospecta.channels.semantic", "prospecta.channels.meta", "prospecta.channels.extract",
]


@pytest.mark.parametrize("module", GONE_MODULES)
def test_a_removed_module_is_gone(module):
    assert importlib.util.find_spec(module) is None


@pytest.mark.parametrize("prompt", [
    "extract-entities", "mapreduce-map", "mapreduce-reduce", "generate-index-text", "rerank-listwise",
    "read-sufficiency", "formulate-queries", "rag-synthesize", "synthesize-grounded"])
def test_a_removed_prompt_is_gone(prompt):
    assert not (ROOT / "prompts" / f"{prompt}.md").exists()


@pytest.mark.parametrize("prompt", ["ask-cited", "expand-searches", "judge-answer"])
def test_the_prompts_the_live_paths_use_stay(prompt):
    assert (ROOT / "prompts" / f"{prompt}.md").exists()


def test_the_legacy_page_is_gone_and_nothing_links_to_it():
    assert not (REPO / "docs" / "legacy.md").exists()
    for path in [REPO / "README.md", *(REPO / "docs").glob("*.md"), *ROOT.rglob("*.py"), *ROOT.rglob("*.sql")]:
        assert "docs/legacy.md" not in path.read_text(encoding="utf-8"), path


def test_memory_has_only_the_surface_the_design_needs():
    for name in ("link_document", "link_pending", "link_pass", "drop_links", "wait_for_links", "set_link_config",
                 "get_link_config", "recall_mapreduce", "recall_synth", "formulate_queries"):
        assert not hasattr(Memory, name), name
    for name in ("retain", "search", "recall", "ask", "expand_and_recall", "recall_for_task", "create_bank",
                 "set_channel_config", "set_recall_config", "index_directory", "index_single_file",
                 "remove_documents", "bank_stats", "start_sweeper", "stop_sweeper", "shutdown", "close"):
        assert hasattr(Memory, name), name
    init = set(inspect.signature(Memory.__init__).parameters)
    assert not {"linker", "rerank_llm", "rerank_model"} & init
    assert {"jev", "rerank_transport", "reranker_model", "decision_model", "synth_llm", "shadow_bank_id"} <= init


def test_the_write_path_is_chunk_only():
    from prospecta import _retain
    real = set(inspect.signature(_retain.retain).parameters) - {"memory"}
    assert real == {"content", "source", "path", "tags", "metadata", "update_mode"}
    assert not hasattr(_retain, "_matching_chunk") and not hasattr(_retain, "_dedupe_strip")


def test_a_search_has_no_mode_and_one_read_path():
    from prospecta import _index
    from prospecta.db import queries
    assert "mode" not in inspect.signature(Memory.recall).parameters
    assert "mode" not in inspect.signature(_index.search).parameters
    for name in ("hybrid_search", "semantic_search", "lexical_search", "body_lexical_search", "HYBRID_SQL",
                 "SEMANTIC_SQL", "LEXICAL_SQL", "BODY_LEXICAL_SQL", "append_formulate_event"):
        assert not hasattr(queries, name), name
    for name in ("_row_to_recalled", "_make_recalled", "extract_filters"):
        assert not hasattr(_index, name), name
    assert "index_text" not in _index.ParsedDocument.__dataclass_fields__


def test_the_channels_are_hybrid_chunk_alone():
    import prospecta.channels as channels
    from prospecta import _scorecut
    from prospecta.channels import REGISTRY, fuse
    assert sorted(REGISTRY) == ["hybrid_chunk"]
    for name in ("DenseChunks", "MetadataScope", "extract_filters", "regex_filters", "promote_scope", "scope_members",
                 "Filters", "RETIRED", "DEFAULT_CHANNEL_CONFIG"):
        assert not hasattr(channels, name), name
    assert "method" not in inspect.signature(fuse).parameters      # floor_union served several channels
    for name in ("fetch_until_cut", "FETCH_PAGE", "CHANNEL_MIN_REL"):
        assert not hasattr(_scorecut, name), name


def test_the_stages_are_the_cross_encoder_stage_3_and_the_decision_model():
    from prospecta import stages
    assert stages.STAGES == ("cross_encoder",)
    for gone in ("SonnetListwise", "SonnetReader", "JevGate", "JevReader", "BlendedReranker", "DEFAULT_RECALL_CONFIG",
                 "blend_order", "read_all", "SONNET_MODEL", "CHEAP_CHANNELS", "JEV_GATE_THRESHOLD", "RETIRED_STAGES",
                 "RETIRED_RECALL_KEYS", "warn_retired_recall_config"):
        assert not hasattr(stages, gone), gone
    for kept in ("CrossEncoderReranker", "JevScore", "JevSufficiency", "STAGE3_THRESHOLD", "JEV_MODEL", "JEV_URL",
                 "openrouter_jev_transport", "openrouter_rerank_transport", "read_recall_config", "run_stages",
                 "validate_recall_config"):
        assert hasattr(stages, kept), kept


def test_the_types_of_the_removed_calls_are_gone():
    from prospecta import _types
    assert not hasattr(_types, "RAGResult")
    for kept in ("RecalledMemory", "ExpandedRecall", "AskResult", "TaskBrief", "SearchSpec", "SearchRun", "Query"):
        assert hasattr(_types, kept), kept


def test_the_tracer_has_no_formulate_event():
    from prospecta.observability import postgres_sink
    assert "formulate_queries" not in postgres_sink._HANDLERS
    assert not hasattr(postgres_sink.PostgresSink, "_handle_formulate")


@pytest.mark.parametrize("argv", [
    ["retain", "x", "--index-text", "q"],
    ["search", "x", "--mode", "semantic"],
    ["eval", "q.md", "--scope-promote"],
    ["link-pass"], ["drop-links"], ["keyword-index"],
])
def test_the_cli_commands_and_options_of_removed_code_are_gone(argv):
    with pytest.raises(SystemExit) as e:
        _build_parser().parse_args(argv)
    assert e.value.code == 2


def test_the_cli_commands_that_stay_parse():
    for argv in (["retain", "x"], ["search", "x", "--limit", "3"], ["eval", "q.md", "--ablate"],
                 ["migrate-bank", "--source-bank", "a"], ["create-bank", "--id", "a", "--embedding-dim", "8"]):
        _build_parser().parse_args(argv)


def test_the_removed_environment_variables_are_not_read():
    text = "\n".join(p.read_text() for p in ROOT.rglob("*.py"))
    for name in ("PROSPECTA_LINKER", "PROSPECTA_LINK_COMPLETENESS", "PROSPECTA_JUDGE_FLOOR_FRAC", "PROSPECTA_BM25_DIR"):
        assert name not in text, name


# ------------------------------------------------------ a new bank works end to end

def test_a_new_bank_works_end_to_end_and_stamps_nothing_legacy(fresh_db):
    llm, rerank = ScriptedLLM(), FakeRerank()
    mem = Memory(database_url=fresh_db, bank_id="fresh", embed=embed, llm=llm, rerank_transport=rerank)
    try:
        mem.create_bank("fresh", embedding_dim=EMBED_DIM)
        channel, recall, link = stored_config(fresh_db, "fresh")
        assert [e["name"] for e in channel] == ["hybrid_chunk"]
        assert recall == {"rerank": {"enabled": True, "stage": "cross_encoder"}}
        assert link == {}              # the column default: nothing writes a link_config any more
        for source, created_on, kind, person, text in CORPUS["augur"]:
            mem.retain(text, source=source, metadata={"created_on": created_on, "source_kind": kind, "project": "atlas"})
        assert llm.calls == []                                   # a write calls no model
        assert set(item_kinds(fresh_db, "fresh")) == {"chunk"}   # and writes chunks only
        assert not any(retired_row_counts(fresh_db, "fresh").values())

        q = QUERIES["augur"][0]
        trace: list = []
        hits = mem.search(q, limit=5, _trace=trace)              # Stage 1 (hybrid_chunk) + Stage 2 (cross_encoder)
        assert "harbour/lamps-decision.md" in sources(hits)
        assert [c["name"] for c in trace[0]["channels"]] == ["hybrid_chunk"]
        assert trace[0]["rerank"]["stage"] == "cross_encoder" and rerank.requests == 1
        assert all(r.match_kind and r.created_on and r.relevance_score is not None for r in hits)
        assert set(hits[0].scores) == {"hybrid_chunk", "rerank", "rrf"}    # no three-channel keys any more

        dated = mem.search(q, limit=8, date_from="2026-03-01", date_to="2026-04-30",
                           excluded_sources=["harbour/lamps-decision.md"])
        assert dated and "harbour/lamps-decision.md" not in sources(dated)
        assert all("2026-03-01" <= r.created_on <= "2026-04-30" for r in dated)

        got = mem.expand_and_recall("what did we decide about the lamps and the glaze?", "chat", gate=False, record=False)
        assert got.error is None and got.items and got.block
        asked = mem.ask(q, limit=4)
        assert asked.items and "[1]" in asked.answer
    finally:
        mem.close()


def test_a_new_bank_builds_the_chunk_index_only(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="fresh", embed=embed)
    try:
        mem.create_bank("fresh", embedding_dim=EMBED_DIM)
        with psycopg.connect(fresh_db) as c:
            names = {r[0] for r in c.execute(
                "SELECT indexname FROM pg_indexes WHERE tablename = 'memory_items' AND indexname LIKE '%fresh%'")}
        assert names == {"memory_items_embedding_fresh_chunk_idx"}
    finally:
        mem.close()


@pytest.mark.parametrize("kwargs", [{"index_text": ["a question?"]}, {"child_chunks": True},
                                    {"generate_questions": True}])
def test_a_write_cannot_ask_for_questions(fresh_db, kwargs):
    mem = Memory(database_url=fresh_db, bank_id="fresh", embed=embed)
    try:
        mem.create_bank("fresh", embedding_dim=EMBED_DIM)
        with pytest.raises(TypeError):
            mem.retain("text", source="a.md", **kwargs)
        assert item_kinds(fresh_db, "fresh") == {}
    finally:
        mem.close()


def test_frontmatter_index_text_is_no_longer_a_write_instruction(fresh_db, tmp_path):
    (tmp_path / "n.md").write_text("---\nindex_text: 'what is this?'\n---\nThe harbour lamps use a warm filter.\n")
    mem = Memory(database_url=fresh_db, bank_id="fresh", embed=embed)
    try:
        mem.create_bank("fresh", embedding_dim=EMBED_DIM)
        mem.index_directory(tmp_path)
        assert set(item_kinds(fresh_db, "fresh")) == {"chunk"}
    finally:
        mem.close()


# ---------------------------------- an old bank opens and searches through hybrid_chunk

@pytest.fixture(params=["augur", "forge"])
def old_bank(request, fresh_db):
    bank = request.param
    mem = build_bank(fresh_db, bank, bank, legacy=True, llm=ScriptedLLM(), rerank_transport=FakeRerank())
    yield bank, mem
    mem.close()


def test_the_fixture_is_an_old_bank(old_bank, fresh_db):
    bank, _ = old_bank
    assert stored_config(fresh_db, bank) == ([], {}, {})
    kinds = item_kinds(fresh_db, bank)
    assert kinds["question"] == kinds["chunk"] > 0
    assert all(n > 0 for n in retired_row_counts(fresh_db, bank).values())


def test_an_empty_channel_config_means_the_hybrid_default(old_bank, fresh_db):
    bank, mem = old_bank
    for q in QUERIES[bank]:
        trace: list = []
        got = mem.search(q, limit=8, _trace=trace)
        assert [c["name"] for c in trace[0]["channels"]] == ["hybrid_chunk"]
        assert got and all(kind_of(fresh_db, r.item_id) == "chunk" for r in got)    # never a question item
        assert not any(r.original_chunk.startswith("What do we know about") for r in got)
        set_stored_config(fresh_db, bank, HYBRID, {}, {})
        explicit = mem.search(q, limit=8)
        set_stored_config(fresh_db, bank, [], {}, {})
        assert sources(got) == sources(explicit)
        assert [round(r.score, 9) for r in got] == [round(r.score, 9) for r in explicit]


def test_an_old_bank_takes_the_stage_1_filters_and_every_entry_point(old_bank, fresh_db):
    bank, mem = old_bank
    q = QUERIES[bank][0]
    dated = mem.search(q, limit=8, date_from="2026-03-01", date_to="2026-04-30")     # used to raise ValueError
    assert all("2026-03-01" <= r.created_on <= "2026-04-30" for r in dated)
    first = mem.search(q, limit=8)[0].source
    assert first not in sources(mem.search(q, limit=8, excluded_sources=[first]))
    assert mem.search(q, limit=3, depth="deep")
    res = mem.expand_and_recall("what did we decide about the lamps and the glaze?", "chat", gate=False, record=False)
    assert res.error is None and res.items
    assert mem.recall([q, QUERIES[bank][1]], limit=4, parallel=True)
    asked = mem.ask(q, limit=4)
    assert asked.items and all(kind_of(fresh_db, i.item_id) == "chunk" for i in asked.items)
    assert mem.recall_for_task(title="Check the rollout", ask="Find the error", record=False).error is None


def test_using_an_old_bank_never_rewrites_its_row_or_its_rows(old_bank, fresh_db):
    bank, mem = old_bank
    before, kinds, links = stored_row(fresh_db, bank), item_kinds(fresh_db, bank), retired_row_counts(fresh_db, bank)
    q = QUERIES[bank][0]
    mem.search(q)
    mem.recall(QUERIES[bank][:2])
    mem.ask(q)
    mem.expand_and_recall("what did we decide?", "chat", gate=False, record=False)
    assert stored_row(fresh_db, bank) == before
    assert stored_config(fresh_db, bank) == ([], {}, {})
    assert item_kinds(fresh_db, bank) == kinds and retired_row_counts(fresh_db, bank) == links


def test_an_old_bank_with_the_rows_of_a_linker_replaces_and_removes_documents(old_bank, fresh_db):
    bank, mem = old_bank
    before = retired_row_counts(fresh_db, bank)
    source, other = CORPUS[bank][0][0], CORPUS[bank][1][0]
    mem.retain("The lamps are now cold white; the warm filter was dropped.", source=source,
               metadata={"created_on": "2026-05-31"})            # same source, new text: the old items go
    assert mem.remove_documents([other]) == 1
    assert mem.search("lamps", limit=5)
    after = retired_row_counts(fresh_db, bank)
    assert after["memory_entities"] == before["memory_entities"]     # the rows nothing reads stay
    assert after["memory_links"] < before["memory_links"]            # those of removed items cascade
    assert item_kinds(fresh_db, bank)["chunk"] > 0


# ---------------------------------------- a stored config that names what was removed

CHANNELS_OF_THE_PAST = [
    {"name": "dense_chunk", "enabled": True, "weight": 4, "params": {}},
    {"name": "question", "enabled": True, "weight": 1, "params": {}},
    {"name": "bm25", "enabled": True, "weight": 1, "params": {}},
    {"name": "keyword", "enabled": True, "weight": 1, "params": {}},
    {"name": "graph", "enabled": True, "weight": 0, "params": {"max_hops": 2, "decay": 0.5}},
    {"name": "meta", "enabled": True, "weight": 3, "params": {"promote": True}},
]
RECALL_OF_THE_PAST = {
    "rerank": {"enabled": True, "stage": "sonnet_listwise", "min_rel_score": 0.15, "pool": 40, "top": 10,
               "blend": {"enabled": True, "weight_rerank": 0.7, "weight_fused": 0.3}},
    "gate": {"enabled": True, "threshold": 2.95},
    "reader": {"enabled": True, "min_rel_score": 0.6, "hop_min_rel_score": 0.4},
    "link_completeness": "connected",
    "fusion": {"method": "floor_union", "floor": 3},
    "evidence": {"min_rel_score": 0.5, "context_tokens": 100000},
    "depth": "standard",
}


@pytest.fixture
def chunk_bank(fresh_db):
    mem = build_bank(fresh_db, "augur-v2", "augur", legacy=False, llm=ScriptedLLM(), rerank_transport=FakeRerank())
    yield mem
    mem.close()


def warnings_of_ours(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING and r.name.startswith("prospecta")]


def test_channels_that_no_longer_exist_are_ignored_beside_hybrid_chunk(chunk_bank, fresh_db, caplog):
    q = QUERIES["augur"][0]
    expected = chunk_bank.search(q, limit=8)
    set_stored_config(fresh_db, "augur-v2", CHANNELS_OF_THE_PAST + HYBRID, HAND_SET_RECALL, {})
    trace: list = []
    with caplog.at_level(logging.DEBUG, logger="prospecta"):
        got = chunk_bank.search(q, limit=8, _trace=trace)
        res = chunk_bank.expand_and_recall("what did we decide about the lamps?", "chat", gate=False, record=False)
    assert expected and sources(got) == sources(expected) and res.error is None and res.items
    assert [c["name"] for c in trace[0]["channels"]] == ["hybrid_chunk"]
    assert {s["name"] for s in trace[0]["skipped"]} == {e["name"] for e in CHANNELS_OF_THE_PAST}
    assert warnings_of_ours(caplog) == []                          # ignored silently


def test_a_config_of_removed_channels_only_is_the_hybrid_default(chunk_bank, fresh_db, caplog):
    q = QUERIES["augur"][0]
    set_stored_config(fresh_db, "augur-v2", HYBRID, HAND_SET_RECALL, {})
    expected = chunk_bank.search(q, limit=8)
    set_stored_config(fresh_db, "augur-v2", CHANNELS_OF_THE_PAST, HAND_SET_RECALL, {})
    with caplog.at_level(logging.DEBUG, logger="prospecta"):
        got = chunk_bank.search(q, limit=8)
        res = chunk_bank.expand_and_recall("what did we decide about the lamps?", "chat", gate=False, record=False)
    assert expected and sources(got) == sources(expected)          # not "nothing": the default runs
    assert res.error is None and res.items
    assert warnings_of_ours(caplog) == []


def test_hybrid_chunk_params_of_a_stored_config_are_honoured(chunk_bank, fresh_db):
    set_stored_config(fresh_db, "augur-v2", [{"name": "hybrid_chunk", "weight": 1,
                                              "params": {"semantic_limit": 3, "stemmed_limit": 2, "exact_limit": 2,
                                                         "fused_limit": 4}}], {}, {})
    assert 0 < len(chunk_bank.search(QUERIES["augur"][0], limit=10)) <= 4


@pytest.mark.parametrize("config", [
    [{"name": "dense_chunk", "weight": 1}], [{"name": "meta", "weight": 1}], [{"name": "bm25", "weight": 1}],
    [{"name": "keyword", "weight": 1}], [{"name": "question", "weight": 1}], [{"name": "graph", "weight": 1}],
    [{"name": "hybrid_chunk", "weight": 1}, {"name": "dense_chunk", "weight": 1}],
])
def test_a_new_channel_config_names_hybrid_chunk_alone(config, chunk_bank):
    with pytest.raises(ValueError):
        chunk_bank.set_channel_config(config)


def test_a_new_channel_config_may_be_empty_which_is_the_default(chunk_bank, fresh_db):
    chunk_bank.set_channel_config([])
    assert stored_config(fresh_db, "augur-v2")[0] == []
    assert chunk_bank.search(QUERIES["augur"][0], limit=3)
    chunk_bank.set_channel_config([{"name": "hybrid_chunk", "weight": 1}])


def test_a_recall_config_of_the_past_is_ignored_silently(chunk_bank, fresh_db, caplog):
    q = QUERIES["augur"][0]
    set_stored_config(fresh_db, "augur-v2", HYBRID, {}, {})
    expected = chunk_bank.search(q, limit=8)                       # no stage at all: Stage 1's order
    llm = chunk_bank._llm
    llm.calls.clear()
    set_stored_config(fresh_db, "augur-v2", HYBRID, RECALL_OF_THE_PAST, {})
    with caplog.at_level(logging.DEBUG, logger="prospecta"):
        got = chunk_bank.search(q, limit=8)
        res = chunk_bank.expand_and_recall("what did we decide about the lamps?", "chat", gate=False, record=False)
        asked = chunk_bank.ask(q, limit=3)
    assert expected and sources(got) == sources(expected)
    assert [round(r.score, 6) for r in got] == [round(r.score, 6) for r in expected]
    assert res.error is None and asked.items
    assert "formulate" not in llm.calls and "synthesize" not in llm.calls     # no rerank / reader model asked
    assert warnings_of_ours(caplog) == []


def test_the_cross_encoder_runs_beside_keys_it_never_read(chunk_bank, fresh_db):
    cfg = {**RECALL_OF_THE_PAST, "rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0.2, "pool": 40,
                                            "blend": {"enabled": True}, "join_top": 5}}
    set_stored_config(fresh_db, "augur-v2", HYBRID, cfg, {})
    trace: list = []
    got = chunk_bank.search(QUERIES["augur"][0], limit=5, _trace=trace)
    assert got and trace[0]["rerank"]["stage"] == "cross_encoder" and trace[0]["rerank"]["fallback_reason"] is None


@pytest.mark.parametrize("config", [
    {"gate": {"enabled": True}}, {"reader": {"enabled": True}}, {"link_completeness": "connected"},
    {"fusion": {"method": "floor_union"}}, {"evidence": {"min_rel_score": 0.5}},
    {"rerank": {"enabled": True, "stage": "sonnet_listwise"}}, {"rerank": {"enabled": True, "stage": "jev_score"}},
    {"rerank": {"enabled": True}}, {"rerank": {"enabled": True, "stage": "cross_encoder", "blend": {"enabled": True}}},
    {"rerank": {"pool": 5}}, {"nonsense": 1},
])
def test_a_new_recall_config_names_what_the_design_reads(config, chunk_bank):
    with pytest.raises(ValueError):
        chunk_bank.set_recall_config(config)


def test_the_recall_config_keys_that_stay_are_configurable(chunk_bank):
    chunk_bank.set_recall_config({
        "rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0.2, "top_n": 5, "min_rel_score": 0.1,
                   "model": "m", "timeout_s": 3, "deduplicate_near_copies": False},
        "models": {"decision_model": "typesafe/jev-1.14", "query_writer_model": "q"},
        "stage3_sufficiency": {"enabled": True, "threshold": 0.5}, "stage0_gate": {"enabled": True},
        "expansion": {"max_searches": 5}, "depth": "deep"})
    chunk_bank.set_recall_config({})


# ----------------------------------------------- the data stays: migrations, tables, rows

def test_the_migration_files_of_the_removed_code_stay():
    names = {p.name[:4] for p in (ROOT / "db" / "migrations").glob("*.sql")}
    assert {"0006", "0008", "0012", "0014", "0016", "0018", "0019"} <= names


def test_the_tables_and_columns_of_the_removed_code_stay_for_existing_banks(fresh_db):
    run_migrations(fresh_db)       # idempotent
    with psycopg.connect(fresh_db) as c:
        tables = {r[0] for r in c.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")}
        banks = {r[0] for r in c.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'banks'")}
        events = {r[0] for r in c.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name = 'retain_events'")}
        indexes = {r[0] for r in c.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'memory_items'")}
    assert {"memory_entities", "memory_item_entities", "memory_links", "memory_link_state", "memory_entity_aliases",
            "memory_alias_state", "memory_link_pairs", "formulate_events"} <= tables
    assert {"channel_config", "recall_config", "link_config"} <= banks
    assert {"index_text_caller_supplied", "index_text_generated", "raw_llm_response"} <= events
    assert "memory_items_exact_tsv_gin" in indexes and "memory_items_keyword_gin" not in indexes


def test_a_write_still_fills_the_retain_event_columns_that_stay_not_null(fresh_db):
    mem = Memory(database_url=fresh_db, bank_id="fresh", embed=embed)
    try:
        mem.create_bank("fresh", embedding_dim=EMBED_DIM)
        mem.retain("The harbour lamps use a warm filter.", source="a.md")
        with psycopg.connect(fresh_db) as c:
            row = c.execute("SELECT items_count, index_text_caller_supplied, index_text_generated FROM retain_events "
                            "WHERE bank_id = 'fresh'").fetchone()
        assert row == (1, False, None)
    finally:
        mem.close()
