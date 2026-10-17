"""What the Hermes plugin (witt3rd/hermes-prospecta, main at 4d37121) imports from this library.

The plugin does not vendor the library: it imports these names at run time. The names below that stay are
pinned, so that a removal here fails in this repository first; the plugin's own test suite is the real gate
and is run against a branch before it merges. REMOVED lists what the plugin imports or calls that the
removal of all remaining legacy took out, so that the plugin follow-up knows what to drop or move."""
from __future__ import annotations

import importlib
import inspect

import pytest

IMPORTS = {
    "prospecta.memory": ["Memory", "_serialize_recall_results"],
    "prospecta._shadow": ["_origin_ids", "run_shadow_recall"],
    "prospecta._tracer": ["NoOpTracer", "RecordingTracer"],
    "prospecta._types": ["ExpandedRecall", "RecalledMemory", "SearchRun", "SearchSpec"],
    "prospecta.cited": ["HEADING", "CITE_LINE", "format_cited_rag_block"],
    "prospecta.cli._common": ["_stage_kwargs"],
    "prospecta.channels.recall": ["read_channel_config"],
    "prospecta.defaults": ["make_default_embedder", "make_default_llm"],
    "prospecta.embed": ["sentence_transformers", "openai"],
    "prospecta.db.migrate": ["run_migrations"],
    "prospecta.stages": ["JEV_MODEL", "JevScore", "openrouter_jev_transport", "openrouter_rerank_transport",
                         "read_recall_config"],
}
MEMORY_METHODS = ["retain", "search", "recall", "ask", "expand_and_recall", "recall_for_task",
                  "create_bank", "set_channel_config", "set_recall_config", "bank_stats", "close", "shutdown",
                  "index_directory", "remove_documents", "start_sweeper", "stop_sweeper"]
# what the plugin passes to Memory(...): its own, and `_stage_kwargs()` / `_recall_stage_kwargs()` of it
MEMORY_KWARGS = ["database_url", "llm", "embed", "bank_id", "tracer", "shadow_bank_id", "shadow_embed",
                 "synth_llm", "jev", "rerank_transport", "reranker_model", "decision_model", "schema"]


# What the plugin imports or calls and this library no longer has (the plugin follow-up drops it or moves it):
REMOVED_MODULES = {
    "prospecta.channels.extract": ["extract_filters", "regex_filters"],   # the plugin's date cue gate
    "prospecta.channels.base": ["Filters"],
    "prospecta._index": ["extract_filters"],                              # the gate patches it: no-op now
}
REMOVED_MEMORY_METHODS = ["recall_synth", "formulate_queries", "recall_mapreduce"]
REMOVED_ARGUMENTS = {                  # Memory methods the plugin calls with arguments that are gone
    "retain": ["index_text", "child_chunks"],
    "search": ["mode"], "recall": ["mode"],
}


@pytest.mark.parametrize("module,names", list(IMPORTS.items()))
def test_the_names_the_plugin_imports_exist(module, names):
    mod = importlib.import_module(module)
    for name in names:
        assert hasattr(mod, name), f"{module}.{name}"


def test_memory_keeps_the_methods_and_arguments_the_plugin_uses():
    from prospecta.memory import Memory
    for name in MEMORY_METHODS:
        assert callable(getattr(Memory, name, None)), name
    params = inspect.signature(Memory.__init__).parameters
    for name in MEMORY_KWARGS:
        assert name in params, name


def test_the_decision_model_holder_still_takes_a_transport_and_a_model():
    from prospecta.stages import JEV_MODEL, JevScore
    jev = JevScore(lambda request, timeout: {}, model="acme/decider", timeout=3.0)
    assert (jev.model, jev.timeout) == ("acme/decider", 3.0) and callable(jev.transport)
    assert JevScore(lambda r, t: {}).model == JEV_MODEL


def test_the_stage_kwargs_stay_a_dict_the_plugin_can_hand_to_memory(monkeypatch):
    from prospecta.cli import _common
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert _common._stage_kwargs() == {}
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-a-secret")
    kw = _common._stage_kwargs()
    assert set(kw) <= set(MEMORY_KWARGS)


def test_what_the_plugin_used_and_the_library_removed_is_gone():
    from prospecta import _retain
    from prospecta.memory import Memory
    for module, names in REMOVED_MODULES.items():
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        for name in names:
            assert not hasattr(mod, name), f"{module}.{name}"
    for name in REMOVED_MEMORY_METHODS:
        assert not hasattr(Memory, name), name
    for method, args in REMOVED_ARGUMENTS.items():
        fn = _retain.retain if method == "retain" else getattr(Memory, method)
        for arg in args:
            assert arg not in inspect.signature(fn).parameters, f"{method}({arg}=)"


def test_the_date_cue_gate_of_the_plugin_finds_nothing_to_wrap():
    """The plugin installs its date cue gate by patching `prospecta._index.extract_filters` and
    feature-detects it: with the function gone the gate is not installed and nothing breaks."""
    import prospecta._index as idx
    assert getattr(idx, "extract_filters", None) is None
