"""T16 — tracer infrastructure tests.

Asserts tracer dispatch fires at all six canonical sites and that the
NoOp/Recording/Composite tracers behave as documented.
"""
from __future__ import annotations

import pytest

from prospecta._tracer import CompositeTracer, NoOpTracer, RecordingTracer


# ---------------------------------------------------------------------------
# Tracer primitives (DB-free)
# ---------------------------------------------------------------------------

def test_noop_tracer_drops_events():
    t = NoOpTracer()
    # Should be quietly silent.
    assert t("retain", {"x": 1}) is None
    assert t("anything", {}) is None


def test_recording_tracer_captures_and_filters():
    t = RecordingTracer()
    t("retain", {"a": 1})
    t("recall", {"b": 2})
    t("retain", {"a": 3})
    assert len(t.events) == 3
    assert t.by_name("retain") == [{"a": 1}, {"a": 3}]
    assert t.by_name("recall") == [{"b": 2}]
    t.clear()
    assert t.events == []
    assert t.by_name("retain") == []


def test_composite_tracer_fans_out_to_all_subtracers():
    r1, r2 = RecordingTracer(), RecordingTracer()
    c = CompositeTracer(r1, r2)
    c("test", {"x": 1})
    assert r1.events == [("test", {"x": 1})]
    assert r2.events == [("test", {"x": 1})]


def test_composite_tracer_isolates_failures():
    """One sub-tracer raising does NOT prevent others from receiving."""
    r1 = RecordingTracer()

    def boom(event, payload):
        raise RuntimeError("nope")

    r2 = RecordingTracer()
    c = CompositeTracer(r1, boom, r2)
    c("x", {})
    assert r1.events == [("x", {})]
    assert r2.events == [("x", {})]


# ---------------------------------------------------------------------------
# Default-tracer contract (Memory wiring)
# ---------------------------------------------------------------------------

def test_default_tracer_is_postgres_sink(memory_with_bank_and_mock_llm):
    """When no tracer kwarg supplied, Memory constructs a PostgresSink."""
    from prospecta.observability import PostgresSink
    mem = memory_with_bank_and_mock_llm
    assert isinstance(mem._tracer, PostgresSink)


def test_caller_supplied_tracer_overrides_default(memory_with_recording_tracer, recording_tracer):
    """The tracer= kwarg wins over the default PostgresSink construction."""
    assert memory_with_recording_tracer._tracer is recording_tracer


# ---------------------------------------------------------------------------
# Live-fire tests (Memory wired with RecordingTracer)
# ---------------------------------------------------------------------------

def test_tracer_fires_on_retain(memory_with_recording_tracer):
    mem = memory_with_recording_tracer
    mem.retain("the body content", source="t1")
    events = mem._tracer.by_name("retain")
    assert len(events) == 1
    assert events[0]["items_count"] == 1                    # one chunk
    assert "index_text_caller_supplied" not in events[0] and "raw_llm_response" not in events[0]
    # no model is called to write a note
    assert mem._tracer.by_name("llm_call") == []


def test_tracer_fires_on_recall_and_ask(populated_corpus_with_recording_tracer):
    mem = populated_corpus_with_recording_tracer
    mem._mock_llm.set_response(lambda messages, *, json_mode: "an answer [1]")
    mem.recall(["about kelly"])
    assert len(mem._tracer.by_name("recall")) == 1 and mem._tracer.by_name("llm_call") == []
    mem.ask("about kelly")
    assert len(mem._tracer.by_name("recall")) == 2
    # llm_call fires for the answer
    assert {p["purpose"] for p in mem._tracer.by_name("llm_call")} == {"ask_cited"}
    assert "formulate_queries" not in {e[0] for e in mem._tracer.events}


def test_tracer_fires_on_sweep(memory_with_recording_tracer, tmp_corpus):
    from prospecta._sweeper import SweeperConfig, run_one_pass
    mem = memory_with_recording_tracer
    run_one_pass(mem, SweeperConfig(corpus_paths=[tmp_corpus]))
    sweeps = mem._tracer.by_name("sweep_pass")
    assert len(sweeps) == 1
    assert sweeps[0]["files_scanned"] == 2
    # per-file index_single_file events also fire
    per_file = mem._tracer.by_name("index_single_file")
    assert len(per_file) >= 2


def test_five_events_fire_end_to_end(memory_with_recording_tracer, tmp_corpus):
    """Smoke that all five named events fire from real call sites."""
    from prospecta._sweeper import SweeperConfig, run_one_pass

    mem = memory_with_recording_tracer
    # retain → fires 'retain' (no model call)
    mem.retain("body content", source="t1")
    # ask → fires 'recall' + 'llm_call'
    mem._mock_llm.set_response(lambda messages, *, json_mode: "synth out [1]")
    mem.ask("body")
    # sweep → fires 'sweep_pass' + 'index_single_file' for each file
    run_one_pass(mem, SweeperConfig(corpus_paths=[tmp_corpus]))

    seen = {e[0] for e in mem._tracer.events}
    assert {"retain", "recall", "sweep_pass", "index_single_file", "llm_call"} <= seen
