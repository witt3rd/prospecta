"""T14 — Background sweeper tests.

Covers run_one_pass + SweeperThread + Memory wiring.

Discipline:
  - Per-test fresh db (fresh_db + memory_with_bank_and_mock_llm).
  - Small sweep_interval_seconds (0.1s) + bounded polling loops.
  - No raw long sleeps; deadlines + early-exit.
  - Tests prefer running run_one_pass synchronously where feasible
    (avoids flaky timing).
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _read_sweep_passes(mem) -> list[dict]:
    """Read all sweep_passes rows for this bank."""
    rows: list[dict] = []
    with mem._pool.connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT corpus_path, files_seen, files_indexed, errors_count, "
            "duration_ms, error "
            "FROM sweep_passes WHERE bank_id = %s ORDER BY started_at ASC",
            (mem._default_bank_id,),
        )
        for r in cur.fetchall():
            rows.append({
                "corpus_path": r[0],
                "files_seen": r[1],
                "files_indexed": r[2],
                "errors_count": r[3],
                "duration_ms": r[4],
                "error": r[5],
            })
    return rows


# ---------------------------------------------------------------------------
# acceptance tests
# ---------------------------------------------------------------------------

def test_sweeper_picks_up_new_file(memory_with_bank_and_mock_llm, tmp_corpus):
    """Background thread picks up a file added on disk after start_sweeper."""
    mem = memory_with_bank_and_mock_llm
    mem.start_sweeper(
        corpus_paths=[tmp_corpus], sweep_interval_seconds=0.1,
    )
    try:
        # write new file with frontmatter index_text so no LLM is needed.
        (tmp_corpus / "new.md").write_text(
            "---\nindex_text: 'kelly birthday'\n---\nbody"
        )
        deadline = time.time() + 5.0
        found = False
        while time.time() < deadline:
            results = mem.search("kelly birthday")
            if results:
                found = True
                break
            time.sleep(0.15)
        assert found, "sweeper failed to pick up the new file"
    finally:
        ok = mem.stop_sweeper(timeout=3.0)
        assert ok


def test_sweeper_skips_indexed_files(memory_with_bank_and_mock_llm, tmp_corpus):
    """A second pass after a first pass reports zero new indexes when files
    haven't changed (P14 unchanged → counted as skipped)."""
    from prospecta._sweeper import SweeperConfig, run_one_pass

    mem = memory_with_bank_and_mock_llm
    cfg = SweeperConfig(corpus_paths=[tmp_corpus])
    first = run_one_pass(mem, cfg)
    second = run_one_pass(mem, cfg)
    assert len(first) == 1 and len(second) == 1
    # First pass indexes both seed files; second pass treats them as unchanged.
    assert first[0].files_indexed == 2
    assert second[0].files_indexed == 0
    assert second[0].files_scanned == 2
    # files_skipped is the unchanged bucket for run_one_pass.
    assert second[0].files_skipped == 2


def test_sweeper_exception_isolation(
    memory_with_bank_and_mock_llm, tmp_corpus, monkeypatch
):
    """Per-file exception in index_single_file does NOT abort the pass."""
    from prospecta import _index, _sweeper

    mem = memory_with_bank_and_mock_llm

    # Make doc1.md raise; doc2.md proceeds via the real implementation.
    real_fn = _index.index_single_file

    def flaky(memory_arg, path, **kwargs):
        if Path(path).name == "doc1.md":
            raise RuntimeError("synthetic failure on doc1.md")
        return real_fn(memory_arg, path, **kwargs)

    # Patch at the Memory delegator level so the sweeper sees the spy
    # (sweeper calls memory.index_single_file).
    monkeypatch.setattr(
        type(mem), "index_single_file",
        lambda self, path, **kw: flaky(self, path, **kw),
        raising=True,
    )

    cfg = _sweeper.SweeperConfig(corpus_paths=[tmp_corpus])
    results = _sweeper.run_one_pass(mem, cfg)
    assert len(results) == 1
    r = results[0]
    assert r.files_scanned == 2
    assert r.errors >= 1
    # the other file still indexed
    assert r.files_indexed == 1
    # error path recorded
    assert any("doc1.md" in p for (p, _err) in r.error_paths)


def test_sweeper_shutdown_clean(memory_with_bank_and_mock_llm, tmp_corpus):
    """start_sweeper + stop_sweeper join cleanly within timeout."""
    mem = memory_with_bank_and_mock_llm
    mem.start_sweeper(
        corpus_paths=[tmp_corpus], sweep_interval_seconds=0.1,
    )
    time.sleep(0.3)
    ok = mem.stop_sweeper(timeout=3.0)
    assert ok
    assert mem._sweeper_thread is None


def test_sweep_pass_appends_event_row(memory_with_bank_and_mock_llm, tmp_corpus):
    """A pass appends a sweep_passes row with correct counters."""
    from prospecta._sweeper import SweeperConfig, run_one_pass

    mem = memory_with_bank_and_mock_llm
    cfg = SweeperConfig(corpus_paths=[tmp_corpus])
    run_one_pass(mem, cfg)

    rows = _read_sweep_passes(mem)
    assert len(rows) == 1
    row = rows[0]
    assert row["corpus_path"] == str(tmp_corpus)
    assert row["files_seen"] == 2
    assert row["files_indexed"] == 2
    assert row["errors_count"] == 0
    assert row["duration_ms"] is not None and row["duration_ms"] >= 0
    assert row["error"] is None


def test_sweeper_state_updates_last_pass_at(
    memory_with_bank_and_mock_llm, tmp_corpus
):
    """sweeper_state row is upserted with last_pass_started_at populated."""
    from prospecta._sweeper import SweeperConfig, run_one_pass
    from prospecta.db.queries import read_sweeper_state

    mem = memory_with_bank_and_mock_llm
    cfg = SweeperConfig(corpus_paths=[tmp_corpus])
    run_one_pass(mem, cfg)

    with mem._pool.connection() as conn:
        state = read_sweeper_state(
            conn,
            bank_id=mem._default_bank_id,
            corpus_path=str(tmp_corpus),
        )
    assert state is not None
    assert state["last_pass_started_at"] is not None
    assert state["last_pass_ended_at"] is not None
    assert state["last_pass_files_indexed"] == 2
    assert state["last_pass_errors"] == 0


def test_run_one_pass_uses_index_single_file(
    memory_with_bank_and_mock_llm, tmp_corpus, monkeypatch
):
    """P7 audit: sweeper must go through Memory.index_single_file."""
    from prospecta._sweeper import SweeperConfig, run_one_pass

    mem = memory_with_bank_and_mock_llm
    calls: list[str] = []
    real = type(mem).index_single_file

    def spy(self, path, **kwargs):
        calls.append(str(path))
        return real(self, path, **kwargs)

    monkeypatch.setattr(type(mem), "index_single_file", spy, raising=True)
    run_one_pass(mem, SweeperConfig(corpus_paths=[tmp_corpus]))

    # Both files visited via the public single-write-path delegator.
    assert any("doc1.md" in c for c in calls)
    assert any("doc2.md" in c for c in calls)


def test_hot_path_survives_sweeper_error(
    memory_with_bank_and_mock_llm, tmp_corpus, monkeypatch
):
    """P14: even when the sweeper sees per-file errors, retain() still works."""
    from prospecta import _index, _sweeper

    mem = memory_with_bank_and_mock_llm
    real_fn = _index.index_single_file

    def flaky(memory_arg, path, **kwargs):
        if Path(path).name == "doc1.md":
            raise RuntimeError("synthetic failure on doc1.md")
        return real_fn(memory_arg, path, **kwargs)

    monkeypatch.setattr(
        type(mem), "index_single_file",
        lambda self, path, **kw: flaky(self, path, **kw),
        raising=True,
    )

    _sweeper.run_one_pass(
        mem, _sweeper.SweeperConfig(corpus_paths=[tmp_corpus]),
    )

    # Hot path still intact.
    doc_id = mem.retain(
        "hot path content",
        index_text=["hot path probe"],
        source="hot/path",
    )
    assert isinstance(doc_id, str) and doc_id
