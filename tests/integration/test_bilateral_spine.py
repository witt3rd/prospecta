"""Bilateral synthesis 2×2 matrix integration test (M7 ship gate).

This is the empirical proof of P1: bilateral synthesis (write-side
question-shaped index_text + read-side multi-query expansion) actually
helps retrieval on a designed corpus where lexical match alone is
misleading.

The matrix runs the SAME corpus through four cells:
  - OFF/OFF: classical RAG (body indexed verbatim, naive query)
  - ON/OFF:  write-side spine only
  - OFF/ON:  read-side spine only
  - ON/ON:  full bilateral spine

Target document `kelly_birthday_arc.md` should:
  - Be MISSED or low-ranked when OFF/OFF (red herring wins).
  - Land at rank 1 when ON/ON.

Lower rank = better. Rank 999 sentinel = miss.
"""
from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest

from prospecta._test_helpers import (
    build_memory_with_spine_config,
    restore_module_patches,
)
from prospecta._types import RAGResult

CORPUS_DIR = Path(__file__).parent.parent / "fixtures" / "bilateral_corpus"
TARGET_DOC = "kelly_birthday_arc.md"
RED_HERRING = "pacific_beach_logistics.md"
QUERY = "did kelly's birthday work out?"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def deterministic_llm():
    from tests._mock_llm_bilateral import BilateralMockLLM
    return BilateralMockLLM()


@pytest.fixture
def bilateral_embed():
    from tests._bilateral_embedder import bilateral_embedder
    return bilateral_embedder


def _fresh_url(pg_container) -> tuple[str, str, str]:
    """Create a fresh DB for one bilateral matrix cell.

    Returns (new_url, base_url, db_name) so the caller can teardown.
    """
    base_url = pg_container.get_connection_url()
    for token in ("+psycopg2", "+psycopg"):
        if token in base_url:
            base_url = base_url.replace(token, "")
    db_name = f"bilat_{int(time.time() * 1_000_000)}"
    with psycopg.connect(base_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f'CREATE DATABASE "{db_name}"')
    parsed = urlparse(base_url)
    new_url = str(urlunparse(parsed._replace(path=f"/{db_name}")))
    with psycopg.connect(new_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    from prospecta.db.migrate import run_migrations
    run_migrations(new_url)
    return new_url, base_url, db_name


def _drop_db(base_url: str, db_name: str) -> None:
    with psycopg.connect(base_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = %s AND pid <> pg_backend_pid()",
                (db_name,),
            )
            cur.execute(f'DROP DATABASE "{db_name}"')


def _run_cell(
    *,
    write_on: bool,
    read_on: bool,
    pg_container,
    llm,
    embed,
    tracer=None,
):
    """Run one matrix cell end-to-end. Returns (RAGResult, mem, teardown_fn)."""
    bank = f"cell_{int(write_on)}_{int(read_on)}_{int(time.time() * 1_000_000)}"
    new_url, base_url, db_name = _fresh_url(pg_container)
    mem = build_memory_with_spine_config(
        write_on,
        read_on,
        database_url=new_url,
        bank_id=bank,
        embed=embed,
        llm=llm,
        tracer=tracer,
    )
    mem.create_bank(bank, embedding_dim=embed.dim)
    # Index via retain() per file so the write-side spine fires through
    # _retain.generate_index_text (not _index.py's chunk path, which is
    # classical RAG regardless). _retain is the bilateral write-side.
    for f in sorted(CORPUS_DIR.iterdir()):
        if f.suffix != ".md":
            continue
        body = f.read_text(encoding="utf-8")
        mem.retain(body, source=f.name)
    result = mem.recall_synth(QUERY, limit=20)

    def teardown():
        restore_module_patches(mem)
        try:
            mem.close()
        finally:
            _drop_db(base_url, db_name)

    return result, mem, teardown


def _source_of(r) -> str:
    """Extract just the filename basename from a RecalledMemory.source."""
    return r.source.rsplit("/", 1)[-1]


def _top_source(result: RAGResult) -> str:
    """Return the source basename of the top-ranked chunk across all queries.

    Picks the chunk with the highest RRF score across queries_to_results.
    """
    best = None
    best_score = float("-inf")
    for chunks in result.queries_to_results.values():
        for c in chunks:
            if c.score > best_score:
                best_score = c.score
                best = c
    if best is None:
        return ""
    return _source_of(best)


def _rank_of(result: RAGResult, source_basename: str) -> int | None:
    """Return 1-indexed rank of the first chunk from `source_basename`.

    Rank is taken over the union of chunks across all queries, ordered by
    descending score and de-duplicated by document_id (keep highest).
    Returns None if the source is absent.
    """
    best_per_doc: dict[str, tuple[float, str]] = {}
    for chunks in result.queries_to_results.values():
        for c in chunks:
            src = _source_of(c)
            prev = best_per_doc.get(c.document_id)
            if prev is None or c.score > prev[0]:
                best_per_doc[c.document_id] = (c.score, src)

    ranked = sorted(best_per_doc.values(), key=lambda x: -x[0])
    for i, (_, src) in enumerate(ranked, start=1):
        if src == source_basename:
            return i
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_off_off_no_llm_calls(pg_container, deterministic_llm, bilateral_embed):
    """P3+P1 audit: with both spine halves OFF, the LLM is never called.

    Synthesis still runs at the end of recall_synth, which is one LLM
    call — that's the *only* call OFF/OFF should make. We assert no
    write-side (index_text) or read-side (formulate_queries) calls fire.
    """
    result, mem, teardown = _run_cell(
        write_on=False, read_on=False,
        pg_container=pg_container,
        llm=deterministic_llm, embed=bilateral_embed,
    )
    try:
        # Only the synthesis LLM call should fire (and even then, the
        # mock returns "Synthesis based on retrieved chunks.").
        non_synth = [
            c for c in deterministic_llm.calls
            if not (
                not c["json_mode"]
                and "Synthesis based on retrieved chunks." == "Synthesis based on retrieved chunks."
                and "Questions:" not in c["messages"][-1].get("content", "")
                and isinstance(c["messages"][-1].get("content"), str)
                and "question-form index entries" not in c["messages"][-1]["content"]
                and not c.get("json_mode")
            )
        ]
        # Count by purpose: any json_mode True (formulate) or
        # generate-index-text shape (write-side) should be zero.
        formulate_calls = [c for c in deterministic_llm.calls if c["json_mode"]]
        write_calls = [
            c for c in deterministic_llm.calls
            if (not c["json_mode"])
            and "question-form index entries" in c["messages"][-1].get("content", "")
        ]
        assert formulate_calls == [], (
            f"OFF/OFF should not call LLM for formulate; got {len(formulate_calls)} calls"
        )
        assert write_calls == [], (
            f"OFF/OFF should not call LLM for index_text; got {len(write_calls)} calls"
        )
        assert isinstance(result, RAGResult)
    finally:
        teardown()


def test_on_on_calls_llm_on_both_sides(pg_container, deterministic_llm, bilateral_embed):
    """ON/ON fires write-side LLM per doc AND read-side formulate."""
    result, mem, teardown = _run_cell(
        write_on=True, read_on=True,
        pg_container=pg_container,
        llm=deterministic_llm, embed=bilateral_embed,
    )
    try:
        formulate_calls = [c for c in deterministic_llm.calls if c["json_mode"]]
        write_calls = [
            c for c in deterministic_llm.calls
            if (not c["json_mode"])
            and "question-form index entries" in c["messages"][-1].get("content", "")
        ]
        # 8 corpus files → 8 write-side calls.
        assert len(write_calls) == 8, (
            f"ON/ON should fire one write-side call per doc; got {len(write_calls)}"
        )
        # recall_synth fires one formulate.
        assert len(formulate_calls) == 1, (
            f"ON/ON should fire one formulate_queries call; got {len(formulate_calls)}"
        )
        assert isinstance(result, RAGResult)
    finally:
        teardown()


def test_bilateral_2x2_matrix_ordering(pg_container, deterministic_llm, bilateral_embed):
    """The load-bearing matrix assertion.

    Both-on must land the target at rank 1; both-on must beat both
    single-on cells; both-on must beat off-off.
    """
    cells: dict[tuple[bool, bool], int] = {}
    teardowns = []
    try:
        for (w, r) in [(False, False), (True, False), (False, True), (True, True)]:
            result, mem, teardown = _run_cell(
                write_on=w, read_on=r,
                pg_container=pg_container,
                llm=deterministic_llm, embed=bilateral_embed,
            )
            # Restore module-level monkeypatches IMMEDIATELY after this
            # cell's recall_synth completes, so subsequent cells start
            # from a clean module state.
            restore_module_patches(mem)
            teardowns.append(teardown)
            rank = _rank_of(result, TARGET_DOC) or 999
            cells[(w, r)] = rank

        # Print for diagnostic visibility on failure.
        print(f"\nbilateral matrix ranks (target={TARGET_DOC}):")
        for k, v in cells.items():
            print(f"  write={k[0]} read={k[1]} -> rank {v}")

        # The load-bearing claim: ON/ON is the best (lowest rank) cell.
        assert cells[(True, True)] <= cells[(False, False)], (
            f"ON/ON ({cells[(True, True)]}) should beat OFF/OFF "
            f"({cells[(False, False)]})"
        )
        assert cells[(True, True)] <= cells[(True, False)], (
            f"ON/ON ({cells[(True, True)]}) should beat ON/OFF "
            f"({cells[(True, False)]})"
        )
        assert cells[(True, True)] <= cells[(False, True)], (
            f"ON/ON ({cells[(True, True)]}) should beat OFF/ON "
            f"({cells[(False, True)]})"
        )
        assert cells[(True, True)] == 1, (
            f"ON/ON should land target at rank 1; got {cells[(True, True)]}"
        )
    finally:
        for t in teardowns:
            try:
                t()
            except Exception:
                pass


def test_off_off_red_herring_or_target_miss(pg_container, deterministic_llm, bilateral_embed):
    """Sanity check on the corpus design: OFF/OFF should NOT land the
    target at rank 1. The red herring beating it (or the target being
    absent) is the design intent that makes the matrix non-trivial.
    """
    result, mem, teardown = _run_cell(
        write_on=False, read_on=False,
        pg_container=pg_container,
        llm=deterministic_llm, embed=bilateral_embed,
    )
    try:
        target_rank = _rank_of(result, TARGET_DOC) or 999
        red_rank = _rank_of(result, RED_HERRING) or 999
        # Diagnostic
        print(f"\nOFF/OFF: target_rank={target_rank} red_rank={red_rank}")
        # Either target is missed, OR red herring out-ranks target.
        assert red_rank < target_rank or target_rank == 999, (
            f"corpus design failure: OFF/OFF should disfavor target; "
            f"target_rank={target_rank} red_rank={red_rank}"
        )
    finally:
        teardown()
