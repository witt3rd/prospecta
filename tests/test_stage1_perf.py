"""Stage 1 latency on a synthetic bank (opt-in: it loads and indexes a lot of data).

    PROSPECTA_PERF_CHUNKS=73000 pytest tests/test_stage1_perf.py -s

builds a bank of that many kind='chunk' items (1536-dim vectors, invented text, both
HNSW indexes, the GIN indexes of migrations 0002, 0018 and 0020), runs the Stage 1 query
warm over a mix of selective queries, and requires the 95th percentile under one
second. Most chunks carry function words ("the", "and", ...), so it also runs the
very-common-word queries of the slice 5 verification ("the", "the and a of", a common
stem): they match tens of thousands of chunks, and each must stay within 1.5 s (before the
stored `exact_tsv` column they re-tokenized every matching chunk: 6 to 7 s at 73,000).
The load and index build takes a couple of minutes at 73,000."""
from __future__ import annotations

import os
import statistics
import time

import psycopg
import pytest

from prospecta.db.queries import stage1_search
from tests._synthetic_bank import load_bank, random_query_vector, vocabulary

N_CHUNKS = int(os.environ.get("PROSPECTA_PERF_CHUNKS") or 0)
DIM = 1536
REPEATS = 20
FUNCTION_WORD_CHUNKS = 0.61             # the share of chunks with "the": 44,500 of 73,000, as in forge-v2
COMMON_P50_LIMIT = 1.5                  # seconds, per very-common-word query

pytestmark = pytest.mark.skipif(not N_CHUNKS, reason="set PROSPECTA_PERF_CHUNKS to run")


def test_stage1_latency_on_a_large_bank(fresh_db):
    load_bank(fresh_db, "perf", n_chunks=N_CHUNKS, dim=DIM, function_words=FUNCTION_WORD_CHUNKS)
    middling = vocabulary()[1500]       # a word in under 1% of the chunks
    queries = ['"electric dreams"', "electric dreams", "0x80070005", "v1.14",
               "harbour OR crew -lamps", "what did the crew say about the harbour lamps",
               middling, f"{middling} OR {vocabulary()[1501]}"]
    common = ["the", "the and a of", vocabulary()[99], vocabulary()[20]]    # ~12% and ~40% of the chunks
    per_query: dict[str, list[float]] = {q: [] for q in queries + common}
    with psycopg.connect(fresh_db) as conn:
        for q in queries + common:       # warm: each query once before it is timed
            stage1_search(conn, bank_id="perf", query_text=q, query_embedding=random_query_vector(DIM, 0))
        for rep in range(REPEATS):
            for q in queries + common:
                started = time.perf_counter()
                rows = stage1_search(conn, bank_id="perf", query_text=q,
                                     query_embedding=random_query_vector(DIM, 1 + rep))
                per_query[q].append(time.perf_counter() - started)
                assert 1 <= len(rows) <= 50
    every = sorted(t for q in queries for t in per_query[q])
    p50, p95 = statistics.median(every), every[int(len(every) * 0.95) - 1]
    print(f"\nstage 1 over {N_CHUNKS} chunks, {len(every)} warm selective calls: "
          f"p50 {p50 * 1000:.1f} ms, p95 {p95 * 1000:.1f} ms, max {every[-1] * 1000:.1f} ms")
    for q, v in per_query.items():
        print(f"  {q!r:50} p50 {statistics.median(v) * 1000:7.1f} ms  max {max(v) * 1000:7.1f} ms")
    assert p95 < 1.0
    for q in common:
        assert statistics.median(per_query[q]) < COMMON_P50_LIMIT, q
