"""Latency of query expansion on a large synthetic bank (opt-in: it loads and indexes a lot of data).

    PROSPECTA_PERF_CHUNKS=73000 pytest tests/test_expand_perf.py -s

builds the 73,000-chunk synthetic bank of tests/test_stage1_perf.py (or, with
PROSPECTA_PERF_BANK_URL, uses a database that already holds the bank `perf`: it is migrated, and
its channel_config and recall_config are SET: a throwaway database only), then runs one
`expand_and_recall` with 20 searches against a fake writer and a fake reranker that each answer
after 0.5 s (PROSPECTA_PERF_WRITER_S / PROSPECTA_PERF_RERANK_S / PROSPECTA_PERF_EMBED_S change
them; the embedder answers instantly by default) and prints the total against the slowest single
search and the pool behaviour. docs/postgres-hybrid.md promises about the slowest one plus the
writer."""
from __future__ import annotations

import json
import os
import statistics
import threading
import time

import psycopg
import pytest

from prospecta._tracer import NoOpTracer
from prospecta.memory import Memory
from prospecta.stages import JevScore
from tests._synthetic_bank import load_bank, random_query_vector, vocabulary

N_CHUNKS = int(os.environ.get("PROSPECTA_PERF_CHUNKS") or 0)
BANK_URL = os.environ.get("PROSPECTA_PERF_BANK_URL")
WRITER_S = float(os.environ.get("PROSPECTA_PERF_WRITER_S", "0.5"))
RERANK_S = float(os.environ.get("PROSPECTA_PERF_RERANK_S", "0.5"))
EMBED_S = float(os.environ.get("PROSPECTA_PERF_EMBED_S", "0"))
RUNS = int(os.environ.get("PROSPECTA_PERF_RUNS", "5"))
DIM = 1536

pytestmark = pytest.mark.skipif(not (N_CHUNKS or BANK_URL), reason="set PROSPECTA_PERF_CHUNKS or PROSPECTA_PERF_BANK_URL")


def twenty_searches() -> list[dict]:
    """20 realistic searches over the synthetic vocabulary: a description of 4 to 8 words of mixed
    frequency (words of rank 60 to 3000: from a few percent of the chunks down to a handful), and
    one or two verbatim terms (a rare word, a needle of the bank)."""
    v = vocabulary()
    needles = ['"electric dreams"', "v1.14", "0x80070005", "harbour"]
    out = []
    for i in range(20):
        ranks = [60 + (i * 97 + k * 233) % 2940 for k in range(4 + i % 5)]
        terms = [v[1500 + (i * 41) % 1500]] + ([needles[i % len(needles)]] if i % 3 == 0 else [])
        out.append({"query": " ".join(v[r] for r in ranks), "terms": terms, "date_from": None, "date_to": None})
    return out


class DelayedWriter:
    def __init__(self, searches, delay):
        self.searches, self.delay = searches, delay

    def __call__(self, messages, *, json_mode=False, model=None):
        time.sleep(self.delay)
        return json.dumps({"searches": self.searches})


RELEVANT_PER_SEARCH = int(os.environ.get("PROSPECTA_PERF_RELEVANT", "8"))


def delayed_rerank(delay):
    """A calibrated cross-encoder: the first few candidates of a request are relevant (0.9 down to 0.4),
    the rest score under the floor (0.05), as a reranker does with a pool of mostly unrelated text."""
    def transport(req, timeout):
        time.sleep(delay)
        n = len(req["documents"])
        return {"model": req["model"], "usage": {"total_tokens": 1, "cost": 0.0},
                "results": [{"index": i, "relevance_score": 0.9 - 0.5 * i / RELEVANT_PER_SEARCH
                             if i < RELEVANT_PER_SEARCH else 0.05} for i in range(n)]}
    return transport


def delayed_embed(delay):
    def embed(texts):
        time.sleep(delay)
        return [random_query_vector(DIM, hash(t) & 0xFFFF) for t in texts]
    return embed


class PoolWatch:
    """Samples the pool while a call runs: the most connections in use at once, the most open."""

    def __init__(self, pool):
        self.pool, self.peak_in_use, self.peak_open, self._stop = pool, 0, 0, threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak_in_use = max(self.peak_in_use, self.pool.in_use)
            self.peak_open = max(self.peak_open, self.pool.open_connections)
            time.sleep(0.002)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


def make_memory(url, searches):
    mem = Memory(database_url=url, bank_id="perf", embed=delayed_embed(EMBED_S),
                 llm=DelayedWriter(searches, WRITER_S), rerank_transport=delayed_rerank(RERANK_S),
                 tracer=NoOpTracer())
    mem.set_channel_config([{"name": "hybrid_chunk", "weight": 1}])
    mem.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder", "min_score": 0.15}})
    return mem


def run_many(mem, prompt, **kw):
    runs = []
    for _ in range(RUNS):
        with PoolWatch(mem._pool) as watch:
            t = time.perf_counter()
            got = mem.expand_and_recall(prompt, gate=False, deadline=60, **kw)
            total = (time.perf_counter() - t) * 1000
        assert got.error is None and all(r.status == "ok" for r in got.runs), got.fallbacks
        runs.append((total, got, watch))
    return runs


def test_twenty_searches_cost_about_the_slowest_one_plus_the_writer(request):
    url = BANK_URL or request.getfixturevalue("fresh_db")
    if not BANK_URL:
        load_bank(url, "perf", n_chunks=N_CHUNKS, dim=DIM, function_words=0.61)
    searches = twenty_searches()
    mem = make_memory(url, searches)
    prompt = "a long instruction that the fake writer turns into twenty searches"
    try:
        mem.expand_and_recall(prompt, gate=False, deadline=60)           # warm the caches and the pool
        one = make_memory(url, searches[:1])
        single = run_many(one, prompt)
        single_search = statistics.median(g.timings_ms["slowest_search"] for _, g, _ in single)
        single_total = statistics.median(t for t, _, _ in single)
        rows = []
        for label, kw in (("default: pool 10, concurrency 8", {}), ("concurrency 10 (the whole pool)", {"max_concurrency": 10}),
                          ("concurrency 4", {"max_concurrency": 4}), ("concurrency 1 (serial)", {"max_concurrency": 1})):
            runs = run_many(mem, prompt, **kw)
            totals = sorted(t for t, _, _ in runs)
            g = runs[len(runs) // 2][1]
            rows.append((label, statistics.median(totals), totals[0], totals[-1],
                         statistics.median(g2.timings_ms["writer"] for _, g2, _ in runs),
                         statistics.median(g2.timings_ms["stage1"] for _, g2, _ in runs),
                         statistics.median(g2.timings_ms["slowest_search"] for _, g2, _ in runs),
                         max(w.peak_in_use for _, _, w in runs), max(w.peak_open for _, _, w in runs),
                         len(g.items)))
        print(f"\n73K-chunk bank ({N_CHUNKS or 'preloaded'}): writer {WRITER_S}s, rerank {RERANK_S}s, "
              f"embed {EMBED_S}s, {RUNS} runs each")
        print(f"one search alone (writer + embed + Stage 1 + rerank): total p50 {single_total:.0f} ms, "
              f"the search itself {single_search:.0f} ms")
        print(f"{'configuration':36} {'total p50':>9} {'min':>6} {'max':>6} {'writer':>7} {'stage1 max':>10} "
              f"{'slowest':>8} {'pool peak in use/open':>22} {'items':>6}")
        for label, p50, lo, hi, w, s1, slow, in_use, opened, n in rows:
            print(f"{label:36} {p50:9.0f} {lo:6.0f} {hi:6.0f} {w:7.0f} {s1:10.0f} {slow:8.0f} "
                  f"{in_use:14d}/{opened:<7d} {n:6d}")
        default = rows[0]
        assert default[7] <= 10 and default[8] <= 10                     # never more connections than the pool has
        assert default[1] < 2 * (WRITER_S * 1000 + single_search) + 500  # about the slowest plus the writer
    finally:
        mem.close()
