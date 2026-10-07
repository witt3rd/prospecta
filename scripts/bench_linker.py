"""Linker cost benchmark on real-shaped notes: CLUSTERS of near-duplicate topical
notes (one big 226-note cluster like Greg's;
several mid-size topical clusters; the rest unrelated), with compressed cosines.
Stub Jev counts calls and PRICES them with a per-call token model: calls, tokens
and cost are a MODEL of the real service (nothing leaves the process), the
neighbour selection, batching and pair cache are the real code under test.

Batching (JevRelationJudge defaults): up to 96 questions (32 candidates x 3 relations) and
an estimated 45,000 input tokens per call. Live wire, verified by the scout: 48, 96, 150
and 300 questions per request answered; 300 questions x 1,000 chars (419 KB) failed loud
with HTTP 400 max_tokens_exceeded: the real limit is input tokens. A 400 halves and retries.

Run it once per code version (the label is only printed): the same script runs
against current main (checkout elsewhere, PYTHONPATH=<dir>) and against this branch.

Measured (STUB cost model, fixed clusters; calls per note at 500/1000/1845/3690 notes):
  main (PR 43):                       5.52 / 8.60 / 5.74 / 4.36   12392 calls, 46.5 USD at 1845
  this branch, 16 questions per call: 3.30 / 5.24 / 3.76 / 3.09   6932 calls, 31.1 USD at 1845
                                      (BENCH_MAX_QUESTIONS=16 BENCH_RELATIONS_PER_CALL=0)
  this branch, default 96 per call:   1.06 / 1.36 / 1.17 / 1.05   2153 calls, 30.8 USD at 1845
Past the cluster plateau (1845 -> 3690) the extra notes cost ~1.0 call each: linear. Cost
is driven by input tokens, which batching does not change; the pair cache and the cosine
floor cut the judged pairs (main 57k judged vs 33k at 1845 notes in the scaled corpus).

  BENCH_LABEL=after PROSPECTA_TEST_PG_URL=postgresql://... python scripts/bench_linker.py 500 1000 1845
"""
from __future__ import annotations

import datetime
import json
import math
import os
import random
import sys
import time
from urllib.parse import urlparse, urlunparse

import psycopg

from prospecta._linker import JevRelationJudge, Linker
from prospecta.db.migrate import run_migrations
from prospecta.memory import Memory

DIM = 64
PRICE_IN, PRICE_OUT = 3e-6, 15e-6   # USD per token (Sonnet-class pricing, a model)
OUT_TOKENS_PER_QUESTION = 8


def corpus(n: int, seed: int = 7):
    """Returns (texts, vectors by text, cluster id by index or -1, planted causal
    partner by index). Cluster sizes: one big (226), six mid (30..60), the rest unrelated. A planted causal partner (near copy) for
    every 10th unrelated-or-cluster note is the typed-link recall target."""
    rng = random.Random(seed)

    def unit(v):
        s = math.sqrt(sum(x * x for x in v))
        return [x / s for x in v]

    def gauss():
        return unit([rng.gauss(0, 1) for _ in range(DIM)])

    def mix(*ws):
        return unit([sum(w * v[i] for w, v in ws) for i in range(DIM)])
    common = gauss()
    # BENCH_CLUSTERS=fixed (default): absolute cluster sizes as in the real corpus
    # (226 + 60,50,45,40,35,30 notes), halved only where n is too small to hold them,
    # so growth in n adds unrelated notes. scaled: every cluster grows with n (the
    # worst case: judging all pairs inside a cluster is inherently quadratic in its size).
    sizes = [226, 60, 50, 45, 40, 35, 30]
    if os.environ.get("BENCH_CLUSTERS", "fixed") == "scaled":
        shares = [sz / 1845 for sz in sizes]
    else:
        k = min(1.0, n / (2 * sum(sizes)))
        shares = [sz * k / n for sz in sizes]
    parents = [gauss() for _ in range(3)]
    centres = [mix((0.7, parents[k % 3]), (0.7, gauss())) for k in range(len(shares))]
    plan = [k for k, sh in enumerate(shares) for _ in range(round(sh * n))]
    plan += [-1] * (n - len(plan))
    rng.shuffle(plan)
    texts, vecs, cluster, partner = [], {}, [], {}
    for i, k in enumerate(plan):
        if k >= 0:   # near-duplicate topical note: cosines ~0.8 inside the cluster
            v = mix((0.55, common), (0.7, centres[k]), (0.45, gauss()))
        else:        # unrelated note
            v = mix((0.55, common), (0.85, gauss()))
        texts.append(f"note {i} body")
        vecs[texts[-1]] = v
        cluster.append(k)
    for i in range(0, n - 1, 10):   # planted partner: the next note becomes a near copy of note i
        j = i + 1
        vecs[texts[j]] = mix((0.93, vecs[texts[i]]), (0.07, gauss()))
        cluster[j] = cluster[i]
        partner[i] = j
    return texts, vecs, cluster, partner


def run(base_url: str, n: int, mode: str):
    notes, vecs, cluster, partner = corpus(n)
    db = f"bench_{mode}_{n}_{int(time.time())}".lower()
    with psycopg.connect(base_url, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{db}"')
    url = urlunparse(urlparse(base_url)._replace(path=f"/{db}"))
    with psycopg.connect(url, autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
    run_migrations(url)
    m = Memory(database_url=url, bank_id="b", llm=None,
               embed=lambda ts: [vecs[t] for t in ts])
    m.create_bank("b", embedding_dim=DIM)
    st = {"calls": 0, "cost": 0.0}

    def jev(req, timeout):
        qs = req["questions"]
        tin = len(json.dumps(req)) // 4
        tout = OUT_TOKENS_PER_QUESTION * len(qs)
        cost = tin * PRICE_IN + tout * PRICE_OUT
        st["calls"] += 1
        st["cost"] += cost
        src = req["state"]["query"]
        ans = {}
        for k, q in qs.items():
            rel, cand = q["instructions"]["title"], q["instructions"]["text"]
            a, b = int(src.split()[1]), int(cand.split()[1])
            planted = partner.get(min(a, b)) == max(a, b)
            same = cluster[a] >= 0 and cluster[a] == cluster[b]
            ans[k] = {"score": 3 if (planted and rel != "caused_by") or (rel == "semantic" and same) else 0}
        return {"model": "stub", "answers": ans,
                "usage": {"input_tokens": tin, "output_tokens": tout, "cost": cost}}
    step_s = {"_temporal": 0.0, "_entities": 0.0, "_semantic": 0.0}
    originals = {k: getattr(Linker, k) for k in step_s}

    def timed(name):
        def run_step(self, *a, **kw):
            t = time.monotonic()
            try:
                return originals[name](self, *a, **kw)
            finally:
                step_s[name] += time.monotonic() - t
        return run_step
    for k in step_s:
        setattr(Linker, k, timed(k))
    try:
        t0 = time.monotonic()
        for i, t in enumerate(notes):
            m.retain(t, source=f"s{i}", index_text=t, metadata={   # one note a day, as a diary
                "created": (datetime.date(2020, 1, 1) + datetime.timedelta(days=i // 2)).isoformat()})
        t_retain = time.monotonic() - t0
        kw = {}   # BENCH_MAX_QUESTIONS / BENCH_RELATIONS_PER_CALL=0 select the other settings
        if os.environ.get("BENCH_MAX_QUESTIONS"):
            kw["max_questions_per_call"] = int(os.environ["BENCH_MAX_QUESTIONS"])
        if os.environ.get("BENCH_RELATIONS_PER_CALL") == "0":
            kw["relations_per_call"] = False
        m._linker = Linker(judge=JevRelationJudge(jev, **kw))
        t1 = time.monotonic()
        with psycopg.connect(url) as c:
            ids = [r[0] for r in c.execute("SELECT id::text FROM documents ORDER BY created_at, id")]
        for d in ids:
            m.link_document(d)
        t_link = time.monotonic() - t1
    finally:
        for k, f in originals.items():
            setattr(Linker, k, f)
    with psycopg.connect(url) as c:
        ex, jd = c.execute("SELECT coalesce(sum((stats->>'candidates_examined')::int),0), "
                           "coalesce(sum((stats->>'candidates_judged')::int),0) "
                           "FROM memory_link_state").fetchone()
        sem = c.execute("SELECT count(*) FROM memory_links WHERE origin='jev' "
                        "AND link_type='SEMANTIC'").fetchone()[0]
        rows = c.execute(
            "SELECT s.source, d.source FROM memory_links l JOIN memory_items a ON a.id=l.src "
            "JOIN documents s ON s.id=a.document_id JOIN memory_items b ON b.id=l.dst "
            "JOIN documents d ON d.id=b.document_id WHERE l.origin='jev' AND l.link_type='CAUSAL'").fetchall()
    got = {(a, b) for a, b in rows}
    hit = sum(1 for a, b in partner.items() if (f"s{a}", f"s{b}") in got)
    m.close()
    return {"mode": mode, "notes": n, "link_s": round(t_link, 1), "semantic_s": round(step_s["_semantic"], 1),
            "temporal_s": round(step_s["_temporal"], 1), "retain_s": round(t_retain, 1),
            "llm_calls": st["calls"], "calls_per_note": round(st["calls"] / n, 2),
            "cost_usd": round(st["cost"], 2), "cost_per_note_usd": round(st["cost"] / n, 5),
            "semantic_links": int(sem),
            "pairs_examined": int(ex), "pairs_judged": int(jd),
            "judged_per_note": round(int(jd) / n, 1),
            "typed_recall": f"{hit}/{len(partner)}"}


if __name__ == "__main__":
    base = os.environ["PROSPECTA_TEST_PG_URL"]
    sizes = [int(x) for x in sys.argv[1:]] or [500, 1000, 1845]
    label = os.environ.get("BENCH_LABEL", "run")
    for n in sizes:
        print(json.dumps(run(base, n, label)), flush=True)
