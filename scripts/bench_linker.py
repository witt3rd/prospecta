"""Linker cost benchmark: import time, LLM-call count / cost and candidate pairs
examined on a synthetic note set, for the old neighbour rule (cosine >= 0.6 x
nearest, no marginal stop) and the current one. Stub Jev counts calls and prices
them; nothing leaves the process.

  PROSPECTA_TEST_PG_URL=postgresql://... python scripts/bench_linker.py 500 1000 1845
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

import prospecta._linker as L
from prospecta._linker import JevRelationJudge, Linker
from prospecta._scorecut import fetch_until_cut
from prospecta.db.migrate import run_migrations
from prospecta.memory import Memory

DIM = 32
PRICE_IN, PRICE_OUT = 3e-6, 15e-6   # USD per token (Sonnet-class pricing)


def corpus(n: int, seed: int = 7):
    """Notes in topics of ~15 with compressed cosines (shared component), plus a
    planted causal partner for every even note (the typed-link recall target)."""
    rng = random.Random(seed)

    def unit(v):
        s = math.sqrt(sum(x * x for x in v))
        return [x / s for x in v]

    def gauss():
        return [rng.gauss(0, 1) for _ in range(DIM)]
    common = unit(gauss())
    topics = [unit(gauss()) for _ in range(max(1, n // 15))]
    notes, vecs, partner = [], {}, {}
    for i in range(n):
        if i % 2 == 1 and i - 1 in vecs_by_idx(vecs, notes):
            base = vecs_by_idx(vecs, notes)[i - 1]
            v = unit([0.93 * b + 0.07 * g for b, g in zip(base, unit(gauss()))])
            partner[i - 1] = i
        else:
            t = topics[rng.randrange(len(topics))]
            v = unit([0.7 * c + 0.6 * tt + 0.45 * g for c, tt, g in zip(common, t, unit(gauss()))])
        text = f"note {i} body"
        notes.append(text)
        vecs[text] = v
    return notes, vecs, partner


def vecs_by_idx(vecs, notes):
    return {i: vecs[t] for i, t in enumerate(notes)}


def run(base_url: str, n: int, mode: str):
    notes, vecs, partner = corpus(n)
    db = f"bench_{mode}_{n}_{int(time.time())}"
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
        cost = tin * PRICE_IN + 3 * PRICE_OUT
        st["calls"] += 1
        st["cost"] += cost
        src = req["state"]["query"]
        ans = {}
        for k, q in qs.items():
            rel, cand = q["instructions"]["title"], q["instructions"]["text"]
            a, b = int(src.split()[1]), int(cand.split()[1])
            planted = partner.get(min(a, b)) == max(a, b)
            ans[k] = {"score": 3 if planted and rel != "caused_by" else 0}
        return {"model": "stub", "answers": ans,
                "usage": {"input_tokens": tin, "output_tokens": 3, "cost": cost}}
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
    old = L.neighbours_until_drop
    if mode == "before":
        L.neighbours_until_drop = lambda fetch, score, rel, page=64: (
            lambda rows: (rows, len(rows)))(fetch_until_cut(fetch, score, 0.6))
    try:
        t0 = time.monotonic()
        for i, t in enumerate(notes):
            m.retain(t, source=f"s{i}", index_text=t, metadata={   # one note a day, as a diary
                "created": (datetime.date(2020, 1, 1) + datetime.timedelta(days=i // 2)).isoformat()})
        t_retain = time.monotonic() - t0
        m._linker = Linker(judge=JevRelationJudge(jev))
        t1 = time.monotonic()
        with psycopg.connect(url) as c:
            ids = [r[0] for r in c.execute("SELECT id::text FROM documents ORDER BY created_at, id")]
        for d in ids:
            m.link_document(d)
        t_link = time.monotonic() - t1
    finally:
        L.neighbours_until_drop = old
        for k, f in originals.items():
            setattr(Linker, k, f)
    with psycopg.connect(url) as c:
        ex, jd = c.execute("SELECT coalesce(sum((stats->>'candidates_examined')::int),0), "
                           "coalesce(sum((stats->>'candidates_judged')::int),0) "
                           "FROM memory_link_state").fetchone()
        rows = c.execute(
            "SELECT s.source, d.source FROM memory_links l JOIN memory_items a ON a.id=l.src "
            "JOIN documents s ON s.id=a.document_id JOIN memory_items b ON b.id=l.dst "
            "JOIN documents d ON d.id=b.document_id WHERE l.origin='jev' AND l.link_type='CAUSAL'").fetchall()
    got = {(a, b) for a, b in rows}
    hit = sum(1 for a, b in partner.items() if (f"s{a}", f"s{b}") in got)
    m.close()
    return {"mode": mode, "notes": n, "link_s": round(t_link, 1), "semantic_s": round(step_s["_semantic"], 1),
            "temporal_s": round(step_s["_temporal"], 1), "retain_s": round(t_retain, 1),
            "llm_calls": st["calls"], "cost_usd": round(st["cost"], 2),
            "pairs_examined": int(ex), "pairs_judged": int(jd),
            "typed_recall": f"{hit}/{len(partner)}"}


if __name__ == "__main__":
    base = os.environ["PROSPECTA_TEST_PG_URL"]
    sizes = [int(x) for x in sys.argv[1:]] or [500, 1000, 1845]
    modes = os.environ.get("BENCH_MODES", "before,after").split(",")
    for mode in modes:
        for n in sizes:
            print(json.dumps(run(base, n, mode)), flush=True)
