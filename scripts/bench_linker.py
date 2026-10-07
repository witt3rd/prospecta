"""Linker cost benchmark on real-shaped notes: DENSE clusters of near-duplicate topical
notes (a Greg-style 226-note cluster + clusters of 170/130/110/100 notes, ~40 percent of
the 1,845 notes), the rest unrelated; a third of the notes have 2-3 chunks (several
anchors per note); a few exact duplicates (same body modulo whitespace / frontmatter);
compressed cosines. The Jev judge is a STUB that counts calls and PRICES them with a
per-call token model; calls, tokens and cost are a MODEL of the real service (nothing
leaves the process). The neighbour selection, batching, mutual / nearest / duplicate
rules and the pair cache are the real code under test.

Ground truth (the generator knows it, the stub judge answers from it): notes of one
cluster are SEMANTICALLY related; every 10th note has a planted CAUSAL partner (a near
copy). Reported recall: planted causal links found; semantic cluster-pair recall
(DIRECT: pair has a link) and cluster CONNECTIVITY (pairs joined by a path of links;
SEMANTIC is transitive within a topical cluster, CAUSAL is not).

Run once per code version (the label is only printed): the same script runs against
current main (git archive into a dir, PYTHONPATH=<dir>) and against this branch.
Results are in docs/graph-links.md.

  BENCH_LABEL=after PROSPECTA_TEST_PG_URL=postgresql://... python scripts/bench_linker.py 500 1000 1845 3690
"""
from __future__ import annotations

import datetime
import json
import math
import os
import random
import re
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


def note_text(i: int, body_of: int, rng) -> str:
    """Note i: 1-3 paragraphs of ~600 chars (chunks) naming note `body_of`; an exact
    duplicate gets the same body with other whitespace and a frontmatter block."""
    r = random.Random(body_of)
    paras = [f"note {body_of} part {k} " + f"note {body_of} lorem ipsum dolor sit amet. " * 12
             for k in range(1 + (r.random() < 0.34) + (r.random() < 0.15))]
    if body_of != i:
        return "---\ntitle: copy\n---\n\n" + "\n\n\n".join(p.replace(" ", "  ") for p in paras)
    return "\n\n".join(paras)


def corpus(n: int, seed: int = 7):
    """Returns (texts, vectors by note index, cluster id by index or -1, planted causal
    partner by index). Cluster sizes: 226 (Greg-style) + 170/130/110/100 at n=1845; a
    planted causal partner (near copy) for every 10th note is the typed-link target;
    every 50th note is followed by an exact duplicate modulo whitespace / frontmatter."""
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
    sizes = [226, 170, 130, 110, 100]
    if os.environ.get("BENCH_CLUSTERS", "fixed") == "scaled":
        shares = [sz / 1845 for sz in sizes]
    else:
        k = min(1.0, n / 1845)
        shares = [sz * k / n for sz in sizes]
    parents = [gauss() for _ in range(3)]
    centres = [mix((0.7, parents[k % 3]), (0.7, gauss())) for k in range(len(shares))]
    plan = [k for k, sh in enumerate(shares) for _ in range(round(sh * n))]
    plan += [-1] * (n - len(plan))
    rng.shuffle(plan)
    vecs, cluster, partner, dup = {}, [], {}, {}
    for i, k in enumerate(plan):
        if k >= 0:   # near-duplicate topical note: cosines ~0.8 inside the cluster
            v = mix((0.55, common), (0.7, centres[k]), (0.45, gauss()))
        else:        # unrelated note
            v = mix((0.55, common), (0.85, gauss()))
        vecs[i] = v
        cluster.append(k)
    hub_n = int(os.environ.get("BENCH_HUB", "0"))   # a hub note with hub_n near neighbours of graded closeness
    if hub_n:   # the last hub_n notes lean on the first unrelated note by a in [0.2, 0.95]; a >= 0.8 is truly related
        hub = next(i for i, k in enumerate(plan) if k < 0)
        cluster[hub] = 99
        for i in range(n - hub_n, n):
            if i != hub:
                a = 0.2 + 0.75 * rng.random()
                vecs[i] = mix((0.55, common), (a, vecs[hub]), (0.45, gauss()))
                cluster[i] = 99 if a >= 0.8 else -1
    for i in range(0, n - 1, 10):   # planted partner: the next note becomes a near copy of note i
        j = i + 1
        vecs[j] = mix((0.93, vecs[i]), (0.07, gauss()))
        cluster[j] = cluster[i]
        partner[i] = j
    for i in range(0, n - 3, 50):   # exact duplicate of note i, three notes later (own vector copy)
        dup[i + 3] = i
        vecs[i + 3] = vecs[i]
        cluster[i + 3] = cluster[i]
        partner.pop(i + 2, None)
    texts = [note_text(i, dup.get(i, i), rng) for i in range(n)]
    return texts, vecs, cluster, partner, dup


def run(base_url: str, n: int, mode: str):
    notes, vecs, cluster, partner, dup = corpus(n)
    db = f"bench_{mode}_{n}_{int(time.time())}".lower()
    with psycopg.connect(base_url, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{db}"')
    url = urlunparse(urlparse(base_url)._replace(path=f"/{db}"))
    with psycopg.connect(url, autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
    run_migrations(url)
    st = {"calls": 0, "cost": 0.0}
    cur = {"note": 0}

    def embed(ts):   # the note's vector plus a small per-text jitter (chunks of one note differ a little)
        out = []
        for t in ts:
            tag = re.search(r"note\s+(\d+)", t)   # a chunk cut mid-paragraph has no tag: the note being retained
            v = vecs[int(tag.group(1)) if tag else cur["note"]]
            r = random.Random(t)
            w = [x + 0.02 * r.gauss(0, 1) / math.sqrt(DIM) for x in v]
            nrm = math.sqrt(sum(x * x for x in w))
            out.append([x / nrm for x in w])
        return out
    m = Memory(database_url=url, bank_id="b", llm=None, embed=embed)
    m.create_bank("b", embedding_dim=DIM)

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
            a, b = (int(re.search(r"note\s+(\d+)", x).group(1)) for x in (src, cand))
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
            cur["note"] = i
            m.retain(t, source=f"s{i}", index_text=f"note {dup.get(i, i)} summary", child_chunks=True, metadata={   # one note a day, as a diary
                "created": (datetime.date(2020, 1, 1) + datetime.timedelta(days=i // 2)).isoformat()})
        t_retain = time.monotonic() - t0
        kw = {}   # BENCH_FLOOR_FRAC sets judge_floor_frac (0 = the unbounded all-pairs pass); BENCH_MAX_QUESTIONS / BENCH_RELATIONS_PER_CALL=0 select the other settings
        if os.environ.get("BENCH_MAX_QUESTIONS"):
            kw["max_questions_per_call"] = int(os.environ["BENCH_MAX_QUESTIONS"])
        if os.environ.get("BENCH_RELATIONS_PER_CALL") == "0":
            kw["relations_per_call"] = False
        m._linker = Linker(judge=JevRelationJudge(jev, **kw),
                          link_completeness=os.environ.get("BENCH_COMPLETENESS") or None,
                          **({"judge_floor_frac": float(os.environ["BENCH_FLOOR_FRAC"])}
                             if os.environ.get("BENCH_FLOOR_FRAC") else {}))
        t1 = time.monotonic()
        with psycopg.connect(url) as c:
            ids = [r[0] for r in c.execute("SELECT id::text FROM documents ORDER BY created_at, id")]
        per_doc = []   # (calls, judged) per document
        with psycopg.connect(url) as c2:
            for d in ids:
                before = st["calls"]
                m.link_document(d)
                jd1 = c2.execute("SELECT coalesce((stats->>'candidates_judged')::int,0) "
                                 "FROM memory_link_state WHERE document_id=%s", (d,)).fetchone()[0]
                per_doc.append((st["calls"] - before, jd1))
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
        sem_rows = c.execute(
            "SELECT s.source, d.source FROM memory_links l JOIN memory_items a ON a.id=l.src "
            "JOIN documents s ON s.id=a.document_id JOIN memory_items b ON b.id=l.dst "
            "JOIN documents d ON d.id=b.document_id WHERE l.origin='jev' AND l.link_type='SEMANTIC'").fetchall()
    got = {(a, b) for a, b in rows}
    direct = {frozenset((a, b)) for a, b in sem_rows}
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in sem_rows:
        parent[find(a)] = find(b)
    members: dict = {}
    for i, k in enumerate(cluster):
        if k >= 0:
            members.setdefault(k, []).append(f"s{i}")
    pairs = d_ok = c_ok = 0
    for ms in members.values():
        for x in range(len(ms)):
            for y in range(x + 1, len(ms)):
                pairs += 1
                d_ok += frozenset((ms[x], ms[y])) in direct
                c_ok += find(ms[x]) == find(ms[y])
    hit = sum(1 for a, b in partner.items() if (f"s{a}", f"s{b}") in got)
    m.close()
    with psycopg.connect(base_url, autocommit=True) as c:   # the run's database is disposable
        c.execute(f'DROP DATABASE "{db}" WITH (FORCE)')
    return {"mode": mode, "notes": n, "link_s": round(t_link, 1), "semantic_s": round(step_s["_semantic"], 1),
            "temporal_s": round(step_s["_temporal"], 1), "retain_s": round(t_retain, 1),
            "llm_calls": st["calls"], "calls_per_note": round(st["calls"] / n, 2),
            "cost_usd": round(st["cost"], 2), "cost_per_note_usd": round(st["cost"] / n, 5),
            "semantic_links": int(sem),
            "pairs_examined": int(ex), "pairs_judged": int(jd),
            "judged_per_note": round(int(jd) / n, 1),
            "worst_doc_judged": max(j for _, j in per_doc), "worst_doc_calls": max(c for c, _ in per_doc),
            "calls_per_note_mean": round(sum(c for c, _ in per_doc) / n, 2),
            "causal_recall": f"{hit}/{len(partner)}",
            "semantic_pair_direct_recall": round(d_ok / max(pairs, 1), 4),
            "semantic_cluster_connectivity": round(c_ok / max(pairs, 1), 4)}


if __name__ == "__main__":
    base = os.environ["PROSPECTA_TEST_PG_URL"]
    sizes = [int(x) for x in sys.argv[1:]] or [500, 1000, 1845]
    label = os.environ.get("BENCH_LABEL", "run")
    for n in sizes:
        print(json.dumps(run(base, n, label)), flush=True)
