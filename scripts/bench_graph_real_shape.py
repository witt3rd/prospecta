"""Measure the graph channel on a REAL-SHAPED synthetic graph (scout V5 defect 3).

Usage: PROSPECTA_TEST_PG_URL=postgresql://... python scripts/bench_graph_real_shape.py [queries]

The private corpus stays on roger; this builds the shape the scout measured: 1,845 notes
(4 chunk + 5 question items each, ~16.6k items), ~250k links, HUBS hub entities of HUB_SIZE+
items, dense topical clusters of 100-250 near-duplicate notes whose items link to many other
items of the cluster (half of them mutual), a TEMPORAL chain, and small entities whose holders
are linked pairwise (ENTITY). Each query is a 10-document pool: most seeds from one cluster
(near-duplicates), a few hub holders and strays. Prints per-stage mean / p90 (seconds) and the
whole channel's. BENCH_OLD=<path to another graph.py> also checks it returns identical candidates. A throwaway database is created and dropped.
"""
from __future__ import annotations

import os
import random
import statistics
import sys
import time

import psycopg

from prospecta.channels import Candidate, GraphExpand, QueryPlan, RecallState
from prospecta.channels import graph as graph_mod
from prospecta.db.migrate import run_migrations

sys.path.insert(0, os.path.dirname(__file__))
import bench_graph_channel as old   # noqa: E402

DOCS = 1845
CHUNKS, QUESTIONS = 4, 5
CLUSTER_LINKS = 8         # per item, to items of other notes of its cluster; half also reversed
HUBS, HUB_SIZE = 5, 5200
SMALL_ENTITIES, SMALL_HOLDERS = 300, 10
SEEDS = 10


def build(url: str) -> list[list[str]]:
    run_migrations(url)
    from prospecta.memory import Memory
    m = Memory(database_url=url, bank_id="b", llm=None, embed=lambda t: [[1.0, 0.0, 0.0]] * len(t))
    m.create_bank("b", embedding_dim=3)
    m.close()
    random.seed(11)
    sizes, left = [], DOCS - 100
    while left >= 100:
        s = min(random.randint(100, 250), left)
        sizes.append(s)
        left -= s
    with psycopg.connect(url) as c:
        c.execute(f"""INSERT INTO documents (id, bank_id, source, original_text, content_hash, created_at)
          SELECT gen_random_uuid(), 'b', 's' || g, 'text ' || g, 'h' || g, now() - (g || ' minutes')::interval
          FROM generate_series(1, {DOCS}) g""")
        c.execute(f"""INSERT INTO memory_items (id, bank_id, document_id, content, original_chunk, embedding, kind, ordinal)
          SELECT gen_random_uuid(), 'b', d.id, 'c', 'chunk', '[1,0,0]',
                 CASE WHEN k < {CHUNKS} THEN 'chunk' ELSE 'question' END, CASE WHEN k < {CHUNKS} THEN k END
          FROM documents d, generate_series(0, {CHUNKS + QUESTIONS - 1}) k""")
        # n: items numbered in document order; cl: the cluster of a document (0 = none)
        c.execute("""CREATE TEMP TABLE dn AS SELECT id, row_number() OVER (ORDER BY created_at DESC) AS d FROM documents""")
        c.execute("""CREATE TEMP TABLE n AS SELECT i.id, i.document_id, dn.d, row_number() OVER (ORDER BY dn.d, i.id) AS r
                     FROM memory_items i JOIN dn ON dn.id = i.document_id""")
        c.execute("CREATE INDEX ON n (r)")
        lo = 1
        bounds = []
        for ci, s in enumerate(sizes):
            bounds.append((lo, lo + s - 1))   # document numbers of cluster ci
            lo += s
        per = CHUNKS + QUESTIONS
        for a, b in bounds:
            ra, rb = (a - 1) * per + 1, b * per
            c.execute(f"""CREATE TEMP TABLE p AS
              SELECT x.r AS ra, {ra} + floor(random() * {rb - ra + 1})::int AS rb, random() < 0.5 AS mutual
              FROM n x, generate_series(1, {CLUSTER_LINKS}) k WHERE x.r BETWEEN {ra} AND {rb}""")
            c.execute("""INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, confidence, origin)
              SELECT 'b', a.id, b.id, 'SEMANTIC', 'RELATED_TO', 0.75 + random() * 0.25, 'pgvector'
              FROM p JOIN n a ON a.r = p.ra JOIN n b ON b.r = p.rb AND b.document_id <> a.document_id
              ON CONFLICT DO NOTHING""")
            c.execute("""INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, confidence, origin)
              SELECT 'b', b.id, a.id, 'SEMANTIC', 'RELATED_TO', 0.75 + random() * 0.25, 'pgvector'
              FROM p JOIN n a ON a.r = p.ra JOIN n b ON b.r = p.rb AND b.document_id <> a.document_id
              WHERE p.mutual ON CONFLICT DO NOTHING""")
            c.execute("DROP TABLE p")
        c.execute("""INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, origin)
          SELECT 'b', a.id, b.id, 'TEMPORAL', 'PRECEDES', 'sql' FROM n a JOIN n b ON b.r = a.r + 1 ON CONFLICT DO NOTHING""")
        c.execute("""INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, origin)
          SELECT 'b', b.id, a.id, 'TEMPORAL', 'SUCCEEDS', 'sql' FROM n a JOIN n b ON b.r = a.r + 1 ON CONFLICT DO NOTHING""")
        total = DOCS * per
        hub_items = []
        for h in range(HUBS):
            eid = c.execute("INSERT INTO memory_entities (bank_id, name, norm, etype) VALUES ('b', %s, %s, 'person') RETURNING id",
                            (f"hub{h}", f"hub{h}")).fetchone()[0]
            members = [r[0] for r in c.execute("SELECT id FROM n ORDER BY random() LIMIT %s", (HUB_SIZE,))]
            with c.cursor() as cur:
                cur.executemany("INSERT INTO memory_item_entities (item_id, entity_id) VALUES (%s, %s)",
                                [(it, eid) for it in members])
            hub_items.append(members)
        for e in range(SMALL_ENTITIES):
            eid = c.execute("INSERT INTO memory_entities (bank_id, name, norm, etype) VALUES ('b', %s, %s, 'other') RETURNING id",
                            (f"e{e}", f"e{e}")).fetchone()[0]
            hold = [r[0] for r in c.execute("SELECT id FROM n ORDER BY random() LIMIT %s", (SMALL_HOLDERS,))]
            with c.cursor() as cur:
                cur.executemany("INSERT INTO memory_item_entities (item_id, entity_id) VALUES (%s, %s)",
                                [(it, eid) for it in hold])
                cur.executemany("""INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, origin)
                                   VALUES ('b', %s, %s, 'ENTITY', 'SHARED_ENTITY', 'sql') ON CONFLICT DO NOTHING""",
                                [(a, b) for i, a in enumerate(hold) for b in hold[i + 1:]])
        c.execute("ANALYZE")
        c.commit()
        nl = c.execute("SELECT count(*) FROM memory_links").fetchone()[0]
        print(f"docs={DOCS} items={total} links={nl} clusters={sizes} hubs={HUBS}x{HUB_SIZE}")
        cdocs = []
        for a, b in bounds:
            cdocs.append([str(r[0]) for r in c.execute("SELECT id FROM dn WHERE d BETWEEN %s AND %s", (a, b))])
        return cdocs


def main() -> None:
    queries = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    base = os.environ["PROSPECTA_TEST_PG_URL"]
    name = f"benchrs_{int(time.time())}"
    with psycopg.connect(base, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{name}"')
    url = base.replace("/postgres?", f"/{name}?") if "/postgres?" in base else base.rsplit("/", 1)[0] + f"/{name}"
    try:
        with psycopg.connect(url, autocommit=True) as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cdocs = build(url)
        with psycopg.connect(url) as c:
            hubdocs = [str(r[0]) for r in c.execute(
                "SELECT DISTINCT document_id FROM memory_items i JOIN memory_item_entities ie ON ie.item_id = i.id "
                "JOIN memory_entities e ON e.id = ie.entity_id WHERE e.name LIKE 'hub%' LIMIT 500")]
            alld = [str(r[0]) for r in c.execute("SELECT id FROM documents")]
            g = GraphExpand()
            times, rows, py = [], [], []
            for q in range(queries):
                cl = random.choice(cdocs)
                pool = random.sample(cl, 6) + random.sample(hubdocs, 2) + random.sample(alld, 2)
                st = RecallState(conn=c, bank_id="b", pool=[
                    Candidate(document_id=d, item_id=None, source="", channel="dense_chunk", rank=i + 1,
                              score=1.0, evidence="") for i, d in enumerate(pool)])
                t0 = time.perf_counter()
                sd = g.seeds(st, graph_mod.DEFAULT_SEED_MIN_REL)
                seed_s = time.perf_counter() - t0
                rows.append({"seed selection (python)": seed_s, **old.stage_times(c, g, st, sd)})
                if os.environ.get("BENCH_OLD"):   # compare with another graph.py (a path): same candidates?
                    import importlib.util
                    sp = importlib.util.spec_from_file_location("graph_old", os.environ["BENCH_OLD"])
                    mod = importlib.util.module_from_spec(sp); sp.loader.exec_module(mod)
                    a = [(x.document_id, x.item_id, round(x.score, 9), x.detail["hops"]) for x in mod.GraphExpand().retrieve(QueryPlan(text="q"), st)]
                    b = [(x.document_id, x.item_id, round(x.score, 9), x.detail["hops"]) for x in g.retrieve(QueryPlan(text="q"), st)]
                    print(f"query {q}: identical={a == b} docs={len(a)} {len(b)} same_docs={ {x[0] for x in a} == {x[0] for x in b} }")
                t0 = time.perf_counter()
                out = g.retrieve(QueryPlan(text="q"), st, 50)
                times.append(time.perf_counter() - t0)
                c.rollback()
            times.sort()
            old.report_stages(rows)
            print(f"graph channel over {queries} queries: mean {statistics.mean(times):.3f}s "
                  f"p90 {times[int(0.9 * (len(times) - 1))]:.3f}s max {times[-1]:.3f}s (last result {len(out)} docs)")
    finally:
        with psycopg.connect(base, autocommit=True) as c:
            c.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


if __name__ == "__main__":
    main()
