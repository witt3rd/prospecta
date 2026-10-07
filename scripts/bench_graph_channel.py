"""Measure the graph channel on a synthetic large links table (F12).

BENCH_STAGES=1 adds per-stage timings (cumulative CTE targets of graph._SQL).
Usage: PROSPECTA_TEST_PG_URL=postgresql://... python scripts/bench_graph_channel.py [docs] [queries] [items_per_doc] [semantic_per_item]

Builds a throwaway database (dropped at the end) with `docs` documents of `items_per_doc` items,
`semantic_per_item` links per item (skewed in-degree, so some items are hubs: PRECEDES/SUCCEEDS
chain + skewed SEMANTIC), and HUBS person hubs of HUB_SIZE items, then times
GraphExpand over random 10-seed pools. Prints mean / p90 / max seconds.
Hub entities (HUBS x HUB_SIZE items) carry no per-anchor links: the channel reaches their holders through the
entity table. Defaults give ~250k links over 60k items, 4 hubs of 5k items, 2 hops, 30 seeds.
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


SEEDS = 30        # seed documents per query (rrf >= 0.5 x best admits many)
HUBS = 4          # hub entities
HUB_SIZE = 5000   # items held by each hub entity
LINK_HUBS = 4     # items that 5k other items link to (memory_links hubs, both link directions)
LINK_HUB_DEGREE = 5000


# Cumulative stage targets of graph._SQL: running the CTE chain up to a target (CTEs that are not
# referenced are never executed) and subtracting the previous cumulative time gives the stage.
STAGES = [("hop 1 edges", "e1"), ("hub join (hop 1)", "a_hubs"), ("hop 1 merge", "h1"),
          ("frontier", "frontier"), ("hub join (hop 2)", "b_hubs"), ("hop 2 edges", "e2"),
          ("hop 2 merge", "h2"), ("reach+scoring prep", "top"), ("final scoring/join", None)]


def stage_times(c, g, state, docs_ws) -> dict[str, float]:
    """Cumulative seconds per target for one query, plus seed selection."""
    docs, ws = docs_ws
    tw = {**graph_mod.DEFAULT_TYPE_WEIGHTS}
    args = {"docs": docs, "ws": ws, "bank": state.bank_id, "decay": graph_mod.DEFAULT_DECAY,
            "tw": __import__("json").dumps(tw), "wmax": max([1.0, *tw.values()]),
            "max_hops": graph_mod.MAX_HOPS, "rel": graph_mod.DEFAULT_NODE_MIN_REL,
            "hub": graph_mod.ENTITY_HUB, "hub_cap": graph_mod.DEFAULT_HUB_CAP}
    cut = graph_mod._SQL.index("SELECT t.item_id")
    out = {}
    # a CTE that is only a prefix of the chain: s0 first, then each target
    for label, target in [("seeds -> s0", "s0")] + STAGES:
        sql = graph_mod._SQL if target is None else graph_mod._SQL[:cut] + f"SELECT count(*) FROM {target}"
        t0 = time.perf_counter()
        with c.cursor() as cur:
            cur.execute(sql, args)
            cur.fetchall()
        out[label] = time.perf_counter() - t0
    c.rollback()
    return out


def report_stages(rows: list[dict[str, float]]) -> None:
    print("stage                    mean(s)   p90(s)   (increment over the previous target)")
    prev = None
    worst = (None, -1.0)
    for label in rows[0]:
        inc = [r[label] - (r[prev] if prev else 0.0) for r in rows]
        inc_s = sorted(inc)
        mean = statistics.mean(inc)
        print(f"  {label:<22} {mean:7.3f}  {inc_s[int(0.9 * (len(inc_s) - 1))]:7.3f}")
        if mean > worst[1]:
            worst = (label, mean)
        prev = None if label == "seed selection (python)" else label
    print(f"slowest stage: {worst[0]} ({worst[1]:.3f}s mean)")


def main() -> None:
    docs = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
    queries = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    per_doc = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    sem = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    base = os.environ["PROSPECTA_TEST_PG_URL"]
    name = f"bench_{int(time.time())}"
    with psycopg.connect(base, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{name}"')
    url = base.replace("/postgres?", f"/{name}?") if "/postgres?" in base else base.rsplit("/", 1)[0] + f"/{name}"
    try:
        with psycopg.connect(url, autocommit=True) as c:
            c.execute("CREATE EXTENSION IF NOT EXISTS vector")
        run_migrations(url)
        from prospecta.memory import Memory
        m = Memory(database_url=url, bank_id="b", llm=None, embed=lambda t: [[1.0, 0.0, 0.0]] * len(t))
        m.create_bank("b", embedding_dim=3)
        m.close()
        random.seed(7)
        with psycopg.connect(url) as c:
            c.execute(f"""
              INSERT INTO documents (id, bank_id, source, original_text, content_hash, created_at)
              SELECT gen_random_uuid(), 'b', 's' || g, 'text ' || g, 'h' || g, now() - (g || ' minutes')::interval
              FROM generate_series(1, {docs}) g""")
            c.execute(f"""
              INSERT INTO memory_items (id, bank_id, document_id, content, original_chunk, embedding, kind, ordinal)
              SELECT gen_random_uuid(), 'b', d.id, 'c', 'chunk', '[1,0,0]', 'chunk', k
              FROM documents d, generate_series(0, {per_doc - 1}) k""")
            c.execute("""CREATE TEMP TABLE n AS SELECT id, row_number() OVER (ORDER BY id) AS r FROM memory_items""")
            c.execute("CREATE INDEX ON n (r)")
            c.execute(f"""
              CREATE TEMP TABLE pairs AS
              SELECT a.r AS ra, 1 + floor({docs * per_doc} * power(random(), 2))::int AS rb
              FROM n a, generate_series(1, {sem}) k""")
            c.execute("""
              INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, confidence, origin)
              SELECT 'b', a.id, b.id, 'SEMANTIC', 'RELATED_TO', 0.6 + random() * 0.4, 'pgvector'
              FROM pairs p JOIN n a ON a.r = p.ra JOIN n b ON b.r = p.rb AND b.r <> a.r
              ON CONFLICT DO NOTHING""")
            c.execute(f"""
              INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, origin)
              SELECT 'b', a.id, b.id, 'TEMPORAL', 'PRECEDES', 'sql' FROM n a JOIN n b ON b.r = a.r + 1
              ON CONFLICT DO NOTHING""")
            c.execute(f"""
              INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, origin)
              SELECT 'b', b.id, a.id, 'TEMPORAL', 'SUCCEEDS', 'sql' FROM n a JOIN n b ON b.r = a.r + 1
              ON CONFLICT DO NOTHING""")
            hubs = []
            for h in range(HUBS):
                eid = c.execute("INSERT INTO memory_entities (bank_id, name, norm, etype) "
                                "VALUES ('b', %s, %s, 'person') RETURNING id", (f"hub{h}", f"hub{h}")).fetchone()[0]
                members = [r[0] for r in c.execute(
                    "SELECT id FROM n WHERE r %% %s = %s ORDER BY r LIMIT %s", (HUBS * 2, h, HUB_SIZE))]
                with c.cursor() as cur:
                    cur.executemany("INSERT INTO memory_item_entities (item_id, entity_id) VALUES (%s, %s)",
                                    [(it, eid) for it in members])
                hubs.append(members)
            link_hubs = [r[0] for r in c.execute(
                "SELECT id FROM n WHERE r %% %s = 1 ORDER BY r LIMIT %s", (docs * per_doc // (LINK_HUBS + 1), LINK_HUBS))]
            for i, hid in enumerate(link_hubs):   # even hubs: in-links; odd hubs: out-links
                c.execute("""
                  INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, confidence, origin)
                  SELECT 'b', CASE WHEN %s THEN a.id ELSE %s END, CASE WHEN %s THEN %s ELSE a.id END,
                         'SEMANTIC', 'RELATED_TO', 0.6 + random() * 0.4, 'pgvector'
                  FROM (SELECT id FROM n WHERE id <> %s ORDER BY random() LIMIT %s) a
                  ON CONFLICT DO NOTHING""", (i % 2 == 0, hid, i % 2 == 0, hid, hid, LINK_HUB_DEGREE))
            c.execute("ANALYZE")
            c.commit()
            nl = c.execute("SELECT count(*) FROM memory_links").fetchone()[0]
            allDocs = [str(r[0]) for r in c.execute("SELECT id FROM documents")]
            print(f"docs={docs} items={docs * per_doc} links={nl}")
            times = []
            stage_rows = []
            g = GraphExpand()
            for q in range(queries):
                pool = random.sample(allDocs, SEEDS)
                if q % 3 == 0:   # a third of the queries seed from hub members
                    pool[0] = str(c.execute("SELECT document_id FROM memory_items WHERE id=%s",
                                            (random.choice(random.choice(hubs)),)).fetchone()[0])
                if q % 3 == 1:   # a third seed from a link hub (5k+ links)
                    pool[0] = str(c.execute("SELECT document_id FROM memory_items WHERE id=%s",
                                            (random.choice(link_hubs),)).fetchone()[0])
                st = RecallState(conn=c, bank_id="b", pool=[
                    Candidate(document_id=d, item_id=None, source="", channel="dense_chunk",
                              rank=i + 1, score=1.0, evidence="") for i, d in enumerate(pool)])
                if os.environ.get("BENCH_STAGES"):
                    t0 = time.perf_counter()
                    sd = g.seeds(st, graph_mod.DEFAULT_SEED_MIN_REL)
                    seed_s = time.perf_counter() - t0
                    stage_rows.append({"seed selection (python)": seed_s, **stage_times(c, g, st, sd)})
                t0 = time.perf_counter()
                out = g.retrieve(QueryPlan(text="q"), st, 50)
                times.append(time.perf_counter() - t0)
                c.rollback()
            times.sort()
            if stage_rows:
                report_stages(stage_rows)
            print(f"graph channel over {queries} queries: mean {statistics.mean(times):.3f}s "
                  f"p90 {times[int(0.9 * (len(times) - 1))]:.3f}s max {times[-1]:.3f}s (last result {len(out)} docs)")
    finally:
        if os.environ.get("BENCH_KEEP"):
            print(f"kept database {name}")
            return
        with psycopg.connect(base, autocommit=True) as c:
            c.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


if __name__ == "__main__":
    main()
