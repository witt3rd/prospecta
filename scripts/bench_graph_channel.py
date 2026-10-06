"""Measure the graph channel on a synthetic large links table (F12).

Usage: PROSPECTA_TEST_PG_URL=postgresql://... python scripts/bench_graph_channel.py [docs] [queries] [items_per_doc] [semantic_per_item]

Builds a throwaway database (dropped at the end) with `docs` documents of `items_per_doc` items,
`semantic_per_item` links per item (skewed in-degree, so some items are hubs: PRECEDES/SUCCEEDS
chain + skewed SEMANTIC), and 3 person hubs of ~190 items, then times
GraphExpand over random 10-seed pools. Prints mean / p90 / max seconds.
Link rows for hub entities follow the current Linker: per-anchor capped links
when run against the old code, entity-table joins when run against the new.
"""
from __future__ import annotations

import os
import random
import statistics
import sys
import time

import psycopg

from prospecta.channels import Candidate, GraphExpand, QueryPlan, RecallState
from prospecta.db.migrate import run_migrations


def main() -> None:
    docs = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
    queries = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    per_doc = int(sys.argv[3]) if len(sys.argv) > 3 else 9
    sem = int(sys.argv[4]) if len(sys.argv) > 4 else 12
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
            for h in range(3):
                eid = c.execute("INSERT INTO memory_entities (bank_id, name, norm, etype) "
                                "VALUES ('b', %s, %s, 'person') RETURNING id", (f"hub{h}", f"hub{h}")).fetchone()[0]
                members = [r[0] for r in c.execute("SELECT id FROM n WHERE r %% 300 = %s ORDER BY r LIMIT 190", (h,))]
                for it in members:
                    c.execute("INSERT INTO memory_item_entities (item_id, entity_id) VALUES (%s, %s)", (it, eid))
                hubs.append(members)
                if os.environ.get("BENCH_OLD_ENTITY_LINKS", "1") == "1":   # old behaviour: capped links per anchor
                    for it in members:
                        for o in random.sample(members, 10):
                            if o != it:
                                c.execute("INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, origin) "
                                          "VALUES ('b', %s, %s, 'ENTITY', 'SHARED_ENTITY', 'sql') ON CONFLICT DO NOTHING",
                                          (it, o))
            c.execute("ANALYZE")
            c.commit()
            nl = c.execute("SELECT count(*) FROM memory_links").fetchone()[0]
            allDocs = [str(r[0]) for r in c.execute("SELECT id FROM documents")]
            print(f"docs={docs} items={docs * per_doc} links={nl}")
            times = []
            g = GraphExpand()
            for q in range(queries):
                pool = random.sample(allDocs, 10)
                if q % 3 == 0:   # a third of the queries seed from a hub member
                    pool[0] = str(c.execute("SELECT document_id FROM memory_items WHERE id=%s",
                                            (random.choice(random.choice(hubs)),)).fetchone()[0])
                st = RecallState(conn=c, bank_id="b", pool=[
                    Candidate(document_id=d, item_id=None, source="", channel="dense_chunk",
                              rank=i + 1, score=1.0, evidence="") for i, d in enumerate(pool)])
                t0 = time.perf_counter()
                out = g.retrieve(QueryPlan(text="q"), st, 50)
                times.append(time.perf_counter() - t0)
                c.rollback()
            times.sort()
            print(f"graph channel over {queries} queries: mean {statistics.mean(times):.3f}s "
                  f"p90 {times[int(0.9 * (len(times) - 1))]:.3f}s max {times[-1]:.3f}s (last result {len(out)} docs)")
    finally:
        with psycopg.connect(base, autocommit=True) as c:
            c.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


if __name__ == "__main__":
    main()
