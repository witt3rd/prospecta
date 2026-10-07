"""GraphExpand: the `expand` channel over memory_links (design 8.7).

Seeds are the best items of the pool the recall channels found. Two bounded
hops follow memory_links in both directions, and the entity table for hub
entities (see _EDGES / _HUBS below). Every step multiplies the score by a decay, the link's type
weight and its confidence; the walk keeps every neighbour item whose score is
at least `node_min_rel` x the best reached score (no count), best first, and
reports them as documents. A document reached
only through its own seed's links is not an expansion and is dropped.

params (all optional): seed_min_rel (0.5: every pool document whose RRF >= 0.5 x
the best is a seed), max_hops (2), decay (0.5), node_min_rel (0.4), hub (30, the Linker's
ENTITY_HUB), hub_cap (200, holders reached per hub entity), type_weights ({"SEMANTIC":1.0,"CAUSAL":0.8,"TEMPORAL":0.5,
"ENTITY":0.5}).
"""
from __future__ import annotations

import json

from prospecta.channels.base import Candidate, QueryPlan, RecallState

DEFAULT_TYPE_WEIGHTS = {"SEMANTIC": 1.0, "CAUSAL": 0.8, "TEMPORAL": 0.5, "ENTITY": 0.5}
DEFAULT_SEED_MIN_REL = 0.5
DEFAULT_DECAY = 0.5
DEFAULT_NODE_MIN_REL = 0.4
MAX_HOPS = 2
ENTITY_HUB = 30         # an entity on more items than this is a hub: no per-anchor links
DEFAULT_HUB_CAP = 200   # holders reached per hub entity per seed item
_RRF_K = 60

# Two explicit, bounded hops instead of a recursive walk over both directions
# of an OR join. Each hop is a UNION ALL of two index scans (src side, dst
# side). After hop 1 every item with a positive score is expanded (no count).
# Scores only fall along a path (decay, weight, confidence are all <= 1), so a
# hop-1 item is expanded only if one more step (decay x the largest type weight)
# could still reach `node_min_rel` x the best hop-1 score: the final cut keeps at
# least that, so this prunes exactly the nodes that cannot survive it (no count). Hub
# entities (more than `hub` items) carry no per-anchor links; hops 1 and 2 reach the
# other holders of a frontier item's hub entities through the memory_item_entities
# join, up to `hub_cap` per entity.
_EDGES = """
    SELECT f.item, f.origin_doc, f.score, f.hops, f.via, f.link_types, f.parent,
           CASE WHEN l.src = f.item THEN l.dst ELSE l.src END AS nid,
           l.link_type, l.subtype, l.confidence
    FROM {frm} f JOIN memory_links l ON l.src = f.item AND l.bank_id = %(bank)s
    UNION ALL
    SELECT f.item, f.origin_doc, f.score, f.hops, f.via, f.link_types, f.parent,
           l.src, l.link_type, l.subtype, l.confidence
    FROM {frm} f JOIN memory_links l ON l.dst = f.item AND l.bank_id = %(bank)s
"""

# Hub holders are fetched once per (entity, kind) -- not once per frontier item --
# and joined to the two best frontier items of distinct documents holding the entity
# (all that the best-score-per-neighbour merge can use; a holder in the best item's
# own document takes the second; a frontier item's own document is left out of its
# neighbours). `{p}` prefixes the CTE names.
_HUBS = """
{p}fe AS MATERIALIZED (   -- frontier item x entity it holds (each item once)
    SELECT s.item, s.origin_doc, s.score, s.hops, s.via, s.link_types, s.parent,
           ie.entity_id, si.kind
    FROM {frm} s
    JOIN memory_items si ON si.id = s.item
    JOIN memory_item_entities ie ON ie.item_id = s.item
    WHERE %(hub_cap)s > 0
),
{p}he AS MATERIALIZED (   -- hub test once per entity, not once per (item, entity)
    SELECT e.entity_id, e.kind
    FROM (SELECT DISTINCT entity_id, kind FROM {p}fe) e
    CROSS JOIN LATERAL (
        SELECT count(*) AS n FROM (SELECT 1 FROM memory_item_entities x
                                   WHERE x.entity_id = e.entity_id LIMIT %(hub)s + 1) q
    ) hc
    WHERE hc.n > %(hub)s
),
{p}ft AS MATERIALIZED (   -- per (entity, kind): the two best frontier items of distinct documents
    SELECT entity_id, kind, item, origin_doc, score, hops, via, link_types, parent
    FROM (
        SELECT d.*, row_number() OVER (PARTITION BY d.entity_id, d.kind ORDER BY d.score DESC, d.item) AS rk
        FROM (SELECT DISTINCT ON (f.entity_id, f.kind, f.origin_doc) f.*
              FROM {p}fe f JOIN {p}he he ON he.entity_id = f.entity_id AND he.kind = f.kind
              ORDER BY f.entity_id, f.kind, f.origin_doc, f.score DESC, f.item) d
    ) r WHERE rk <= 2
),
{p}hh AS MATERIALIZED (   -- holders: each hub entity's items read once, the best hub_cap per kind
    SELECT entity_id, kind, item_id, doc
    FROM (
        SELECT h.entity_id, h.kind, ie2.item_id, bi.document_id AS doc,
               row_number() OVER (PARTITION BY h.entity_id, h.kind ORDER BY ie2.n DESC, ie2.item_id) AS rn
        FROM {p}he h
        JOIN memory_item_entities ie2 ON ie2.entity_id = h.entity_id
        JOIN memory_items bi ON bi.id = ie2.item_id AND bi.kind = h.kind
        WHERE true {skip}
    ) r WHERE rn <= %(hub_cap)s
    {reuse}
),
{p}hubs AS (   -- each holder joins the (at most two) best frontier items of its entity: once per hub
    SELECT t.item, t.origin_doc, t.score, t.hops, t.via, t.link_types, t.parent, hh.item_id AS nid,
           'ENTITY'::text AS link_type, 'SHARED_ENTITY'::text AS subtype, 1.0::real AS confidence
    FROM {p}ft t
    JOIN {p}hh hh ON hh.entity_id = t.entity_id AND hh.kind = t.kind AND hh.doc <> t.origin_doc
)
"""

_SQL = """
WITH seed(doc, w) AS (
    SELECT * FROM unnest(%(docs)s::uuid[], %(ws)s::float8[])
),
s0 AS (
    SELECT m.id AS item, m.document_id AS origin_doc, s.w AS score, 0 AS hops,
           ARRAY[]::text[] AS via, ARRAY[]::text[] AS link_types, m.id AS parent
    FROM seed s JOIN memory_items m ON m.document_id = s.doc AND m.bank_id = %(bank)s
),
e1 AS (""" + _EDGES.format(frm="s0") + """),
""" + _HUBS.format(frm="s0", p="a_", skip="", reuse="") + """,
h1 AS (
    SELECT DISTINCT ON (nid) nid AS item, origin_doc, 1 AS hops, item AS parent,
           score * %(decay)s * COALESCE((%(tw)s::jsonb ->> link_type)::float8, 0.0) * confidence AS score,
           ARRAY[subtype || ':' || nid::text] AS via, ARRAY[link_type] AS link_types
    FROM (SELECT * FROM e1 UNION ALL SELECT * FROM a_hubs) e
    ORDER BY nid, score * COALESCE((%(tw)s::jsonb ->> link_type)::float8, 0.0) * confidence DESC
),
best1 AS (
    SELECT COALESCE(max(h.score), 0) AS b
    FROM h1 h JOIN memory_items m ON m.id = h.item WHERE m.document_id <> h.origin_doc
),
frontier AS (
    SELECT h.* FROM h1 h, best1
    WHERE h.score > 0 AND %(max_hops)s >= 2
      AND h.score * %(decay)s * %(wmax)s >= %(rel)s * best1.b
),
""" + _HUBS.format(frm="frontier", p="b_",
    skip="AND NOT EXISTS (SELECT 1 FROM a_he o WHERE o.entity_id = h.entity_id AND o.kind = h.kind)",
    reuse="UNION ALL SELECT a.entity_id, a.kind, a.item_id, a.doc FROM a_hh a JOIN b_he h USING (entity_id, kind)") + """,
e2 AS (""" + _EDGES.format(frm="frontier") + """
    UNION ALL
    SELECT * FROM b_hubs
),
h2 AS (
    SELECT DISTINCT ON (nid) nid AS item, origin_doc, 2 AS hops,
           score * %(decay)s * COALESCE((%(tw)s::jsonb ->> link_type)::float8, 0.0) * confidence AS score,
           via || (subtype || ':' || nid::text) AS via, link_types || link_type AS link_types
    FROM e2, best1
    WHERE nid <> parent
      AND score * %(decay)s * COALESCE((%(tw)s::jsonb ->> link_type)::float8, 0.0) * confidence
          >= %(rel)s * best1.b   -- below the final cut whatever the rest of the walk finds
    ORDER BY nid, score * COALESCE((%(tw)s::jsonb ->> link_type)::float8, 0.0) * confidence DESC
),
reached AS (
    SELECT DISTINCT ON (item) item AS item_id, hops, score, via, link_types, origin_doc
    FROM (SELECT item, origin_doc, hops, score, via, link_types FROM h1 WHERE score > 0
          UNION ALL SELECT * FROM h2 WHERE score > 0) u
    ORDER BY item, score DESC
),
top AS (
    SELECT r.*, m.document_id FROM reached r JOIN memory_items m ON m.id = r.item_id
    WHERE m.document_id <> r.origin_doc
),
kept AS MATERIALIZED (   -- the cut runs before the content and source joins
    SELECT t.* FROM top t WHERE t.score >= %(rel)s * (SELECT max(score) FROM top)
)
SELECT t.item_id, t.document_id, d.source, m.content, m.original_chunk, m.metadata,
       t.hops, t.score, t.via, t.link_types
FROM kept t
JOIN memory_items m ON m.id = t.item_id
JOIN documents d ON d.id = t.document_id
ORDER BY t.score DESC, d.source, t.item_id
"""


class GraphExpand:
    name = "graph"
    kind = "expand"
    takes_params = True

    def __init__(self, **params):
        self.params = params

    def seeds(self, state: RecallState, rel: float) -> tuple[list[str], list[float]]:
        """Seed documents of the pool: RRF over every channel's list; every
        document at or above `rel` x the best is a seed (no count)."""
        acc: dict[str, float] = {}
        for c in state.pool:
            acc[c.document_id] = acc.get(c.document_id, 0.0) + 1.0 / (_RRF_K + c.rank)
        ranked = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))
        if not ranked:
            return [], []
        best = ranked[0][1]
        top = [kv for kv in ranked if kv[1] >= rel * best]
        return [d for d, _ in top], [s / best for _, s in top]

    def retrieve(self, plan: QueryPlan, state: RecallState, limit: int | None = None) -> list[Candidate]:
        p = self.params
        docs, ws = self.seeds(state, float(p.get("seed_min_rel", DEFAULT_SEED_MIN_REL)))
        if not docs:
            return []
        tw = {**DEFAULT_TYPE_WEIGHTS, **(p.get("type_weights") or {})}
        with state.conn.cursor() as cur:
            cur.execute(_SQL, {
                "docs": docs, "ws": ws, "bank": state.bank_id,
                "decay": float(p.get("decay", DEFAULT_DECAY)),
                "tw": json.dumps(tw),
                "wmax": max([1.0, *tw.values()]),
                "max_hops": int(p.get("max_hops", MAX_HOPS)),
                "rel": float(p.get("node_min_rel", DEFAULT_NODE_MIN_REL)),
                "hub": int(p.get("hub", ENTITY_HUB)),
                "hub_cap": int(p.get("hub_cap", DEFAULT_HUB_CAP)),
            })
            cols = [c.name for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        out: list[Candidate] = []
        seen: set[str] = set()
        for r in rows:   # items arrive best first; one candidate per document
            doc = str(r["document_id"])
            if doc in seen:
                continue
            seen.add(doc)
            out.append(Candidate(
                document_id=doc, item_id=str(r["item_id"]), source=r.get("source") or "",
                channel=self.name, rank=len(out) + 1, score=float(r["score"]),
                evidence=r["original_chunk"],
                detail={"hops": int(r["hops"]), "link_types": list(r["link_types"]),
                        "via": list(r["via"]), "content": r["content"],
                        "metadata": r.get("metadata") or {}},
            ))
            if limit is not None and len(out) >= limit:
                break
        return out
