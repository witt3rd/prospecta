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
# and then joined to every frontier item that holds the entity (the frontier item's
# own document is left out of its neighbours). `{p}` prefixes the CTE names.
_HUBS = """
{p}he AS MATERIALIZED (
    SELECT ie.entity_id, si.kind
    FROM (SELECT DISTINCT item FROM {frm}) s
    JOIN memory_items si ON si.id = s.item
    JOIN memory_item_entities ie ON ie.item_id = s.item
    CROSS JOIN LATERAL (
        SELECT count(*) AS n FROM (SELECT 1 FROM memory_item_entities x
                                   WHERE x.entity_id = ie.entity_id LIMIT %(hub)s + 1) q
    ) hc
    WHERE hc.n > %(hub)s AND %(hub_cap)s > 0
    GROUP BY ie.entity_id, si.kind
),
{p}hh AS MATERIALIZED (
    SELECT he.entity_id, he.kind, b.item_id, bi.document_id AS doc
    FROM {p}he he
    CROSS JOIN LATERAL (
        SELECT ie2.item_id FROM memory_item_entities ie2
        JOIN memory_items bi ON bi.id = ie2.item_id
        WHERE ie2.entity_id = he.entity_id AND bi.kind = he.kind
        ORDER BY ie2.n DESC, ie2.item_id LIMIT %(hub_cap)s
    ) b
    JOIN memory_items bi ON bi.id = b.item_id
),
{p}hubs AS (
    SELECT s.item, s.origin_doc, s.score, s.hops, s.via, s.link_types, s.parent, hh.item_id AS nid,
           'ENTITY'::text AS link_type, 'SHARED_ENTITY'::text AS subtype, 1.0::real AS confidence
    FROM {frm} s
    JOIN memory_items si ON si.id = s.item
    JOIN memory_item_entities ie ON ie.item_id = s.item
    JOIN {p}hh hh ON hh.entity_id = ie.entity_id AND hh.kind = si.kind AND hh.doc <> s.origin_doc
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
""" + _HUBS.format(frm="s0", p="a_") + """,
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
""" + _HUBS.format(frm="frontier", p="b_") + """,
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
