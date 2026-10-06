"""GraphExpand: the `expand` channel over memory_links (design 8.7).

Seeds are the best items of the pool the recall channels found. One hop is a
join over memory_links (both directions); a second hop is the recursive step of
WITH RECURSIVE. Every step multiplies the score by a decay, the link's type
weight and its confidence; the walk keeps at most `node_cap` neighbour items
(Jev-Mem's 60), best first, and reports them as documents. A document reached
only through its own seed's links is not an expansion and is dropped.

params (all optional): seeds (10), max_hops (2, at most 2), decay (0.5),
node_cap (60), type_weights ({"SEMANTIC":1.0,"CAUSAL":0.8,"TEMPORAL":0.5,
"ENTITY":0.5}).
"""
from __future__ import annotations

import json

from prospecta.channels.base import Candidate, QueryPlan, RecallState

DEFAULT_TYPE_WEIGHTS = {"SEMANTIC": 1.0, "CAUSAL": 0.8, "TEMPORAL": 0.5, "ENTITY": 0.5}
DEFAULT_SEEDS = 10
DEFAULT_DECAY = 0.5
DEFAULT_NODE_CAP = 60
MAX_HOPS = 2
_RRF_K = 60

_SQL = """
WITH RECURSIVE seed(doc, w) AS (
    SELECT * FROM unnest(%(docs)s::uuid[], %(ws)s::float8[])
),
walk(item_id, origin_doc, hops, score, via, link_types, path) AS (
    SELECT m.id, m.document_id, 0, s.w, ARRAY[]::text[], ARRAY[]::text[], ARRAY[m.id]
    FROM seed s JOIN memory_items m ON m.document_id = s.doc AND m.bank_id = %(bank)s
  UNION ALL
    SELECT n.id, w.origin_doc, w.hops + 1,
           w.score * %(decay)s * COALESCE((%(tw)s::jsonb ->> l.link_type)::float8, 0.0)
                   * l.confidence,
           w.via || (l.subtype || ':' || n.id::text),
           w.link_types || l.link_type,
           w.path || n.id
    FROM walk w
    JOIN memory_links l ON (l.src = w.item_id OR l.dst = w.item_id)
    JOIN memory_items n ON n.id = CASE WHEN l.src = w.item_id THEN l.dst ELSE l.src END
    WHERE w.hops < %(max_hops)s
      AND l.bank_id = %(bank)s
      AND n.id <> ALL (w.path)
),
reached AS (
    SELECT DISTINCT ON (w.item_id) w.item_id, w.hops, w.score, w.via, w.link_types, w.origin_doc
    FROM walk w
    WHERE w.hops > 0 AND w.score > 0
    ORDER BY w.item_id, w.score DESC
)
SELECT r.item_id, m.document_id, d.source, m.content, m.original_chunk, m.metadata,
       r.hops, r.score, r.via, r.link_types
FROM reached r
JOIN memory_items m ON m.id = r.item_id
JOIN documents d ON d.id = m.document_id
WHERE m.document_id <> r.origin_doc
ORDER BY r.score DESC, d.source, r.item_id
LIMIT %(cap)s
"""


class GraphExpand:
    name = "graph"
    kind = "expand"
    takes_params = True

    def __init__(self, **params):
        self.params = params

    def seeds(self, state: RecallState, n: int) -> tuple[list[str], list[float]]:
        """Best seed documents of the pool: RRF over every channel's list."""
        acc: dict[str, float] = {}
        for c in state.pool:
            acc[c.document_id] = acc.get(c.document_id, 0.0) + 1.0 / (_RRF_K + c.rank)
        top = sorted(acc.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
        if not top:
            return [], []
        best = top[0][1]
        return [d for d, _ in top], [s / best for _, s in top]

    def retrieve(self, plan: QueryPlan, state: RecallState, limit: int) -> list[Candidate]:
        p = self.params
        docs, ws = self.seeds(state, int(p.get("seeds", DEFAULT_SEEDS)))
        if not docs:
            return []
        tw = {**DEFAULT_TYPE_WEIGHTS, **(p.get("type_weights") or {})}
        with state.conn.cursor() as cur:
            cur.execute(_SQL, {
                "docs": docs, "ws": ws, "bank": state.bank_id,
                "decay": float(p.get("decay", DEFAULT_DECAY)),
                "tw": json.dumps(tw),
                "max_hops": min(int(p.get("max_hops", MAX_HOPS)), MAX_HOPS),
                "cap": int(p.get("node_cap", DEFAULT_NODE_CAP)),
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
            if len(out) >= limit:
                break
        return out
