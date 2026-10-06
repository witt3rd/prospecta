"""MetadataScope (kind `filter`, design 8.5) and scope promotion.

The channel returns the notes inside the extracted filter (documents.person /
created_on), ranked by their best item's dense score; empty when no filter was
extracted. Promotion never excludes: after fusion, when the filter is `hard`
and its set holds at most PROMOTE_MAX notes, every member moves to the front."""
from __future__ import annotations

from prospecta.channels.base import Candidate, QueryPlan, RecallState
from prospecta.channels.fusion import FusedDoc
from prospecta.db.queries import _meta_param, _vec_literal

PROMOTE_MAX = 12
NAME = "meta"

_SQL = """
SELECT DISTINCT ON (d.id) d.id AS document_id, d.source, m.id AS item_id,
       m.content, m.original_chunk, m.metadata, d.person, d.created_on,
       1 - (m.embedding <=> %(qe)s::vector) AS sem_score
FROM documents d
JOIN memory_items m ON m.document_id = d.id
WHERE d.bank_id = %(bank_id)s
  AND (%(people)s::text[] IS NULL OR d.person = ANY(%(people)s::text[]))
  AND (%(date_from)s::date IS NULL OR d.created_on >= %(date_from)s::date)
  AND (%(date_to)s::date IS NULL OR d.created_on <= %(date_to)s::date)
  AND (%(meta)s::jsonb IS NULL OR m.metadata @> %(meta)s::jsonb)
ORDER BY d.id, m.embedding <=> %(qe)s::vector
"""


class MetadataScope:
    name = NAME
    kind = "filter"

    def retrieve(self, plan: QueryPlan, state: RecallState, limit: int) -> list[Candidate]:
        f = plan.filters
        if not (f.people or f.date_from or f.date_to):
            return []
        qe = state.embed_query(plan.text)
        with state.conn.cursor() as cur:
            cur.execute(_SQL, {
                "bank_id": state.bank_id, "qe": _vec_literal(qe),
                "people": list(f.people) or None,
                "date_from": f.date_from, "date_to": f.date_to,
                "meta": _meta_param(state.metadata_filter),
            })
            cols = [c.name for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        rows.sort(key=lambda r: (-float(r["sem_score"]), r["source"] or "", str(r["document_id"])))
        out = []
        for r in rows[:limit]:
            sem = float(r["sem_score"])
            out.append(Candidate(
                document_id=str(r["document_id"]), item_id=str(r["item_id"]),
                source=r["source"] or "", channel=self.name, rank=len(out) + 1,
                score=sem, evidence=r["original_chunk"],
                detail={"cosine": sem, "content": r["content"],
                        "metadata": r["metadata"] or {}, "person": r["person"],
                        "created_on": r["created_on"].isoformat() if r["created_on"] else None,
                        "filter_hit": True, "set_size": len(rows)},
            ))
        return out


def scope_members(plan: QueryPlan, candidates: list[Candidate], limit: int) -> list[Candidate]:
    """The filter set to promote: [] unless hard and the set is complete
    (fewer than the channel limit were returned) and has <= PROMOTE_MAX notes."""
    if not plan.filters.hard:
        return []
    members = [c for c in candidates if c.channel == NAME]
    if not members or len(members) > PROMOTE_MAX or len(members) >= limit:
        return []
    return members


def promote_scope(fused: list[FusedDoc], members: list[Candidate]) -> list[FusedDoc]:
    """Move every member to the front (best filter rank first), then the rest
    in fused order. Members missing from the pool are added. Nothing is
    removed, so the filter can never hard-exclude a note."""
    if not members:
        return fused
    by_doc = {f.document_id: f for f in fused}
    front: list[FusedDoc] = []
    for c in sorted(members, key=lambda c: c.rank):
        front.append(by_doc.get(c.document_id) or FusedDoc(
            document_id=c.document_id, source=c.source, score=0.0, best=c,
            ranks={c.channel: c.rank}, scores={c.channel: c.score},
        ))
    ids = {f.document_id for f in front}
    return front + [f for f in fused if f.document_id not in ids]
