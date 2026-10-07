"""MetadataScope (kind `filter`, design 8.5) and scope promotion.

The channel returns the notes inside the extracted filter (documents.person /
created_on), ranked by their best item's dense score; empty when no filter was
extracted. Promotion never excludes and has no count limit: after fusion and
the stages, when the filter is `hard`, every member of the filter set gets a
score boost of `weight * top_score * cosine` (top_score = best fused score,
cosine = the member's filter-channel score). The boost, not a count, decides
how far members rise, so a large filter set stays sensible. A member whose
cosine is 0 gets no boost: the boost multiplies a real score."""
from __future__ import annotations

from dataclasses import replace

from prospecta.channels.base import Candidate, QueryPlan, RecallState
from prospecta.channels.fusion import FusedDoc
from prospecta._entities import resolve_entities
from prospecta.db.queries import _meta_param, _vec_literal

DEFAULT_PROMOTE = False       # scope-promotion re-sort is opt-in (meta param `promote`)
DEFAULT_PROMOTE_WEIGHT = 1.0   # tuning weight (meta param `promote_weight`), not a cap
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

    def retrieve(self, plan: QueryPlan, state: RecallState, limit: int | None = None) -> list[Candidate]:
        f = plan.filters
        if not (f.people or f.date_from or f.date_to):
            return []
        qe = state.embed_query(plan.text)
        people = list(f.people)
        if people:   # name matching, not exact equality: Nelson reaches "Mr. Nelson"
            people = list(dict.fromkeys(people + resolve_entities(state.conn, state.bank_id, people).people))
        with state.conn.cursor() as cur:
            cur.execute(_SQL, {
                "bank_id": state.bank_id, "qe": _vec_literal(qe),
                "people": people or None,
                "date_from": f.date_from, "date_to": f.date_to,
                "meta": _meta_param(state.metadata_filter),
            })
            cols = [c.name for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        rows.sort(key=lambda r: (-float(r["sem_score"]), r["source"] or "", str(r["document_id"])))
        out = []
        for r in rows:   # the whole filter set; scope is not a count
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


def scope_members(plan: QueryPlan, candidates: list[Candidate]) -> list[Candidate]:
    """The filter set to promote: [] unless the filter is hard. No size limit."""
    if not plan.filters.hard:
        return []
    return [c for c in candidates if c.channel == NAME]


def promote_scope(fused: list[FusedDoc], members: list[Candidate],
                  weight: float = DEFAULT_PROMOTE_WEIGHT) -> list[FusedDoc]:
    """Add `weight * top_score * cosine` to every member's score and re-sort
    (stable). Members missing from the pool are added with score 0 first.
    Nothing is removed, so the filter can never hard-exclude a note. A member
    with cosine 0 gets no boost (the boost multiplies a real score); accepted."""
    if not members or weight <= 0:
        return fused
    top = max((f.score for f in fused), default=0.0)
    if top <= 0:
        top = 1.0
    by_doc = {f.document_id: f for f in fused}
    boost = {c.document_id: weight * top * max(0.0, c.score) for c in members}
    out = []
    for f in fused:
        b = boost.get(f.document_id)
        out.append(replace(f, score=f.score + b) if b else f)
    for c in sorted(members, key=lambda c: c.rank):
        if c.document_id not in by_doc:
            out.append(FusedDoc(
                document_id=c.document_id, source=c.source,
                score=boost[c.document_id], best=c,
                ranks={c.channel: c.rank}, scores={c.channel: c.score},
            ))
    out.sort(key=lambda f: -f.score)
    return out
