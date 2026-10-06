"""Built-in channels over the existing semantic code: DenseChunks and
AnticipatedQuestions, the same cosine search as SEMANTIC_SQL restricted to
one memory_items.kind. Each repeats the kind predicate literally so the
per-kind partial HNSW index (migration 0004) can serve it."""
from __future__ import annotations

from prospecta.channels.base import Candidate, QueryPlan, RecallState
from prospecta.db.queries import _meta_param, _vec_literal

# Over-fetch items so several chunks of one note still leave `limit` notes.
_ITEM_OVERFETCH = 3

_SQL = """
SELECT m.id, m.document_id, m.content, m.original_chunk, d.source, m.metadata,
       1 - (m.embedding <=> %(qe)s::vector) AS sem_score
FROM memory_items m
JOIN documents d ON d.id = m.document_id
WHERE m.bank_id = %(bank_id)s
  AND m.kind = '{kind}'
  AND (%(meta)s::jsonb IS NULL OR m.metadata @> %(meta)s::jsonb)
ORDER BY m.embedding <=> %(qe)s::vector
LIMIT %(n)s
"""
DENSE_CHUNK_SQL = _SQL.format(kind="chunk")
QUESTION_SQL = _SQL.format(kind="question")


class _Semantic:
    name: str
    kind = "recall"
    _sql: str

    def retrieve(self, plan: QueryPlan, state: RecallState, limit: int) -> list[Candidate]:
        qe = state.embed_query(plan.text)
        with state.conn.cursor() as cur:
            cur.execute(self._sql, {
                "bank_id": state.bank_id,
                "qe": _vec_literal(qe),
                "meta": _meta_param(state.metadata_filter),
                "n": int(limit) * _ITEM_OVERFETCH,
            })
            cols = [c.name for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        out: list[Candidate] = []
        seen: set[str] = set()
        for r in rows:  # best item per document, document-level rank
            doc = str(r["document_id"])
            if doc in seen:
                continue
            seen.add(doc)
            sem = float(r["sem_score"])
            out.append(Candidate(
                document_id=doc, item_id=str(r["id"]), source=r.get("source") or "",
                channel=self.name, rank=len(out) + 1, score=sem,
                evidence=r["original_chunk"],
                detail={"cosine": sem, "content": r["content"],
                        "metadata": r.get("metadata") or {}},
            ))
            if len(out) >= limit:
                break
        return out


class DenseChunks(_Semantic):
    name = "dense_chunk"
    _sql = DENSE_CHUNK_SQL


class AnticipatedQuestions(_Semantic):
    name = "question"
    _sql = QUESTION_SQL
