"""Built-in channels over the existing semantic code: DenseChunks and
AnticipatedQuestions, the same cosine search as SEMANTIC_SQL restricted to
one memory_items.kind. Each repeats the kind predicate literally so the
per-kind partial HNSW index (migration 0004) can serve it."""
from __future__ import annotations

import logging

from prospecta._scorecut import CHANNEL_MIN_REL, fetch_until_cut
from prospecta.channels.base import Candidate, QueryPlan, RecallState
from prospecta.db.queries import _meta_param, _vec_literal

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

logger = logging.getLogger(__name__)
_warned_exact = False


def _scan_everything(cur) -> None:
    """Make the next query in this transaction see every row the HNSW index
    holds: pgvector >= 0.8 keeps scanning (iterative_scan) until LIMIT rows are
    found; older versions have no such mode, so the index is bypassed and the
    search is exact. Either way a short page means the source is exhausted."""
    global _warned_exact
    cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    row = cur.fetchone()
    version = tuple(int(x) for x in (row[0] if row else "0").split(".")[:2])
    if version >= (0, 8):
        cur.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
        cur.execute("SET LOCAL hnsw.max_scan_tuples = 2147483647")
        cur.execute("SET LOCAL hnsw.ef_search = 1000")
        return
    cur.execute("SET LOCAL enable_indexscan = off")
    if not _warned_exact:
        _warned_exact = True
        logger.warning(
            "pgvector %s has no hnsw.iterative_scan (needs >= 0.8): semantic channels "
            "fall back to an exact, index-bypassing scan so that no qualifying item "
            "beyond the HNSW ef_search reach is dropped. Upgrade pgvector to restore "
            "index speed.", ".".join(map(str, version)) if row else "unknown")


class _Semantic:
    name: str
    kind = "recall"
    takes_params = True
    _sql: str

    def __init__(self, min_rel: float | None = None, **_legacy):
        """`min_rel`: keep every item whose cosine >= min_rel x the best cosine
        (no count). Legacy `limit` in a stored config is ignored."""
        self.min_rel = CHANNEL_MIN_REL \
            if min_rel is None else float(min_rel)

    def retrieve(self, plan: QueryPlan, state: RecallState, limit: int | None = None) -> list[Candidate]:
        qe = state.embed_query(plan.text)

        def fetch(n: int) -> list[dict]:
            with state.conn.transaction(), state.conn.cursor() as cur:
                _scan_everything(cur)
                cur.execute(self._sql, {
                    "bank_id": state.bank_id,
                    "qe": _vec_literal(qe),
                    "meta": _meta_param(state.metadata_filter),
                    "n": n,
                })
                cols = [c.name for c in cur.description]
                rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            rows.sort(key=lambda r: -float(r["sem_score"]))
            return rows

        rows = fetch_until_cut(fetch, lambda r: float(r["sem_score"]), self.min_rel)
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
        return out


class DenseChunks(_Semantic):
    name = "dense_chunk"
    _sql = DENSE_CHUNK_SQL


class AnticipatedQuestions(_Semantic):
    name = "question"
    _sql = QUESTION_SQL
