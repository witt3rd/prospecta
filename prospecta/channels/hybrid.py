"""HybridChunks: Stage 1 of the PostgreSQL hybrid architecture (docs/postgres-hybrid.md).

One round trip runs three retrievers over a bank's kind='chunk' items, each with
its own reserved slots: semantic (embedding, top 25), stemmed lexical ('english'
body_tsv, top 20) and unstemmed exact / phrase ('simple', top 20). The query text
is read by websearch_to_tsquery, so "exact phrase", OR and -exclude work. The
candidates are fused in SQL into the top 50 with an RRF fallback score and carry
an `exact_snippet` (ts_headline) whenever a lexical retriever matched. Per-call
filters (date window, anti-echo source list) and the bank's metadata filter are
applied inside every retriever (queries.stage1_search).

The slot counts and the HNSW ef_search are the channel's params (the plan's
`stage1_postgres` section). Opt-in, like bm25 and keyword: add
{"name": "hybrid_chunk", "weight": 1} to a bank's config.
"""
from __future__ import annotations

from prospecta.channels.base import Candidate, QueryPlan, RecallState
from prospecta.db.queries import (
    STAGE1_EXACT_LIMIT, STAGE1_FUSED_LIMIT, STAGE1_HNSW_EF_SEARCH,
    STAGE1_SEMANTIC_LIMIT, STAGE1_STEMMED_LIMIT, stage1_search,
)


class HybridChunks:
    name = "hybrid_chunk"
    kind = "recall"
    takes_params = True

    def __init__(self, semantic_limit: int = STAGE1_SEMANTIC_LIMIT,
                 stemmed_limit: int = STAGE1_STEMMED_LIMIT,
                 exact_limit: int = STAGE1_EXACT_LIMIT,
                 fused_limit: int = STAGE1_FUSED_LIMIT,
                 hnsw_ef_search: int = STAGE1_HNSW_EF_SEARCH):
        self.limits = {"semantic_limit": semantic_limit, "stemmed_limit": stemmed_limit,
                       "exact_limit": exact_limit, "fused_limit": fused_limit,
                       "hnsw_ef_search": hnsw_ef_search}
        for key, value in self.limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"hybrid_chunk channel: {key} must be an integer >= 1")

    def retrieve(self, plan: QueryPlan, state: RecallState, limit: int | None = None) -> list[Candidate]:
        rows = stage1_search(
            state.conn, bank_id=state.bank_id, query_text=plan.text,
            query_embedding=state.embed_query(plan.text),
            date_from=state.date_from, date_to=state.date_to,
            excluded_sources=state.excluded_sources,
            metadata_filter=state.metadata_filter, **self.limits)
        out: list[Candidate] = []
        seen: set[str] = set()
        for r in rows:  # fused order; best chunk per document, document-level rank
            doc = str(r["document_id"])
            if doc in seen:
                continue
            seen.add(doc)
            out.append(Candidate(
                document_id=doc, item_id=str(r["id"]), source=r.get("source") or "",
                channel=self.name, rank=len(out) + 1, score=float(r["rrf_score"]),
                evidence=r["original_chunk"],
                detail={"content": r["content"], "metadata": r.get("metadata") or {},
                        "created_on": r["created_on"],
                        "sem_rank": r["sem_rank"], "stem_rank": r["stem_rank"],
                        "exact_rank": r["exact_rank"], "exact_snippet": r["exact_snippet"]},
            ))
        return out
