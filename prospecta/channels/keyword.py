"""KeywordChunks: a plain keyword channel over the stored passage text.

Raw words of the question against the raw words of each stored item's
`original_chunk` (every kind: a bank holding only question-kind items, the case the
`bm25` channel returns nothing on, still has its passage text here): lower-cased, OR semantics, no stemming and no stop-word
removal (Postgres 'simple' configuration), so it matches what was written and
is empty only when no passage shares a single word with the question. Stemming
is on only when asked (`params: {"stemming": true}`, 'english' configuration).

It is never silently empty over a bank that has passages: no overlap is
reported in `last_note` (and so in the recall trace), and a question with no
usable word at all raises (the trace records the error) instead of returning
nothing. It is a separate channel from `bm25`, which stems and drops stop words.
Opt-in, like bm25: add {"name": "keyword", "weight": 1} to a bank's config.
"""
from __future__ import annotations

import re

from prospecta.channels.base import Candidate, QueryPlan, RecallState
from prospecta.db.keyword_index import has_keyword_index, word_vector as _vec
from prospecta.db.queries import _meta_param

_HEX_PREFIX = re.compile(r"(?=.*\d)[0-9a-f]{6,}")

# ONE word rule for query and passage, defined once in db/keyword_index.py (also the
# stored index's expression).
_SQL = """
WITH q AS (SELECT to_tsquery('simple', %(tsq)s) AS tsq)
SELECT m.id, m.document_id, m.content, m.original_chunk, d.source, m.metadata,
       ts_rank_cd(__VEC__, q.tsq) AS score
FROM memory_items m JOIN documents d ON d.id = m.document_id, q
WHERE m.bank_id = %(bank_id)s
  AND __VEC__ @@ q.tsq
  AND (%(meta)s::jsonb IS NULL OR m.metadata @> (%(meta)s::jsonb - '_allow'))
  AND (%(meta)s::jsonb->'_allow' IS NULL OR (
        (%(meta)s::jsonb->'_allow'->'audience_key' IS NULL OR COALESCE(m.metadata->>'audience_key','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'audience_key')))
        AND (%(meta)s::jsonb->'_allow'->'project' IS NULL OR COALESCE(m.metadata->>'project','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'project')))))
ORDER BY score DESC, m.id
"""

_COUNT_SQL = "SELECT count(*) FROM memory_items WHERE bank_id = %s"


def keyword_terms(conn, text: str, stemming: bool = False) -> list[str]:
    """The question's words by the same rule as the stored passages, in order."""
    cfg = "'english'" if stemming else "'simple'"
    with conn.cursor() as cur:
        cur.execute("SELECT lexeme FROM unnest(" + _vec(cfg, "%s") .replace("%s", "%(t)s") + ")"
                    " ORDER BY positions", {"t": text})
        return list(dict.fromkeys(r[0] for r in cur.fetchall()))


class KeywordChunks:
    name = "keyword"
    kind = "recall"
    takes_params = True

    def __init__(self, stemming: bool = False, **_ignored):
        if not isinstance(stemming, bool):
            raise ValueError("keyword channel: stemming must be true or false")
        self.stemming = stemming
        self.last_note: str | None = None
        self.used_index = False

    def retrieve(self, plan: QueryPlan, state: RecallState, limit: int | None = None) -> list[Candidate]:
        self.last_note = None
        terms = keyword_terms(state.conn, plan.text, self.stemming)
        with state.conn.cursor() as cur:
            cur.execute(_COUNT_SQL, (state.bank_id,))
            passages = cur.fetchone()[0]
        if not passages:
            return []                      # nothing stored: empty is the truth
        if not terms:
            raise ValueError("keyword channel: the question has no usable words "
                             f"(bank has {passages} passages)")
        cfg = "english" if self.stemming else "simple"
        tsq = " | ".join("'" + t.replace("'", "''") + "'" + (":*" if _HEX_PREFIX.fullmatch(t) else "")
                         for t in terms)
        with state.conn.cursor() as cur:
            # simple config: literal, so the planner can use the stored index; the
            # stemmed variant (or a bank without a valid index) is the plain scan.
            cfg_sql = "'simple'" if cfg == "simple" else "%(cfg)s"
            self.used_index = cfg == "simple" and has_keyword_index(state.conn)
            cur.execute(_SQL.replace("__VEC__", _vec(cfg_sql, "m.original_chunk")), {"cfg": cfg, "tsq": tsq, "bank_id": state.bank_id,
                               "meta": _meta_param(state.metadata_filter)})
            cols = [c.name for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        out: list[Candidate] = []
        seen: set[str] = set()
        for r in rows:                     # best chunk per document, document-level rank
            doc = str(r["document_id"])
            if doc in seen:
                continue
            seen.add(doc)
            out.append(Candidate(
                document_id=doc, item_id=str(r["id"]), source=r.get("source") or "",
                channel=self.name, rank=len(out) + 1, score=float(r["score"]),
                evidence=r["original_chunk"],
                detail={"keyword": float(r["score"]), "content": r["content"],
                        "metadata": r.get("metadata") or {}},
            ))
        if not out:
            self.last_note = (f"no passage shares a word with the question ({passages} passages, "
                              f"{'stemmed' if self.stemming else 'unstemmed'} match)")
        return out
