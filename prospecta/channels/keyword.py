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
Phrase matching (same word rule, same index): a text in double quotes must appear
as an exact phrase (words adjacent, in order, `phraseto_tsquery`), identifiers,
versions and paths inside it included. With quotes present only passages holding
every phrase are returned; unquoted words still count towards the rank. An unquoted
multi-word question keeps every any-word hit and ranks passages higher the more
adjacent pairs of its words they hold (adjacency boost, never a filter).
Opt-in, like bm25: add {"name": "keyword", "weight": 1} to a bank's config.
"""
from __future__ import annotations

import re

from prospecta.channels.base import Candidate, QueryPlan, RecallState
from prospecta.db.keyword_index import has_keyword_index, word_vector as _vec
from prospecta.db.queries import _meta_param

_QUOTED = re.compile(r'"([^"]*)"')
_MAX_PAIRS = 16

_HEX_PREFIX = re.compile(r"(?=.*\d)[0-9a-f]{6,}")

# ONE word rule for query and passage, defined once in db/keyword_index.py (also the
# stored index's expression).
_SQL = """
WITH q AS (SELECT to_tsquery('simple', %(tsq)s) AS tsq)
SELECT m.id, m.document_id, m.content, m.original_chunk, d.source, m.metadata,
       ts_rank_cd(__VEC__, q.tsq) + (__ADJ__) AS score
FROM memory_items m JOIN documents d ON d.id = m.document_id, q
WHERE m.bank_id = %(bank_id)s
  AND __VEC__ @@ q.tsq__PHRASES__
  AND (%(meta)s::jsonb IS NULL OR m.metadata @> (%(meta)s::jsonb - '_allow'))
  AND (%(meta)s::jsonb->'_allow' IS NULL OR (
        (%(meta)s::jsonb->'_allow'->'audience_key' IS NULL OR COALESCE(m.metadata->>'audience_key','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'audience_key')))
        AND (%(meta)s::jsonb->'_allow'->'project' IS NULL OR COALESCE(m.metadata->>'project','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'project')))))
ORDER BY (__ADJ__)::int DESC, score DESC, m.id
"""

_COUNT_SQL = "SELECT count(*) FROM memory_items WHERE bank_id = %s"


def keyword_terms(conn, text: str, stemming: bool = False) -> list[str]:
    """The question's words by the same rule as the stored passages, in order."""
    cfg = "'english'" if stemming else "'simple'"
    with conn.cursor() as cur:
        cur.execute("SELECT lexeme FROM unnest(" + _vec(cfg, "%s") .replace("%s", "%(t)s") + ")"
                    " ORDER BY positions", {"t": text})
        return list(dict.fromkeys(r[0] for r in cur.fetchall()))


def split_phrases(text: str) -> tuple[list[str], str]:
    """(quoted phrases, the rest of the text). An unbalanced quote is plain text."""
    phrases = [p.strip() for p in _QUOTED.findall(text) if p.strip()]
    return phrases, _QUOTED.sub(" ", text)


def _adjacent(cfg_sql: str, param: str) -> str:
    """True when the passage holds the phrase in `param` under the word rule.

    Checked on the two vectors the rule is made of, separately, so a phrase never
    spans the join between them (the stored index only narrows the candidates).
    """
    return (f"(to_tsvector({cfg_sql}::regconfig, m.original_chunk) @@ phraseto_tsquery({cfg_sql}::regconfig, %({param})s)"
            f" OR to_tsvector({cfg_sql}::regconfig, translate(m.original_chunk, '/\\', '  '))"
            f" @@ phraseto_tsquery({cfg_sql}::regconfig, translate(%({param})s, '/\\', '  ')))")


def _phrase_sql(cfg_sql: str, phrases: list[str], loose: list[str], params: dict) -> tuple[str, str]:
    """(extra WHERE clauses, adjacency score expression)."""
    where = ""
    for i, ph in enumerate(phrases):
        params[f"ph{i}"] = ph
        where += f"\n  AND {_adjacent(cfg_sql, f'ph{i}')}"
    pairs = []
    for i, (a, b) in enumerate(zip(loose, loose[1:])):
        if a == b or i >= _MAX_PAIRS:
            continue
        params[f"pr{i}"] = f"{a} {b}"
        pairs.append(f"({_adjacent(cfg_sql, f'pr{i}')})::int")
    return where, (" + ".join(pairs) if pairs else "0::int")


def _word_sequence(conn, text: str, cfg_sql: str) -> list[str]:
    """The text's words in order, repeats kept (for adjacent pairs)."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT w FROM (SELECT lexeme, unnest(positions) AS pos "
                    f"FROM unnest(to_tsvector({cfg_sql}::regconfig, %(t)s))) x(w, pos) ORDER BY pos",
                    {"t": text})
        return [r[0] for r in cur.fetchall()]


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
        phrases, rest = split_phrases(plan.text)
        cfg_lit = "'simple'" if cfg == "simple" else "%(cfg)s"
        params = {"cfg": cfg, "tsq": tsq, "bank_id": state.bank_id,
                  "meta": _meta_param(state.metadata_filter)}
        phrases = [p for p in phrases if _word_sequence(state.conn, p, "'simple'" if cfg == "simple" else f"'{cfg}'")]
        loose = _word_sequence(state.conn, rest, "'simple'" if cfg == "simple" else f"'{cfg}'")
        where, adj = _phrase_sql(cfg_lit, phrases, loose, params)
        sql = (_SQL.replace("__PHRASES__", where).replace("__ADJ__", adj)
               .replace("__VEC__", _vec(cfg_lit, "m.original_chunk")))
        with state.conn.cursor() as cur:
            # simple config: literal, so the planner can use the stored index; the
            # stemmed variant (or a bank without a valid index) is the plain scan.
            self.used_index = cfg == "simple" and has_keyword_index(state.conn)
            cur.execute(sql, params)
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
            self.last_note = (f"no passage {'holds the quoted phrase' if phrases else 'shares a word with the question'} ({passages} passages, "
                              f"{'stemmed' if self.stemming else 'unstemmed'} match)")
        return out
