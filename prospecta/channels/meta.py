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

import re
import unicodedata
from dataclasses import replace

from prospecta.channels.base import Candidate, QueryPlan, RecallState
from prospecta.channels.fusion import FusedDoc
from prospecta.db.queries import _meta_param, _vec_literal

DEFAULT_PROMOTE = False       # scope-promotion re-sort is opt-in (meta param `promote`)
DEFAULT_PROMOTE_WEIGHT = 1.0   # tuning weight (meta param `promote_weight`), not a cap
NAME = "meta"

# --- person-name matching: a name in a question reaches the documents.person that is that person ---
# Honorifics and articles are dropped, case / whitespace / diacritics fold, "Last, First" order is
# irrelevant, and a name matches when every core token of the shorter name equals (or, for 3+
# letters, prefixes) a token of the other: "Nelson" reaches "Mr. Nelson". Every plausible match is
# returned (ambiguity widens the set; nothing is dropped).
HONORIFICS = frozenset({
    "mr", "mrs", "ms", "miss", "mx", "dr", "prof", "professor", "sir", "dame", "lady", "lord",
    "master", "madam", "madame", "mme", "mlle", "monsieur", "sr", "sra", "srta", "don", "dona",
    "rev", "reverend", "fr", "father", "sister", "br", "brother", "capt", "captain", "sgt",
    "lt", "col", "gen", "hon", "judge", "rabbi", "imam", "jr", "sr.", "ii", "iii", "the",
    "a", "an",
})
_FROM = "àáâãäåāăąçćčďèéêëēěęğìíîïīıłñńňòóôõöøōőřśşšťùúûüūůűýÿźżž"
_TO = "aaaaaaaaacccdeeeeeeegiiiiiilnnnooooooooorsssstuuuuuuuyyzzz"
_SQL_FOLD = "translate(lower({col}), '%s', '%s')" % (_FROM, _TO)
_PEOPLE_SQL = f"""
SELECT DISTINCT d.person FROM documents d
WHERE d.bank_id = %(bank)s AND d.person IS NOT NULL
  AND {_SQL_FOLD.format(col='d.person')} LIKE ANY(%(pats)s)
"""


def _fold(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def _tokens(name: str) -> list[str]:
    """Core tokens: folded, honorifics/articles removed, initials kept only if nothing longer remains."""
    toks = [t for t in _fold(name).split() if t not in HONORIFICS]
    long = [t for t in toks if len(t) > 1]
    return long or toks


def _tok_match(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)))


def name_matches(query: str, candidate: str) -> bool:
    q, c = _tokens(query), _tokens(candidate)
    if not q or not c:
        return False
    short, longer = (q, c) if len(q) <= len(c) else (c, q)
    return all(any(_tok_match(s, t) for t in longer) for s in short)


def resolve_people(conn, bank_id: str, names: list[str]) -> list[str]:
    """The documents.person values of the bank that the given names match (see above)."""
    given = list(dict.fromkeys(n.strip() for n in names if n.strip()))
    if not given:
        return []
    toks = sorted({t for n in given for t in _tokens(n) if len(t) >= 2})
    pats = [f"%{t.replace('%', '').replace('_', '')}%" for t in toks] or ["%"]
    with conn.cursor() as cur:
        cur.execute(_PEOPLE_SQL, {"bank": bank_id, "pats": pats})
        people = [r[0] for r in cur.fetchall() if any(name_matches(g, r[0]) for g in given)]
    return sorted(set(people))

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
  AND (%(meta)s::jsonb IS NULL OR m.metadata @> (%(meta)s::jsonb - '_allow'))
  AND (%(meta)s::jsonb->'_allow' IS NULL OR (
        (%(meta)s::jsonb->'_allow'->'audience_key' IS NULL OR COALESCE(m.metadata->>'audience_key','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'audience_key')))
        AND (%(meta)s::jsonb->'_allow'->'project' IS NULL OR COALESCE(m.metadata->>'project','') IN
            (SELECT jsonb_array_elements_text(%(meta)s::jsonb->'_allow'->'project')))))
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
            people = list(dict.fromkeys(people + resolve_people(state.conn, state.bank_id, people)))
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
