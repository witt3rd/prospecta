"""ONE entity-name resolver. A name resolves by MATCHING, not exact normalised
equality: honorifics/titles and articles are dropped, case/whitespace/diacritics
fold, "Last, First" order is irrelevant, and a name matches an entity when every
core token of the shorter name equals (or, for 3+ letters, prefixes) a token of
the other. Alias rows resolve to their entity. Every plausible match is returned
(ambiguity widens the set; nothing is dropped)."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

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

_ENTITIES_SQL = f"""
SELECT e.id, e.name, e.norm FROM memory_entities e
WHERE e.bank_id = %(bank)s AND {_SQL_FOLD.format(col='e.norm')} LIKE ANY(%(pats)s)
UNION
SELECT e.id, e.name, e.norm FROM memory_entities e
JOIN memory_entity_aliases a ON a.entity_id = e.id
WHERE e.bank_id = %(bank)s AND a.bank_id = %(bank)s
  AND ({_SQL_FOLD.format(col='a.norm')} LIKE ANY(%(pats)s) OR a.norm = ANY(%(norms)s))
"""
_PEOPLE_SQL = f"""
SELECT DISTINCT d.person FROM documents d
WHERE d.bank_id = %(bank)s AND d.person IS NOT NULL
  AND {_SQL_FOLD.format(col='d.person')} LIKE ANY(%(pats)s)
"""
_ALIASES_SQL = """
SELECT a.alias, a.norm FROM memory_entity_aliases a
WHERE a.bank_id = %(bank)s AND a.entity_id = ANY(%(ids)s)
"""


def fold(name: str) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def tokens(name: str) -> list[str]:
    """Core tokens: folded, honorifics/articles removed, initials kept only if
    nothing longer remains."""
    toks = [t for t in fold(name).split() if t not in HONORIFICS]
    long = [t for t in toks if len(t) > 1]
    return long or toks


def _tok_match(a: str, b: str) -> bool:
    return a == b or (min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)))


def matches(query: str, candidate: str) -> bool:
    q, c = tokens(query), tokens(candidate)
    if not q or not c:
        return False
    short, longer = (q, c) if len(q) <= len(c) else (c, q)
    return all(any(_tok_match(s, t) for t in longer) for s in short)


@dataclass
class Resolution:
    ids: list = field(default_factory=list)       # every matching entity id
    names: list[str] = field(default_factory=list)  # the given names + matched entity names + alias strings
    norms: list[str] = field(default_factory=list)  # lower-cased whitespace-collapsed forms of `names`
    people: list[str] = field(default_factory=list)  # documents.person values that match


def _clean(n: str) -> str:
    return re.sub(r"\s+", " ", n.strip().lower())


def resolve_entities(conn, bank_id: str, names: list[str]) -> Resolution:
    given = list(dict.fromkeys(n.strip() for n in names if n.strip()))
    if not given:
        raise ValueError("entity resolution needs a name")
    toks = sorted({t for n in given for t in tokens(n) if len(t) >= 2})
    pats = [f"%{t.replace('%', '').replace('_', '')}%" for t in toks] or ["%"]
    p = {"bank": bank_id, "pats": pats, "norms": sorted({_clean(n) for n in given})}
    written, norms = list(given), [_clean(n) for n in given]
    ids: list = []
    with conn.cursor() as cur:
        cur.execute(_ENTITIES_SQL, p)
        for eid, name, norm in cur.fetchall():
            if any(matches(g, name) or matches(g, norm) for g in given) or norm in p["norms"]:
                ids.append(eid); written.append(name); norms.append(norm)
        # alias rows: a row that matches by name resolves to its entity
        cur.execute("SELECT a.entity_id, a.alias, a.norm FROM memory_entity_aliases a "
                    f"WHERE a.bank_id = %(bank)s AND ({_SQL_FOLD.format(col='a.norm')} LIKE ANY(%(pats)s))", p)
        for eid, alias, norm in cur.fetchall():
            if any(matches(g, alias) for g in given) and eid not in ids:
                ids.append(eid)
        ids = list(dict.fromkeys(ids))
        if ids:
            cur.execute(_ALIASES_SQL, {"bank": bank_id, "ids": ids})
            for alias, norm in cur.fetchall():
                written.append(alias); norms.append(norm)
            cur.execute("SELECT name, norm FROM memory_entities WHERE id = ANY(%s)", (ids,))
            for name, norm in cur.fetchall():
                written.append(name); norms.append(norm)
        cur.execute(_PEOPLE_SQL, p)
        people = [r[0] for r in cur.fetchall() if any(matches(g, r[0]) for g in given)]
    return Resolution(ids, list(dict.fromkeys(written)), sorted(set(norms)), sorted(set(people)))
