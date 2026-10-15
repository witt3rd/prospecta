"""Bound the links of a note.

A *link* here is the relation between two notes (documents) that grows with how many
notes are alike: the rows of memory_links whose origin is 'jev' or 'pgvector' (semantic
and causal) and the TEMPORALLY_CLOSE rows (every note within three days of the same
day), both directions and every relation type between the same two notes counted once. Its strength is
(confidence, cosine), the link's own, compared as stored (confidence is REAL).
NEXT, PRECEDES/SUCCEEDS and entity links are bounded by their own rules and are
never counted or trimmed.

Two users:
- the Linker, when it writes the links of a note (`Linker.max_links_per_note`):
  it never creates a link past a note's cap; a new link that is stronger than the
  weakest of a note that is exactly at its cap takes its place; a note already
  over its cap (a bank linked before the cap existed) only refuses new links, its
  stored links are left alone;
- `trim_bank` (CLI `prospecta trim-links`), which removes the weakest links of
  notes over the cap from a bank that already holds more.
"""
from __future__ import annotations

import struct

SIMILARITY_ORIGINS = ("jev", "pgvector")
# the two kinds of link that grow with the number of alike notes; each has its own budget of
# `cap` links per note, so a day of same-day notes cannot crowd out the semantic links
FAMILIES = {
    "similarity": "l.origin = ANY(%(o)s)",
    "temporal": "l.subtype = 'TEMPORALLY_CLOSE'",
}
DEFAULT_MAX_LINKS_PER_NOTE = 50

_PAIRS = """
SELECT other, max(confidence), max(cos) FROM (
    SELECT b.document_id AS other, l.confidence,
           COALESCE((l.evidence->>'cosine')::float8, l.confidence) AS cos
    FROM memory_links l JOIN memory_items a ON a.id = l.src JOIN memory_items b ON b.id = l.dst
    WHERE l.bank_id = %(bank)s AND ({pred})
      AND a.document_id = %(doc)s AND b.document_id <> %(doc)s
  UNION ALL
    SELECT a.document_id, l.confidence,
           COALESCE((l.evidence->>'cosine')::float8, l.confidence)
    FROM memory_links l JOIN memory_items a ON a.id = l.src JOIN memory_items b ON b.id = l.dst
    WHERE l.bank_id = %(bank)s AND ({pred})
      AND b.document_id = %(doc)s AND a.document_id <> %(doc)s
) x GROUP BY other
"""

_MATCH = """
FROM memory_links l JOIN memory_items a ON a.id = l.src JOIN memory_items b ON b.id = l.dst
WHERE l.bank_id = %(bank)s AND ({pred})
  AND ((a.document_id = %(doc)s AND b.document_id = ANY(%(others)s))
    OR (b.document_id = %(doc)s AND a.document_id = ANY(%(others)s)))
"""


def f32(x: float) -> float:
    """x as REAL stores it, so a new strength compares equal to a stored equal one."""
    return struct.unpack("f", struct.pack("f", float(x)))[0]


class Sources:
    """Document sources, the stable tie-break between equally strong links (ids are random)."""

    def __init__(self, cur):
        self.cur, self.known = cur, {}

    def load(self, ids):
        miss = [i for i in ids if i not in self.known]
        if miss:
            self.cur.execute("SELECT id, source FROM documents WHERE id = ANY(%s)", (miss,))
            for i, s in self.cur.fetchall():
                self.known[i] = s or ""
            for i in miss:
                self.known.setdefault(i, "")

    def key(self, doc):
        self.load([doc])
        return (self.known[doc], str(doc))


def pair_strengths(cur, bank_id: str, doc, family: str = "similarity") -> dict:
    """Neighbour note -> (confidence, cosine) of the similarity links of `doc`."""
    cur.execute(_PAIRS.format(pred=FAMILIES[family]), {"bank": bank_id, "doc": doc, "o": list(SIMILARITY_ORIGINS)})
    return {o: (f32(c), float(k)) for o, c, k in cur.fetchall()}


def weakest_first(pairs: dict, src: Sources) -> list:
    src.load(list(pairs))
    return sorted(pairs, key=lambda o: (pairs[o][0], pairs[o][1], src.known[o], str(o)))


def delete_pairs(cur, bank_id: str, doc, others: list, dry_run: bool = False,
                 family: str = "similarity") -> int:
    """Remove (or count) every similarity link row between `doc` and each of `others`."""
    if not others:
        return 0
    args = {"bank": bank_id, "doc": doc, "others": list(others), "o": list(SIMILARITY_ORIGINS)}
    if dry_run:
        cur.execute("SELECT count(*) " + _MATCH.format(pred=FAMILIES[family]), args)
        return cur.fetchone()[0]
    cur.execute("DELETE FROM memory_links WHERE (src, dst, link_type, subtype) IN "
                "(SELECT l.src, l.dst, l.link_type, l.subtype " + _MATCH.format(pred=FAMILIES[family]) + ")", args)
    return cur.rowcount


def make_room(pairs: dict, strength: tuple, cap: int, src: Sources):
    """(ok, victim): may a note holding `pairs` take one more link of `strength`? Under the
    cap yes; exactly at it only by replacing its weakest, when the new one is stronger; over
    it (linked before the cap existed) no."""
    if len(pairs) < cap:
        return True, None
    if len(pairs) > cap:
        return False, None
    weakest = weakest_first(pairs, src)[0]
    return (strength > pairs[weakest]), weakest


def trim_bank(conn, bank_id: str, keep: int = DEFAULT_MAX_LINKS_PER_NOTE, dry_run: bool = False,
              on_progress=None) -> dict:
    """Trim a bank to the strongest `keep` similarity links per note, without locking it.

    `conn` must be autocommit. One small transaction per over-cap note, deleting only
    that note's weakest links by primary key (row locks only; readers and the Linker
    go on; no table lock, no VACUUM FULL). Notes are visited most-linked first, then by
    id, so the result is deterministic; deleting never raises a degree, so one pass
    leaves every note at `keep` or fewer. dry_run counts and deletes nothing.

    Returns notes_over, notes_trimmed, pairs_removed, rows_removed, and the space those
    rows take: bytes_reclaimable (table + indexes, an estimate from the average row of
    memory_links). A plain VACUUM keeps that space for the table's own reuse; giving it
    back to the file system takes a rewrite (pg_repack, or VACUUM FULL in a window)."""
    if keep < 1:
        raise ValueError("keep must be at least 1")
    out = {"keep": keep, "dry_run": dry_run, "notes_over": 0, "notes_trimmed": 0,
           "pairs_removed": 0, "rows_removed": 0, "bytes_reclaimable": 0}
    with conn.cursor() as cur:
        cur.execute("SELECT pg_total_relation_size('memory_links'), "
                    "GREATEST(reltuples, 0)::bigint FROM pg_class WHERE oid = 'memory_links'::regclass")
        total_bytes, est_rows = cur.fetchone()
        if est_rows == 0:
            cur.execute("SELECT count(*) FROM memory_links")
            est_rows = cur.fetchone()[0]
    visited = 0
    for family, pred in FAMILIES.items():
        with conn.cursor() as cur:
            cur.execute("""
                WITH p AS (SELECT DISTINCT least(a.document_id, b.document_id) AS x,
                                           greatest(a.document_id, b.document_id) AS y
                           FROM memory_links l JOIN memory_items a ON a.id = l.src
                           JOIN memory_items b ON b.id = l.dst
                           WHERE l.bank_id = %(bank)s AND (""" + pred + """)
                             AND a.document_id <> b.document_id)
                SELECT d, count(*) FROM (SELECT x AS d FROM p UNION ALL SELECT y FROM p) q
                GROUP BY d HAVING count(*) > %(keep)s ORDER BY count(*) DESC, d""",
                        {"bank": bank_id, "keep": keep, "o": list(SIMILARITY_ORIGINS)})
            over = [r[0] for r in cur.fetchall()]
        out["notes_over"] += len(over)
        removed: set = set()   # dry run: pairs already counted as gone
        for doc in over:
            with conn.transaction(), conn.cursor() as cur:
                src = Sources(cur)
                pairs = {k: v for k, v in pair_strengths(cur, bank_id, doc, family).items()
                         if frozenset((doc, k)) not in removed}
                excess = len(pairs) - keep
                if excess > 0:
                    victims = weakest_first(pairs, src)[:excess]
                    out["rows_removed"] += delete_pairs(cur, bank_id, doc, victims, dry_run, family)
                    out["pairs_removed"] += len(victims)
                    out["notes_trimmed"] += 1
                    if dry_run:
                        removed.update(frozenset((doc, v)) for v in victims)
            visited += 1
            if on_progress:
                on_progress(dict(out, visited=visited, remaining=out["notes_over"] - visited))
    if est_rows:
        out["bytes_reclaimable"] = int(out["rows_removed"] * total_bytes / est_rows)
    return out
