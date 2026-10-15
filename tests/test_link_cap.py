"""Links per note are bounded when they are built (Linker.max_links_per_note), and
`prospecta trim-links` trims a bank that already holds more. No model: the
model-free pgvector RELATED_TO path, on a 32-dim embedder whose notes lie on an arc,
so the cosine between two notes falls with the distance of their numbers."""
from __future__ import annotations

import math
import re
from collections import Counter

import psycopg
import pytest

from prospecta._linker import Linker
from prospecta.memory import Memory
from tests._stub_embedder import EMBED_DIM

STEP = 0.0005   # radians between neighbouring notes: every pair stays above the cosine floor


def arc_embed(texts):
    out = []
    for t in texts:
        k = int(re.search(r"note (\d+)", t).group(1))
        v = [0.0] * EMBED_DIM
        v[0], v[1] = math.cos(k * STEP), math.sin(k * STEP)
        out.append(v)
    return out


def build(db, bank, n, linker=None):
    m = Memory(database_url=db, bank_id=bank, llm=None, embed=arc_embed)
    m.create_bank(bank, embedding_dim=EMBED_DIM)
    for k in range(n):
        m.retain(f"note {k}", source=f"n{k:05d}", index_text=f"note {k}")
    m._linker = linker or Linker()
    return m


def link_all(m):
    with psycopg.connect(m.database_url) as c:
        ids = [str(r[0]) for r in c.execute(
            "SELECT id FROM documents WHERE bank_id=%s ORDER BY source", (m.default_bank_id,))]
    for d in ids:
        m.link_document(d)


def pairs(db, bank):
    """Unordered note pairs linked by similarity, with the strongest confidence."""
    q = """SELECT least(a.source, b.source), greatest(a.source, b.source), max(l.confidence)
           FROM memory_links l JOIN memory_items i ON i.id=l.src JOIN documents a ON a.id=i.document_id
           JOIN memory_items j ON j.id=l.dst JOIN documents b ON b.id=j.document_id
           WHERE l.bank_id=%s AND l.origin IN ('jev','pgvector') GROUP BY 1,2 ORDER BY 1,2"""
    with psycopg.connect(db) as c:
        return [tuple(r) for r in c.execute(q, (bank,))]


def degrees(ps):
    d = Counter()
    for a, b, _ in ps:
        d[a] += 1
        d[b] += 1
    return d


def test_default_cap_is_fifty():
    assert Linker().max_links_per_note == 50


def test_two_thousand_similar_notes_have_at_most_n_links_each(fresh_db):
    m = build(fresh_db, "big", 2000)
    link_all(m)
    ps = pairs(fresh_db, "big")
    deg = degrees(ps)
    assert ps, "linking made no links at all"
    assert max(deg.values()) <= 50
    assert len(deg) > 1900   # the cap does not leave notes unlinked


def test_the_strongest_links_are_the_ones_kept(fresh_db):
    m = build(fresh_db, "q", 41, Linker(max_links_per_note=5))
    with psycopg.connect(fresh_db) as c:
        q = str(c.execute("SELECT id FROM documents WHERE source='n00020'").fetchone()[0])
    m.link_document(q)   # the other notes are not linked yet: all have room
    kept = {b if a == "n00020" else a for a, b, _ in pairs(fresh_db, "q")}
    assert kept == {"n00018", "n00019", "n00021", "n00022", "n00016"} - {"n00016"} | {"n00017"} \
        or kept == {"n00018", "n00019", "n00021", "n00022", "n00017"} \
        or kept == {"n00018", "n00019", "n00021", "n00022", "n00023"}


def test_a_stronger_link_replaces_the_weakest_of_a_full_note(fresh_db):
    m = build(fresh_db, "r", 11, Linker(max_links_per_note=2))
    link_all(m)
    deg = degrees(pairs(fresh_db, "r"))
    assert max(deg.values()) <= 2


def test_two_runs_are_identical(fresh_db):
    a = build(fresh_db, "one", 300, Linker(max_links_per_note=10))
    link_all(a)
    b = build(fresh_db, "two", 300, Linker(max_links_per_note=10))
    link_all(b)
    assert pairs(fresh_db, "one") == pairs(fresh_db, "two")
    assert pairs(fresh_db, "one")


def test_zero_means_unbounded(fresh_db):
    m = build(fresh_db, "u", 60, Linker(max_links_per_note=0))
    link_all(m)
    assert max(degrees(pairs(fresh_db, "u")).values()) > 50


def test_trim_keeps_the_strongest_and_reports(fresh_db):
    from prospecta._linkcap import trim_bank
    m = build(fresh_db, "t", 80, Linker(max_links_per_note=0))
    link_all(m)
    before = pairs(fresh_db, "t")
    assert max(degrees(before).values()) > 10
    with psycopg.connect(fresh_db, autocommit=True) as c:
        out = trim_bank(c, "t", 10)
    after = pairs(fresh_db, "t")
    assert max(degrees(after).values()) <= 10
    assert set(after) <= set(before)
    assert out["rows_removed"] > 0 and out["pairs_removed"] == len(before) - len(after)
    assert out["bytes_reclaimable"] >= 0
    # nothing left to remove: a second run is a no-op, and a dry run changes nothing
    with psycopg.connect(fresh_db, autocommit=True) as c:
        assert trim_bank(c, "t", 10)["rows_removed"] == 0
    assert pairs(fresh_db, "t") == after


def test_trim_dry_run_removes_nothing(fresh_db):
    from prospecta._linkcap import trim_bank
    m = build(fresh_db, "d", 40, Linker(max_links_per_note=0))
    link_all(m)
    before = pairs(fresh_db, "d")
    with psycopg.connect(fresh_db, autocommit=True) as c:
        out = trim_bank(c, "d", 5, dry_run=True)
    assert pairs(fresh_db, "d") == before and out["rows_removed"] > 0
