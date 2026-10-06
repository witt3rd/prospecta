"""Bm25Chunks: a real BM25 channel over the chunk items (design 8.6).

Postgres `ts_rank_cd` is not BM25, and `websearch_to_tsquery` is AND-only. This
channel scores a bank's `kind='chunk'` items (`original_chunk`) with Okapi BM25
(Lucene variant, k1=1.5, b=0.75, as bm25s does), stemmed, stop words removed,
OR semantics, entirely in process: no extension and no change to the database.

Per-bank index: built from the bank's chunk items, persisted as gzip JSON under
`$PROSPECTA_BM25_DIR`, else `$PROSPECTA_DATA_DIR/bm25` (the bank's own scratch/
data dir); with neither set nothing is written to disk (never `~/.cache`). Kept in memory, and
rebuilt whenever the bank's chunk fingerprint (count, newest created_at, id
hash) differs from the one stored. The index holds postings and item ids only;
the evidence text is fetched from the database for the top hits.

Backend seam: `Bm25Backend` is the interface (`search(conn, bank_id, query, n)`
-> ranked `(item_id, score)`). `InProcessBm25` is the implementation below. A
`pg_search` backend (`CREATE INDEX ... USING bm25`, `original_chunk ||| :text`
ordered by `pdb.score(id)`) is `PgSearchBm25`; migration 0012 creates its index
only where the extension is installable. `AutoBm25` (the default) uses it when
that index exists and otherwise stays on `InProcessBm25`.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Protocol

from prospecta.channels.base import Candidate, QueryPlan, RecallState
from prospecta.db.queries import _meta_param

K1 = 1.5
B = 0.75
_ITEM_OVERFETCH = 3
_FORMAT = 1

STOPWORDS = frozenset("""a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its itself just me
more most my myself no nor not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these they this those through to
too under until up very was we were what when where which while who whom why will with would you your
yours yourself yourselves""".split())

_TOKEN = re.compile(r"[a-z0-9]+")
_SUFFIXES = ("ization", "ational", "fulness", "ousness", "iveness", "ations", "ingly", "ation",
             "ments", "ement", "ness", "ment", "ings", "edly", "able", "ible", "ing", "ies",
             "ied", "est", "ers", "ed", "er", "ly", "es", "s")


def stem(w: str) -> str:
    """A light English suffix stripper (enough to match plural/tense variants)."""
    if len(w) <= 3 or w.isdigit():
        return w
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    if len(w) > 3 and w.endswith("e"):
        w = w[:-1]
    if len(w) > 3 and w[-1] == w[-2] and w[-1] not in "ls":
        w = w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class Bm25Index:
    """Okapi BM25 over a fixed set of documents (item ids)."""

    def __init__(self, ids: list[str], lengths: list[int],
                 postings: dict[str, list[tuple[int, int]]]) -> None:
        self.ids = ids
        self.lengths = lengths
        self.postings = postings
        self.avgdl = (sum(lengths) / len(lengths)) if lengths else 0.0

    @classmethod
    def build(cls, items: list[tuple[str, str]]) -> "Bm25Index":
        ids, lengths = [], []
        postings: dict[str, list[tuple[int, int]]] = {}
        for item_id, text in items:
            toks = tokenize(text)
            n = len(ids)
            ids.append(item_id)
            lengths.append(len(toks))
            for term, tf in Counter(toks).items():
                postings.setdefault(term, []).append((n, tf))
        return cls(ids, lengths, postings)

    def search(self, query: str, n: int) -> list[tuple[str, float]]:
        """Top `n` (item_id, score), OR semantics, score > 0 only; ties by id."""
        N = len(self.ids)
        scores: dict[int, float] = {}
        for term in set(tokenize(query)):
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = math.log(1 + (N - len(plist) + 0.5) / (len(plist) + 0.5))
            for doc, tf in plist:
                norm = K1 * (1 - B + B * self.lengths[doc] / self.avgdl) if self.avgdl else K1
                scores[doc] = scores.get(doc, 0.0) + idf * tf * (K1 + 1) / (tf + norm)
        ranked = sorted(((s, self.ids[d]) for d, s in scores.items() if s > 0),
                        key=lambda x: (-x[0], x[1]))
        return [(i, s) for s, i in ranked[:n]]

    def to_json(self) -> dict:
        return {"format": _FORMAT, "ids": self.ids, "lengths": self.lengths,
                "postings": self.postings}

    @classmethod
    def from_json(cls, d: dict) -> "Bm25Index":
        if d.get("format") != _FORMAT:
            raise ValueError("unknown bm25 index format")
        return cls(d["ids"], d["lengths"],
                   {t: [(a, b) for a, b in p] for t, p in d["postings"].items()})


class Bm25Backend(Protocol):
    """The seam between the channel and a BM25 implementation."""

    def search(self, conn, bank_id: str, query: str, n: int) -> list[tuple[str, float]]: ...


_FINGERPRINT_SQL = """
SELECT count(*), coalesce(max(created_at)::text, ''),
       coalesce(sum(hashtextextended(id::text, 0)::numeric), 0)::text
FROM memory_items WHERE bank_id = %s AND kind = 'chunk'
"""
_ITEMS_SQL = "SELECT id, original_chunk FROM memory_items WHERE bank_id = %s AND kind = 'chunk' ORDER BY id"


def index_dir() -> Path | None:
    """Where the index is persisted: PROSPECTA_BM25_DIR, else
    PROSPECTA_DATA_DIR/bm25, else None (in-memory only, no files)."""
    explicit = os.environ.get("PROSPECTA_BM25_DIR")
    if explicit:
        return Path(explicit)
    data = os.environ.get("PROSPECTA_DATA_DIR")
    return Path(data) / "bm25" if data else None


class InProcessBm25:
    """Per-bank in-process BM25, persisted and rebuilt when the chunks change."""

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory else None
        self._mem: dict[tuple[str, str], tuple[list[str], Bm25Index]] = {}
        self.builds = 0  # how many times an index was rebuilt from the database

    def _path(self, bank_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", bank_id)
        base = self.directory or index_dir()
        return base / f"{safe}.bm25.json.gz" if base else None

    def _fingerprint(self, conn, bank_id: str) -> list[str]:
        with conn.cursor() as cur:
            cur.execute(_FINGERPRINT_SQL, (bank_id,))
            return [str(x) for x in cur.fetchone()]

    def _load(self, bank_id: str, fp: list[str]) -> Bm25Index | None:
        path = self._path(bank_id)
        if path is None:
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as f:
                blob = json.load(f)
            if blob.get("fingerprint") == fp:
                return Bm25Index.from_json(blob["index"])
        except (OSError, ValueError, KeyError, EOFError):
            pass
        return None

    def _save(self, bank_id: str, fp: list[str], idx: Bm25Index) -> None:
        path = self._path(bank_id)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "wb") as raw, gzip.open(raw, "wt", encoding="utf-8") as f:
                json.dump({"fingerprint": fp, "index": idx.to_json()}, f)
            os.replace(tmp, path)
        except OSError:
            pass  # persistence is an optimisation; the in-memory index still serves

    def index_for(self, conn, bank_id: str) -> Bm25Index:
        fp = self._fingerprint(conn, bank_id)
        key = (str(self.directory or index_dir()), bank_id)
        cached = self._mem.get(key)
        if cached and cached[0] == fp:
            return cached[1]
        idx = self._load(bank_id, fp)
        if idx is None:
            with conn.cursor() as cur:
                cur.execute(_ITEMS_SQL, (bank_id,))
                items = [(str(i), t) for i, t in cur.fetchall()]
            idx = Bm25Index.build(items)
            self.builds += 1
            self._save(bank_id, fp, idx)
        self._mem[key] = (fp, idx)
        return idx

    def search(self, conn, bank_id: str, query: str, n: int) -> list[tuple[str, float]]:
        return self.index_for(conn, bank_id).search(query, n)


PG_SEARCH_INDEX = "memory_items_original_chunk_bm25"

_PG_SEARCH_SQL = """
SELECT id, pdb.score(id) FROM memory_items
WHERE bank_id = %s AND kind = 'chunk' AND original_chunk ||| %s
ORDER BY pdb.score(id) DESC, id LIMIT %s
"""


class PgSearchBm25:
    """BM25 via the pg_search extension's index (migration 0012)."""

    def available(self, conn) -> bool:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_indexes WHERE indexname = %s", (PG_SEARCH_INDEX,))
            return cur.fetchone() is not None

    def search(self, conn, bank_id: str, query: str, n: int) -> list[tuple[str, float]]:
        with conn.cursor() as cur:
            cur.execute(_PG_SEARCH_SQL, (bank_id, query, int(n)))
            return [(str(i), float(s)) for i, s in cur.fetchall()]


class AutoBm25:
    """pg_search when its index exists, else the in-process bm25s backend."""

    def __init__(self, pg=None, fallback=None) -> None:
        self.pg = pg or PgSearchBm25()
        self.fallback = fallback or InProcessBm25()

    def choose(self, conn) -> Bm25Backend:
        try:
            return self.pg if self.pg.available(conn) else self.fallback
        except Exception:
            if hasattr(conn, "rollback"):
                conn.rollback()
            return self.fallback

    def search(self, conn, bank_id: str, query: str, n: int) -> list[tuple[str, float]]:
        return self.choose(conn).search(conn, bank_id, query, n)


_DEFAULT_BACKEND = AutoBm25()

_HITS_SQL = """
SELECT m.id, m.document_id, m.content, m.original_chunk, d.source, m.metadata
FROM memory_items m JOIN documents d ON d.id = m.document_id
WHERE m.bank_id = %(bank_id)s AND m.kind = 'chunk' AND m.id = ANY(%(ids)s::uuid[])
  AND (%(meta)s::jsonb IS NULL OR m.metadata @> %(meta)s::jsonb)
"""


class Bm25Chunks:
    name = "bm25"
    kind = "recall"
    backend: Bm25Backend = _DEFAULT_BACKEND

    def retrieve(self, plan: QueryPlan, state: RecallState, limit: int) -> list[Candidate]:
        hits = self.backend.search(state.conn, state.bank_id, plan.text,
                                   int(limit) * _ITEM_OVERFETCH * (4 if state.metadata_filter else 1))
        if not hits:
            return []
        with state.conn.cursor() as cur:
            cur.execute(_HITS_SQL, {"bank_id": state.bank_id, "ids": [i for i, _ in hits],
                                    "meta": _meta_param(state.metadata_filter)})
            cols = [c.name for c in cur.description]
            rows = {str(r[0]): dict(zip(cols, r)) for r in cur.fetchall()}
        out: list[Candidate] = []
        seen: set[str] = set()
        for item_id, score in hits:  # best chunk per document, document-level rank
            r = rows.get(item_id)
            if r is None:
                continue
            doc = str(r["document_id"])
            if doc in seen:
                continue
            seen.add(doc)
            out.append(Candidate(
                document_id=doc, item_id=item_id, source=r.get("source") or "",
                channel=self.name, rank=len(out) + 1, score=float(score),
                evidence=r["original_chunk"],
                detail={"bm25": float(score), "content": r["content"],
                        "metadata": r.get("metadata") or {}},
            ))
            if len(out) >= limit:
                break
        return out
