"""The exact-identifier gate: a rare identifier in the question comes back first.

Why it exists. Recall finds a record by what it MEANS (the dense legs) or by the
WORDS a full-text query shares with it (the lexical legs). Neither is made for an
identifier: a file name, a hex id, a ticket number, a project name inside a
path.

* A dense embedder reads `deliverable-7f3a91c04b2e` as noise after the word
  `deliverable`, so every other deliverable is as near as the one asked for.
* Postgres full text reads an identifier as its parser shapes it: a file name
  (`deliverable-7f3a91c04b2e.pr`) is ONE host token, `scripts/audit-ledger.py`
  one path token, a link one url token. The question `deliverable-7f3a91c04b2e`
  is parsed as a different token (a hyphenated compound), so the two never meet.
* A record's own id (a commission's `commission:ask_<uuid>:0`) is kept in the
  item metadata and in `documents.source`; it is in no text at all.

What it does. When the question holds a token that is SHAPED like an identifier
(`identifier_tokens`: it mixes letters and digits, or joins pieces with one of
`- _ . : / @ # +`), it looks for that token as written, case-insensitive, with
a non-alphanumeric (or the end of the text) on both sides, in the chunk text and
in `documents.source`, under the same metadata filter and allowed-audience rule
as every other leg. A token found in more than `MAX_DOCS` documents is not an
identifier of any of them and is ignored. The documents that hold a token come
first, ahead of everything the other legs ranked: the ones that hold more of the
question's identifiers first, each group in the order the other legs gave. The
rest follow as they were. A question with no identifier-shaped token is not
touched, and costs nothing.

The scan is a substring match over the bank's items (no index): about 0.2 s for
200 000 items per token, at most `MAX_TOKENS` tokens a question. A trigram index
would make it instant; it needs an extension the shared-database mode never
creates, so none is assumed.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from prospecta.channels.base import Candidate
from prospecta.channels.fusion import FusedDoc
from prospecta.db.queries import META_FILTER_SQL, _meta_param

logger = logging.getLogger(__name__)

MAX_TOKENS = 4      # identifier tokens looked up per question
MAX_DOCS = 50       # a token in more documents than this is not an identifier
ROW_CAP = 2000      # matched items after which a token is surely common: stop scanning
GATE = 1.0          # added to the boost so that a gated score clears every ungated one
MIN_LEN, MAX_LEN = 4, 200

_PIECES = re.compile(r"[\w.:/@#+-]+")
_EDGE = re.compile(r"^\W+|\W+$")
_INNER = re.compile(r"[^\W_][_.:/@#+-]+[^\W_]")     # alphanumeric, joiner(s), alphanumeric


def _shaped(tok: str) -> bool:
    if not MIN_LEN <= len(tok) <= MAX_LEN:
        return False
    if any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok):
        return True                                  # 7f3a91c04b2e, ITEM-2001, v2.14.1
    if _INNER.search(tok):
        return True                                  # fleet-docker-janitor, a.b, ask_x
    return tok.isdigit() and len(tok) >= 6


def identifier_tokens(text: str) -> list[str]:
    """The identifier-shaped tokens of a question, as written, each once (any
    case), in order, at most MAX_TOKENS. Prose gives none."""
    seen: set[str] = set()
    out: list[str] = []
    for piece in _PIECES.findall(text or ""):
        tok = _EDGE.sub("", piece)
        key = tok.lower()
        if key in seen or not _shaped(tok):
            continue
        seen.add(key)
        out.append(tok)
        if len(out) == MAX_TOKENS:
            break
    return out


def like_pattern(tok: str) -> str:
    """`%tok%` for ILIKE, with LIKE's own characters taken literally."""
    return "%" + re.sub(r"([\\%_])", r"\\\1", tok) + "%"


def regex_pattern(tok: str) -> str:
    """The token as written, with no alphanumeric on either side (a POSIX
    regex for `~*`)."""
    return "(^|[^[:alnum:]])" + re.sub(r"([^\w])", r"\\\1", tok) + "($|[^[:alnum:]])"


@dataclass(frozen=True)
class ExactHit:
    """A document that holds identifier tokens of the question, with the item
    that shows it (the chunk holding the token, else the document's first)."""

    document_id: str
    item_id: str
    source: str
    content: str
    original_chunk: str
    metadata: dict
    tags: list
    matched: tuple[str, ...]            # the question's tokens this document holds

    @property
    def score(self) -> float:
        return float(len(self.matched))

    def row(self) -> dict:
        """The shape `hybrid_search` returns, with every leg's score 0."""
        return {"id": self.item_id, "document_id": self.document_id, "content": self.content,
                "original_chunk": self.original_chunk, "source": self.source,
                "metadata": self.metadata, "tags": self.tags, "sem_score": 0.0,
                "lex_content_score": 0.0, "lex_body_score": 0.0, "rrf_score": 0.0}


_TEXT_SQL = f"""
SELECT m.id, m.document_id, m.kind, m.ordinal
FROM memory_items m
WHERE m.bank_id = %(bank_id)s
  AND m.original_chunk ILIKE %(like)s AND m.original_chunk ~* %(rx)s
  {META_FILTER_SQL}
LIMIT %(cap)s
"""

_SOURCE_SQL = """
SELECT d.id FROM documents d
WHERE d.bank_id = %(bank_id)s AND d.source ILIKE %(like)s AND d.source ~* %(rx)s
LIMIT %(cap)s
"""

_FIRST_ITEM_SQL = f"""
SELECT DISTINCT ON (m.document_id) m.id, m.document_id
FROM memory_items m
WHERE m.bank_id = %(bank_id)s AND m.document_id = ANY(%(docs)s::uuid[])
  {META_FILTER_SQL}
ORDER BY m.document_id, (m.kind <> 'chunk'), m.ordinal NULLS LAST, m.id
"""

_ROWS_SQL = """
SELECT m.id, m.document_id, m.content, m.original_chunk, d.source, m.metadata, m.tags
FROM memory_items m JOIN documents d ON d.id = m.document_id
WHERE m.id = ANY(%(ids)s::uuid[])
"""


def find_exact(conn, bank_id: str, text: str, metadata_filter: dict | None = None) -> list[ExactHit]:
    """The documents that hold an identifier token of `text`, those holding more
    tokens first, then by document id. [] when the question has none, when every
    token is common, or when the lookup failed (it never sinks the recall)."""
    tokens = identifier_tokens(text)
    if not tokens:
        return []
    try:
        return _find(conn, bank_id, tokens, _meta_param(metadata_filter))
    except Exception:
        logger.warning("exact-identifier lookup failed; the other legs answer", exc_info=True)
        conn.rollback()
        return []


def _find(conn, bank_id: str, tokens: list[str], meta: str | None) -> list[ExactHit]:
    held: dict[str, list[str]] = {}                  # document -> the tokens it holds
    shown: dict[str, tuple] = {}                     # document -> (sort key, item id) of a text match
    with conn.cursor() as cur:
        for tok in tokens:
            args = {"bank_id": bank_id, "like": like_pattern(tok), "rx": regex_pattern(tok),
                    "cap": ROW_CAP, "meta": meta}
            items = cur.execute(_TEXT_SQL, args).fetchall()
            sources = cur.execute(_SOURCE_SQL, {k: args[k] for k in ("bank_id", "like", "rx", "cap")}
                                  ).fetchall()
            if len(items) >= ROW_CAP or len(sources) >= ROW_CAP:
                continue                             # surely in too many documents
            docs: dict[str, tuple] = {}              # document -> sort key of its best matching item
            for item_id, doc_id, kind, ordinal in items:
                key = (kind != "chunk", ordinal if ordinal is not None else 1 << 30, str(item_id))
                if str(doc_id) not in docs or key < docs[str(doc_id)]:
                    docs[str(doc_id)] = key
            matched = set(docs) | {str(d) for (d,) in sources}
            if not matched or len(matched) > MAX_DOCS:
                continue                             # in no document, or in too many
            for doc_id in matched:
                held.setdefault(doc_id, []).append(tok)
                if doc_id in docs and (doc_id not in shown or docs[doc_id] < shown[doc_id][0]):
                    shown[doc_id] = (docs[doc_id], docs[doc_id][2])
        if not held:
            return []
        bare = [d for d in held if d not in shown]   # held by its id (documents.source) alone
        if bare:
            for item_id, doc_id in cur.execute(
                    _FIRST_ITEM_SQL, {"bank_id": bank_id, "docs": bare, "meta": meta}).fetchall():
                shown[str(doc_id)] = (None, str(item_id))
        rows = {str(r[0]): r for r in cur.execute(
            _ROWS_SQL, {"ids": [item for _, item in shown.values()]}).fetchall()}
    hits = []
    for doc_id, toks in held.items():
        if doc_id not in shown:
            continue                                 # none of its items passes the filter
        r = rows[shown[doc_id][1]]
        hits.append(ExactHit(doc_id, str(r[0]), r[4] or "", r[2], r[3], r[5] or {},
                             list(r[6] or []), tuple(toks)))
    hits.sort(key=lambda h: (-len(h.matched), h.document_id))
    return hits


def _boost(top: float, n: float) -> float:
    """What a document holding `n` tokens adds to its score: more than the whole
    spread of the ungated scores, so it always clears them, `n` times over."""
    return n * (max(top, 0.0) + GATE)


def promote_rows(rows: list[dict], hits: list[ExactHit]) -> list[dict]:
    """The legacy recall's rows (`hybrid_search`) with the exact documents first:
    more tokens first, each group in the order it came. A document that none of
    the legs returned is added from its hit. A promoted row carries `exact` (the
    number of tokens it holds) and `gate_boost` (added to its score); nothing
    else of it changes. The input rows are not touched."""
    if not hits:
        return rows
    top = max((float(r["rrf_score"]) for r in rows), default=0.0)
    by_doc = {h.document_id: h for h in hits}
    present = {str(r["document_id"]) for r in rows}
    pool = [dict(r) for r in rows] + [h.row() for h in hits if h.document_id not in present]
    gated, rest = [], []
    for pos, r in enumerate(pool):
        h = by_doc.get(str(r["document_id"]))
        if h is None:
            rest.append(r)
            continue
        r["exact"] = h.score
        r["gate_boost"] = _boost(top, h.score)
        gated.append((-h.score, -float(r["rrf_score"]), pos, r))
    gated.sort(key=lambda g: g[:3])
    return [g[3] for g in gated] + rest


def promote_fused(fused: list[FusedDoc], hits: list[ExactHit]) -> list[FusedDoc]:
    """The channel pipeline's fused documents with the exact ones first (the
    same rule as `promote_rows`). A promoted document's evidence is the item that
    holds the token, and its `scores` and `ranks` gain an `exact` entry; one the
    channels did not return is added."""
    if not hits:
        return fused
    top = max((f.score for f in fused), default=0.0)
    by_doc = {f.document_id: f for f in fused}
    promoted = {h.document_id for h in hits}
    gated = []
    for rank, h in enumerate(hits, start=1):
        cand = Candidate(
            document_id=h.document_id, item_id=h.item_id, source=h.source, channel="exact",
            rank=rank, score=h.score, evidence=h.original_chunk,
            detail={"content": h.content, "metadata": h.metadata, "matched": list(h.matched)})
        f = by_doc.get(h.document_id)
        gated.append(FusedDoc(
            document_id=h.document_id, source=h.source,
            score=(f.score if f else 0.0) + _boost(top, h.score),
            best=cand, ranks={**(f.ranks if f else {}), "exact": rank},
            scores={**(f.scores if f else {}), "exact": h.score}))
    gated.sort(key=lambda f: (-f.score, f.source, f.document_id))
    return gated + [f for f in fused if f.document_id not in promoted]


def trace_entries(hits: list[ExactHit], latency_ms: int) -> tuple[dict, list[dict]]:
    """The `channels` entry and the `candidates` entries of a channel-pipeline
    trace for a gate that ran (the shape of a channel's own)."""
    channel = {"name": "exact", "kind": "gate", "weight": None, "n": len(hits),
               "latency_ms": latency_ms, "cost_usd": 0.0, "error": None}
    cands = [{"document_id": h.document_id, "item_id": h.item_id, "channel": "exact",
              "rank": i, "score": h.score} for i, h in enumerate(hits, start=1)]
    return channel, cands
