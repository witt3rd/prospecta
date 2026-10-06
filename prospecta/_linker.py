"""The Linker: typed links and entities between memory items (design 8.7).

Runs after retain (asynchronously through Memory's link worker, or on demand
with Memory.link_document / link_pending). For one document it

1. stitches NEXT links between its chunks and writes temporal links to the
   neighbouring notes (PRECEDES / SUCCEEDS / TEMPORALLY_CLOSE), all in SQL from
   the note's date and person, no model call;
2. extracts entities with the injected `llm` (Sonnet), fills
   memory_entities / memory_item_entities and writes SHARED_ENTITY links to the
   other items that mention them (a join, capped per entity);
3. links semantically related items: the top neighbours of each anchor item by
   pgvector, in SQL. With a `judge` (JevRelationJudge) Jev answers the typed
   relation questions (semantic, causes, caused_by; threshold 0.6) and the rows
   carry origin='jev'; without one (or when it fails) the neighbours above a
   cosine floor become RELATED_TO links with origin='pgvector'.

The anchor items of a document are its chunk items, else its question items.
A document is done when its memory_link_state row exists; replacing the
document's items removes that row and cascades its links away.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Protocol

from prospecta._template import render_prompt
from prospecta.stages import (
    JEV_BATCH, JEV_INPUT_BYTES, JEV_MODEL, JEV_PASSAGE_CHARS, JEV_TIMEOUT_S,
    JevTransport, call_llm, parse_json_object, totals,
)

logger = logging.getLogger(__name__)

NEIGHBOURS = 10            # candidates per anchor, by pgvector
JEV_THRESHOLD = 0.6        # Jev-Mem's relation probability threshold
VECTOR_FLOOR = 0.75        # cosine floor of the model-free RELATED_TO fallback
VECTOR_TOP = 5
TEMPORAL_DAYS = 3
TEMPORAL_CLOSE_CAP = 5
ENTITY_CAP = 10            # items linked per shared entity
ETYPES = {"person", "org", "place", "project", "product", "event", "other"}

_ANCHORS = """
SELECT id, kind, content, original_chunk, ordinal FROM (
    SELECT m.*, bool_or(m.kind = 'chunk') OVER () AS has_chunk
    FROM memory_items m WHERE m.document_id = %(doc)s
) x WHERE (has_chunk AND kind = 'chunk') OR (NOT has_chunk AND kind = 'question')
ORDER BY ordinal NULLS LAST, id
"""

_NEXT = """
INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, confidence, origin)
SELECT %(bank)s, a.id, b.id, 'TEMPORAL', 'NEXT', 1.0, 'sql'
FROM memory_items a JOIN memory_items b
  ON b.document_id = a.document_id AND b.kind = 'chunk' AND b.ordinal = a.ordinal + 1
WHERE a.document_id = %(doc)s AND a.kind = 'chunk'
ON CONFLICT DO NOTHING
"""

# Temporal links between the representative item of this note and its
# chronological neighbours of the same person; dates from prospecta_doc_date.
_TEMPORAL = """
WITH me AS (
    SELECT d.id, d.bank_id, d.created_at,
           prospecta_doc_date(d.document_metadata, d.created_at) AS dt,
           d.document_metadata->>'person' AS person
    FROM documents d WHERE d.id = %(doc)s
),
others AS (
    SELECT d.id, d.created_at, prospecta_doc_date(d.document_metadata, d.created_at) AS dt
    FROM documents d, me
    WHERE d.bank_id = me.bank_id AND d.id <> me.id
      AND d.document_metadata->>'person' IS NOT DISTINCT FROM me.person
),
prev AS (SELECT o.id FROM others o, me WHERE (o.dt, o.created_at, o.id) < (me.dt, me.created_at, me.id)
         ORDER BY o.dt DESC, o.created_at DESC, o.id DESC LIMIT 1),
nxt AS (SELECT o.id FROM others o, me WHERE (o.dt, o.created_at, o.id) > (me.dt, me.created_at, me.id)
        ORDER BY o.dt, o.created_at, o.id LIMIT 1),
near AS (SELECT o.id, abs(o.dt - me.dt) AS days FROM others o, me
          WHERE abs(o.dt - me.dt) <= %(days)s ORDER BY abs(o.dt - me.dt), o.id LIMIT %(cap)s),
rep AS (   -- one representative item per note: the first anchor
    SELECT DISTINCT ON (d.id) d.id AS doc, m.id AS item
    FROM documents d JOIN memory_items m ON m.document_id = d.id
    WHERE d.bank_id = %(bank)s
      AND d.id IN (SELECT id FROM me UNION SELECT id FROM prev
                   UNION SELECT id FROM nxt UNION SELECT id FROM near)
    ORDER BY d.id, (m.kind = 'chunk') DESC, m.ordinal NULLS LAST, m.id
),
edges(a, b, subtype, conf) AS (
    SELECT p.id, me.id, 'PRECEDES', 1.0::real FROM prev p, me
    UNION ALL SELECT me.id, p.id, 'SUCCEEDS', 1.0 FROM prev p, me
    UNION ALL SELECT me.id, n.id, 'PRECEDES', 1.0 FROM nxt n, me
    UNION ALL SELECT n.id, me.id, 'SUCCEEDS', 1.0 FROM nxt n, me
    UNION ALL SELECT me.id, c.id, 'TEMPORALLY_CLOSE', (1.0 / (1 + c.days))::real FROM near c, me
)
INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, confidence, origin, evidence)
SELECT %(bank)s, ra.item, rb.item, 'TEMPORAL', e.subtype, e.conf, 'sql',
       jsonb_build_object('person', (SELECT person FROM me))
FROM edges e JOIN rep ra ON ra.doc = e.a JOIN rep rb ON rb.doc = e.b
ON CONFLICT DO NOTHING
"""

_SHARED_ENTITY = """
INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, confidence, origin, evidence)
SELECT %(bank)s, x.a_item, x.b_item, 'ENTITY', 'SHARED_ENTITY', 1.0, 'sql',
       jsonb_build_object('entities', jsonb_agg(DISTINCT x.name))
FROM (
    SELECT a.item_id AS a_item, b.item_id AS b_item, e.name
    FROM memory_item_entities a
    JOIN memory_items ai ON ai.id = a.item_id
    JOIN memory_entities e ON e.id = a.entity_id
    CROSS JOIN LATERAL (
        SELECT ie.item_id FROM memory_item_entities ie
        JOIN memory_items bi ON bi.id = ie.item_id
        WHERE ie.entity_id = a.entity_id AND bi.document_id <> ai.document_id
          AND bi.kind = ai.kind
        ORDER BY bi.created_at DESC, bi.id LIMIT %(cap)s
    ) b
    WHERE ai.document_id = %(doc)s
) x
GROUP BY x.a_item, x.b_item
ON CONFLICT DO NOTHING
"""

_NEIGHBOUR_SQL = """
SELECT n.id, n.document_id, n.content, n.original_chunk,
       1 - (n.embedding <=> a.embedding) AS cos
FROM memory_items a
CROSS JOIN LATERAL (
    SELECT m.id, m.document_id, m.content, m.original_chunk, m.embedding
    FROM memory_items m
    WHERE m.bank_id = %(bank)s AND m.kind = '{kind}' AND m.document_id <> %(doc)s
    ORDER BY m.embedding <=> a.embedding
    LIMIT %(n)s
) n
WHERE a.id = %(item)s
"""


@dataclass
class RelationHit:
    candidate: int          # index into the candidates asked about
    link_type: str
    subtype: str
    confidence: float
    forward: bool = True    # src -> candidate (False: candidate -> src)


class RelationJudge(Protocol):
    def judge(self, source_text: str, candidates: list[str],
              calls: list[dict]) -> list[RelationHit]: ...


# relation question -> (link_type, subtype, forward)
RELATIONS = {
    "semantic": ("SEMANTIC", "RELATED_TO", True),
    "causes": ("CAUSAL", "LEADS_TO", True),
    "caused_by": ("CAUSAL", "LEADS_TO", False),
}
_RELATION_TASK = {
    "semantic": "Is this record about the same subject as the query, closely enough that someone reading one should be pointed to the other?",
    "causes": "Does the query text lead to, or cause, what this record describes?",
    "caused_by": "Is what the query text describes caused or led to by what this record describes?",
}
_RELATION_CRITERIA = [
    "No", "Barely", "Probably", "Clearly",
]


class JevRelationJudge:
    """Jev (System One, `score` questions, 0..3) answers the relation questions
    of Jev-Mem's linker: semantic, causes, caused_by, one question per
    (candidate, relation). score / 3 is the confidence; a link needs at least
    `threshold` (0.6). Same wire shape and limits as the Jev reranker."""

    def __init__(self, transport: JevTransport, model: str = JEV_MODEL,
                 timeout: float = JEV_TIMEOUT_S, threshold: float = JEV_THRESHOLD):
        self.transport, self.model, self.timeout = transport, model, timeout
        self.threshold = threshold

    def _question(self, relation: str, cand: str) -> dict:
        return {"type": "score",
                "instructions": {"task": _RELATION_TASK[relation], "kind": "note",
                                 "title": relation, "text": cand[:JEV_PASSAGE_CHARS]},
                "criteria": _RELATION_CRITERIA}

    def judge(self, source_text: str, candidates: list[str],
              calls: list[dict]) -> list[RelationHit]:
        asks = [(ci, rel) for ci in range(len(candidates)) for rel in RELATIONS]
        batches: list[list[tuple[int, str]]] = []
        cur: list[tuple[int, str]] = []
        for a in asks:
            trial = cur + [a]
            if cur and (len(trial) > JEV_BATCH or len(json.dumps(
                    self._request(source_text, candidates, trial)).encode()) > JEV_INPUT_BYTES):
                batches.append(cur)
                cur = [a]
            else:
                cur = trial
        if cur:
            batches.append(cur)
        hits: list[RelationHit] = []
        for batch in batches:
            req = self._request(source_text, candidates, batch)
            t0 = time.monotonic()
            rec: dict = {"purpose": "jev_relations", "model": self.model, "tokens_in": None,
                         "tokens_out": None, "cost_usd": None, "json_mode": False,
                         "messages_count": len(batch), "prompt_text": json.dumps(req),
                         "response_text": None, "error": None}
            try:
                resp = self.transport(req, self.timeout)
                rec["response_text"] = json.dumps(resp)
                u = resp.get("usage") or {}
                rec.update(model=resp.get("model") or self.model,
                           tokens_in=u.get("input_tokens"),
                           tokens_out=u.get("output_tokens"), cost_usd=u.get("cost"))
                answers = resp.get("answers")
                if not isinstance(answers, dict) or set(answers) != {
                        f"p{n}" for n in range(len(batch))}:
                    raise ValueError("answers are not exactly the asked ids")
                for n, (ci, rel) in enumerate(batch):
                    a = answers[f"p{n}"]
                    v = a.get("score") if isinstance(a, dict) else None
                    if isinstance(v, bool) or not isinstance(v, (int, float)) \
                            or not 0 <= v <= 3:
                        raise ValueError("answer does not score every question in 0..3")
                    conf = float(v) / 3.0
                    if conf >= self.threshold:
                        lt, st, fwd = RELATIONS[rel]
                        hits.append(RelationHit(ci, lt, st, conf, fwd))
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                rec["duration_ms"] = int((time.monotonic() - t0) * 1000)
                calls.append(rec)
        return hits

    def _request(self, source_text, candidates, batch) -> dict:
        return {"model": self.model, "state": {"query": source_text},
                "questions": {f"p{n}": self._question(rel, candidates[ci])
                              for n, (ci, rel) in enumerate(batch)}}


@dataclass
class Linker:
    """llm: Sonnet callable for entity extraction (None = no entities);
    judge: a RelationJudge (None = model-free pgvector RELATED_TO links)."""
    llm: Any = None
    judge: RelationJudge | None = None
    model: str | None = None
    neighbours: int = NEIGHBOURS
    temporal_days: int = TEMPORAL_DAYS
    entity_cap: int = ENTITY_CAP
    vector_floor: float = VECTOR_FLOOR
    asynchronous: bool = True      # Memory runs it on a worker thread after retain

    # ------------------------------------------------------------ one document
    def link_document(self, conn, bank_id: str, document_id: str,
                      calls: list[dict] | None = None) -> dict:
        """Link one document; commits. Returns stats. Errors in a step are
        recorded in the state row and never raise."""
        calls = calls if calls is not None else []
        stats: dict = {"temporal": 0, "entities": 0, "entity_links": 0,
                       "semantic": 0, "errors": []}
        with conn.cursor() as cur:
            cur.execute(_ANCHORS, {"doc": document_id})
            anchors = [dict(zip([c.name for c in cur.description], r)) for r in cur.fetchall()]
        if anchors:
            for name, step in (("temporal", self._temporal), ("entities", self._entities),
                               ("semantic", self._semantic)):
                try:
                    step(conn, bank_id, document_id, anchors, stats, calls)
                    conn.commit()
                except Exception as exc:
                    conn.rollback()
                    stats["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
                    logger.warning("linker %s failed for %s: %s", name, document_id, exc)
        stats.update(totals(calls))
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO memory_link_state (document_id, bank_id, status, stats, error) "
                "VALUES (%s, %s, %s, %s::jsonb, %s) ON CONFLICT (document_id) DO UPDATE "
                "SET status = EXCLUDED.status, stats = EXCLUDED.stats, "
                "error = EXCLUDED.error, linked_at = now()",
                (document_id, bank_id, "error" if stats["errors"] else "linked",
                 json.dumps(stats), "; ".join(stats["errors"]) or None))
        conn.commit()
        return stats

    # --------------------------------------------------------------- the steps
    def _temporal(self, conn, bank_id, doc, anchors, stats, calls):
        with conn.cursor() as cur:
            cur.execute(_NEXT, {"bank": bank_id, "doc": doc})
            n = cur.rowcount
            cur.execute(_TEMPORAL, {"bank": bank_id, "doc": doc,
                                    "days": self.temporal_days, "cap": TEMPORAL_CLOSE_CAP})
            stats["temporal"] = n + cur.rowcount

    def _entities(self, conn, bank_id, doc, anchors, stats, calls):
        if self.llm is None:
            return
        with conn.cursor() as cur:
            cur.execute("SELECT original_text FROM documents WHERE id = %s", (doc,))
            text = cur.fetchone()[0]
        prompt = render_prompt("extract-entities", {"text": text})
        raw = call_llm(self.llm, [{"role": "user", "content": prompt}],
                       purpose="extract_entities", calls=calls, model=self.model)
        found = parse_entities(raw)
        if found:
            with conn.cursor() as cur:
                for name, etype in found:
                    norm = normalise(name)
                    cur.execute(
                        "INSERT INTO memory_entities (bank_id, name, norm, etype) "
                        "VALUES (%s, %s, %s, %s) ON CONFLICT (bank_id, norm, etype) "
                        "DO UPDATE SET name = memory_entities.name RETURNING id",
                        (bank_id, name, norm, etype))
                    eid = cur.fetchone()[0]
                    hits = 0
                    for a in anchors:   # the items whose text names it
                        n = len(re.findall(re.escape(norm), normalise(
                            a["original_chunk"] or a["content"])))
                        if n:
                            hits += 1
                            cur.execute(
                                "INSERT INTO memory_item_entities (item_id, entity_id, n) "
                                "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                                (a["id"], eid, n))
                    if not hits:   # named in the note, not in a single chunk: first anchor
                        cur.execute(
                            "INSERT INTO memory_item_entities (item_id, entity_id, n) "
                            "VALUES (%s, %s, 1) ON CONFLICT DO NOTHING", (anchors[0]["id"], eid))
                    stats["entities"] += 1
        with conn.cursor() as cur:
            cur.execute(_SHARED_ENTITY, {"bank": bank_id, "doc": doc, "cap": self.entity_cap})
            stats["entity_links"] = cur.rowcount

    def _semantic(self, conn, bank_id, doc, anchors, stats, calls):
        for a in anchors:
            kind = "chunk" if a["kind"] == "chunk" else "question"
            with conn.cursor() as cur:
                cur.execute(_NEIGHBOUR_SQL.format(kind=kind), {
                    "bank": bank_id, "doc": doc, "n": self.neighbours, "item": a["id"]})
                cands = [dict(zip([c.name for c in cur.description], r)) for r in cur.fetchall()]
            if not cands:
                continue
            hits: list[RelationHit] | None = None
            origin = "jev"
            if self.judge is not None:
                try:
                    hits = self.judge.judge(
                        a["original_chunk"] or a["content"],
                        [c["original_chunk"] or c["content"] for c in cands], calls)
                except Exception as exc:
                    stats["errors"].append(f"jev: {type(exc).__name__}: {exc}")
            if hits is None:   # no judge, or Jev could not answer: pgvector neighbours
                origin = "pgvector"
                hits = [RelationHit(i, "SEMANTIC", "RELATED_TO", float(c["cos"]))
                        for i, c in enumerate(cands[:VECTOR_TOP])
                        if float(c["cos"]) >= self.vector_floor]
            with conn.cursor() as cur:
                for h in hits:
                    c = cands[h.candidate]
                    src, dst = (a["id"], c["id"]) if h.forward else (c["id"], a["id"])
                    cur.execute(
                        "INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, "
                        "confidence, origin, evidence) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb) "
                        "ON CONFLICT DO NOTHING",
                        (bank_id, src, dst, h.link_type, h.subtype, h.confidence, origin,
                         json.dumps({"cosine": float(c["cos"])})))
                    stats["semantic"] += cur.rowcount


def normalise(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().lower())


def parse_entities(raw: str) -> list[tuple[str, str]]:
    obj = parse_json_object(raw)
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for e in obj.get("entities") or []:
        if isinstance(e, str):
            e = {"name": e}
        if not isinstance(e, dict) or not isinstance(e.get("name"), str):
            continue
        name = e["name"].strip()
        etype = str(e.get("type", "other")).strip().lower()
        etype = etype if etype in ETYPES else "other"
        key = (normalise(name), etype)
        if name and key not in seen:
            seen.add(key)
            out.append((name, etype))
    return out


# ---------------------------------------------------------------- the worker

def pending_documents(conn, bank_id: str, limit: int) -> list[str]:
    """Documents with items and no memory_link_state row, oldest first."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT d.id::text FROM documents d "
            "WHERE d.bank_id = %s "
            "  AND NOT EXISTS (SELECT 1 FROM memory_link_state s WHERE s.document_id = d.id) "
            "  AND EXISTS (SELECT 1 FROM memory_items m WHERE m.document_id = d.id) "
            "ORDER BY d.created_at, d.id LIMIT %s", (bank_id, limit))
        return [r[0] for r in cur.fetchall()]
