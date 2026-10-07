"""The Linker: typed links and entities between memory items (design 8.7).

Runs after retain (asynchronously through Memory's link worker, or on demand
with Memory.link_document / link_pending). For one document it

1. stitches NEXT links between its chunks and writes temporal links to the
   neighbouring notes (PRECEDES / SUCCEEDS / TEMPORALLY_CLOSE), all in SQL from
   the note's date and person, no model call;
2. extracts entities with the injected `llm` (Sonnet), fills
   memory_entities / memory_item_entities and writes SHARED_ENTITY links to the
   other items that mention them (a join; entities held by more than ENTITY_HUB items are hubs and get no
   per-anchor links: GraphExpand reaches their holders through the entity table);
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

import hashlib
import json
import logging
import re
import random
import time
from dataclasses import dataclass
from typing import Any, Protocol

import psycopg.errors

from prospecta._template import render_prompt
from prospecta.channels.graph import ENTITY_HUB
from prospecta.stages import (
    CHARS_PER_TOKEN, JEV_MODEL, JEV_TIMEOUT_S, fit_passage,
    JevTransport, call_llm, parse_json_object, totals,
)

logger = logging.getLogger(__name__)

NEIGHBOUR_MIN_REL = 0.9    # candidates per anchor: neighbours with cosine >= 0.9 x the nearest, ending at the first
                           # marginal drop larger than (1 - 0.9) x the nearest (no count). Embedding cosines are
                           # compressed (0.5..0.9), so 0.6 admitted nearly every note: the import was quadratic.
NEIGHBOUR_MIN_COS = 0.5    # absolute floor: a neighbour below this cosine is never judged, whatever the relative stop
                           # says (cosines of unrelated notes sit at 0.2..0.4; related ones above 0.5). Configurable.
JEV_MAX_QUESTIONS_PER_CALL = 96   # questions per Jev request (verified live: 48, 96, 150, 300 answered; config)
JEV_MAX_INPUT_TOKENS = 45_000     # estimated input tokens per request (verified: 56.6k ok, ~110k+ fails 400); split above it
JUDGE_RETRIES = 3          # attempts per judge call before the failure is surfaced (state error, pgvector fallback)
JUDGE_BACKOFF_S = 0.5
HNSW_EF_MAX = 1000         # pgvector's hnsw.ef_search ceiling: beyond it the fetch is exhaustive
NEIGHBOUR_PAGE = 16        # rows per HNSW fetch (throughput; doubles until the stop rule is decided)
JUDGE_TOP_K = 32           # neighbour notes judged per note: one call at 96 questions (32 x 3 relations)
JUDGE_NEAREST = 3          # a note's nearest neighbours are judged even when not mutual
JUDGE_FLOOR_FRAC = 0.25    # all-pairs: judge a candidate above median + frac x (best - median) of the anchor's neighbours; 0 = off
JEV_THRESHOLD = 0.6        # Jev-Mem's relation probability threshold
VECTOR_FLOOR = 0.75        # cosine floor of the model-free RELATED_TO fallback
TEMPORAL_DAYS = 3
TEMPORAL_MIN_REL = 0.5     # TEMPORALLY_CLOSE: every note within TEMPORAL_DAYS whose closeness 1/(1+days) >= 0.5 x the closest's
DEADLOCK_RETRIES = 6       # attempts per step on deadlock / serialization failure
DEADLOCK_BACKOFF_S = 0.05  # first backoff; doubles with jitter
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
ORDER BY a.id, b.id
ON CONFLICT DO NOTHING
"""

# Temporal links between the representative item of this note and its
# chronological neighbours of the same person; dates from prospecta_doc_date.
_TEMPORAL = """
WITH me AS (
    SELECT d.id, d.bank_id, d.created_at,
           prospecta_doc_date(d.document_metadata, d.created_at, d.created_on) AS dt,
           COALESCE(d.person, d.document_metadata->>'person') AS person
    FROM documents d WHERE d.id = %(doc)s
),
others AS (
    SELECT d.id, d.created_at, prospecta_doc_date(d.document_metadata, d.created_at, d.created_on) AS dt
    FROM documents d, me
    WHERE d.bank_id = me.bank_id AND d.id <> me.id
      AND COALESCE(d.person, d.document_metadata->>'person') IS NOT DISTINCT FROM me.person
),
prev AS (SELECT o.id FROM others o, me WHERE (o.dt, o.created_at, o.id) < (me.dt, me.created_at, me.id)
         ORDER BY o.dt DESC, o.created_at DESC, o.id DESC LIMIT 1),
nxt AS (SELECT o.id FROM others o, me WHERE (o.dt, o.created_at, o.id) > (me.dt, me.created_at, me.id)
        ORDER BY o.dt, o.created_at, o.id LIMIT 1),
closest AS (SELECT min(abs(o.dt - me.dt)) AS days FROM others o, me
            WHERE abs(o.dt - me.dt) <= %(days)s),
near AS (SELECT o.id, abs(o.dt - me.dt) AS days FROM others o, me, closest cl
          WHERE abs(o.dt - me.dt) <= %(days)s
            AND 1.0 / (1 + abs(o.dt - me.dt)) >= %(rel)s * (1.0 / (1 + cl.days))),
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
ORDER BY ra.item, rb.item, e.subtype
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
    CROSS JOIN LATERAL (   -- a hub entity gets no per-anchor links (a cap would drop holders)
        SELECT count(*) AS n FROM (SELECT 1 FROM memory_item_entities x
                                   WHERE x.entity_id = a.entity_id LIMIT %(cap)s + 1) q
    ) hc
    CROSS JOIN LATERAL (
        SELECT ie.item_id FROM memory_item_entities ie
        JOIN memory_items bi ON bi.id = ie.item_id
        WHERE ie.entity_id = a.entity_id AND bi.document_id <> ai.document_id
          AND bi.kind = ai.kind
        ORDER BY bi.created_at DESC, bi.id LIMIT %(cap)s
    ) b
    WHERE ai.document_id = %(doc)s AND hc.n <= %(cap)s
) x
GROUP BY x.a_item, x.b_item
ORDER BY x.a_item, x.b_item
ON CONFLICT DO NOTHING
"""

_NEIGHBOUR_SQL = """
SELECT n.id, n.document_id, n.content, n.original_chunk,
       1 - (n.embedding::vector({dim}) <=> a.embedding::vector({dim})) AS cos
FROM memory_items a
CROSS JOIN LATERAL (
    SELECT m.id, m.document_id, m.content, m.original_chunk, m.embedding
    FROM memory_items m
    WHERE m.bank_id = %(bank)s AND m.kind = '{kind}' AND m.document_id <> %(doc)s
    ORDER BY m.embedding::vector({dim}) <=> a.embedding::vector({dim})   -- the index expression
    LIMIT %(n)s
) n
WHERE a.id = %(item)s
"""


class _TooLarge(Exception):
    """Jev answered HTTP 400 max_tokens_exceeded: the request is split and retried."""


def _is_max_tokens(exc: Exception) -> bool:
    body = ""
    if hasattr(exc, "read"):   # urllib HTTPError: the body names the error
        try:
            body = exc.read().decode(errors="replace")
        except Exception:
            pass
    return "max_tokens_exceeded" in f"{exc} {body}"


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
    `threshold` (0.6).

    Batching (verified on the live wire by the scout: 48, 96, 150 and 300
    questions per request were answered; 300 questions of 1,000 chars, 419 KB,
    failed loud with HTTP 400 max_tokens_exceeded): the real limit is input
    TOKENS, not questions. A request holds up to `max_questions_per_call`
    questions (96) and an estimated `max_input_tokens` (45,000) at most; with
    `relations_per_call` all relation questions of a candidate travel together.
    A larger set SPLITS into several requests; no candidate or text is dropped
    or cut. A 400 max_tokens_exceeded halves the request and retries."""

    def __init__(self, transport: JevTransport, model: str = JEV_MODEL,
                 timeout: float = JEV_TIMEOUT_S, threshold: float = JEV_THRESHOLD,
                 max_questions_per_call: int = JEV_MAX_QUESTIONS_PER_CALL,
                 max_input_tokens: int = JEV_MAX_INPUT_TOKENS,
                 relations_per_call: bool = True):
        self.transport, self.model, self.timeout = transport, model, timeout
        self.threshold = threshold
        self.max_questions_per_call = max_questions_per_call
        self.max_input_tokens = max_input_tokens
        self.relations_per_call = relations_per_call

    def _question(self, relation: str, cand: str) -> dict:
        return {"type": "score",
                "instructions": {"task": _RELATION_TASK[relation], "kind": "note",
                                 "title": relation, "text": fit_passage(cand, relation)},
                "criteria": _RELATION_CRITERIA}

    def _units(self, candidates: list[str]) -> list[list[tuple[int, str]]]:
        """The indivisible groups of questions: a candidate's relation questions
        together (when relations_per_call and they fit one request), else single."""
        together = self.relations_per_call and self.max_questions_per_call >= len(RELATIONS)
        if together:
            return [[(ci, rel) for rel in RELATIONS] for ci in range(len(candidates))]
        return [[(ci, rel)] for ci in range(len(candidates)) for rel in RELATIONS]

    def _batches(self, source_text, candidates) -> list[list[tuple[int, str]]]:
        base = len(json.dumps(self._request(source_text, candidates, [])).encode())
        batches, cur, q, size = [], [], 0, base
        for unit in self._units(candidates):
            usize = sum(len(json.dumps(self._question(rel, candidates[ci])).encode()) + 12
                        for ci, rel in unit)
            if cur and (q + len(unit) > self.max_questions_per_call
                        or (size + usize) // CHARS_PER_TOKEN > self.max_input_tokens):
                batches.append(cur)
                cur, q, size = [], 0, base
            cur += unit
            q += len(unit)
            size += usize
        if cur:
            batches.append(cur)
        return batches

    def judge(self, source_text: str, candidates: list[str],
              calls: list[dict]) -> list[RelationHit]:
        hits: list[RelationHit] = []
        pending = self._batches(source_text, candidates)
        while pending:
            batch = pending.pop(0)
            try:
                hits.extend(self._send(source_text, candidates, batch, calls))
            except _TooLarge:
                units = self._regroup(batch)
                if len(units) < 2:
                    logger.error("jev link request of one candidate is over the token limit "
                                 "(max_tokens_exceeded); full source text: %s; full candidate "
                                 "text: %s", source_text, candidates[batch[0][0]])
                    raise
                mid = len(units) // 2
                logger.warning("jev max_tokens_exceeded on %d questions: halving", len(batch))
                pending[:0] = [[a for u in units[:mid] for a in u],
                               [a for u in units[mid:] for a in u]]
        return hits

    @staticmethod
    def _regroup(batch):
        """The batch's asks grouped by candidate, in order (a candidate stays whole
        when it is the only one it can be split into)."""
        groups: list[list[tuple[int, str]]] = []
        for ask in batch:
            if groups and groups[-1][0][0] == ask[0]:
                groups[-1].append(ask)
            else:
                groups.append([ask])
        if len(groups) < 2 and len(batch) > 1:   # one candidate, several questions
            return [[a] for a in batch]
        return groups

    def _send(self, source_text, candidates, batch, calls) -> list[RelationHit]:
        hits: list[RelationHit] = []
        req = self._request(source_text, candidates, batch)
        t0 = time.monotonic()
        rec: dict = {"purpose": "jev_relations", "model": self.model, "tokens_in": None,
                     "tokens_out": None, "cost_usd": None, "json_mode": False,
                     "messages_count": len(batch), "prompt_text": json.dumps(req),
                     "response_text": None, "error": None}
        try:
            try:
                resp = self.transport(req, self.timeout)
            except Exception as exc:
                if _is_max_tokens(exc):
                    raise _TooLarge(str(exc)) from exc
                raise
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
    neighbour_min_rel: float = NEIGHBOUR_MIN_REL
    neighbour_min_cos: float = NEIGHBOUR_MIN_COS
    temporal_min_rel: float = TEMPORAL_MIN_REL
    temporal_days: int = TEMPORAL_DAYS
    entity_hub: int = ENTITY_HUB
    vector_floor: float = VECTOR_FLOOR
    judge_top_k: int = JUDGE_TOP_K          # a note judges at most its top-K neighbour notes ...
    judge_nearest: int = JUDGE_NEAREST      # ... always its nearest few, the rest only if mutual
    judge_floor_frac: float = JUDGE_FLOOR_FRAC   # all-pairs relative floor (see anchor_floor); 0 judges every qualifying pair
    link_completeness: str | None = None    # 'all-pairs' | 'connected'; None = the bank's recall_config, else 'all-pairs'
    asynchronous: bool = True      # Memory runs it on a worker thread after retain

    # ------------------------------------------------------------ one document
    def link_document(self, conn, bank_id: str, document_id: str,
                      calls: list[dict] | None = None) -> dict:
        """Link one document; commits. Returns stats. Errors in a step are
        recorded in the state row and never raise. A session advisory lock on
        the document id serialises concurrent linkers of the same document."""
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(hashtextextended(%s, 0))",
                        (f"link:{document_id}",))
        try:
            return self._link_locked(conn, bank_id, document_id, calls)
        finally:
            try:
                conn.rollback()
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(hashtextextended(%s, 0))",
                                (f"link:{document_id}",))
                conn.commit()
            except psycopg.Error:   # closed connection: Postgres drops the lock itself
                logger.warning("linker could not release the lock of %s", document_id)

    def _link_locked(self, conn, bank_id, document_id, calls) -> dict:
        calls = calls if calls is not None else []
        stats: dict = {"temporal": 0, "entities": 0, "entity_links": 0,
                       "semantic": 0, "errors": []}
        with conn.cursor() as cur:
            cur.execute(_ANCHORS, {"doc": document_id})
            anchors = [dict(zip([c.name for c in cur.description], r)) for r in cur.fetchall()]
        if not anchors:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM documents WHERE id = %s", (document_id,))
                gone = cur.fetchone() is None
            conn.commit()
            if gone:   # deleted or replaced meanwhile: nothing to link, no state row to write
                logger.warning("linker: document %s no longer exists, skipped", document_id)
                stats["skipped"] = "document gone"
                return stats
        if anchors:
            for name, step in (("temporal", self._temporal), ("entities", self._entities),
                               ("semantic", self._semantic)):
                try:
                    self._retrying(conn, name, lambda: self._run_step(
                        step, conn, bank_id, document_id, anchors, stats, calls))
                except Exception as exc:
                    conn.rollback()
                    stats["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
                    logger.error("linker %s failed for %s (recorded; link_pending "
                                 "will retry): %s", name, document_id, exc)
        stats.update(totals(calls))

        def write_state():
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO memory_link_state (document_id, bank_id, status, stats, error) "
                    "VALUES (%s, %s, %s, %s::jsonb, %s) ON CONFLICT (document_id) DO UPDATE "
                    "SET status = EXCLUDED.status, stats = EXCLUDED.stats, "
                    "error = EXCLUDED.error, linked_at = now()",
                    (document_id, bank_id, "error" if stats["errors"] else "linked",
                     json.dumps(stats), "; ".join(stats["errors"]) or None))
            conn.commit()
        self._retrying(conn, "state", write_state)
        return stats

    @staticmethod
    def _run_step(step, conn, bank_id, doc, anchors, stats, calls):
        before = dict(stats)
        try:
            step(conn, bank_id, doc, anchors, stats, calls)
            conn.commit()
        except Exception:
            stats.clear()
            stats.update(before)   # a retried step must not double count
            raise

    @staticmethod
    def _retrying(conn, name, fn):
        """Run fn; on deadlock / serialization failure roll back and retry with
        exponential backoff and jitter. Other errors propagate at once."""
        delay = DEADLOCK_BACKOFF_S
        for attempt in range(1, DEADLOCK_RETRIES + 1):
            try:
                return fn()
            except (psycopg.errors.DeadlockDetected, psycopg.errors.SerializationFailure,
                    psycopg.errors.ForeignKeyViolation) as exc:   # FK: a target deleted concurrently; the retry re-reads
                conn.rollback()
                if attempt == DEADLOCK_RETRIES:
                    raise
                logger.warning("linker %s: %s, retry %d/%d", name, type(exc).__name__,
                               attempt, DEADLOCK_RETRIES - 1)
                time.sleep(delay * (1 + random.random()))
                delay *= 2

    # --------------------------------------------------------------- the steps
    def _temporal(self, conn, bank_id, doc, anchors, stats, calls):
        with conn.cursor() as cur:
            cur.execute(_NEXT, {"bank": bank_id, "doc": doc})
            n = cur.rowcount
            cur.execute(_TEMPORAL, {"bank": bank_id, "doc": doc,
                                    "days": self.temporal_days, "rel": self.temporal_min_rel})
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
        aliases = parse_aliases(raw)
        touched: set[str] = set()
        if found:
            with conn.cursor() as cur:
                for name, etype in sorted(found, key=lambda e: (normalise(e[0]), e[1])):
                    norm = normalise(name)
                    for canon, als in aliases:   # persons first: aliases resolve to them
                        if normalise(canon) == norm and etype == "person":
                            self._add_aliases(cur, bank_id, name, norm, als, touched, stats)
                    eid = self._alias_target(cur, bank_id, norm) if etype == "person" else None
                    if eid is not None:   # a known alias of a person: the person's mentions
                        self._attach(cur, anchors, eid, norm)
                        stats["entities"] += 1
                        continue
                    cur.execute(
                        "INSERT INTO memory_entities (bank_id, name, norm, etype) "
                        "VALUES (%s, %s, %s, %s) ON CONFLICT (bank_id, norm, etype) "
                        "DO UPDATE SET name = memory_entities.name RETURNING id",
                        (bank_id, name, norm, etype))
                    eid = cur.fetchone()[0]
                    self._attach(cur, anchors, eid, norm)
                    stats["entities"] += 1
        with conn.cursor() as cur:
            touched.add(doc)
            stats["entity_links"] = 0
            for d in sorted(touched):   # this note, and notes whose mentions an alias moved
                cur.execute(_SHARED_ENTITY, {"bank": bank_id, "doc": d, "cap": self.entity_hub})
                stats["entity_links"] += cur.rowcount
            cur.execute(
                "INSERT INTO memory_alias_state (document_id, bank_id, status, stats) "
                "VALUES (%s, %s, 'done', %s::jsonb) ON CONFLICT (document_id) DO UPDATE "
                "SET status = 'done', stats = EXCLUDED.stats, error = NULL, done_at = now()",
                (doc, bank_id, json.dumps({"aliases": stats.get("aliases", 0)})))

    @staticmethod
    def _attach(cur, anchors, eid, norm):
        hits = 0
        for a in anchors:   # the items whose text names it
            n = len(re.findall(re.escape(norm), normalise(
                a["original_chunk"] or a["content"])))
            if n:
                hits += 1
                cur.execute(
                    "INSERT INTO memory_item_entities (item_id, entity_id, n) "
                    "VALUES (%s, %s, %s) ON CONFLICT (item_id, entity_id) "
                    "DO UPDATE SET n = GREATEST(memory_item_entities.n, EXCLUDED.n)",
                    (a["id"], eid, n))
        if not hits:   # named in the note, not in a single chunk: first anchor
            cur.execute(
                "INSERT INTO memory_item_entities (item_id, entity_id, n) "
                "VALUES (%s, %s, 1) ON CONFLICT DO NOTHING", (anchors[0]["id"], eid))

    @staticmethod
    def _alias_target(cur, bank_id, norm):
        cur.execute("SELECT entity_id FROM memory_entity_aliases WHERE bank_id = %s AND norm = %s",
                    (bank_id, norm))
        r = cur.fetchone()
        return r[0] if r else None

    def _add_aliases(self, cur, bank_id, name, norm, alias_names, touched, stats):
        """Record alias rows resolving to the person `name`, and move onto the
        person the mentions of any entity already extracted under an alias."""
        cur.execute(
            "INSERT INTO memory_entities (bank_id, name, norm, etype) "
            "VALUES (%s, %s, %s, 'person') ON CONFLICT (bank_id, norm, etype) "
            "DO UPDATE SET name = memory_entities.name RETURNING id", (bank_id, name, norm))
        canon = cur.fetchone()[0]
        for alias in sorted(set(alias_names), key=normalise):
            an = normalise(alias)
            if an == norm or self._alias_target(cur, bank_id, an) not in (None, canon):
                continue
            cur.execute(
                "INSERT INTO memory_entity_aliases (bank_id, norm, alias, entity_id) "
                "VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING", (bank_id, an, alias, canon))
            stats["aliases"] = stats.get("aliases", 0) + cur.rowcount
            cur.execute(   # an entity already extracted under the alias becomes the person
                "SELECT id FROM memory_entities WHERE bank_id = %s AND norm = %s "
                "AND etype = 'person' AND id <> %s", (bank_id, an, canon))
            for (old,) in cur.fetchall():
                cur.execute("SELECT m.item_id, i.document_id::text FROM memory_item_entities m "
                            "JOIN memory_items i ON i.id = m.item_id WHERE m.entity_id = %s", (old,))
                for _, d in cur.fetchall():
                    touched.add(d)
                cur.execute(
                    "INSERT INTO memory_item_entities (item_id, entity_id, n) "
                    "SELECT item_id, %s, n FROM memory_item_entities WHERE entity_id = %s "
                    "ON CONFLICT (item_id, entity_id) DO UPDATE "
                    "SET n = memory_item_entities.n + EXCLUDED.n", (canon, old))
                cur.execute("DELETE FROM memory_item_entities WHERE entity_id = %s", (old,))

    def backfill_aliases(self, conn, bank_id: str, limit: int = 100,
                         after: str | None = None) -> tuple[int, str | None]:
        """Resumable alias backfill: documents with items and no done alias
        state, by document id after `after`. Returns (processed, last id);
        pass the last id back to resume. Errors are recorded and skipped."""
        with conn.cursor() as cur:
            cur.execute(
                "SELECT d.id::text FROM documents d WHERE d.bank_id = %s "
                "AND (%s::uuid IS NULL OR d.id > %s::uuid) "
                "AND NOT EXISTS (SELECT 1 FROM memory_alias_state s "
                "                WHERE s.document_id = d.id AND s.status = 'done') "
                "AND EXISTS (SELECT 1 FROM memory_items m WHERE m.document_id = d.id) "
                "ORDER BY d.id LIMIT %s", (bank_id, after, after, limit))
            ids = [r[0] for r in cur.fetchall()]
        last = after
        for doc in ids:
            last = doc
            with conn.cursor() as cur:
                cur.execute(_ANCHORS, {"doc": doc})
                anchors = [dict(zip([c.name for c in cur.description], r)) for r in cur.fetchall()]
            try:
                self._retrying(conn, "aliases", lambda: self._run_step(
                    self._entities, conn, bank_id, doc, anchors, {"entities": 0}, []))
            except Exception as exc:
                conn.rollback()
                logger.error("alias backfill failed for %s (recorded): %s", doc, exc)
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO memory_alias_state (document_id, bank_id, status, error) "
                        "VALUES (%s, %s, 'error', %s) ON CONFLICT (document_id) DO UPDATE "
                        "SET status = 'error', error = EXCLUDED.error, done_at = now()",
                        (doc, bank_id, f"{type(exc).__name__}: {exc}"))
                conn.commit()
        return len(ids), last

    def link_pass(self, conn, bank_id: str, mode: str = "all-pairs", limit: int | None = None,
                  calls: list[dict] | None = None, on_progress=None) -> dict:
        """Upgrade already linked documents to `mode` ('all-pairs'): re-run the semantic
        step only, for every linked document whose state does not record that
        completeness. Never touches retain (own connection, no lock beyond the link
        rows), RESUMABLE: a finished document records its completeness in its link
        state and is skipped next run, and the pair cache (memory_link_pairs) means a
        pair judged before is never asked again, so an interrupted pass loses nothing.
        Returns {"documents", "judged", "cached", "failed", "remaining"}; `limit`
        bounds the documents of this run; `on_progress(dict)` is called after each."""
        if mode not in ("connected", "all-pairs"):
            raise ValueError(f"mode must be 'connected' or 'all-pairs', got {mode!r}")
        calls = calls if calls is not None else []
        out = {"documents": 0, "judged": 0, "cached": 0, "failed": 0, "remaining": 0}
        if mode == "connected" or self.judge is None:   # all-pairs is a superset: nothing to upgrade to
            return out
        todo_sql = ("FROM memory_link_state s WHERE s.bank_id = %s "
                    "AND s.stats->>'completeness' IS DISTINCT FROM %s")
        with conn.cursor() as cur:
            cur.execute("SELECT s.document_id::text " + todo_sql + " ORDER BY s.document_id",
                        (bank_id, mode))
            ids = [r[0] for r in cur.fetchall()]
        total = len(ids)
        out["remaining"] = total
        for doc in ids[:limit]:
            with conn.cursor() as cur:
                cur.execute(_ANCHORS, {"doc": doc})
                anchors = [dict(zip([c.name for c in cur.description], r)) for r in cur.fetchall()]
            stats: dict = {"semantic": 0, "errors": []}
            try:
                if anchors:
                    self._retrying(conn, "pass", lambda: self._run_step(
                        lambda *a: self._semantic(*a, mode=mode), conn, bank_id, doc,
                        anchors, stats, calls))
                if stats["errors"]:
                    raise RuntimeError("; ".join(stats["errors"]))
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE memory_link_state SET stats = stats || %s::jsonb, linked_at = now() "
                        "WHERE document_id = %s",
                        (json.dumps({"completeness": mode,
                                     "pass_judged": stats.get("candidates_judged", 0)}), doc))
                conn.commit()
            except Exception as exc:   # recorded in the log; the document stays to do next run
                conn.rollback()
                out["failed"] += 1
                logger.error("link pass failed for %s (left for the next run): %s: %s",
                             doc, type(exc).__name__, exc)
            else:
                out["judged"] += stats.get("candidates_judged", 0)
                out["cached"] += stats.get("candidates_cached", 0)
                out["remaining"] -= 1
            out["documents"] += 1
            if on_progress:
                on_progress(dict(out))
        return out

    def _judge_with_retry(self, source, texts, calls):
        delay = JUDGE_BACKOFF_S
        for attempt in range(1, JUDGE_RETRIES + 1):
            try:
                return self.judge.judge(source, texts, calls)
            except Exception as exc:
                if attempt == JUDGE_RETRIES:
                    raise
                logger.warning("linker judge: %s: %s, retry %d/%d", type(exc).__name__, exc,
                               attempt, JUDGE_RETRIES - 1)
                time.sleep(delay)
                delay *= 2

    @staticmethod
    def text_hash(text: str) -> str:
        """sha256 of the text modulo frontmatter, whitespace and case: two chunks
        with the same hash are the same content."""
        return hashlib.sha256(normalise_text(text).encode()).hexdigest()

    @staticmethod
    def _pair_hash(th_a: str, th_b: str) -> str:
        lo, hi = sorted((th_a, th_b))
        return hashlib.sha256(f"{lo}:{hi}".encode()).hexdigest()

    @staticmethod
    def _text(item) -> str:
        return item["original_chunk"] or item["content"]

    def _doc_reps(self, conn, docs, kind):
        """The representative item of each document: its first anchor of `kind`
        (the order of _ANCHORS). One symmetric pair per document pair."""
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT ON (document_id) id, document_id, content, original_chunk "
                "FROM memory_items WHERE document_id = ANY(%s) AND kind = %s "
                "ORDER BY document_id, ordinal NULLS LAST, id", (list(docs), kind))
            return {r[1]: dict(zip(("id", "document_id", "content", "original_chunk"), r))
                    for r in cur.fetchall()}

    def _mutual(self, conn, bank_id, kind, dim, me_doc, rep) -> bool:
        """Is `me_doc` among the top `judge_top_k` neighbour documents of `rep`?"""
        with conn.transaction(), conn.cursor() as cur:
            cur.execute(f"SET LOCAL hnsw.ef_search = {max(2 * self.judge_top_k, 40)}")
            cur.execute(_NEIGHBOUR_SQL.format(kind=kind, dim=dim), {
                "bank": bank_id, "doc": rep["document_id"], "n": self.judge_top_k,
                "item": rep["id"]})
            return any(str(r[1]) == str(me_doc) for r in cur.fetchall())

    def _completeness(self, conn, bank_id, mode=None) -> str:
        mode = mode or self.link_completeness
        if mode is None:
            from prospecta.stages import read_recall_config
            mode = read_recall_config(conn, bank_id).get("link_completeness", "all-pairs")
        if mode not in ("connected", "all-pairs"):
            raise ValueError(f"link_completeness must be 'connected' or 'all-pairs', got {mode!r}")
        return mode

    def anchor_floor(self, cands) -> float:
        """The per-anchor similarity floor of the all-pairs pass, relative to the
        anchor's own neighbour distribution (no count cap): median + judge_floor_frac
        x (best - median) of the candidates' cosines. In a dense cluster the median
        is close to the best, so the floor sits high and only distinctly related
        pairs are judged; in a sparse region the spread is wide and the floor falls."""
        if self.judge_floor_frac <= 0 or len(cands) < 2:
            return float("-inf")
        cos = sorted(float(c["cos"]) for c in cands)
        median = cos[len(cos) // 2]
        return median + self.judge_floor_frac * (cos[-1] - median)

    def _judged_edges(self, conn, bank_id, me, cands, kind, dim, stats, calls,
                      all_pairs=False, floor=float("-inf")):
        """Relation edges (src, dst, link_type, subtype, confidence) between the
        representative item `me` and each candidate representative in `cands`
        (best first).

        Fewer pairs, no model call for:
        - a pair with the same normalised text (a duplicate): RELATED_TO at 1.0;
        - a pair already judged, from either side or in an earlier run: read back
          from memory_link_pairs (keyed by the text-hash pair, so it is global);
        - a pair outside the candidate's top-K reach: a pair is judged only when it
          is MUTUAL (each in the other's top `judge_top_k` neighbours) or the
          candidate is among my `judge_nearest` best, so no note loses its nearest
          links. SEMANTIC (same subject) is taken as transitive inside a dense
          cluster: members beyond a note's top-K are reached through their cluster
          neighbours; CAUSAL / LEADS_TO is NOT transitive, so it is kept only for
          the judged pairs (the nearest, mutual ones).
        With all_pairs (link_completeness = 'all-pairs', the default) every qualifying
        candidate is judged: no top-K cap, no mutual test (duplicates and the cache still
        apply). 'connected' is the cheaper explicit setting.
        With all_pairs, `floor` (anchor_floor) skips judging a candidate whose cosine is
        below it, except the `judge_nearest` best; counted in candidates_below_floor,
        never silent. Cached pairs are still read back (free).
        The rest are judged in batched calls; each judgment is stored once."""
        th_me = self.text_hash(self._text(me))
        edges: list[tuple] = []
        fresh_rows: list[tuple] = []
        pending = []   # (rank, cand, th_c, hash, me_is_lo)
        for rank, c in enumerate(cands):
            th_c = self.text_hash(self._text(c))
            if th_c == th_me:
                stats["candidates_duplicate"] = stats.get("candidates_duplicate", 0) + 1
                edges += [(me["id"], c["id"], "SEMANTIC", "RELATED_TO", 1.0, rank)]
                continue
            pending.append((rank, c, th_c, self._pair_hash(th_me, th_c), th_me < th_c))
        hashes = [p[3] for p in pending]
        cached = {}
        if hashes:
            with conn.cursor() as cur:
                cur.execute("SELECT pair_hash, hits FROM memory_link_pairs "
                            "WHERE pair_hash = ANY(%s)", (hashes,))
                cached = dict(cur.fetchall())
        todo = []
        for p in pending:
            rank, c, th_c, h, me_lo = p
            if h in cached:
                edges += [(me["id"], c["id"], lt, st, float(cf), rank, (bool(fw) == me_lo))
                          for lt, st, cf, fw in cached[h]]
                stats["candidates_cached"] = stats.get("candidates_cached", 0) + 1
            elif all_pairs and rank >= self.judge_nearest and float(c["cos"]) < floor:
                stats["candidates_below_floor"] = stats.get("candidates_below_floor", 0) + 1
            elif all_pairs or rank < self.judge_nearest or self._mutual(
                    conn, bank_id, kind, dim, me["document_id"], c):
                todo.append(p)
            else:
                stats["candidates_not_mutual"] = stats.get("candidates_not_mutual", 0) + 1
        if todo:
            got = self._judge_with_retry(self._text(me), [self._text(p[1]) for p in todo], calls)
            with conn.cursor() as cur:
                for k, (rank, c, th_c, h, me_lo) in enumerate(todo):
                    mine = [x for x in got if x.candidate == k]
                    edges += [(me["id"], c["id"], x.link_type, x.subtype, x.confidence, rank,
                               x.forward) for x in mine]
                    lo, hi = (me, c) if me_lo else (c, me)
                    cur.execute(   # stored relative to lo -> hi
                        "INSERT INTO memory_link_pairs (pair_hash, bank_id, src, dst, hits) "
                        "VALUES (%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING",
                        (h, bank_id, lo["id"], hi["id"], json.dumps(
                            [[x.link_type, x.subtype, x.confidence, x.forward == me_lo]
                             for x in mine])))
        stats["candidates_judged"] = stats.get("candidates_judged", 0) + len(todo)
        return edges

    def _semantic(self, conn, bank_id, doc, anchors, stats, calls, mode=None):
        with conn.cursor() as cur:   # the HNSW indexes are on embedding::vector(dim)
            cur.execute("SELECT vector_dims(embedding) FROM memory_items WHERE id = %s",
                        (anchors[0]["id"],))
            dim = int(cur.fetchone()[0])
        kind = "chunk" if anchors[0]["kind"] == "chunk" else "question"
        best: dict = {}   # candidate document -> best cosine over my anchors
        per_anchor: list = []   # (anchor, its neighbour items, best first)
        for a in anchors:
            def fetch(n, a=a):
                from prospecta.channels.semantic import _scan_everything

                def query(exhaustive):
                    with conn.transaction(), conn.cursor() as cur:
                        if exhaustive:
                            _scan_everything(cur)   # a short page must mean the source is exhausted
                        else:   # plain HNSW top-n: index cost, independent of the bank size
                            cur.execute(f"SET LOCAL hnsw.ef_search = {max(2 * n, 40)}")
                        cur.execute(_NEIGHBOUR_SQL.format(kind=kind, dim=dim), {
                            "bank": bank_id, "doc": doc, "n": n, "item": a["id"]})
                        return sorted((dict(zip([c.name for c in cur.description], r))
                                       for r in cur.fetchall()), key=lambda c: -float(c["cos"]))
                if n > HNSW_EF_MAX // 2:
                    return query(True)
                rows = query(False)
                # a short page may be the index reach (ef_search) rather than the end of the
                # source: confirm with the exhaustive scan before treating it as the end
                return rows if len(rows) >= n else query(True)
            cands, examined = neighbours_until_drop(
                fetch, lambda c: float(c["cos"]), self.neighbour_min_rel,
                floor=self.neighbour_min_cos)
            stats["candidates_examined"] = stats.get("candidates_examined", 0) + examined
            per_anchor.append((a, cands))
            for c in cands:
                d = c["document_id"]
                best[d] = max(best.get(d, 0.0), float(c["cos"]))
        if not best:
            return
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], str(kv[0])))
        groups = None   # [(me item, candidate items, edges)]
        origin = "jev"
        if self.judge is not None:
            try:
                stats["completeness"] = self._completeness(conn, bank_id, mode)
                if stats["completeness"] == "all-pairs":   # chunk level: every anchor vs its neighbours
                    groups = []
                    for a, cands in per_anchor:
                        me = dict(a, document_id=doc)
                        groups.append((me, cands, self._judged_edges(
                            conn, bank_id, me, cands, kind, dim, stats, calls, all_pairs=True,
                            floor=self.anchor_floor(cands))))
                else:   # 'connected': one representative pair per note, the top judge_top_k notes
                    reps = self._doc_reps(conn, [d for d, _ in ranked], kind)
                    me = dict(anchors[0], document_id=doc)
                    cands = [dict(reps[d], cos=cos) for d, cos in ranked if d in reps]
                    groups = [(me, cands, self._judged_edges(
                        conn, bank_id, me, cands[:self.judge_top_k], kind, dim, stats, calls))]
            except Exception as exc:   # surfaced: state error row, retried by link_pending
                stats["errors"].append(f"jev: {type(exc).__name__}: {exc}")
                groups = None
        if groups is None:   # no judge, or Jev could not answer: pgvector neighbours
            origin = "pgvector"
            reps = self._doc_reps(conn, [d for d, _ in ranked], kind)
            me = dict(anchors[0], document_id=doc)
            cands = [dict(reps[d], cos=cos) for d, cos in ranked if d in reps]
            groups = [(me, cands, [(me["id"], c["id"], "SEMANTIC", "RELATED_TO", c["cos"], i)
                                   for i, c in enumerate(cands) if c["cos"] >= self.vector_floor])]
        with conn.cursor() as cur:
            rows = []
            for _me, cands, edges in groups:
                for e in edges:   # SEMANTIC is symmetric: one judgment writes both directions
                    src, dst, lt, st, conf, rank = e[:6]
                    fwd = e[6] if len(e) > 6 else True
                    cos = cands[rank]["cos"]
                    if lt == "SEMANTIC":
                        rows += [(src, dst, lt, st, conf, cos), (dst, src, lt, st, conf, cos)]
                    else:
                        rows.append((src, dst, lt, st, conf, cos) if fwd
                                    else (dst, src, lt, st, conf, cos))
            for src, dst, lt, st, conf, cos in sorted(rows, key=lambda r: r[:4]):
                cur.execute(
                    "INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, "
                    "confidence, origin, evidence) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb) "
                    "ON CONFLICT DO NOTHING",
                    (bank_id, src, dst, lt, st, conf, origin, json.dumps({"cosine": cos})))
                stats["semantic"] += cur.rowcount


def normalise_text(text: str) -> str:
    """Text modulo YAML frontmatter, whitespace and case."""
    t = re.sub(r"\A\s*---\n.*?\n---\s*(\n|\Z)", "", text or "", flags=re.S)
    return re.sub(r"\s+", " ", t.strip().lower())


def neighbours_until_drop(fetch, score, rel: float,
                          page: int = NEIGHBOUR_PAGE, floor: float = 0.0) -> tuple[list, int]:
    """Candidates from `fetch(n)` (best first, an HNSW top-n): the leading rows
    with score >= rel x the best, ending at the first marginal drop larger than
    (1 - rel) x the best and >= the absolute `floor`. Grows n (doubling) only until that stop is decided, so
    the rows touched per anchor do not grow with the bank. Returns
    (qualifying rows, rows examined)."""
    n = page
    while True:
        rows = fetch(n)
        if not rows:
            return [], 0
        best = score(rows[0])
        if best < floor:
            return [], len(rows)
        if rel <= 0 or best <= 0:
            if len(rows) < n:
                return [r for r in rows if score(r) >= floor], len(rows)
        else:
            cut, gap = max(rel * best, floor), (1 - rel) * best
            out = [rows[0]]
            for prev, r in zip(rows, rows[1:]):
                if score(r) < cut or score(prev) - score(r) > gap:
                    return out, len(rows)
                out.append(r)
            if len(rows) < n:
                return out, len(rows)
        n *= 2


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


def parse_aliases(raw: str) -> list[tuple[str, list[str]]]:
    """(person name, aliases) for every person entity that lists aliases."""
    obj = parse_json_object(raw)
    out: list[tuple[str, list[str]]] = []
    for e in obj.get("entities") or []:
        if not isinstance(e, dict) or not isinstance(e.get("name"), str):
            continue
        if str(e.get("type", "")).strip().lower() != "person":
            continue
        als = e.get("aliases")
        if isinstance(als, str):
            als = [als]
        als = [a.strip() for a in als or [] if isinstance(a, str) and a.strip()]
        if als:
            out.append((e["name"].strip(), als))
    return out


# ---------------------------------------------------------------- the worker

def pending_documents(conn, bank_id: str, limit: int) -> list[str]:
    """Documents with items and no finished link state (no row, or a row with
    status 'error': a failed step is retried), oldest first."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT d.id::text FROM documents d "
            "WHERE d.bank_id = %s "
            "  AND NOT EXISTS (SELECT 1 FROM memory_link_state s "
            "                  WHERE s.document_id = d.id AND s.status <> 'error') "
            "  AND EXISTS (SELECT 1 FROM memory_items m WHERE m.document_id = d.id) "
            "ORDER BY EXISTS (SELECT 1 FROM memory_link_state s WHERE s.document_id = d.id), "
            "d.created_at, d.id LIMIT %s", (bank_id, limit))
        return [r[0] for r in cur.fetchall()]
