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
    JEV_BATCH, JEV_INPUT_BYTES, JEV_MODEL, JEV_TIMEOUT_S, fit_passage,
    JevTransport, call_llm, parse_json_object, totals,
)

logger = logging.getLogger(__name__)

NEIGHBOUR_MIN_REL = 0.9    # candidates per anchor: neighbours with cosine >= 0.9 x the nearest, ending at the first
                           # marginal drop larger than (1 - 0.9) x the nearest (no count). Embedding cosines are
                           # compressed (0.5..0.9), so 0.6 admitted nearly every note: the import was quadratic.
JUDGE_RETRIES = 3          # attempts per judge call before the failure is surfaced (state error, pgvector fallback)
JUDGE_BACKOFF_S = 0.5
HNSW_EF_MAX = 1000         # pgvector's hnsw.ef_search ceiling: beyond it the fetch is exhaustive
NEIGHBOUR_PAGE = 16        # rows per HNSW fetch (throughput; doubles until the stop rule is decided)
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
                                 "title": relation, "text": fit_passage(cand, relation)},
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
    neighbour_min_rel: float = NEIGHBOUR_MIN_REL
    temporal_min_rel: float = TEMPORAL_MIN_REL
    temporal_days: int = TEMPORAL_DAYS
    entity_hub: int = ENTITY_HUB
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
            except (psycopg.errors.DeadlockDetected, psycopg.errors.SerializationFailure) as exc:
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

    def _semantic(self, conn, bank_id, doc, anchors, stats, calls):
        with conn.cursor() as cur:   # the HNSW indexes are on embedding::vector(dim)
            cur.execute("SELECT vector_dims(embedding) FROM memory_items WHERE id = %s",
                        (anchors[0]["id"],))
            dim = int(cur.fetchone()[0])
        for a in anchors:
            kind = "chunk" if a["kind"] == "chunk" else "question"

            def fetch(n, kind=kind, a=a, dim=dim):
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
                fetch, lambda c: float(c["cos"]), self.neighbour_min_rel)
            stats["candidates_examined"] = stats.get("candidates_examined", 0) + examined
            stats["candidates_judged"] = stats.get("candidates_judged", 0) + (
                len(cands) if self.judge is not None else 0)
            if not cands:
                continue
            hits: list[RelationHit] | None = None
            origin = "jev"
            if self.judge is not None:
                try:
                    hits = self._judge_with_retry(
                        a["original_chunk"] or a["content"],
                        [c["original_chunk"] or c["content"] for c in cands], calls)
                except Exception as exc:   # surfaced: state error row, retried by link_pending
                    stats["errors"].append(f"jev: {type(exc).__name__}: {exc}")
            if hits is None:   # no judge, or Jev could not answer: pgvector neighbours
                origin = "pgvector"
                hits = [RelationHit(i, "SEMANTIC", "RELATED_TO", float(c["cos"]))
                        for i, c in enumerate(cands)
                        if float(c["cos"]) >= self.vector_floor]
            with conn.cursor() as cur:
                def edge(h):
                    c = cands[h.candidate]
                    return ((a["id"], c["id"]) if h.forward else (c["id"], a["id"])) + (
                        h.link_type, h.subtype)
                for h in sorted(hits, key=edge):
                    c = cands[h.candidate]
                    src, dst = (a["id"], c["id"]) if h.forward else (c["id"], a["id"])
                    cur.execute(
                        "INSERT INTO memory_links (bank_id, src, dst, link_type, subtype, "
                        "confidence, origin, evidence) VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb) "
                        "ON CONFLICT DO NOTHING",
                        (bank_id, src, dst, h.link_type, h.subtype, h.confidence, origin,
                         json.dumps({"cosine": float(c["cos"])})))
                    stats["semantic"] += cur.rowcount


def neighbours_until_drop(fetch, score, rel: float,
                          page: int = NEIGHBOUR_PAGE) -> tuple[list, int]:
    """Candidates from `fetch(n)` (best first, an HNSW top-n): the leading rows
    with score >= rel x the best, ending at the first marginal drop larger than
    (1 - rel) x the best. Grows n (doubling) only until that stop is decided, so
    the rows touched per anchor do not grow with the bank. Returns
    (qualifying rows, rows examined)."""
    n = page
    while True:
        rows = fetch(n)
        if not rows:
            return [], 0
        best = score(rows[0])
        if rel <= 0 or best <= 0:
            if len(rows) < n:
                return rows, len(rows)
        else:
            cut, gap = rel * best, (1 - rel) * best
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
