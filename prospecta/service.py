"""Small HTTP service around Memory + SourceIndexer (stdlib only).

    python -m prospecta.service            # reads the environment, see below

Endpoints (JSON):
    GET  /healthz
    Every JSON body may carry "bank" (and a GET its ?bank=): the bank the call is about. Absent
    means the process's default bank (PROSPECTA_BANK), so a host that never sends it sees no
    change. An unknown bank is 404 (a /retain with "create": true creates it); a bank whose
    embedding dimension differs from the service's embedder is 409. One connection pool serves
    every bank.
    POST /retain    {"content", "source", "bank"?, "metadata"?: {"created_on", "project", "person",
                     "kind"}, "tags"?: [...], "create"?: true}
                                          -> {"document_id", "status": "created" | "replaced" |
                                             "unchanged"}. `source` is the idempotency key: the
                                             same source with new content replaces the old note
                                             ("replaced"), the same content again changes nothing
                                             ("unchanged"), the same content under another source
                                             is 409. Nothing is truncated; /retain has no body cap.
    POST /lookup    {"questions": [...], "limit": 10, "bank"?, "kind"?, "project"?, "person"?,
                     "audience_key"?,
                     "allowed_audience_keys"?, "allowed_projects"?, "explain"?: true,
                     "date_from"?, "date_to"?, "excluded_sources"?, "reranker_model"?,
                     "decision_model"?}
                                          -> ranked results per question, each with its citation
                                             fields (created_on, match_kind, exact_snippet,
                                             relevance_score); a standalone note (from /retain)
                                             comes back as kind "chunk" with source, document_id
                                             and content; with explain, also
                                             channels_run / channels_failed per question and
                                             channels (rank, score, weight) per hit
                    {"task": {"title", "ask", "done"?, "project"?}, "limit": 10, ...}
                                          -> the task brief's recall: {"task": {"queries": [...],
                                             "context": "<cited block>", "results": [...]}}
    POST /index     {"kind", "record_id"}  or  {"kind", "reindex": true, "concurrency"?}
    POST /remove    {"kind", "record_id"}
    POST /reconcile {"references": [{"kind","record_id"} | {"kind","project","title"}], "concurrency"?}
    GET  /coverage?kind=&project=&bank=
    POST /lung-memory/1                   the adapter for lung's `lung-memory/1` wire (the Lean spec
                                          lung/spec/LungSpec/MemoryWire.lean, PART B): the request is
                                          {"op": "recall", "task", "banks": [...]},
                                          {"op": "search", "query", "banks": [...]} or
                                          {"op": "retain", "bank", "text", "source"}; the replies
                                          are exactly what lung's decoder reads, HTTP 200 for every
                                          well-formed request (a failure is status "unavailable"
                                          with a reason), 400 for one that parses as none of the
                                          three. See docs/source-contract.md.

Environment: DATABASE_URL, PROSPECTA_SCHEMA (shared-database mode),
PROSPECTA_SOURCE_SCHEMA / PROSPECTA_SOURCE_VIEWS (the source contract),
PROSPECTA_INDEX_CONCURRENCY (worker threads for reconcile/reindex, default 8, max 32),
PROSPECTA_DB_POOL_MAX (the one shared connection pool, default 10),
PROSPECTA_DB_POOL_IDLE_SECONDS (close an idle connection after this, default 60),
PROSPECTA_REQUEST_TIMEOUT (seconds a request waits for a connection, default 30),
PROSPECTA_ZDR_ATTESTED (exactly "1": the operator asserts the deployment's OpenRouter key is
zero-data-retention; /lung-memory/1 then says "attested", else "unattested"; Prospecta does not
enforce it),
PROSPECTA_BANK (default "prospecta"), PROSPECTA_SERVICE_HOST (127.0.0.1),
PROSPECTA_SERVICE_PORT (8080), PROSPECTA_SERVICE_TOKEN (bearer token; required
to bind a non-loopback address), plus the usual embedder/LLM settings.
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import hmac
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from prospecta._index import _delete_prior_docs_at_source_diff_hash
from prospecta._types import DocumentSourceConflictError
from prospecta.memory import BankConfigConflict
from prospecta.db.pool import PoolTimeout
from prospecta.db.queries import SELECT_BANK, validate_bank_id
from prospecta.source import MAX_INDEX_CONCURRENCY, SourceConfig, SourceContractError, SourceIndexer

logger = logging.getLogger("prospecta.service")
MAX_BODY = 1 << 20          # every route but /retain: a note is never refused for its size
RETAIN_METADATA_KEYS = ("created_on", "project", "person", "kind")
MAX_LIMIT = 100
LUNG_PATH = "/lung-memory/1"
LUNG_SEARCH_LIMIT = 10      # the same default as /lookup's `limit`


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _explain_query(trace: dict, ranked: list[dict]) -> dict:
    """Additive explain fields for one question: channels_run, channels_failed
    (failed or skipped, each with a reason) and, on every ranked hit, `channels`
    (name, rank, raw score, weight applied). Mutates `ranked` to add `channels`."""
    weights = {c["name"]: c["weight"] for c in trace["channels"]}
    by_doc: dict[str, dict[str, tuple[int, float]]] = {}
    for c in trace["candidates"]:
        cur = by_doc.setdefault(c["document_id"], {})
        if c["channel"] not in cur or c["rank"] < cur[c["channel"]][0]:
            cur[c["channel"]] = (c["rank"], c["score"])
    for r in ranked:
        found = by_doc.get(r["_doc"], {})
        r["channels"] = [{"channel": n, "rank": found[n][0], "score": float(found[n][1]),
                          "weight": weights[n]}
                         for n in (c["name"] for c in trace["channels"]) if n in found]
    failed = [{"channel": c["name"], "reason": f"failed: {c['error']}"}
              for c in trace["channels"] if c["error"]]
    failed += [{"channel": s["name"], "reason": f"skipped: {s['reason']}"}
               for s in trace.get("skipped", [])]
    return {"channels_run": [c["name"] for c in trace["channels"] if not c["error"]],
            "channels_failed": failed}


class _LungUnavailable(Exception):
    """A lung-memory/1 call that cannot be answered; `reason` is one word of the wire's table."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def zdr_attested_from_env() -> bool:
    """Whether the operator declares the deployment zero-data-retention: PROSPECTA_ZDR_ATTESTED=1, only."""
    return os.environ.get("PROSPECTA_ZDR_ATTESTED") == "1"


class Service:
    """The service logic, separate from HTTP so it can be tested directly."""

    def __init__(self, memory, indexer: SourceIndexer | None, *, zdr_attested: bool = False):
        self.memory = memory        # the default bank's Memory; its pool serves every bank
        self.indexer = indexer
        self.zdr_attested = zdr_attested    # the operator's claim, said back on /lung-memory/1
        self._dim: int | None = None

    def _embedding_dim(self) -> int:
        if self._dim is None:
            self._dim = _embedding_dim(self.memory)
        return self._dim

    def _bank(self, body: dict, *, create: bool = False, embeds: bool = True):
        """The Memory for this call's bank: the default's when `bank` is absent (or names it),
        else a view of it on that bank, sharing the one connection pool and the embedder.
        An unknown bank is 404 unless `create`; one built for another embedding dimension is 409."""
        name = body.get("bank")
        if name is None or name == self.memory.default_bank_id:
            return self.memory
        if not isinstance(name, str) or not name:
            raise ApiError(400, "bank must be a non-empty string")
        try:
            validate_bank_id(name)
        except ValueError as e:
            raise ApiError(400, str(e)) from None
        mem = copy.copy(self.memory)
        mem._default_bank_id = name
        mem._shadow_bank_id = None      # shadow reads belong to the default bank's migration
        mem._shadow_embed = None
        with mem._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(SELECT_BANK, {"bank_id": name})
                row = cur.fetchone()
            conn.rollback()
        if row is None:
            if not create:
                raise ApiError(404, f"no such bank: {name}")
            if mem._embed is None:
                raise ApiError(503, "no embedder available to create a bank")
            try:
                mem.create_bank(name, embedding_dim=self._embedding_dim())
            except BankConfigConflict as e:
                raise ApiError(409, str(e)) from None
        elif embeds and mem._embed is not None and row[1] != self._embedding_dim():
            raise ApiError(409, f"bank {name} has embedding_dim {row[1]}; the service's embedder "
                                f"makes {self._embedding_dim()}")
        return mem

    def _indexer_for(self, mem) -> SourceIndexer:
        ix = self._need_indexer()
        if mem is self.memory:
            return ix
        return SourceIndexer(mem, ix.config, ix.concurrency)

    def _need_indexer(self) -> SourceIndexer:
        if self.indexer is None:
            raise ApiError(409, "no source configured (set PROSPECTA_SOURCE_SCHEMA)")
        return self.indexer

    @staticmethod
    def _recall_options(body: dict) -> dict:
        """The per-call Stage 1 filters and model overrides a lookup may carry."""
        opts: dict = {}
        for key in ("date_from", "date_to"):
            v = body.get(key)
            if v is None:
                continue
            try:
                opts[key] = datetime.date.fromisoformat(v)
            except (TypeError, ValueError):
                raise ApiError(400, f"{key} must be a YYYY-MM-DD date") from None
        ex = body.get("excluded_sources")
        if ex is not None:
            if not isinstance(ex, list) or not all(isinstance(x, str) for x in ex):
                raise ApiError(400, "excluded_sources must be a list of strings")
            opts["excluded_sources"] = ex
        for key in ("reranker_model", "decision_model"):
            v = body.get(key)
            if v is not None:
                if not isinstance(v, str) or not v:
                    raise ApiError(400, f"{key} must be a non-empty string")
                opts[key] = v
        return opts

    def _lookup_task(self, body: dict, memory) -> dict:
        task = body["task"]
        if not isinstance(task, dict) or not all(
                isinstance(task.get(k), str) and task[k].strip() for k in ("title", "ask")):
            raise ApiError(400, "task needs a title and an ask (non-empty strings)")
        for k in ("done", "project"):
            if task.get(k) is not None and not isinstance(task[k], str):
                raise ApiError(400, f"task.{k} must be a string")
        if "limit" in body:
            raise ApiError(400, "limit does not apply to a task: its block is sized by relevance "
                                "(max_tokens is the safety valve)")
        max_tokens = body.get("max_tokens", 0)
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 0:
            raise ApiError(400, "max_tokens must be an integer >= 0 (0: no valve)")
        try:
            got = memory.recall_for_task(
                title=task["title"], ask=task["ask"], done=task.get("done"),
                project=task.get("project"), max_tokens=max_tokens, **self._recall_options(body))
        except ValueError as e:
            raise ApiError(400, str(e)) from None
        return {"task": {"queries": got.queries, "context": got.block,
                         "searches": [{"query": s.query, "terms": list(s.terms),
                                       "date_from": s.date_from, "date_to": s.date_to}
                                      for s in got.searches],
                         "results": [_cited_hit(h) for h in got.items],
                         "fallbacks": got.fallbacks, "error": got.error}}

    def lookup(self, body: dict) -> dict:
        memory = self._bank(body)
        if body.get("task") is not None:
            return self._lookup_task(body, memory)
        qs = body.get("questions")
        if not isinstance(qs, list) or not qs or not all(isinstance(q, str) and q.strip() for q in qs):
            raise ApiError(400, "questions must be a non-empty list of strings")
        if len(qs) > 50:
            raise ApiError(400, "at most 50 questions per call")
        limit = body.get("limit", 10)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIMIT:
            raise ApiError(400, f"limit must be an integer 1..{MAX_LIMIT}")
        flt = {k: body[k] for k in ("kind", "project", "person", "audience_key")
               if body.get(k) is not None}
        if not all(isinstance(v, str) for v in flt.values()):
            raise ApiError(400, "filters must be strings")
        allow = {}
        for field, key in (("allowed_audience_keys", "audience_key"), ("allowed_projects", "project")):
            v = body.get(field)
            if v is None:
                continue
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                raise ApiError(400, f"{field} must be a list of strings")
            allow[key] = v          # [] allows nothing; "" allows records with no value
        if allow:
            flt["_allow"] = allow
        explain = body.get("explain", False)
        if not isinstance(explain, bool):
            raise ApiError(400, "explain must be true or false")
        options = self._recall_options(body)
        out = []
        for q in qs:
            traces: list[dict] = []
            try:
                hits = memory.recall([q], limit=limit * 3, metadata_filter=flt or None,
                                          **options, **({"_trace": traces} if explain else {}))
            except ValueError as e:     # e.g. a depth that is neither standard nor deep
                raise ApiError(400, str(e)) from None
            best: dict[tuple, dict] = {}
            for h in hits:
                md = h.metadata or {}
                if "record_id" not in md or "kind" not in md:
                    # a standalone note (from /retain): no host holds its text, so it comes back
                    # whole, as its best chunk, under its source and document id
                    key = ("chunk", h.document_id)
                    if key not in best or h.score > best[key]["score"]:
                        best[key] = {"kind": "chunk", "record_id": None, "revision": None,
                                     "source": h.source, "document_id": h.document_id,
                                     "content": h.original_chunk, "score": float(h.score),
                                     "created_on": h.created_on, "match_kind": h.match_kind,
                                     "exact_snippet": h.exact_snippet,
                                     "relevance_score": h.relevance_score}
                        if explain:
                            best[key]["_doc"] = h.document_id
                    continue
                key = (md["kind"], md["record_id"], md.get("revision"))
                if key not in best or h.score > best[key]["score"]:
                    best[key] = {"kind": md["kind"], "record_id": md["record_id"],
                                 "revision": md.get("revision"), "score": float(h.score),
                                 "created_on": h.created_on, "match_kind": h.match_kind,
                                 "exact_snippet": h.exact_snippet,
                                 "relevance_score": h.relevance_score}
                    if explain:
                        best[key]["_doc"] = h.document_id
            ranked = sorted(best.values(), key=lambda r: -r["score"])[:limit]
            entry = {"question": q}
            if explain:
                entry.update(_explain_query(traces[0], ranked) if traces
                             else {"channels_run": [], "channels_failed": []})
                for r in ranked:
                    r.pop("_doc")
            entry["results"] = ranked
            out.append(entry)
        return {"results": out}

    @staticmethod
    def _concurrency(body: dict) -> int | None:
        c = body.get("concurrency")
        if c is None:
            return None
        if not isinstance(c, int) or isinstance(c, bool) or not 1 <= c <= MAX_INDEX_CONCURRENCY:
            raise ApiError(400, f"concurrency must be an integer 1..{MAX_INDEX_CONCURRENCY}")
        return c

    def index(self, body: dict) -> dict:
        ix = self._indexer_for(self._bank(body))
        kind = _str(body, "kind")
        if body.get("reindex"):
            return ix.reindex_kind(kind, self._concurrency(body))
        return ix.index_record(kind, _str(body, "record_id")).as_dict()

    def remove(self, body: dict) -> dict:
        ix = self._indexer_for(self._bank(body, embeds=False))
        return ix.remove_record(_str(body, "kind"), _str(body, "record_id")).as_dict()

    def reconcile(self, body: dict) -> dict:
        ix = self._indexer_for(self._bank(body))
        refs = body.get("references")
        if not isinstance(refs, list) or not refs or not all(isinstance(r, dict) for r in refs):
            raise ApiError(400, "references must be a non-empty list of objects")
        return ix.reconcile(refs, self._concurrency(body))

    def coverage(self, kind: str | None, project: str | None, bank: str | None = None) -> dict:
        ix = self._indexer_for(self._bank({"bank": bank}, embeds=False))
        return {"coverage": [c.as_dict() for c in ix.coverage(kind, project)]}

    def retain(self, body: dict) -> dict:
        """Save a standalone note, keyed by `source` (see the module docstring)."""
        content, source = body.get("content"), body.get("source")
        if not isinstance(content, str) or not content.strip():
            raise ApiError(400, "content is required (non-empty string)")
        if not isinstance(source, str) or not source:
            raise ApiError(400, "source is required (non-empty string): it is the note's key")
        meta = body.get("metadata")
        if meta is None:
            meta = {}
        if not isinstance(meta, dict) or set(meta) - set(RETAIN_METADATA_KEYS):
            raise ApiError(400, f"metadata must be an object with only {', '.join(RETAIN_METADATA_KEYS)}")
        for k, v in meta.items():
            if not isinstance(v, str) or not v:
                raise ApiError(400, f"metadata.{k} must be a non-empty string")
        if "created_on" in meta:
            try:
                datetime.date.fromisoformat(meta["created_on"])
            except ValueError:
                raise ApiError(400, "metadata.created_on must be a YYYY-MM-DD date") from None
        tags = body.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise ApiError(400, "tags must be a list of strings")
        create = body.get("create", False)
        if not isinstance(create, bool):
            raise ApiError(400, "create must be true or false")
        mem = self._bank(body, create=create)
        if mem._embed is None:
            raise ApiError(503, "no embedder available; configure PROSPECTA_EMBEDDER")
        bank = mem.default_bank_id
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with mem._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT id, content_hash, tags, document_metadata FROM documents "
                            "WHERE bank_id = %s AND source = %s", (bank, source))
                prior = cur.fetchall()
            conn.rollback()
        same = [r for r in prior if r[1] == digest]
        if same:
            _, _, have_tags, have_meta = same[0]
            if list(have_tags or []) == tags and all(
                    (have_meta or {}).get(k) == v for k, v in meta.items()):
                return {"document_id": str(same[0][0]), "status": "unchanged"}
        try:
            doc_id = mem.retain(content, source=source, tags=tags, metadata=meta)
        except DocumentSourceConflictError as e:
            raise ApiError(409, f"the same content is already stored under source "
                                f"{e.existing_source!r}") from None
        dropped = 0
        if any(r[1] != digest for r in prior):
            with mem._pool.connection() as conn:     # the old text under this key goes away
                dropped = _delete_prior_docs_at_source_diff_hash(
                    conn, bank_id=bank, source=source, keep_doc_id=doc_id)
                conn.commit()
        return {"document_id": doc_id,
                "status": "replaced" if prior or dropped else "created"}


    # ------------------------------------------------------------------
    # POST /lung-memory/1: lung's wire over the calls above (docs/source-contract.md)
    # ------------------------------------------------------------------

    def lung_unavailable(self, reason: str) -> dict:
        return {"status": "unavailable", "reason": reason, "retention": self._lung_retention()}

    def _lung_retention(self) -> str:
        return "attested" if self.zdr_attested else "unattested"

    @staticmethod
    def lung_parse(body: dict) -> tuple:
        """The request as the Lean `Request.ofJson` reads it, or ApiError(400): an op of recall
        (task, banks), search (query, banks) or retain (bank, text, source; bank and source non-empty).
        Other fields are ignored."""
        def strs(v):
            return isinstance(v, list) and all(isinstance(x, str) for x in v)
        op = body.get("op")
        if op in ("recall", "search"):
            key = "task" if op == "recall" else "query"
            if isinstance(body.get(key), str) and strs(body.get("banks")):
                return op, body[key], body["banks"]
        elif op == "retain":
            bank, text, source = body.get("bank"), body.get("text"), body.get("source")
            if all(isinstance(x, str) for x in (bank, text, source)) and bank and source:
                return op, bank, text, source
        raise ApiError(400, "not a lung-memory/1 request")

    def lung_memory(self, body: dict) -> dict:
        """One lung-memory/1 call: the reply as lung's decoder reads it. Raises ApiError(400) only for
        a request that is none of the three ops; every other outcome, a failure included, is a reply."""
        req = self.lung_parse(body)
        try:
            if req[0] == "retain":
                return self._lung_retain(*req[1:])
            if req[0] == "recall":
                records = self._lung_records(req[2], lambda mem: self._lung_task_hits(mem, req[1]))
            else:
                records = self._lung_records(req[2], lambda mem: self._lung_search_hits(mem, req[1]))
        except _LungUnavailable as e:
            return self.lung_unavailable(e.reason)
        except PoolTimeout:
            return self.lung_unavailable("timeout")
        except Exception:
            logger.exception("lung-memory/1 call failed")
            return self.lung_unavailable("provider")
        return {"status": "ok", "records": records, "retention": self._lung_retention()}

    @staticmethod
    def _lung_task_hits(mem, task: str) -> list:
        """Mode 1 for a task: the writer's searches over the bank, merged (a blank task finds nothing)."""
        got = mem.expand_and_recall(task, kind="task")
        if got.error:
            logger.warning("lung-memory/1 recall failed: %s", got.error)
            raise _LungUnavailable("provider")
        if got.deadline_hit:
            raise _LungUnavailable("timeout")
        return got.items

    @staticmethod
    def _lung_search_hits(mem, query: str) -> list:
        """The questions recall of /lookup for one query: best chunk per note, best first."""
        if not query.strip():
            return []
        hits = mem.recall([query], limit=LUNG_SEARCH_LIMIT * 3)
        best: dict[str, object] = {}
        for h in hits:
            if h.document_id not in best or h.score > best[h.document_id].score:
                best[h.document_id] = h
        return sorted(best.values(), key=lambda h: -h.score)[:LUNG_SEARCH_LIMIT]

    def _lung_records(self, banks: list[str], fetch) -> list[dict]:
        """`fetch(memory)` over each named bank (once each), as records. A bank that does not exist
        has no notes. A record needs an id, a source and a date: a hit without one is dropped, never
        defaulted. Ids are distinct. Every record's bank is the name the request used."""
        records: list[dict] = []
        seen: set[str] = set()
        dropped = 0
        for name in dict.fromkeys(banks):
            try:
                mem = self._bank({"bank": name})
            except ApiError as e:
                if e.status in (400, 404):      # not a bank name, or not made yet: nothing in it
                    continue
                raise _LungUnavailable("provider") from None
            for h in fetch(mem):
                doc_id, source, day = str(h.document_id or ""), h.source, h.created_on
                if not (doc_id and source and day):
                    dropped += 1
                    continue
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                records.append({"bank": name, "id": doc_id, "text": h.original_chunk,
                                "source": source, "date": str(day)})
        if dropped:
            logger.warning("lung-memory/1: %d hit(s) without an id, source or date left out", dropped)
        return records

    def _lung_retain(self, bank: str, text: str, source: str) -> dict:
        """/retain with `create`: the note keyed by `source` in `bank`, the bank made on first use."""
        try:
            validate_bank_id(bank)
        except ValueError:
            return self.lung_unavailable("malformed")
        if not text.strip():
            return self.lung_unavailable("malformed")
        try:
            out = self.retain({"content": text, "source": source, "bank": bank, "create": True})
        except ApiError as e:
            logger.warning("lung-memory/1 retain refused: %s %s", e.status, e.message)
            return self.lung_unavailable("malformed" if e.status == 400 else "provider")
        return {"status": "retained", "id": out["document_id"], "retention": self._lung_retention()}


def _embedding_dim(memory) -> int:
    """The dimension a bank made by this service has: PROSPECTA_EMBEDDING_DIM, else what the embedder makes."""
    dim = os.environ.get("PROSPECTA_EMBEDDING_DIM")
    return int(dim) if dim else len(memory._embed(["probe"])[0])


def _cited_hit(h) -> dict:
    """One recalled note as a lookup result: where and when it is from, why it matched."""
    md = h.metadata or {}
    return {"source": h.source, "created_on": h.created_on, "kind": md.get("kind") or h.kind,
            "record_id": md.get("record_id"), "revision": md.get("revision"),
            "project": md.get("project"), "match_kind": h.match_kind,
            "exact_snippet": h.exact_snippet, "relevance_score": h.relevance_score,
            "found_by": list(h.found_by) if h.found_by else None}


def _str(body: dict, key: str) -> str:
    v = body.get(key)
    if not isinstance(v, str) or not v:
        raise ApiError(400, f"{key} is required (string)")
    return v


def make_handler(service: Service, token: str | None):
    class Handler(BaseHTTPRequestHandler):
        server_version = "prospecta"

        def log_message(self, fmt, *args):
            logger.info("%s %s", self.address_string(), fmt % args)

        def _send(self, status: int, payload: dict):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self) -> bool:
            if not token:
                return True
            got = self.headers.get("Authorization", "")
            return hmac.compare_digest(got.encode(), f"Bearer {token}".encode())

        def _handle(self, method: str):
            parts = urlsplit(self.path)
            try:
                if parts.path == "/healthz" and method == "GET":
                    return self._send(200, {"status": "ok"})
                if not self._authorized():
                    raise ApiError(401, "unauthorized")
                body: dict = {}
                if method == "POST":
                    n = int(self.headers.get("Content-Length") or 0)
                    if n > MAX_BODY and parts.path not in ("/retain", LUNG_PATH):
                        raise ApiError(413, "body too large")
                    try:
                        body = json.loads(self.rfile.read(n) or b"{}")
                    except ValueError:
                        raise ApiError(400, "body must be JSON") from None
                    if not isinstance(body, dict):
                        raise ApiError(400, "body must be a JSON object")
                route = (method, parts.path)
                if route == ("POST", "/lookup"):
                    return self._send(200, service.lookup(body))
                if route == ("POST", "/retain"):
                    return self._send(200, service.retain(body))
                if route == ("POST", LUNG_PATH):
                    return self._send(200, service.lung_memory(body))
                if route == ("POST", "/index"):
                    return self._send(200, service.index(body))
                if route == ("POST", "/remove"):
                    return self._send(200, service.remove(body))
                if route == ("POST", "/reconcile"):
                    return self._send(200, service.reconcile(body))
                if route == ("GET", "/coverage"):
                    q = parse_qs(parts.query)
                    return self._send(200, service.coverage(
                        (q.get("kind") or [None])[0], (q.get("project") or [None])[0],
                        (q.get("bank") or [None])[0]))
                raise ApiError(404, "not found")
            except ApiError as e:
                if parts.path == LUNG_PATH and e.status == 400:     # lung reads one shape, always
                    return self._send(400, service.lung_unavailable("malformed"))
                self._send(e.status, {"error": e.message})
            except PoolTimeout as e:
                logger.warning("request waited too long for a database connection: %s", e)
                self._send(503, {"error": "database busy, try again"})
            except Exception:
                logger.exception("request failed")
                self._send(500, {"error": "internal error"})

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    return Handler


class Server(ThreadingHTTPServer):
    """One thread per request; none of them keeps a database connection (they
    borrow from the shared pool). A deep accept backlog so a burst of clients
    queues in the kernel instead of being reset."""
    request_queue_size = 512
    daemon_threads = True


def build_service() -> tuple[Service, str | None]:
    """Construct Memory (migrated, bank ensured) and the source indexer from env."""
    from types import SimpleNamespace

    from prospecta.cli import _common
    from prospecta.db.migrate import run_migrations

    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is required")
    bank = os.environ.get("PROSPECTA_BANK", "prospecta")
    run_migrations(url)          # shared mode: creates/checks the schema, nothing outside it
    memory = _common.make_memory(SimpleNamespace(database_url=url, bank=bank))
    if memory._embed is None:
        raise SystemExit("no embedder available; configure PROSPECTA_EMBEDDER")
    memory.create_bank(bank, embedding_dim=_embedding_dim(memory))
    cfg = SourceConfig.from_env()
    indexer = None
    if cfg is not None:
        indexer = SourceIndexer(memory, cfg)
        try:
            indexer.check_contract()
        except SourceContractError as e:
            raise SystemExit(f"source contract check failed: {e}") from e
    return Service(memory, indexer, zdr_attested=zdr_attested_from_env()), os.environ.get("PROSPECTA_SERVICE_TOKEN") or None


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("PROSPECTA_SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("PROSPECTA_SERVICE_PORT", "8080"))
    service, token = build_service()
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        print("refusing to listen on a non-loopback address without PROSPECTA_SERVICE_TOKEN",
              file=sys.stderr)
        return 2
    srv = Server((host, port), make_handler(service, token))
    logger.info("prospecta service on %s:%s", host, port)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
        service.memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
