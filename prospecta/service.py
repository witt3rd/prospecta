"""Small HTTP service around Memory + SourceIndexer (stdlib only).

    python -m prospecta.service            # reads the environment, see below

Endpoints (JSON):
    GET  /healthz
    POST /lookup    {"questions": [...], "limit": 10, "kind"?, "project"?, "person"?,
                     "audience_key"?,
                     "allowed_audience_keys"?, "allowed_projects"?}
                                          -> ranked results per question
    POST /index     {"kind", "record_id"}  or  {"kind", "reindex": true}
    POST /remove    {"kind", "record_id"}
    POST /reconcile {"references": [{"kind","record_id"} | {"kind","project","title"}]}
    GET  /coverage?kind=&project=

Environment: DATABASE_URL, PROSPECTA_SCHEMA (shared-database mode),
PROSPECTA_SOURCE_SCHEMA / PROSPECTA_SOURCE_VIEWS (the source contract),
PROSPECTA_BANK (default "prospecta"), PROSPECTA_SERVICE_HOST (127.0.0.1),
PROSPECTA_SERVICE_PORT (8080), PROSPECTA_SERVICE_TOKEN (bearer token; required
to bind a non-loopback address), plus the usual embedder/LLM settings.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from prospecta.source import SourceConfig, SourceContractError, SourceIndexer

logger = logging.getLogger("prospecta.service")
MAX_BODY = 1 << 20
MAX_LIMIT = 100


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


class Service:
    """The service logic, separate from HTTP so it can be tested directly."""

    def __init__(self, memory, indexer: SourceIndexer | None):
        self.memory = memory
        self.indexer = indexer

    def _need_indexer(self) -> SourceIndexer:
        if self.indexer is None:
            raise ApiError(409, "no source configured (set PROSPECTA_SOURCE_SCHEMA)")
        return self.indexer

    def lookup(self, body: dict) -> dict:
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
        out = []
        for q in qs:
            hits = self.memory.recall([q], limit=limit * 3, metadata_filter=flt or None)
            best: dict[tuple, dict] = {}
            for h in hits:
                md = h.metadata or {}
                if "record_id" not in md or "kind" not in md:
                    continue          # not a source record (e.g. standalone notes)
                key = (md["kind"], md["record_id"], md.get("revision"))
                if key not in best or h.score > best[key]["score"]:
                    best[key] = {"kind": md["kind"], "record_id": md["record_id"],
                                 "revision": md.get("revision"), "score": float(h.score)}
            ranked = sorted(best.values(), key=lambda r: -r["score"])[:limit]
            out.append({"question": q, "results": ranked})
        return {"results": out}

    def index(self, body: dict) -> dict:
        ix = self._need_indexer()
        kind = _str(body, "kind")
        if body.get("reindex"):
            return ix.reindex_kind(kind)
        return ix.index_record(kind, _str(body, "record_id")).as_dict()

    def remove(self, body: dict) -> dict:
        return self._need_indexer().remove_record(_str(body, "kind"), _str(body, "record_id")).as_dict()

    def reconcile(self, body: dict) -> dict:
        ix = self._need_indexer()
        refs = body.get("references")
        if not isinstance(refs, list) or not refs or not all(isinstance(r, dict) for r in refs):
            raise ApiError(400, "references must be a non-empty list of objects")
        if len(refs) > 500:
            raise ApiError(400, "at most 500 references per call")
        return ix.reconcile(refs)

    def coverage(self, kind: str | None, project: str | None) -> dict:
        return {"coverage": [c.as_dict() for c in self._need_indexer().coverage(kind, project)]}


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
            try:
                parts = urlsplit(self.path)
                if parts.path == "/healthz" and method == "GET":
                    return self._send(200, {"status": "ok"})
                if not self._authorized():
                    raise ApiError(401, "unauthorized")
                body: dict = {}
                if method == "POST":
                    n = int(self.headers.get("Content-Length") or 0)
                    if n > MAX_BODY:
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
                if route == ("POST", "/index"):
                    return self._send(200, service.index(body))
                if route == ("POST", "/remove"):
                    return self._send(200, service.remove(body))
                if route == ("POST", "/reconcile"):
                    return self._send(200, service.reconcile(body))
                if route == ("GET", "/coverage"):
                    q = parse_qs(parts.query)
                    return self._send(200, service.coverage(
                        (q.get("kind") or [None])[0], (q.get("project") or [None])[0]))
                raise ApiError(404, "not found")
            except ApiError as e:
                self._send(e.status, {"error": e.message})
            except Exception:
                logger.exception("request failed")
                self._send(500, {"error": "internal error"})

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    return Handler


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
    dim = os.environ.get("PROSPECTA_EMBEDDING_DIM")
    memory.create_bank(bank, embedding_dim=int(dim) if dim else len(memory._embed(["probe"])[0]))
    cfg = SourceConfig.from_env()
    indexer = None
    if cfg is not None:
        indexer = SourceIndexer(memory, cfg)
        try:
            indexer.check_contract()
        except SourceContractError as e:
            raise SystemExit(f"source contract check failed: {e}") from e
    return Service(memory, indexer), os.environ.get("PROSPECTA_SERVICE_TOKEN") or None


def main() -> int:
    logging.basicConfig(level=logging.INFO)
    host = os.environ.get("PROSPECTA_SERVICE_HOST", "127.0.0.1")
    port = int(os.environ.get("PROSPECTA_SERVICE_PORT", "8080"))
    service, token = build_service()
    if host not in ("127.0.0.1", "localhost", "::1") and not token:
        print("refusing to listen on a non-loopback address without PROSPECTA_SERVICE_TOKEN",
              file=sys.stderr)
        return 2
    srv = ThreadingHTTPServer((host, port), make_handler(service, token))
    logger.info("prospecta service on %s:%s", host, port)
    try:
        srv.serve_forever()
    finally:
        srv.server_close()
        service.memory.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
