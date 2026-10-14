"""Index records IN PLACE from views a host application provides.

Prospecta knows nothing about the host: it only reads a view contract (see
docs/source-contract.md) from a configured schema. It never copies record text
into a store of record. What it keeps are derived items (chunks, anticipated
questions, embeddings, entities, links) plus one reference row per record
(source_refs: kind, record id, revision, filter columns; no text).

All operations are idempotent and keyed by (kind, record_id, revision).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field

from prospecta._chunker import chunk_paragraphs
from prospecta._retain import CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS
from prospecta.db.queries import _vec_literal, delete_memory_items_for_document

logger = logging.getLogger(__name__)

# The fixed columns every source view must provide, with the Postgres type
# family each is read as (the view may use any castable type, e.g. uuid ids).
SOURCE_COLUMNS = (
    "kind", "record_id", "revision", "title", "text",
    "person", "project", "created_date", "audience_key", "deleted_at",
)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
DEFAULT_VIEWS = ("records",)


class SourceContractError(RuntimeError):
    """A configured source view is missing or lacks contract columns."""


def _ident(name: str, what: str) -> str:
    if not _IDENT.match(name or ""):
        raise SourceContractError(f"invalid {what} {name!r}")
    return name


@dataclass(frozen=True)
class SourceConfig:
    schema: str
    views: tuple[str, ...] = DEFAULT_VIEWS

    def __post_init__(self):
        _ident(self.schema, "source schema")
        for v in self.views:
            _ident(v, "source view")

    @classmethod
    def from_env(cls) -> "SourceConfig | None":
        """PROSPECTA_SOURCE_SCHEMA (+ PROSPECTA_SOURCE_VIEWS, comma list)."""
        schema = (os.environ.get("PROSPECTA_SOURCE_SCHEMA") or "").strip()
        if not schema:
            return None
        views = tuple(v.strip() for v in
                      (os.environ.get("PROSPECTA_SOURCE_VIEWS") or "").split(",") if v.strip())
        return cls(schema, views or DEFAULT_VIEWS)

    def union_sql(self, where: str = "") -> str:
        cols = ("kind::text AS kind, record_id::text AS record_id, revision::text AS revision, "
                "title::text AS title, text::text AS text, person::text AS person, "
                "project::text AS project, created_date::date AS created_date, "
                "audience_key::text AS audience_key, deleted_at::timestamptz AS deleted_at")
        parts = [f'SELECT {cols} FROM "{self.schema}"."{v}" {where}' for v in self.views]
        return " UNION ALL ".join(parts)


@dataclass
class IndexResult:
    kind: str
    record_id: str
    status: str            # indexed | unchanged | removed | absent
    revision: str | None = None
    items: int = 0

    def as_dict(self) -> dict:
        return {"kind": self.kind, "record_id": self.record_id, "status": self.status,
                "revision": self.revision, "items": self.items}


@dataclass
class Coverage:
    """host views vs derived items, for one kind (and optionally one project)."""
    kind: str
    project: str | None
    live: int = 0                         # records the views show as not deleted
    indexed: int = 0                      # of those, current in Prospecta
    missing: list[dict] = field(default_factory=list)   # {record_id, revision}
    stale: list[dict] = field(default_factory=list)     # {record_id, revision, indexed_revision}
    orphaned: list[dict] = field(default_factory=list)  # derived, but gone/deleted in the view

    def as_dict(self) -> dict:
        return {"kind": self.kind, "project": self.project, "live": self.live,
                "indexed": self.indexed, "missing": self.missing, "stale": self.stale,
                "orphaned": self.orphaned}


class SourceIndexer:
    """Index/remove/coverage over the host views, writing into a Memory's bank."""

    def __init__(self, memory, config: SourceConfig):
        self._m = memory
        self.config = config

    # -- contract ---------------------------------------------------------
    def check_contract(self) -> None:
        """Raise SourceContractError unless every view exists with all columns
        and is readable by this role."""
        with self._m._pool.connection() as conn:
            for v in self.config.views:
                rows = conn.execute(
                    "SELECT a.attname FROM pg_attribute a "
                    "WHERE a.attrelid = to_regclass(%s) AND a.attnum > 0 AND NOT a.attisdropped",
                    (f'"{self.config.schema}"."{v}"',)).fetchall()
                have = {r[0] for r in rows}
                if not have:
                    raise SourceContractError(
                        f"source view {self.config.schema}.{v} does not exist")
                missing = [c for c in SOURCE_COLUMNS if c not in have]
                if missing:
                    raise SourceContractError(
                        f"source view {self.config.schema}.{v} lacks columns: {', '.join(missing)}")
                try:
                    conn.execute(f'SELECT 1 FROM "{self.config.schema}"."{v}" LIMIT 0')
                except Exception as e:
                    conn.rollback()
                    raise SourceContractError(
                        f"source view {self.config.schema}.{v} is not readable: {e}") from e
            conn.rollback()

    # -- reading the view -------------------------------------------------
    def _fetch(self, conn, kind: str, record_id: str) -> dict | None:
        sql = (f"SELECT * FROM ({self.config.union_sql()}) s "
               "WHERE kind = %s AND record_id = %s LIMIT 1")
        cur = conn.execute(sql, (kind, record_id))
        row = cur.fetchone()
        return dict(zip(SOURCE_COLUMNS, row)) if row else None

    # -- index one --------------------------------------------------------
    def index_record(self, kind: str, record_id: str) -> IndexResult:
        return self._sync(kind, record_id, lambda conn: self._fetch(conn, kind, record_id))

    def _sync(self, kind: str, record_id: str, load) -> IndexResult:
        """Bring one reference in line with the view; `load(conn)` returns the
        record to index, or None when there is none."""
        m = self._m
        bank = m._default_bank_id
        with m._pool.connection() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                         (f"source:{bank}:{kind}:{record_id}",))
            try:
                rec = load(conn)
            except Exception:
                conn.rollback()
                raise
            ref = conn.execute(
                "SELECT revision, document_id FROM source_refs "
                "WHERE bank_id = %s AND kind = %s AND record_id = %s",
                (bank, kind, record_id)).fetchone()
            if rec is None or rec["deleted_at"] is not None:
                if ref is None:
                    conn.commit()
                    return IndexResult(kind, record_id, "absent")
                self._retire(conn, ref[1])
                conn.commit()
                return IndexResult(kind, record_id, "removed", ref[0])
            revision = rec["revision"]
            if ref is not None and ref[0] == revision:
                conn.commit()
                return IndexResult(kind, record_id, "unchanged", revision)
            if m._embed is None:
                raise RuntimeError("indexing requires an `embed` callable")
            if ref is not None:
                self._retire(conn, ref[1])
            items = self._build_items(rec)
            doc_id = self._write(conn, bank, rec, items)
            conn.commit()
        m._enqueue_link(doc_id)
        return IndexResult(kind, record_id, "indexed", revision, len(items))

    # -- document scope / reconcile ----------------------------------------
    def _fetch_document(self, conn, kind: str, project: str, title: str) -> dict | None:
        """Every live row of `kind` in `project` with `title`, as one record.
        Its text is the rows' texts in (created_date, record_id) order; its
        revision is a hash of the rows' (record_id, revision) pairs, so any
        piece added, removed or changed moves it. None when no live row remains."""
        sql = (f"SELECT * FROM ({self.config.union_sql()}) s WHERE kind = %s "
               "AND project IS NOT DISTINCT FROM %s AND title IS NOT DISTINCT FROM %s "
               "AND deleted_at IS NULL")
        rows = [dict(zip(SOURCE_COLUMNS, r)) for r in conn.execute(sql, (kind, project, title))]
        if not rows:
            return None
        rows.sort(key=lambda r: (r["created_date"] is None, r["created_date"], r["record_id"]))
        if len({r["audience_key"] or "" for r in rows}) > 1:
            raise ValueError("pieces of one document have different audience keys")
        people = {r["person"] for r in rows}
        dates = [r["created_date"] for r in rows if r["created_date"]]
        rev = hashlib.sha256(json.dumps([[r["record_id"], r["revision"]] for r in rows]).encode()).hexdigest()
        return {"kind": kind, "record_id": document_record_id(project, title), "revision": rev,
                "title": title, "text": "\n\n".join((r["text"] or "").strip() for r in rows),
                "person": people.pop() if len(people) == 1 else None, "project": project,
                "created_date": min(dates) if dates else None,
                "audience_key": rows[0]["audience_key"], "deleted_at": None}

    def reconcile(self, refs: list[dict]) -> dict:
        """Re-read the views for each reference ({kind, record_id} or
        {kind, project, title}) and index, leave or retire it. Idempotent."""
        counts = {"indexed": 0, "unchanged": 0, "removed": 0, "absent": 0}
        unreadable: list[dict] = []
        for ref in refs:
            try:
                kind = ref.get("kind") if isinstance(ref, dict) else None
                if not isinstance(kind, str) or not kind:
                    raise ValueError("kind is required (string)")
                if isinstance(ref.get("record_id"), str) and ref["record_id"]:
                    res = self.index_record(kind, ref["record_id"])
                elif isinstance(ref.get("project"), str) and isinstance(ref.get("title"), str):
                    project, title = ref["project"], ref["title"]
                    res = self._sync(kind, document_record_id(project, title),
                                     lambda conn: self._fetch_document(conn, kind, project, title))
                else:
                    raise ValueError("reference needs record_id, or project and title")
                counts[res.status] += 1
            except Exception as e:
                logger.warning("reconcile: cannot read %r: %s", ref, e)
                unreadable.append({"reference": ref, "error": str(e)})
        return {**counts, "unreadable": unreadable}

    def _build_items(self, rec: dict) -> list[dict]:
        m = self._m
        title = (rec["title"] or "").strip()
        text = rec["text"] or ""
        body = f"{title}\n\n{text}".strip() if title else text.strip()
        meta = ref_metadata(rec)
        items: list[dict] = []
        questions: list[str] = []
        if body and m._llm is not None:
            from prospecta._index_text import generate_index_text
            questions, _p, _r = generate_index_text(
                body, m._llm, context={"source": f"{rec['kind']}:{rec['record_id']}",
                                       **({"person": rec["person"]} if rec["person"] else {})})
        chunks = chunk_paragraphs(body, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS) if body else []
        texts = list(questions) + [c.content for c in chunks]
        vecs = m._embed(texts) if texts else []
        if len(vecs) != len(texts):
            raise RuntimeError(f"embed() returned {len(vecs)} vectors for {len(texts)} inputs")
        for q, vec in zip(questions, vecs):
            items.append({"content": q, "original_chunk": chunks[0].content if chunks else body[:CHUNK_MAX_CHARS],
                          "embedding": vec, "metadata": meta, "llm_generated": True, "kind": "question"})
        for c, vec in zip(chunks, vecs[len(questions):]):
            items.append({"content": c.content, "original_chunk": c.content, "embedding": vec,
                          "metadata": meta, "llm_generated": False, "kind": "chunk",
                          "ordinal": c.ordinal, "char_start": c.char_start, "char_end": c.char_end})
        return items

    def _write(self, conn, bank: str, rec: dict, items: list[dict]) -> str:
        meta = ref_metadata(rec)
        chash = hashlib.sha256(
            f"{rec['kind']}\x00{rec['record_id']}\x00{rec['revision']}".encode()).hexdigest()
        doc_id = str(conn.execute(
            "INSERT INTO documents (bank_id, source, original_text, content_hash, "
            "document_metadata, created_on, person, source_kind) "
            "VALUES (%s, %s, '', %s, %s::jsonb, %s, %s, %s) RETURNING id",
            (bank, f"{rec['kind']}:{rec['record_id']}", chash, json.dumps(meta),
             rec["created_date"], rec["person"], rec["kind"])).fetchone()[0])
        for it in items:
            conn.execute(
                "INSERT INTO memory_items (bank_id, document_id, content, original_chunk, "
                "embedding, metadata, llm_generated, kind, ordinal, char_start, char_end) "
                "VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb, %s, %s, %s, %s, %s)",
                (bank, doc_id, it["content"], it["original_chunk"], _vec_literal(it["embedding"]),
                 json.dumps(it["metadata"]), it["llm_generated"], it["kind"],
                 it.get("ordinal"), it.get("char_start"), it.get("char_end")))
        conn.execute(
            "INSERT INTO source_refs (bank_id, kind, record_id, revision, document_id, "
            "person, project, created_date, audience_key) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (bank, rec["kind"], rec["record_id"], rec["revision"], doc_id,
             rec["person"], rec["project"], rec["created_date"], rec["audience_key"]))
        return doc_id

    @staticmethod
    def _retire(conn, document_id) -> None:
        """Drop every derived item of a record (the ref row cascades)."""
        delete_memory_items_for_document(conn, document_id=str(document_id))
        conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))

    # -- remove / reindex -------------------------------------------------
    def remove_record(self, kind: str, record_id: str) -> IndexResult:
        """Retire a record's derived items now, whatever the view says."""
        bank = self._m._default_bank_id
        with self._m._pool.connection() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                         (f"source:{bank}:{kind}:{record_id}",))
            ref = conn.execute(
                "SELECT revision, document_id FROM source_refs "
                "WHERE bank_id = %s AND kind = %s AND record_id = %s",
                (bank, kind, record_id)).fetchone()
            if ref is not None:
                self._retire(conn, ref[1])
            conn.commit()
        return IndexResult(kind, record_id, "removed" if ref else "absent",
                           ref[0] if ref else None)

    def reindex_kind(self, kind: str) -> dict:
        """Index every record of `kind` the views show, retire the ones that are
        deleted or gone. Idempotent: current records are left untouched."""
        bank = self._m._default_bank_id
        with self._m._pool.connection() as conn:
            seen = [r[0] for r in conn.execute(
                f"SELECT record_id FROM ({self.config.union_sql()}) s WHERE kind = %s", (kind,))]
            known = [r[0] for r in conn.execute(
                "SELECT record_id FROM source_refs WHERE bank_id = %s AND kind = %s", (bank, kind))]
            conn.rollback()
        counts: dict[str, int] = {"indexed": 0, "unchanged": 0, "removed": 0, "absent": 0}
        for rid in sorted(set(seen) | set(known)):
            counts[self.index_record(kind, rid).status] += 1
        return {"kind": kind, **counts}

    # -- coverage ---------------------------------------------------------
    def coverage(self, kind: str | None = None, project: str | None = None) -> list[Coverage]:
        """One Coverage per (kind, project): the host's live records against the
        derived items, naming exactly which references are missing or stale."""
        bank = self._m._default_bank_id
        with self._m._pool.connection() as conn:
            where, params = [], []
            if kind:
                where.append("kind = %s"); params.append(kind)
            if project:
                where.append("project = %s"); params.append(project)
            w = ("WHERE " + " AND ".join(where)) if where else ""
            host = conn.execute(
                "SELECT kind, project, record_id, revision, deleted_at FROM "
                f"({self.config.union_sql()}) s {w}", params).fetchall()
            rw, rp = ["bank_id = %s"], [bank]
            if kind:
                rw.append("kind = %s"); rp.append(kind)
            if project:
                rw.append("project = %s"); rp.append(project)
            refs = conn.execute(
                "SELECT kind, project, record_id, revision FROM source_refs WHERE "
                + " AND ".join(rw), rp).fetchall()
            conn.rollback()
        derived = {(k, rid): (rev, p) for k, p, rid, rev in refs}
        out: dict[tuple, Coverage] = {}

        def cov(k: str, p: str | None) -> Coverage:
            return out.setdefault((k, p), Coverage(k, p))
        host_keys = set()
        for k, p, rid, rev, deleted in sorted(host, key=lambda r: (r[0], r[1] or "", r[2])):
            host_keys.add((k, rid))
            c = cov(k, p)
            have = derived.get((k, rid))
            if deleted is not None:
                if have is not None:
                    c.orphaned.append({"record_id": rid, "indexed_revision": have[0]})
                continue
            c.live += 1
            if have is None:
                c.missing.append({"record_id": rid, "revision": rev})
            elif have[0] != rev:
                c.stale.append({"record_id": rid, "revision": rev, "indexed_revision": have[0]})
            else:
                c.indexed += 1
        for (k, rid), (have, p) in sorted(derived.items()):
            if (k, rid) not in host_keys:
                cov(k, p).orphaned.append({"record_id": rid, "indexed_revision": have})
        if kind and not any(k == kind for k, _ in out):
            cov(kind, project)
        return [out[key] for key in sorted(out, key=lambda x: (x[0], x[1] or ""))]


def document_record_id(project: str, title: str) -> str:
    """The record id under which a (project, title) document is kept."""
    return "doc:" + json.dumps([project, title], ensure_ascii=False)


def ref_metadata(rec: dict) -> dict:
    """Metadata stamped on every derived item (and used for lookup filters)."""
    meta = {"kind": rec["kind"], "record_id": rec["record_id"], "revision": rec["revision"]}
    for k in ("person", "project", "audience_key"):
        if rec.get(k):
            meta[k] = rec[k]
    if rec.get("created_date"):
        meta["created_date"] = rec["created_date"].isoformat()
    return meta
