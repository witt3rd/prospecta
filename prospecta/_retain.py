"""Bilateral-spine write path: Memory.retain() implementation.

Replace-on-source-match (docs/schema.md, "Re-retain"):
  - Same content_hash + same source → REPLACE: keep documents row, delete
    existing memory_items, regenerate fresh memory_items.
  - Same content_hash + DIFFERENT source → raise DocumentSourceConflictError.
  - New content_hash → INSERT new documents row.

A note is indexed as its own chunks: kind='chunk' items whose content and
original_chunk are the chunk text, with no model call.

P-principles enforced here:
  P3 — zero provider imports; uses the memory.embed callable.
  P5 — original content preserved verbatim in documents.original_text.
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from prospecta._chunker import chunk_paragraphs
from prospecta._filters import filter_fields, today
from prospecta._types import DocumentSourceConflictError
from prospecta.db.queries import (
    delete_memory_items_for_document,
    upsert_memory_items,
)

if TYPE_CHECKING:
    from prospecta.memory import Memory


logger = logging.getLogger(__name__)

CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP_CHARS = 100


def retain(
    memory: "Memory",
    content: str,
    *,
    source: str | None = None,
    path: Path | str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    update_mode: str = "append",
) -> str:
    """Write a note. Returns document_id (str UUID).

    Default: the note is written as kind='chunk' items of at most 1,000 chars
    at paragraph boundaries (small overlap), each carrying the parent
    document_id and its ordinal / char offsets, and no model is called.

    documents.created_on is always stamped: the note's own date (frontmatter
    or metadata) else today's date in PROSPECTA_TIMEZONE (UTC when unset).
    """
    if memory._embed is None:
        raise RuntimeError(
            "Memory.retain requires an `embed` callable; construct Memory(embed=...)"
        )

    t_start = time.monotonic()
    bank_id = memory._default_bank_id

    if not isinstance(content, str):
        raise TypeError(f"content must be str, got {type(content).__name__}")
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Resolve source (substrate-opacity: opaque string from caller's POV).
    # ------------------------------------------------------------------
    resolved_source = source
    path_obj: Path | None = None
    if resolved_source is None and path is not None:
        path_obj = Path(path)
        resolved_source = str(path_obj)
    if resolved_source is None:
        resolved_source = f"retain://{content_hash[:12]}"

    # Convenience: if a path is supplied and doesn't exist, write content.
    # Substrate-opacity holds — the library does not introspect the source
    # string downstream, this is just a write-through helper.
    if path is not None:
        path_obj = Path(path) if path_obj is None else path_obj
        if not path_obj.exists():
            try:
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                path_obj.write_text(content, encoding="utf-8")
            except OSError as e:
                logger.warning("retain: failed to write path %s: %s", path_obj, e)

    tags_list = list(tags or [])
    extra_metadata = dict(metadata or {})

    # ------------------------------------------------------------------
    # Re-retain check (docs/schema.md, "Re-retain").
    # ------------------------------------------------------------------
    with memory._pool.connection() as conn:
        with conn.cursor() as cur:
            # Writers of the same content serialise here (released at commit):
            # the check-then-insert below must not race the UNIQUE constraint.
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                        (f"retain:{bank_id}:{content_hash}",))
            cur.execute(
                "SELECT id, source FROM documents "
                "WHERE bank_id = %(bank_id)s AND content_hash = %(content_hash)s",
                {"bank_id": bank_id, "content_hash": content_hash},
            )
            row = cur.fetchone()

        if row is not None:
            doc_id_existing, existing_source = row
            if existing_source != resolved_source:
                # Different source → conflict. The library treats source
                # as opaque; only the caller knows the right routing.
                raise DocumentSourceConflictError(
                    content_hash=content_hash,
                    existing_source=existing_source,
                    attempted_source=resolved_source,
                )
            # Same source → REPLACE: keep documents row (bump updated_at,
            # refresh tags + metadata to current call), drop existing
            # memory_items, regenerate.
            document_id = str(doc_id_existing)
            _refresh_document(
                conn,
                document_id=document_id,
                original_text=content,
                tags=tags_list,
                metadata=extra_metadata,
            )
            delete_memory_items_for_document(conn, document_id=document_id)
        else:
            # New document.
            document_id = _insert_document(
                conn,
                bank_id=bank_id,
                source=resolved_source,
                original_text=content,
                content_hash=content_hash,
                tags=tags_list,
                metadata=extra_metadata,
            )

        # ------------------------------------------------------------------
        # Embed + write the chunks.
        # ------------------------------------------------------------------
        body, body_off = _body_and_offset(content)
        chunks = chunk_paragraphs(body, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS)
        vectors = memory._embed([c.content for c in chunks]) if chunks else []
        if len(vectors) != len(chunks):
            raise RuntimeError(
                f"embed() returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        items = [{
            "content": c.content,
            "original_chunk": c.content,
            "embedding": list(vec),
            "metadata": {**extra_metadata},
            "tags": tags_list,
            "update_mode": update_mode,
            "llm_generated": False,
            "kind": "chunk",
            "ordinal": c.ordinal,
            "char_start": c.char_start + body_off,
            "char_end": c.char_end + body_off,
        } for c, vec in zip(chunks, vectors)]
        upsert_memory_items(
            conn,
            bank_id=bank_id,
            document_id=document_id,
            items=items,
        )

        duration_ms = int((time.monotonic() - t_start) * 1000)
        conn.commit()

    # T16: tracer dispatch (was direct append_retain_event; default
    # PostgresSink performs the row write).
    try:
        memory._tracer("retain", {
            "bank_id": bank_id,
            "document_id": document_id,
            "items_count": len(items),
            "duration_ms": duration_ms,
            "error": None,
            "source": resolved_source,
        })
    except Exception:  # pragma: no cover — tracer must not break retain
        logger.exception("tracer raised; ignoring")

    return document_id


# ---------------------------------------------------------------------------
# helpers (kept local to retain to avoid mutating _index.py — T9 frozen)
# ---------------------------------------------------------------------------

def _frontmatter(text: str) -> dict:
    from prospecta._parser import parse_frontmatter
    return parse_frontmatter(text).frontmatter or {}


def _doc_metadata(metadata: dict, original_text: str) -> dict:
    """document_metadata: the caller's metadata over the frontmatter-derived
    filter fields (created, created_on, person), so temporal links and
    headers follow the note's own date. Caller keys always win."""
    fields = filter_fields(metadata, _frontmatter(original_text))
    derived: dict = {}
    if fields["created_on"] is not None:
        derived["created"] = fields["created_on"].isoformat()
        derived["created_on"] = fields["created_on"].isoformat()
    if fields["person"] is not None:
        derived["person"] = fields["person"]
    return {**derived, **(metadata or {})}


def _body_and_offset(content: str) -> tuple[str, int]:
    """The note without its YAML frontmatter, and the body's offset in content."""
    from prospecta._parser import parse_frontmatter
    body = parse_frontmatter(content).body
    if not body.strip():
        return content, 0
    off = content.find(body)
    return body, max(off, 0)


def _insert_document(
    conn,
    *,
    bank_id: str,
    source: str,
    original_text: str,
    content_hash: str,
    tags: list[str],
    metadata: dict,
) -> str:
    import json as _json
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents
                (bank_id, source, original_text, content_hash,
                 tags, document_metadata, created_on, person, source_kind)
            VALUES (%(bank_id)s, %(source)s, %(original_text)s, %(content_hash)s,
                    %(tags)s, %(metadata)s::jsonb,
                    COALESCE(%(created_on)s, %(today)s), %(person)s, %(source_kind)s)
            RETURNING id
            """,
            {
                "bank_id": bank_id,
                "source": source,
                "original_text": original_text,
                "content_hash": content_hash,
                "tags": list(tags),
                "metadata": _json.dumps(_doc_metadata(metadata, original_text)),
                "today": today(),
                **filter_fields(metadata, _frontmatter(original_text)),
            },
        )
        return str(cur.fetchone()[0])


def _refresh_document(
    conn,
    *,
    document_id: str,
    original_text: str,
    tags: list[str],
    metadata: dict,
) -> None:
    import json as _json
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE documents
            SET original_text = %(original_text)s,
                tags = %(tags)s,
                document_metadata = %(metadata)s::jsonb,
                created_on = COALESCE(%(created_on)s, created_on, %(today)s),
                person = %(person)s, source_kind = %(source_kind)s,
                updated_at = now()
            WHERE id = %(id)s
            """,
            {
                "id": document_id,
                "original_text": original_text,
                "tags": list(tags),
                "metadata": _json.dumps(_doc_metadata(metadata, original_text)),
                "today": today(),
                **filter_fields(metadata, _frontmatter(original_text)),
            },
        )
