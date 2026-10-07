"""Bilateral-spine write path: Memory.retain() implementation.

Replace-on-source-match per schema.md §6:
  - Same content_hash + same source → REPLACE: keep documents row, delete
    existing memory_items, regenerate fresh memory_items.
  - Same content_hash + DIFFERENT source → raise DocumentSourceConflictError.
  - New content_hash → INSERT new documents row.

P-principles enforced here:
  P1 — memory_items.content holds question-shaped index_text strings; the
       full body lives in documents.original_text.
  P3 — zero provider imports; uses memory.llm and memory.embed callables.
  P4 — caller-supplied index_text bypasses LLM; prompt_override bypasses
       default prompt.
  P5 — original content preserved verbatim in documents.original_text
       (and memory_items.original_chunk = full body for LLM-generated
       paths, matching the T10 frontmatter override pattern).
"""
from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from prospecta._chunker import chunk_paragraphs
from prospecta._filters import filter_fields
from prospecta._index_text import generate_index_text
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
    index_text: str | list[str] | None = None,
    index_text_prompt_override: str | None = None,
    source: str | None = None,
    path: Path | str | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    update_mode: str = "append",
    child_chunks: bool = False,
) -> str:
    """Write-side bilateral spine. Returns document_id (str UUID).

    See plan-v2.md §3.4 for the canonical signature & semantics.

    child_chunks (default False = unchanged behaviour): additionally write
    the note as kind='chunk' items of at most 1,000 chars at paragraph
    boundaries (small overlap), each carrying the parent document_id and its
    ordinal / char offsets. The question items are written either way.
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
    # Resolve index_text (P4: caller wins).
    # ------------------------------------------------------------------
    caller_supplied: bool
    raw_llm_response: str | None = None
    rendered_prompt: str | None = None
    index_text_generated: list[str] | None = None
    if index_text is not None:
        caller_supplied = True
        if isinstance(index_text, str):
            index_text_list = [index_text]
        elif isinstance(index_text, list):
            index_text_list = [str(x) for x in index_text]
        else:
            raise TypeError(
                f"index_text must be str | list[str] | None, got {type(index_text).__name__}"
            )
        # Strip/dedupe to match LLM-path discipline.
        index_text_list = _dedupe_strip(index_text_list)
        if not index_text_list:
            raise ValueError("index_text resolved to empty list after stripping")
    else:
        caller_supplied = False
        if memory._llm is None:
            raise RuntimeError(
                "Memory.retain requires an `llm` callable when index_text is not "
                "supplied; construct Memory(llm=...) or pass index_text=..."
            )
        prompt_context: dict = {
            "tags": tags_list,
            "source": resolved_source,
            **extra_metadata,
        }
        index_text_list, rendered_prompt, raw_llm_response = generate_index_text(
            content,
            memory._llm,
            prompt_override=index_text_prompt_override,
            context=prompt_context,
        )
        index_text_generated = list(index_text_list)
        # T16 + 0003: tracer event for the LLM call carries verbatim
        # prompt + response so PostgresSink can durably persist them
        # (opt-in via PostgresSink(persist_llm_text=...)).
        try:
            memory._tracer("llm_call", {
                "bank_id": bank_id,
                "purpose": "index_text",
                "json_mode": False,
                "duration_ms": int((time.monotonic() - t_start) * 1000),
                "messages_count": 1,
                "prompt_text": rendered_prompt,
                "response_text": raw_llm_response,
            })
        except Exception:  # pragma: no cover
            logger.exception("tracer raised on llm_call (index_text); ignoring")

    # ------------------------------------------------------------------
    # Re-retain check (schema.md §6).
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
        # Embed + write memory_items.
        # ------------------------------------------------------------------
        vectors = memory._embed(index_text_list)
        if len(vectors) != len(index_text_list):
            raise RuntimeError(
                f"embed() returned {len(vectors)} vectors for "
                f"{len(index_text_list)} inputs"
            )

        body, body_off = _body_and_offset(content)
        chunks = chunk_paragraphs(body, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS)
        items = []
        for text, vec in zip(index_text_list, vectors):
            items.append({
                "content": text,
                # The full note stays in documents.original_text; a question
                # item carries only its matching chunk (frontmatter-free).
                "original_chunk": _matching_chunk(text, chunks, body),
                "embedding": list(vec),
                "metadata": {
                    **extra_metadata,
                    "index_text_caller_supplied": caller_supplied,
                },
                "tags": tags_list,
                "update_mode": update_mode,
                "llm_generated": not caller_supplied,
            })
        if child_chunks:
            chunk_vectors = memory._embed([c.content for c in chunks]) if chunks else []
            if len(chunk_vectors) != len(chunks):
                raise RuntimeError(
                    f"embed() returned {len(chunk_vectors)} vectors for "
                    f"{len(chunks)} chunks"
                )
            for c, vec in zip(chunks, chunk_vectors):
                items.append({
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
                })
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
            "index_text_caller_supplied": caller_supplied,
            "duration_ms": duration_ms,
            "raw_llm_response": raw_llm_response,
            "error": None,
            "source": resolved_source,
            "index_text_generated": index_text_generated,
        })
    except Exception:  # pragma: no cover — tracer must not break retain
        logger.exception("tracer raised; ignoring")

    memory._enqueue_link(document_id)
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


def _matching_chunk(query: str, chunks: list, fallback: str) -> str:
    """The chunk sharing the most words with the question text."""
    if not chunks:
        return fallback
    words = set(re.findall(r"\w{3,}", query.lower()))
    best = max(chunks, key=lambda c: len(words & set(re.findall(r"\w{3,}", c.content.lower()))))
    return best.content


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
                    %(created_on)s, %(person)s, %(source_kind)s)
            RETURNING id
            """,
            {
                "bank_id": bank_id,
                "source": source,
                "original_text": original_text,
                "content_hash": content_hash,
                "tags": list(tags),
                "metadata": _json.dumps(_doc_metadata(metadata, original_text)),
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
                created_on = %(created_on)s, person = %(person)s,
                source_kind = %(source_kind)s,
                updated_at = now()
            WHERE id = %(id)s
            """,
            {
                "id": document_id,
                "original_text": original_text,
                "tags": list(tags),
                "metadata": _json.dumps(_doc_metadata(metadata, original_text)),
                **filter_fields(metadata, _frontmatter(original_text)),
            },
        )


def _dedupe_strip(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for s in items:
        ss = s.strip()
        if not ss or ss in seen:
            continue
        seen.add(ss)
        out.append(ss)
    return out
