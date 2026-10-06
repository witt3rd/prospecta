"""Bank embedding-dimension migration (design 8.4): old bank -> '<bank>-v2'.

A bank's embedding dimension is fixed, so a change is a NEW bank. The old
bank is only ever READ here; nothing in it is modified. The backfill:

  * copies each source document (same text, hash, source, tags, metadata)
    into the target bank, recording ``migrated_from_document_id`` in the new
    document's ``document_metadata``;
  * chunks ``original_text`` (paragraph chunks, 1,000 chars, small overlap)
    and inserts ``kind='chunk'`` items;
  * copies each source ``kind='question'`` item's text, re-embedded (no LLM
    call), as ``kind='question'``;
  * embeds through the injected callable in batches of ``batch_size`` (128).

RESUMABLE BY DOCUMENT ID: a document and all its items are written in one
transaction, and a source document counts as done when the target holds a
document with ``migrated_from_document_id`` equal to its id. Re-running
skips the done ones and continues; killing it at any point loses nothing.
Embedder failures (e.g. HTTP 429) are retried with exponential backoff and,
if they persist, abort the run cleanly so it can be resumed.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import psycopg

from prospecta._chunker import chunk_paragraphs
from prospecta._retain import CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS
from prospecta.db.queries import _vec_literal, validate_bank_id

if TYPE_CHECKING:
    from prospecta._types import EmbedCallable

logger = logging.getLogger(__name__)

DEFAULT_TARGET_DIM = 1536
DEFAULT_TARGET_MODEL = "openai/text-embedding-3-large@1536"
BATCH_SIZE = 128
MIGRATED_FROM_KEY = "migrated_from_document_id"


@dataclass
class MigrationReport:
    source_bank: str
    target_bank: str
    documents_total: int = 0       # source documents
    documents_already_done: int = 0
    documents_migrated: int = 0    # this run
    chunks_inserted: int = 0
    questions_inserted: int = 0
    embed_batches: int = 0
    embed_retries: int = 0
    remaining: int = 0             # source documents not yet migrated
    notes: list[str] = field(default_factory=list)


def retrying_embed(
    embed: "EmbedCallable",
    *,
    retries: int = 6,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    on_retry: Callable[[], None] | None = None,
) -> "EmbedCallable":
    """Wrap an embedder with exponential backoff (rate limits, transient errors)."""

    def wrapped(texts: list[str]) -> list[list[float]]:
        attempt = 0
        while True:
            try:
                return embed(texts)
            except Exception as e:
                if attempt >= retries:
                    raise
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "embed failed (%s: %s); retry %d/%d in %.1fs",
                    type(e).__name__, e, attempt + 1, retries, delay,
                )
                attempt += 1
                if on_retry:
                    on_retry()
                sleep(delay)

    return wrapped


def default_target_bank(source_bank: str) -> str:
    return f"{source_bank}-v2"


def _embed_all(embed, texts: list[str], batch_size: int, dim: int, report) -> list[list[float]]:
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        vecs = embed(batch)
        report.embed_batches += 1
        if len(vecs) != len(batch):
            raise RuntimeError(
                f"embed() returned {len(vecs)} vectors for {len(batch)} texts"
            )
        for v in vecs:
            if len(v) != dim:
                raise RuntimeError(
                    f"embed() returned a {len(v)}-dim vector; target bank is {dim}-dim"
                )
        out.extend(list(v) for v in vecs)
    return out


def migrate_bank(
    memory,
    source_bank: str,
    *,
    target_bank: str | None = None,
    embed: "EmbedCallable",
    embedding_dim: int = DEFAULT_TARGET_DIM,
    embedding_model_id: str | None = DEFAULT_TARGET_MODEL,
    batch_size: int = BATCH_SIZE,
    max_documents: int | None = None,
    retries: int = 6,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> MigrationReport:
    """Create the target bank if needed and backfill it from `source_bank`.

    `memory` is a Memory (only its connection pool and create_bank are used;
    its own embedder is NOT used). `max_documents` bounds this run (for
    staged runs / tests); run again to continue.
    """
    target_bank = target_bank or default_target_bank(source_bank)
    validate_bank_id(source_bank)
    validate_bank_id(target_bank)
    if target_bank == source_bank:
        raise ValueError("target bank must differ from the source bank")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    with memory._pool.cursor() as cur:
        cur.execute("SELECT embedding_dim FROM banks WHERE bank_id = %s", (source_bank,))
        row = cur.fetchone()
    if row is None:
        raise ValueError(f"source bank {source_bank!r} does not exist")

    # Creates the bank + its HNSW indexes; BankConfigConflict if it exists
    # with another dimension (a stale/mistyped target is refused, not reused).
    memory.create_bank(
        target_bank, embedding_dim=embedding_dim,
        embedding_model_id=embedding_model_id,
    )

    report = MigrationReport(source_bank, target_bank)
    embed = retrying_embed(
        embed, retries=retries, base_delay=base_delay, sleep=sleep,
        on_retry=lambda: setattr(report, "embed_retries", report.embed_retries + 1),
    )

    with memory._pool.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents WHERE bank_id = %s", (source_bank,))
        report.documents_total = cur.fetchone()[0]
        cur.execute(_PENDING_COUNT, {"src": source_bank, "dst": target_bank})
        pending = cur.fetchone()[0]
    report.documents_already_done = report.documents_total - pending

    last_id = None
    budget = max_documents
    while budget is None or budget > 0:
        # Take documents until the group holds >= batch_size texts (or the
        # budget/source runs out); embed the group in batches; write each
        # document atomically.
        group: list[dict] = []
        texts = 0
        while texts < batch_size and (budget is None or len(group) < budget):
            docs = _fetch_pending(memory, source_bank, target_bank, last_id, limit=16)
            if not docs:
                break
            for d in docs:
                last_id = d["id"]
                d["chunks"] = chunk_paragraphs(
                    d["original_text"], CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS
                )
                d["questions"] = _fetch_questions(memory, d["id"])
                group.append(d)
                texts += len(d["chunks"]) + len(d["questions"])
                if (budget is not None and len(group) >= budget) or texts >= batch_size:
                    break
        if not group:
            break

        flat = [c.content for d in group for c in d["chunks"]]
        flat += [q["content"] for d in group for q in d["questions"]]
        vecs = _embed_all(embed, flat, batch_size, embedding_dim, report)

        pos = 0
        vec_for = []
        for d in group:
            cv = vecs[pos:pos + len(d["chunks"])]
            pos += len(d["chunks"])
            vec_for.append([cv, None])
        for i, d in enumerate(group):
            qv = vecs[pos:pos + len(d["questions"])]
            pos += len(d["questions"])
            vec_for[i][1] = qv

        for d, (cv, qv) in zip(group, vec_for):
            _write_document(memory, target_bank, d, cv, qv, report)
            report.documents_migrated += 1
        if budget is not None:
            budget -= len(group)

    with memory._pool.cursor() as cur:
        cur.execute(_PENDING_COUNT, {"src": source_bank, "dst": target_bank})
        report.remaining = cur.fetchone()[0]
    return report


_PENDING = """
    FROM documents s
    WHERE s.bank_id = %(src)s
      AND NOT EXISTS (
          SELECT 1 FROM documents t
          WHERE t.bank_id = %(dst)s
            AND t.document_metadata ->> 'migrated_from_document_id' = s.id::text
      )
"""
_PENDING_COUNT = "SELECT count(*) " + _PENDING


def _fetch_pending(memory, source_bank, target_bank, after_id, limit):
    sql = (
        "SELECT s.id::text, s.source, s.original_text, s.content_hash, s.tags, "
        "s.document_metadata, s.retain_params, s.created_at "
        + _PENDING
        + (" AND s.id > %(after)s::uuid" if after_id else "")
        + " ORDER BY s.id LIMIT %(limit)s"
    )
    with memory._pool.cursor() as cur:
        cur.execute(sql, {"src": source_bank, "dst": target_bank,
                          "after": after_id, "limit": limit})
        cols = ["id", "source", "original_text", "content_hash", "tags",
                "document_metadata", "retain_params", "created_at"]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _fetch_questions(memory, document_id):
    with memory._pool.cursor() as cur:
        cur.execute(
            "SELECT content, original_chunk, context, metadata, tags, "
            "update_mode, llm_generated FROM memory_items "
            "WHERE document_id = %s AND kind = 'question' ORDER BY created_at, id",
            (document_id,),
        )
        cols = ["content", "original_chunk", "context", "metadata", "tags",
                "update_mode", "llm_generated"]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _write_document(memory, target_bank, d, chunk_vecs, q_vecs, report) -> None:
    import json

    meta = dict(d["document_metadata"] or {})
    meta[MIGRATED_FROM_KEY] = d["id"]
    with memory._pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO documents (bank_id, source, original_text, content_hash,
                                       tags, document_metadata, retain_params,
                                       created_at)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                ON CONFLICT (bank_id, content_hash) DO NOTHING
                RETURNING id
                """,
                (target_bank, d["source"], d["original_text"], d["content_hash"],
                 list(d["tags"] or []), json.dumps(meta),
                 json.dumps(d["retain_params"] or {}), d["created_at"]),
            )
            row = cur.fetchone()
            if row is None:
                # Same content_hash already in the target (not from this
                # migration's marker): skip rather than duplicate or clobber.
                report.notes.append(
                    f"document {d['id']} skipped: content_hash already in {target_bank}"
                )
                conn.commit()
                return
            new_id = row[0]
            for c, vec in zip(d["chunks"], chunk_vecs):
                cur.execute(
                    _INSERT_ITEM,
                    {"bank": target_bank, "doc": new_id, "content": c.content,
                     "orig": c.content, "ctx": None, "emb": _vec_literal(vec),
                     "meta": json.dumps(meta_for_item(d)), "tags": list(d["tags"] or []),
                     "mode": "append", "llm": False, "kind": "chunk",
                     "ordinal": c.ordinal, "cs": c.char_start, "ce": c.char_end},
                )
            for q, vec in zip(d["questions"], q_vecs):
                cur.execute(
                    _INSERT_ITEM,
                    {"bank": target_bank, "doc": new_id, "content": q["content"],
                     "orig": q["original_chunk"], "ctx": q["context"],
                     "emb": _vec_literal(vec),
                     "meta": json.dumps(q["metadata"] or {}),
                     "tags": list(q["tags"] or []), "mode": q["update_mode"],
                     "llm": q["llm_generated"], "kind": "question",
                     "ordinal": None, "cs": None, "ce": None},
                )
        conn.commit()
    report.chunks_inserted += len(d["chunks"])
    report.questions_inserted += len(d["questions"])


def meta_for_item(d) -> dict:
    return {k: v for k, v in (d["document_metadata"] or {}).items()
            if k != MIGRATED_FROM_KEY}


_INSERT_ITEM = """
    INSERT INTO memory_items
        (bank_id, document_id, content, original_chunk, context, embedding,
         metadata, tags, update_mode, llm_generated,
         kind, ordinal, char_start, char_end)
    VALUES
        (%(bank)s, %(doc)s, %(content)s, %(orig)s, %(ctx)s, %(emb)s::vector,
         %(meta)s::jsonb, %(tags)s, %(mode)s, %(llm)s,
         %(kind)s, %(ordinal)s, %(cs)s, %(ce)s)
"""
