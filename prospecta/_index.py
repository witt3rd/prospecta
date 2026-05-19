"""Internal index module — directory walk, parse, chunk, embed, write.

T9 vertical slice: classical RAG over Postgres. NO LLM-generated index_text
yet (spine ships T11/T13). For now, memory_items.content == chunk text,
which lets the lexical CTE work and seeds the slice end-to-end.

Public Memory methods (index_directory, search, remove_documents) delegate
here. The retain() method stays a stub in memory.py for T11 to flesh out.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Literal, Protocol

from prospecta._chunker import chunk_text
from prospecta._ignore import should_ignore
from prospecta._parser import parse_frontmatter
from prospecta._types import IndexStats, RecalledMemory
from prospecta.db.queries import (
    delete_documents_by_source,
    delete_memory_items_for_document,
    hybrid_search,
    lexical_search,
    semantic_search,
    update_document_source,
    upsert_document,
    upsert_memory_items,
)

if TYPE_CHECKING:
    from prospecta.memory import Memory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ParserPlugin Protocol (decision-record-1 §addendum 2026-05-19)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedDocument:
    original_text: str
    index_text: str | list[str] | None = None
    metadata: dict | None = None
    tags: list[str] | None = None


class ParserPlugin(Protocol):
    file_patterns: list[str]

    def parse(self, path: Path) -> Iterator[ParsedDocument]: ...


# ---------------------------------------------------------------------------
# Bundled markdown parser
# ---------------------------------------------------------------------------

class _MarkdownParser:
    """Bundled markdown parser. Honors `index_text:` frontmatter (T10 will
    use this on the spine write path; here we record it in metadata for
    forward compatibility but still index the body as chunks)."""

    file_patterns = ["*.md", "*.markdown"]

    def parse(self, path: Path) -> Iterator[ParsedDocument]:
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as e:
            logger.warning("Skipping unreadable file %s: %s", path, e)
            return
        parsed = parse_frontmatter(content)
        fm = parsed.frontmatter or {}
        # Cast frontmatter values to JSON-safe shapes.
        metadata: dict = {}
        if "tags" in fm:
            t = fm["tags"]
            if isinstance(t, str):
                t = [s.strip() for s in t.split(",")]
            if isinstance(t, list):
                metadata["frontmatter_tags"] = [str(x) for x in t]
        if "index_text" in fm:
            # Forward-compatible: record presence; spine consumes in T10.
            metadata["has_caller_index_text"] = True
        # Tags surface for documents.tags
        tags = metadata.get("frontmatter_tags", []) or []
        yield ParsedDocument(
            original_text=parsed.body if parsed.body else content,
            index_text=fm.get("index_text"),
            metadata=metadata,
            tags=tags,
        )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _file_matches(path: Path, patterns: list[str]) -> bool:
    import fnmatch
    name = path.name
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _select_parser(path: Path, parsers: list[ParserPlugin]) -> ParserPlugin | None:
    for p in parsers:
        if _file_matches(path, getattr(p, "file_patterns", [])):
            return p
    return None


def _crawl(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*")):
        if p.is_file() and not should_ignore(p, root):
            yield p


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# index_directory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SingleFileIndexResult:
    """Outcome of indexing a single file. P7 single write path.

    status:
      - "new":       a brand-new document was inserted.
      - "replaced":  a prior document for the same source was replaced
                     (or its source bookkeeping updated for same-hash).
      - "unchanged": the same file (same hash) was already indexed; no-op.
      - "skipped":   no parser matched, parser produced no docs, etc.
      - "error":     a recoverable error fired (e.g. parser exception).
    """

    status: str  # one of: new | replaced | unchanged | skipped | error
    document_ids: tuple[str, ...] = ()
    items_added: int = 0
    error: str | None = None


def index_single_file(
    memory: "Memory",
    path: str | Path,
    *,
    source_prefix: str | None = None,
    root: Path | None = None,
    parsers: list[ParserPlugin] | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> SingleFileIndexResult:
    """Index ONE file. Same write semantics as index_directory's per-file body.

    P7: this is the single write path for filesystem-derived indexing. Both
    index_directory (loop) and the sweeper (drift detection) call this.

    `root` is used only to compute a relative source string when
    `source_prefix` is supplied. When `root` is None, `source_prefix` falls
    back to file basename.
    """
    if memory._embed is None:
        raise RuntimeError(
            "Memory.index_single_file requires an `embed` callable; "
            "construct Memory(embed=...)"
        )
    file_path = Path(path)
    parsers = list(parsers or [])
    parsers.append(_MarkdownParser())

    parser = _select_parser(file_path, parsers)
    if parser is None:
        return SingleFileIndexResult(status="skipped")

    try:
        file_bytes = file_path.read_bytes()
    except OSError as e:
        return SingleFileIndexResult(status="error", error=repr(e))
    content_hash = _hash_bytes(file_bytes)

    # Substrate-opacity: source is caller-supplied opaque string.
    if source_prefix:
        try:
            if root is not None and root.is_dir():
                rel = file_path.relative_to(root)
            else:
                rel = file_path.name
        except ValueError:
            rel = file_path
        source = f"{source_prefix}/{rel}"
    else:
        source = str(file_path)

    try:
        parsed_iter = list(parser.parse(file_path))
    except Exception as e:
        logger.warning("Parser failed for %s: %s", file_path, e)
        return SingleFileIndexResult(status="error", error=repr(e))

    if not parsed_iter:
        return SingleFileIndexResult(status="skipped")

    return _write_parsed_docs(
        memory,
        file_path=file_path,
        parsed_iter=parsed_iter,
        content_hash=content_hash,
        source=source,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def _write_parsed_docs(
    memory: "Memory",
    *,
    file_path: Path,
    parsed_iter: list[ParsedDocument],
    content_hash: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> SingleFileIndexResult:
    """Inner write-path. Returns aggregated SingleFileIndexResult per file."""
    bank_id = memory._default_bank_id
    statuses: list[str] = []
    document_ids: list[str] = []
    total_items_added = 0

    for parsed in parsed_iter:
        if len(parsed_iter) > 1:
            doc_hash = hashlib.sha256(
                (content_hash + parsed.original_text).encode("utf-8")
            ).hexdigest()
        else:
            doc_hash = content_hash

        with memory._pool.connection() as conn:
            doc_id, _replaced, prior_source = upsert_document(
                conn,
                bank_id=bank_id,
                source=source,
                content_hash=doc_hash,
                original_text=parsed.original_text,
                metadata=parsed.metadata or {},
                tags=list(parsed.tags or []),
            )
            existed = prior_source is not None or (
                _replaced is False
                and _doc_already_had_items(conn, doc_id)
            )

            if existed:
                if prior_source != source:
                    update_document_source(
                        conn,
                        document_id=doc_id,
                        source=source,
                        original_text=parsed.original_text,
                        metadata=parsed.metadata or {},
                        tags=list(parsed.tags or []),
                    )
                    statuses.append("replaced")
                else:
                    statuses.append("unchanged")
                document_ids.append(str(doc_id))
                conn.commit()
                continue

            replaced_count = _delete_prior_docs_at_source_diff_hash(
                conn, bank_id=bank_id, source=source, keep_doc_id=doc_id,
            )
            if replaced_count > 0:
                statuses.append("replaced")
            else:
                statuses.append("new")
            document_ids.append(str(doc_id))

            # T10: frontmatter `index_text:` override (P4 — caller wins).
            override = parsed.index_text
            if override is not None:
                if isinstance(override, str):
                    override_texts = [override]
                elif isinstance(override, list):
                    override_texts = [str(x) for x in override]
                else:
                    override_texts = [str(override)]
                vectors = memory._embed(override_texts)
                if len(vectors) != len(override_texts):
                    raise RuntimeError(
                        f"embed() returned {len(vectors)} vectors "
                        f"for {len(override_texts)} inputs"
                    )
                items = []
                for text, vec in zip(override_texts, vectors):
                    items.append({
                        "content": text,
                        "original_chunk": text,
                        "embedding": list(vec),
                        "metadata": {
                            **(parsed.metadata or {}),
                            "index_text_caller_supplied": True,
                        },
                        "tags": list(parsed.tags or []),
                        "update_mode": "append",
                        "llm_generated": False,
                    })
                upsert_memory_items(
                    conn,
                    bank_id=bank_id,
                    document_id=doc_id,
                    items=items,
                )
                total_items_added += len(items)
                conn.commit()
                continue

            # Chunk + embed (no override path)
            if parsed.original_text.strip():
                chunks = list(
                    chunk_text(
                        parsed.original_text,
                        file_path,
                        chunk_size=chunk_size,
                        overlap=chunk_overlap,
                    )
                )
            else:
                chunks = []

            if not chunks:
                conn.commit()
                continue

            texts = [c.content for c in chunks]
            vectors = memory._embed(texts)
            if len(vectors) != len(texts):
                raise RuntimeError(
                    f"embed() returned {len(vectors)} vectors for {len(texts)} inputs"
                )

            items = []
            for c, vec in zip(chunks, vectors):
                items.append({
                    "content": c.content,
                    "original_chunk": c.content,
                    "embedding": list(vec),
                    "metadata": {
                        **(parsed.metadata or {}),
                        **c.to_metadata(),
                    },
                    "tags": list(parsed.tags or []),
                    "update_mode": "append",
                    "llm_generated": False,
                })
            upsert_memory_items(
                conn,
                bank_id=bank_id,
                document_id=doc_id,
                items=items,
            )
            total_items_added += len(items)
            conn.commit()

    # Aggregate per-doc statuses into one file-level status.
    # Priority: new > replaced > unchanged > skipped.
    if "new" in statuses:
        agg = "new"
    elif "replaced" in statuses:
        agg = "replaced"
    elif "unchanged" in statuses:
        agg = "unchanged"
    else:
        agg = "skipped"
    return SingleFileIndexResult(
        status=agg,
        document_ids=tuple(document_ids),
        items_added=total_items_added,
    )


def index_directory(
    memory: "Memory",
    path: str | Path,
    *,
    source_prefix: str | None = None,
    parsers: list[ParserPlugin] | None = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> IndexStats:
    """Walk a directory; delegate per-file write to index_single_file (P7)."""
    if memory._embed is None:
        raise RuntimeError(
            "Memory.index_directory requires an `embed` callable; "
            "construct Memory(embed=...)"
        )
    root = Path(path)

    docs_added = 0
    docs_replaced = 0
    items_added = 0
    files_scanned = 0
    files_skipped = 0
    files_unchanged = 0
    errors = 0

    for file_path in _crawl(root):
        files_scanned += 1
        result = index_single_file(
            memory,
            file_path,
            source_prefix=source_prefix,
            root=root,
            parsers=parsers,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if result.status == "new":
            docs_added += 1
        elif result.status == "replaced":
            docs_replaced += 1
        elif result.status == "unchanged":
            files_unchanged += 1
        elif result.status == "error":
            errors += 1
        else:
            files_skipped += 1
        items_added += result.items_added

    return IndexStats(
        documents_added=docs_added,
        documents_replaced=docs_replaced,
        memory_items_added=items_added,
        files_scanned=files_scanned,
        files_skipped=files_skipped,
        files_unchanged=files_unchanged,
        errors=errors,
    )


def _doc_already_had_items(conn, doc_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM memory_items WHERE document_id = %s LIMIT 1",
            (doc_id,),
        )
        return cur.fetchone() is not None


def _delete_prior_docs_at_source_diff_hash(
    conn, *, bank_id: str, source: str, keep_doc_id: str
) -> int:
    """Replace-on-source-match: delete any other documents in this bank
    that share the same source but a *different* content_hash than the one
    we just inserted. Returns count of deleted documents.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            DELETE FROM documents
            WHERE bank_id = %(bank_id)s
              AND source = %(source)s
              AND id <> %(keep)s
            """,
            {"bank_id": bank_id, "source": source, "keep": keep_doc_id},
        )
        return cur.rowcount or 0


# ---------------------------------------------------------------------------
# remove_documents
# ---------------------------------------------------------------------------

def remove_documents(memory: "Memory", sources: list[str]) -> int:
    with memory._pool.connection() as conn:
        n = delete_documents_by_source(
            conn, bank_id=memory._default_bank_id, sources=sources
        )
        conn.commit()
    return n


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def search(
    memory: "Memory",
    text: str,
    *,
    mode: Literal["hybrid", "semantic", "lexical"] = "hybrid",
    limit: int = 10,
    metadata_filter: dict | None = None,
    rrf_k: int = 60,
) -> list[RecalledMemory]:
    if mode not in ("hybrid", "semantic", "lexical"):
        raise ValueError(f"Invalid mode: {mode!r}")
    if mode in ("hybrid", "semantic") and memory._embed is None:
        raise RuntimeError(
            "Memory.search requires an `embed` callable for "
            f"mode={mode!r}; construct Memory(embed=...)"
        )
    bank_id = memory._default_bank_id

    if mode == "hybrid":
        qvec = memory._embed([text])[0]
        with memory._pool.connection() as conn:
            rows = hybrid_search(
                conn,
                bank_id=bank_id,
                query_text=text,
                query_embedding=list(qvec),
                limit=limit,
                rrf_k=rrf_k,
                metadata_filter=metadata_filter,
            )
        return [_row_to_recalled(r, bank_id, mode="hybrid") for r in rows]

    if mode == "semantic":
        qvec = memory._embed([text])[0]
        with memory._pool.connection() as conn:
            rows = semantic_search(
                conn,
                bank_id=bank_id,
                query_embedding=list(qvec),
                limit=limit,
                metadata_filter=metadata_filter,
            )
        out = []
        for r in rows:
            sem = float(r.get("sem_score") or 0.0)
            out.append(_make_recalled(
                r, bank_id, sem_score=sem, lex_score=0.0, rrf=sem,
            ))
        return out

    # lexical
    with memory._pool.connection() as conn:
        rows = lexical_search(
            conn,
            bank_id=bank_id,
            query_text=text,
            limit=limit,
            metadata_filter=metadata_filter,
        )
    out = []
    for r in rows:
        lex = float(r.get("lex_score") or 0.0)
        out.append(_make_recalled(
            r, bank_id, sem_score=0.0, lex_score=lex, rrf=lex,
        ))
    return out


def _row_to_recalled(row: dict, bank_id: str, *, mode: str) -> RecalledMemory:
    sem = float(row.get("sem_score") or 0.0)
    lex = float(row.get("lex_score") or 0.0)
    rrf = float(row.get("rrf_score") or 0.0)
    return _make_recalled(row, bank_id, sem_score=sem, lex_score=lex, rrf=rrf)


def _make_recalled(
    row: dict, bank_id: str,
    *, sem_score: float, lex_score: float, rrf: float,
) -> RecalledMemory:
    scores = {
        "semantic": float(sem_score),
        "lexical": float(lex_score),
        "rrf": float(rrf),
    }
    return RecalledMemory(
        content=row["content"],
        original_chunk=row["original_chunk"],
        source=row.get("source") or "",
        score=float(rrf),
        scores=scores,
        metadata=row.get("metadata") or {},
        bank_id=bank_id,
        document_id=str(row["document_id"]),
    )
