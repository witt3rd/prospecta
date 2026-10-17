"""Internal index module — directory walk, parse, chunk, embed, write, and search.

A file is written as kind='chunk' items (content == original_chunk == the chunk text), with no model
call. search() is Mode 2A: Stage 1 (the bank's hybrid_chunk channel) then Stage 2.

Public Memory methods (index_directory, index_single_file, search, remove_documents) delegate here.
"""
from __future__ import annotations

import dataclasses
import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Protocol

from prospecta._chunker import chunk_text
from prospecta.cited import document_citations, match_kind
from prospecta._scorecut import check_depth, resolve_depth
from prospecta.stages import StageDeps, read_recall_config, run_stages
from prospecta.channels import QueryPlan, RecallState, read_channel_config, run_channels
from prospecta._ignore import should_ignore
from prospecta._filters import filter_fields
from prospecta._parser import parse_frontmatter
from prospecta._types import IndexStats, RecalledMemory
from prospecta.db.queries import (
    delete_documents_by_source,
    update_document_source,
    upsert_document,
    upsert_memory_items,
)

if TYPE_CHECKING:
    from prospecta.memory import Memory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ParserPlugin Protocol (a caller-supplied parser for a source format; see prospecta/_parser.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedDocument:
    original_text: str
    metadata: dict | None = None
    tags: list[str] | None = None


class ParserPlugin(Protocol):
    file_patterns: list[str]

    def parse(self, path: Path) -> Iterator[ParsedDocument]: ...


# ---------------------------------------------------------------------------
# Bundled markdown parser
# ---------------------------------------------------------------------------

class _MarkdownParser:
    """Bundled markdown parser: the body is chunked; frontmatter gives the note's tags, date,
    person and kind."""

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
        # Filter fields (documents.created_on / person / source_kind, 0005)
        for k, v in filter_fields(fm).items():
            if v is not None:
                metadata[k] = v.isoformat() if k == "created_on" else v
        # Tags surface for documents.tags
        tags = metadata.get("frontmatter_tags", []) or []
        yield ParsedDocument(
            original_text=parsed.body if parsed.body else content,
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

            # Chunk + embed
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
                    "kind": "chunk",
                    "ordinal": c.chunk_index,
                    "char_start": c.start_char,
                    "char_end": c.end_char,
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

def search(memory: "Memory", text: str, **kwargs) -> list[RecalledMemory]:
    """Hits for `text`, each carrying its citation fields (created_on, kind); see _search."""
    results = _search(memory, text, **kwargs)
    if not results:
        return results
    with memory._pool.connection() as conn:
        cited = document_citations(conn, sorted({r.document_id for r in results}))
        conn.rollback()
    return [dataclasses.replace(
        r, created_on=cited.get(r.document_id, (None, None))[0],
        kind=cited.get(r.document_id, (None, None))[1],
        relevance_score=r.score if r.relevance_score is None else r.relevance_score)
        for r in results]


def _search(
    memory: "Memory",
    text: str,
    *,
    limit: int = 10,
    metadata_filter: dict | None = None,
    rrf_k: int = 60,
    _trace: list | None = None,
    depth: str | None = None,
    date_from=None,
    date_to=None,
    excluded_sources: list[str] | None = None,
    reranker_model: str | None = None,
    decision_model: str | None = None,
) -> list[RecalledMemory]:
    """Stage 1 (the bank's hybrid_chunk channel: an unset or empty channel_config is its defaults)
    then Stage 2. `date_from` / `date_to` (the note's date, inclusive) and `excluded_sources` (the
    anti-echo list of documents.source values) are Stage 1 filters. `reranker_model` /
    `decision_model` override, for this call, the Memory's, the bank's recall_config and the
    environment's choice of the stage 2 / Jev model."""
    check_depth(depth)
    if memory._embed is None:
        raise RuntimeError(
            "Memory.search requires an `embed` callable; construct Memory(embed=...)"
        )
    bank_id = memory._default_bank_id
    with memory._pool.connection() as conn:
        channel_config = read_channel_config(conn, bank_id)
        conn.rollback()
    return _search_channels(
        memory, text, bank_id, channel_config, limit=limit,
        metadata_filter=metadata_filter, rrf_k=rrf_k, trace=_trace,
        depth=depth,
        stage1_filters={"date_from": date_from, "date_to": date_to,
                        "excluded_sources": excluded_sources},
        reranker_model=reranker_model, decision_model=decision_model,
    )


def _search_channels(
    memory: "Memory", text: str, bank_id: str, config: list[dict],
    *, limit: int, metadata_filter: dict | None, rrf_k: int, trace: list | None,
    recall_cfg: dict | None = None, depth: str | None = None,
    stage1_filters: dict | None = None, reranker_model: str | None = None,
    decision_model: str | None = None,
) -> list[RecalledMemory]:
    """Recall through the bank's channel (banks.channel_config, resolved by read_channel_config).
    `recall_cfg` overrides the bank's stage config ({} = no stages); None reads it."""
    plan = QueryPlan(text=text)
    with memory._pool.connection() as conn:
        state = RecallState(
            conn=conn, bank_id=bank_id, embed=memory._embed,
            metadata_filter=metadata_filter, **(stage1_filters or {}),
        )
        if recall_cfg is None:
            recall_cfg = read_recall_config(conn, bank_id)
        fused, tr = run_channels(state, plan, config, k=rrf_k)
        if recall_cfg:
            fused, st = run_stages(
                state, text, fused, recall_cfg,
                StageDeps(jev=memory._jev,
                          rerank_transport=memory._rerank_transport,
                          reranker_model=reranker_model or memory._reranker_model,
                          decision_model=decision_model or memory._decision_model),
                depth=depth,
            )
            tr.update(rerank=st["rerank"], calls=st["calls"],
                      fallback_reason=st.get("fallback_reason"),
                      accounting={k: st[k] for k in
                                  ("n_llm_calls", "tokens_in", "tokens_out", "cost_usd")})
        tr["depth"] = resolve_depth(depth, recall_cfg)
    if trace is not None:
        trace.append(tr)
    return recalled_from_fused(
        fused[:limit], bank_id, state.channel_lists, [c["name"] for c in tr["channels"]], recall_cfg)


def recalled_from_fused(fused, bank_id: str, channel_lists: dict, channel_names: list[str],
                        recall_cfg: dict | None) -> list[RecalledMemory]:
    """The fused documents as RecalledMemory, in order. `channel_lists` is what the channel
    returned (the hybrid_chunk chunk carries the snippet that belongs with its own text);
    `channel_names` are the channels that ran; `recall_cfg` says whether stages ran (scores
    gain rerank) and whether the cross-encoder graded (relevance_score)."""
    hybrid = {c.document_id: c for c in channel_lists.get("hybrid_chunk", [])}
    rerank_cfg = (recall_cfg or {}).get("rerank") or {}
    cross_encoder = rerank_cfg.get("enabled") and rerank_cfg.get("stage") == "cross_encoder"
    out = []
    for f in fused:
        h = hybrid.get(f.document_id)   # the Stage 1 chunk: its snippet belongs with its own text
        b = h or f.best
        d = h.detail if h else {}
        scores = {n: 0.0 for n in channel_names}
        if recall_cfg:
            scores.update(rerank=0.0)
        scores.update({n: float(v) for n, v in f.scores.items()})
        scores["rrf"] = float(f.score)
        out.append(RecalledMemory(
            content=b.detail.get("content") or b.evidence or "",
            original_chunk=b.evidence or "",
            source=f.source,
            score=float(f.score),
            scores=scores,
            metadata=b.detail.get("metadata") or {},
            bank_id=bank_id,
            document_id=f.document_id,
            match_kind=match_kind(
                exact=d.get("exact_rank") is not None,
                lexical=d.get("stem_rank") is not None,
                semantic=d.get("sem_rank") is not None,
                date=d.get("date_rank") is not None),
            relevance_score=float(f.scores["rerank"]) if cross_encoder and "rerank" in f.scores
            else float(f.score),
            exact_snippet=d.get("exact_snippet"),
            item_id=b.item_id,
        ))
    return out
