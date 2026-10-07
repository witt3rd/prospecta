"""Shadow reads (design 8.4 step 5): run old and new recall side by side.

``Memory(shadow_bank_id=..., shadow_embed=...)`` makes every ``recall()``
also query the shadow (new, e.g. 1536-d) bank with its own embedder. The
caller still gets only the primary (old bank) results; both runs are stored
as ``recall_events`` rows (one under each bank_id, mode ``<mode>``) whose
``trace`` carries ``{"shadow": {id, role, peer_bank, ...}}`` so they pair up
and can be compared offline. The shadow path can never fail a recall: an
error is logged and recorded in the trace instead.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prospecta._types import EmbedCallable, Query, RecalledMemory
    from prospecta.memory import Memory

logger = logging.getLogger(__name__)


def _origin_ids(memory: "Memory", bank_id: str, results) -> list[str]:
    """Document ids of `results` mapped to the ORIGINAL (pre-migration) id."""
    ids = [str(r.document_id) for r in results]
    if not ids:
        return []
    with memory._pool.cursor() as cur:
        cur.execute(
            "SELECT id::text, COALESCE(document_metadata ->> 'migrated_from_document_id', id::text) "
            "FROM documents WHERE bank_id = %s AND id = ANY(%s::uuid[])",
            (bank_id, ids),
        )
        m = dict(cur.fetchall())
    return [m.get(i, i) for i in ids]


def run_shadow_recall(
    memory: "Memory",
    queries: "list[Query]",
    primary_results: "list[RecalledMemory]",
    *,
    mode: str,
    limit: int,
    metadata_filter: dict | None,
    rrf_k: int,
    primary_duration_ms: int,
    depth: str | None = None,
) -> None:
    """Run the shadow recall and store both sides in recall_events. Never raises."""
    from prospecta._tracer import NoOpTracer
    from prospecta.memory import Memory, _serialize_recall_results

    shadow_bank = memory._shadow_bank_id
    old_bank = memory._default_bank_id
    shadow_id = str(uuid.uuid4())
    error = None
    flat: "list[RecalledMemory]" = []
    t0 = time.monotonic()
    try:
        shadow_mem = Memory(
            database_url=memory.database_url, embed=memory._shadow_embed,
            bank_id=shadow_bank, tracer=NoOpTracer(),
        )
        try:
            for q in queries:
                flat.extend(shadow_mem.search(
                    q.text, mode=mode, limit=limit,
                    metadata_filter=metadata_filter, rrf_k=rrf_k,
                    depth=depth,
                ))
        finally:
            shadow_mem.close()
    except Exception as e:  # shadow must never break the real recall
        logger.exception("shadow recall against %s failed", shadow_bank)
        error = f"{type(e).__name__}: {e}"
    shadow_ms = int((time.monotonic() - t0) * 1000)

    try:
        old_origin = _origin_ids(memory, old_bank, primary_results)
        new_origin = _origin_ids(memory, shadow_bank, flat) if error is None else []
        old_set, new_set = set(old_origin), set(new_origin)
        union = old_set | new_set
        overlap = {
            "old_origin_document_ids": old_origin,
            "new_origin_document_ids": new_origin,
            "jaccard": (len(old_set & new_set) / len(union)) if union else 1.0,
        }
        texts = [q.text for q in queries]
        for role, bank, peer, results, ms, err in (
            ("old", old_bank, shadow_bank, primary_results, primary_duration_ms, None),
            ("new", shadow_bank, old_bank, flat, shadow_ms, error),
        ):
            memory._tracer("recall", {
                "bank_id": bank,
                "queries": texts,
                "mode": mode,
                "n_results": len(results),
                "duration_ms": ms,
                "trace": {"shadow": {
                    "id": shadow_id, "role": role, "peer_bank": peer,
                    "error": err, **overlap,
                }},
                "results": _serialize_recall_results(results),
                "synthesis": None,
                "depth": depth,
            })
    except Exception:
        logger.exception("could not store shadow recall events")
