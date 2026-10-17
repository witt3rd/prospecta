"""prospecta.Memory — the public API: write (retain, index_directory), read (search, recall, ask,
expand_and_recall, recall_for_task), banks (create_bank, set_channel_config, set_recall_config,
bank_stats) and the sweeper. docs/postgres-hybrid.md is the design."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

import psycopg

from typing import Literal

from prospecta.bank_defaults import new_bank_configs
from prospecta._types import (
    AskResult, EmbedCallable, ExpandedRecall, LLMCallable, Query, RecalledMemory, TaskBrief, Tracer,
)
from prospecta.db.pool import ConnectionPool
from prospecta.db.schema import check_schema_writable, resolve_schema, scoped_url
from prospecta.db.queries import (
    INSERT_BANK,
    SELECT_BANK,
    SELECT_BANK_STATS,
    ensure_kind_hnsw_indexes,
    validate_bank_id,
)

logger = logging.getLogger(__name__)


def _serialize_recall_results(results: "list[RecalledMemory]") -> list[dict]:
    """Serialize a flat list of RecalledMemory for the recall_events.results
    JSONB column (migration 0003).

    One dict per result, in flat-list order (rank is 1-indexed within the
    flat list, the queries' results one after the other):

      - source: documents.source (file path / URL / conv ID)
      - document_id: UUID string
      - rank: 1-indexed position in flat list
      - scores: {hybrid_chunk, rrf} and, when Stage 2 graded the note, rerank
      - content_preview: first 200 chars of RecalledMemory.content
        (NOT a P5 violation — full content stays in memory_items;
        this is the inspection-only audit trail per 0003 column comment)
    """
    PREVIEW_LEN = 200
    out: list[dict] = []
    for i, r in enumerate(results, start=1):
        content = r.content if isinstance(r.content, str) else str(r.content)
        out.append({
            "source": r.source,
            "document_id": str(r.document_id),
            "rank": i,
            "scores": dict(r.scores) if r.scores is not None else {},
            "content_preview": content[:PREVIEW_LEN],
        })
    return out


def _channel_trace_payload(traces: "list[dict]") -> dict:
    """Merge per-query channel traces into the recall tracer payload
    (recall_events.plan/channels/fusion + recall_event_candidates)."""
    if not traces:
        return {}
    channels: list[dict] = []
    candidates: list[dict] = []
    for qi, t in enumerate(traces):
        channels.extend({**c, "query_index": qi} for c in t["channels"])
        candidates.extend({**c, "query_index": qi} for c in t["candidates"])
    out = {
        "depth": traces[0].get("depth"),
        "plan": {"queries": [t["plan"] for t in traces]},
        "channels": channels,
        "fusion": {**traces[0]["fusion"], "per_query": [t["fusion"] for t in traces]},
        "candidates": candidates,
    }
    per = [{**t["rerank"], "query_index": qi} for qi, t in enumerate(traces) if t.get("rerank")]
    if per:
        out["rerank"] = {"per_query": per}
    accs = [t["accounting"] for t in traces if t.get("accounting")]
    if accs:
        def _sum(k):
            v = [a[k] for a in accs if a.get(k) is not None]
            return sum(v) if v else None
        out.update({k: _sum(k) for k in ("n_llm_calls", "tokens_in", "tokens_out", "cost_usd")})
    return out


class BankConfigConflict(Exception):
    """Raised when create_bank is called with a different embedding_dim
    than an existing bank with the same bank_id."""


@dataclass(frozen=True)
class BankStats:
    """Bank stats returned by Memory.bank_stats()."""

    bank_id: str
    documents: int
    memory_items: int
    last_retain_at: Any  # datetime or None


class Memory:
    """The public Memory API for prospecta (see the module docstring)."""

    def __init__(
        self,
        *,
        database_url: str,
        llm: LLMCallable | None = None,
        embed: EmbedCallable | None = None,
        bank_id: str = "prospecta",
        tracer: Tracer | None = None,
        shadow_bank_id: str | None = None,
        shadow_embed: EmbedCallable | None = None,
        synth_llm: LLMCallable | None = None,
        jev: "Any | None" = None,
        rerank_transport: "Any | None" = None,
        reranker_model: str | None = None,
        decision_model: str | None = None,
        schema: str | None = None,
    ) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        # Shared-database mode: `schema` (or $PROSPECTA_SCHEMA) pins every
        # connection to one schema; refuse to start if it is not writable.
        self._schema = resolve_schema(schema)
        if self._schema is not None:
            check_schema_writable(database_url, self._schema)
            database_url = scoped_url(database_url, self._schema)
        self._database_url = database_url
        self._llm = llm
        self._embed = embed
        # `synth_llm` answers Memory.ask, else `llm`; `jev` is a
        # prospecta.stages.JevScore, the decision model of Stage 0's gate and Stage 3 (None = no Jev).
        self._synth_llm = synth_llm
        self._jev = jev
        # Stage 2 cross-encoder (recall_config.rerank.stage = cross_encoder): its
        # transport (prospecta.stages.openrouter_rerank_transport) and the
        # call-argument model overrides for it and for Jev (the decision model).
        self._rerank_transport = rerank_transport
        self._reranker_model = reranker_model
        self._decision_model = decision_model
        self._default_bank_id = bank_id
        # Shadow reads (embedding migration): recall() also queries this bank
        # with shadow_embed and stores both runs in recall_events.
        if shadow_bank_id is not None:
            validate_bank_id(shadow_bank_id)
            if shadow_embed is None:
                raise ValueError("shadow_bank_id requires shadow_embed")
            if shadow_bank_id == bank_id:
                raise ValueError("shadow_bank_id must differ from bank_id")
        self._shadow_bank_id = shadow_bank_id
        self._shadow_embed = shadow_embed
        self._pool = ConnectionPool(database_url=database_url)
        # T16: tracer fan-out. Default = PostgresSink (canonical persistence
        # path for events). Caller may supply NoOpTracer, RecordingTracer,
        # CompositeTracer, or any Tracer-shaped callable to override.
        if tracer is not None:
            self._tracer = tracer
        else:
            from prospecta.observability import PostgresSink
            self._tracer = PostgresSink(pool=self._pool, bank_id=bank_id)
        # T14: sweeper thread state (lazy; created in start_sweeper).
        self._sweeper_thread = None  # type: ignore[assignment]
        self._sweeper_config = None  # type: ignore[assignment]

    @property
    def schema(self) -> str | None:
        return self._schema

    @property
    def database_url(self) -> str:
        return self._database_url

    @property
    def default_bank_id(self) -> str:
        return self._default_bank_id

    def create_bank(
        self,
        bank_id: str,
        *,
        embedding_dim: int,
        embedding_model_id: str | None = None,
        mission: str | None = None,
        retain_mission: str | None = None,
    ) -> None:
        """Create a bank with the given embedding dimensionality.

        Idempotent for identical configs (same bank_id + same embedding_dim
        is a no-op). Raises BankConfigConflict if bank exists with different
        embedding_dim. A bank that exists is never changed (its stored
        channel_config / recall_config stay as they are).

        A bank created here runs the hybrid architecture out of the box
        (prospecta.bank_defaults): channel_config = [hybrid_chunk] (Stage 1),
        recall_config = a cross_encoder rerank (Stage 2).

        Also creates the bank's partial HNSW index over its chunk items.
        """
        validate_bank_id(bank_id)
        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive int, got {embedding_dim!r}")

        with self._pool.cursor() as cur:
            cur.execute(SELECT_BANK, {"bank_id": bank_id})
            existing = cur.fetchone()

            if existing is not None:
                existing_dim = existing[1]
                if existing_dim != embedding_dim:
                    raise BankConfigConflict(
                        f"Bank {bank_id!r} exists with embedding_dim={existing_dim}, "
                        f"refusing to recreate with embedding_dim={embedding_dim}"
                    )
                logger.debug("Bank %s already exists with matching config", bank_id)
                return

            cur.execute(
                INSERT_BANK,
                {
                    "bank_id": bank_id,
                    "embedding_dim": embedding_dim,
                    "embedding_model_id": embedding_model_id,
                    "mission": mission,
                    "retain_mission": retain_mission,
                    **{k: json.dumps(v) for k, v in new_bank_configs().items()},
                },
            )

        # CREATE INDEX CONCURRENTLY cannot run inside a transaction.
        self._create_hnsw_index(bank_id, embedding_dim)

    def set_channel_config(self, config: "list[dict]", bank_id: str | None = None) -> None:
        """Set a bank's Stage 1 channel (banks.channel_config): `[{"name": "hybrid_chunk", "weight": 1,
        "params": {...}}]`, the slot counts and the HNSW ef_search being the params. `hybrid_chunk` is
        the only channel; `[]` means its defaults (what a bank that never set one runs). Any other name
        is refused (docs/recall-stages.md); a bank that stored one before still recalls, with that
        entry ignored."""
        import json as _json

        from prospecta.channels import validate_channel_config

        bank_id = bank_id or self._default_bank_id
        validate_bank_id(bank_id)
        validate_channel_config(config)
        with self._pool.cursor() as cur:
            cur.execute(
                "UPDATE banks SET channel_config = %s::jsonb, updated_at = now() "
                "WHERE bank_id = %s",
                (_json.dumps(config), bank_id),
            )
            if cur.rowcount == 0:
                raise ValueError(f"No such bank: {bank_id!r}")

    def _emit_stage_calls(self, traces: "list[dict]") -> None:
        """One llm_call event per model call a recall stage made."""
        for t in traces:
            for c in t.get("calls") or []:
                try:
                    self._tracer("llm_call", {"bank_id": self._default_bank_id, **c})
                except Exception:  # pragma: no cover
                    logger.exception("tracer raised on llm_call; ignoring")

    def set_recall_config(self, config: dict, bank_id: str | None = None) -> None:
        """Set a bank's recall stages (banks.recall_config): the cross_encoder rerank, the
        models, the Stage 0 / Stage 3 settings, query expansion, the depth. `{}` turns every stage
        off. A key the design does not read is refused; docs/recall-stages.md lists the keys. (A bank
        that stored a key it no longer reads keeps it: it is ignored.)"""
        import json as _json

        from prospecta.stages import validate_recall_config
        validate_recall_config(config)
        bank_id = bank_id or self._default_bank_id
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE banks SET recall_config = %s::jsonb, updated_at = now() "
                    "WHERE bank_id = %s", (_json.dumps(config), bank_id))
            conn.commit()

    def _create_hnsw_index(self, bank_id: str, embedding_dim: int) -> None:
        """Create the bank's partial HNSW index over its chunk items.

        Uses CREATE INDEX CONCURRENTLY which CANNOT run in a transaction,
        so opens a dedicated autocommit connection. bank_id was validated by
        create_bank; embedding_dim too.
        """
        with psycopg.connect(self._database_url, autocommit=True) as conn:
            ensure_kind_hnsw_indexes(conn, bank_id, embedding_dim)
        logger.info("Created the HNSW index of bank %s", bank_id)

    def bank_stats(self, bank_id: str | None = None) -> BankStats:
        """Return basic counts for a bank (defaults to default_bank_id)."""
        bank_id = bank_id or self._default_bank_id
        validate_bank_id(bank_id)
        with self._pool.cursor() as cur:
            cur.execute(SELECT_BANK_STATS, {"bank_id": bank_id})
            row = cur.fetchone()
            if row is None:
                raise ValueError(f"No such bank: {bank_id!r}")
            return BankStats(
                bank_id=row[0],
                documents=row[1],
                memory_items=row[2],
                last_retain_at=row[3],
            )

    def close(self) -> None:
        """Close the connection pool."""
        self._pool.close()

    # ------------------------------------------------------------------
    # T9 — index/search/remove (delegate to _index)
    # ------------------------------------------------------------------

    def index_directory(self, path, **kwargs):
        """Walk a directory, parse + chunk + embed + index each file.

        See prospecta._index.index_directory for full signature.
        """
        from prospecta import _index
        return _index.index_directory(self, path, **kwargs)

    def index_single_file(self, path, **kwargs):
        """Index ONE file. P7 single write path (used by retain AND sweeper).

        See prospecta._index.index_single_file for full signature.
        """
        from prospecta import _index
        return _index.index_single_file(self, path, **kwargs)

    def search(self, text: str, **kwargs):
        """Mode 2A: search the default bank. Stage 1 (the bank's hybrid_chunk channel) then Stage 2
        (the cross-encoder, when the bank's recall_config asks for it); see prospecta._index.search.
        Each RecalledMemory carries its citation fields and scores {hybrid_chunk, rrf, rerank}."""
        from prospecta import _index
        return _index.search(self, text, **kwargs)

    def remove_documents(self, sources: list[str]) -> int:
        """Explicit removal by caller-supplied source list (P5 substrate-opacity).

        Caller decides which sources are stale. The library does not introspect.
        """
        from prospecta import _index
        return _index.remove_documents(self, sources)

    def retain(self, content: str, **kwargs) -> str:
        """Write a note. Returns document_id (str UUID).

        See prospecta._retain.retain for full signature and semantics:
          - the note is indexed as kind='chunk' items, no LLM call
          - replace-on-source-match (docs/schema.md, "Re-retain")
          - DocumentSourceConflictError on hash collision with different source
        """
        from prospecta import _retain
        return _retain.retain(self, content, **kwargs)

    # ------------------------------------------------------------------
    # Read: recall (many queries), expand_and_recall (Mode 1), ask (Mode 2B)
    # ------------------------------------------------------------------

    def recall(
        self,
        queries: "list[Query] | list[str]",
        *,
        limit: int = 10,
        metadata_filter: dict | None = None,
        rrf_k: int = 60,
        depth: "Literal['standard','deep'] | None" = None,
        date_from: "date | str | None" = None,
        date_to: "date | str | None" = None,
        excluded_sources: "list[str] | None" = None,
        reranker_model: str | None = None,
        decision_model: str | None = None,
        parallel: bool = False,
        _by_query: "list[list[RecalledMemory]] | None" = None,
        _trace: "list[dict] | None" = None,
    ) -> "list[RecalledMemory]":
        """Run retrieval for multiple queries; return FLAT list[RecalledMemory].

        `_trace`, when given, receives the per-query channel traces, so a caller can explain the
        hits. `_by_query` receives each query's own results, in query order.

        `date_from` / `date_to` (the note's date, inclusive) and `excluded_sources`
        (the anti-echo list of documents.source values) are pushed into Stage 1;
        `reranker_model` / `decision_model` override the Stage 2 / Jev model for this call
        (see docs/recall-stages.md). `parallel=True` runs the queries concurrently, each on its
        own connection.

        `depth` ('standard' | 'deep'; None = the bank's recall_config.depth,
        else 'standard') sets the rerank pool cut: 0.15 vs 0.05 x the best fused
        score. It is recorded in recall_events.depth.

        Each query runs its own search(); results concatenated preserving
        query order and intra-query rank. Duplicate memory_item_ids across
        queries are KEPT (caller may use this signal).

        Fires a 'recall' tracer event with the durable results array (migration 0003).
        """
        import time as _time

        coerced: list[Query] = []
        for q in queries:
            if isinstance(q, str):
                coerced.append(Query(text=q))
            else:
                coerced.append(q)

        t_start = _time.monotonic()
        flat: list[RecalledMemory] = []
        traces: list[dict] = []

        def one(q):
            own_trace: list[dict] = []
            return self.search(
                q.text,
                limit=limit,
                metadata_filter=metadata_filter,
                rrf_k=rrf_k,
                _trace=own_trace,
                depth=depth,
                date_from=date_from,
                date_to=date_to,
                excluded_sources=excluded_sources,
                reranker_model=reranker_model,
                decision_model=decision_model,
            ), own_trace

        if parallel and len(coerced) > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(coerced)) as pool:
                ran = list(pool.map(one, coerced))
        else:
            ran = [one(q) for q in coerced]
        for results, own_trace in ran:
            flat.extend(results)
            traces.extend(own_trace)
        if _by_query is not None:
            _by_query.extend(results for results, _ in ran)
        duration_ms = int((_time.monotonic() - t_start) * 1000)
        self._emit_stage_calls(traces)
        if _trace is not None:
            _trace.extend(traces)

        if self._shadow_bank_id is None:
            try:
                self._tracer("recall", {
                    "bank_id": self._default_bank_id,
                    "queries": [q.text for q in coerced],
                    "mode": "hybrid",
                    "n_results": len(flat),
                    "duration_ms": duration_ms,
                    "trace": None,
                    "results": _serialize_recall_results(flat),
                    "synthesis": None,
                    **_channel_trace_payload(traces),
                })
            except Exception:  # pragma: no cover
                logger.exception("tracer raised on recall; ignoring")
        else:
            # With a shadow bank, the old/new PAIR of events replaces the
            # single event (the old side is stored by run_shadow_recall).
            from prospecta._shadow import run_shadow_recall
            run_shadow_recall(
                self, coerced, flat, limit=limit,
                metadata_filter=metadata_filter, rrf_k=rrf_k,
                primary_duration_ms=duration_ms, depth=depth,
            )

        return flat

    def _emit_calls(self, calls: "list[dict]") -> None:
        """One llm_call event per model call record."""
        self._emit_stage_calls([{"calls": calls}])

    def expand_and_recall(
        self,
        prompt: "str | TaskBrief",
        kind: "Literal['chat','task']" = "chat",
        *,
        agent_last: str | None = None,
        project: str | None = None,
        date_from: "date | str | None" = None,
        date_to: "date | str | None" = None,
        excluded_sources: "list[str] | None" = None,
        metadata_filter: dict | None = None,
        deadline: float | None = None,
        max_searches: int | None = None,
        max_concurrency: int | None = None,
        max_tokens: int | None = None,
        query_writer_model: str | None = None,
        decision_model: str | None = None,
        reranker_model: str | None = None,
        decision: "Callable | None" = None,
        gate: bool | None = None,
        needs_memory_threshold: float | None = None,
        min_score: float | None = None,
        depth: "Literal['standard','deep'] | None" = None,
        near_copy_cosine: float | None = None,
        record: bool | None = None,
        now: "date | None" = None,
    ) -> "ExpandedRecall":
        """Mode 1 (docs/postgres-hybrid.md): automatic recall by query expansion, the one
        entry point for a chat message (`kind="chat"`, `prompt` a string, `agent_last` the
        agent's last turn for follow-ups) and for a task (`kind="task"`; `prompt` a TaskBrief
        of title / ask / done / project, or the brief's text).

        1. Gate (chat only): the decision model's `needs_memory` (a noul in 0..1; at or above
           `needs_memory_threshold`, 0.5, the message goes on) skips acknowledgements, go-aheads,
           greetings and live-state questions. `decision(prompt, agent_last)` injects the caller's
           own decision (a number in 0..1, a bool, a GateDecision, or a mapping with needs_memory);
           else Memory(jev=...) is asked; with neither the gate is not run. A failing gate lets the
           message through. `gate=False` turns it off. A task is never gated.
        2. Writer: the query-writer model (`query_writer_model`, else recall_config.models
           .query_writer_model, else PROSPECTA_QUERY_WRITER_MODEL, else
           openrouter/google/gemini-2.5-flash-lite; it is the Memory's `llm`) expands the prompt into
           N searches {query, terms (verbatim proper names, identifiers, versions, paths, error
           codes and "quoted phrases", never common words), date_from, date_to}: about 3 for a
           short chat message, up to 10 to 20 for an elaborate one, never more than `max_searches`
           (20). A term that is a plain English word is dropped unless the prompt quoted it, a
           search left with neither query nor terms is dropped, and when every search is dropped
           nothing is asked (a skipped result, an empty block). Names, identifiers and quoted
           phrases of the prompt that no search carries are put back into the first search. A bad
           reply (not JSON, no search, a timeout) falls back to ONE search: the prompt itself.
        3. Every search runs Stage 1 (the bank's hybrid_chunk channel; its `terms` feed the exact
           retriever, its `query` the semantic and stemmed ones) in parallel, at most
           `max_concurrency` (8, and never more than the pool) at once, each with the same filters:
           `metadata_filter` (scope), `excluded_sources` (anti-echo), and the search's own period, else
           the caller's `date_from` / `date_to`.
        4. Each search's candidates are reranked against its OWN query (the cross-encoder when the
           bank's recall_config.rerank is a cross_encoder stage; `min_score` is its floor); a search
           whose rerank fails keeps Stage 1's order, cut to the candidates within recall_config
           .expansion.fallback_min_rel (0.5) of its best.
        5. The records merge: one record once (best score, `found_by` the searches that found it),
           near-copies collapsed (by wording: token Jaccard over 0.88; then by meaning: chunk
           embedding cosine at or above `near_copy_cosine`, else recall_config.expansion
           .near_copy_cosine, else PROSPECTA_NEAR_COPY_COSINE, else 0.90), the kept record taking
           the searches of the copies and listing them (`collapsed_ids`), best first, the `project`'s
           own notes first and other projects' labeled. Sized by relevance, never by a count;
           `max_tokens` (0 = off) is only a safety valve.
        6. One cited block (`format_cited_rag_block`): the searches that ran, the line telling the
           agent to cite [n], then the numbered items.

        Everything is inside `deadline` seconds (7): a search still running then is dropped. It never
        raises: the returned ExpandedRecall carries `block`, `searches`, per-stage `timings_ms`,
        `fallbacks` and `accounting`. `record` (default: for a task, not for a chat) writes the
        llm_call events and one `recall` event. Defaults for the numbers also come from the bank's
        recall_config (`expansion`: max_searches, max_concurrency, deadline_s, max_tokens,
        fallback_min_rel, near_copy_cosine; `stage0_gate`: enabled, needs_memory_threshold)."""
        from prospecta._expand_run import run_expansion
        return run_expansion(
            self, prompt, kind, agent_last=agent_last, project=project, date_from=date_from,
            date_to=date_to, excluded_sources=excluded_sources, metadata_filter=metadata_filter,
            deadline=deadline, max_searches=max_searches, max_concurrency=max_concurrency,
            max_tokens=max_tokens, query_writer_model=query_writer_model, decision_model=decision_model,
            reranker_model=reranker_model, decision=decision, gate=gate,
            needs_memory_threshold=needs_memory_threshold, min_score=min_score, depth=depth,
            near_copy_cosine=near_copy_cosine, record=record, now=now)

    def recall_for_task(
        self,
        *,
        title: str,
        ask: str,
        done: str | None = None,
        project: str | None = None,
        **options,
    ) -> "ExpandedRecall":
        """Recall for a non-interactive task: `expand_and_recall(TaskBrief(...), kind="task")`,
        the same entry point as a chat message (a thin wrapper that checks the brief). `options`
        are expand_and_recall's (date_from, date_to, excluded_sources, deadline, max_tokens,
        query_writer_model, reranker_model, ...). A task is never gated and its block is sized by
        relevance. The result has `.queries`, `.items` and `.block`."""
        for name, value in (("title", title), ("ask", ask)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} is required (non-empty string)")
        return self.expand_and_recall(
            TaskBrief(title=title, ask=ask, done=done, project=project), kind="task",
            project=project, **options)

    def ask(
        self,
        question: str,
        *,
        limit: int = 8,
        sufficiency: bool | None = None,
        sufficiency_threshold: float | None = None,
        depth: "Literal['standard','deep'] | None" = None,
        metadata_filter: dict | None = None,
        date_from: "date | str | None" = None,
        date_to: "date | str | None" = None,
        excluded_sources: "list[str] | None" = None,
        reranker_model: str | None = None,
        decision_model: str | None = None,
    ) -> "AskResult":
        """Mode 2B: interactive RAG. The question runs Stage 1 + Stage 2 (no needs-memory
        gate: the user asked) and the best `limit` notes go to the synthesis model
        (`synth_llm`, else `llm`), which answers from them alone and cites [n] for every
        statement. AskResult.citations ties each [n] to its source, date and the verbatim
        sentence of the chunk that supports it (plus the exact-match snippet when a lexical
        retriever fired); `rendered` is the answer followed by its sources.

        Stage 3 (off unless `sufficiency=True` or recall_config.stage3_sufficiency.enabled):
        the decision model (`decision_model`, else the Memory's, else recall_config.models
        .decision_model, else PROSPECTA_DECISION_MODEL, else typesafe/jev-1.14; needs
        Memory(jev=...)) scores whether the evidence suffices; under the threshold
        (`sufficiency_threshold`, else recall_config.stage3_sufficiency.threshold, else 0.70)
        ONE broadened retrieval runs (deep rerank pool, twice the limit, the date window
        dropped; excluded_sources and the metadata filter stay), its notes join the
        evidence, and the model answers. A failing gate never blocks the answer: it is
        recorded in AskResult.gate and `sufficient` is None. With no evidence at all the
        answer is `not in memory` and no model is called.
        """
        import time as _time

        from prospecta.cited import build_citations, format_cited_rag_block, render_answer
        from prospecta._llmutil import llm_call_record, llm_text
        from prospecta._template import render_prompt
        from prospecta.stages import STAGE3_THRESHOLD, read_recall_config

        llm = self._synth_llm or self._llm
        if llm is None:
            raise RuntimeError(
                "Memory.ask requires an `llm` callable; construct Memory(llm=...)")
        with self._pool.connection() as conn:
            cfg = read_recall_config(conn, self._default_bank_id)
            conn.rollback()
        gate_cfg = cfg.get("stage3_sufficiency") or {}
        bank_decision_model = (cfg.get("models") or {}).get("decision_model")
        gate_on = gate_cfg.get("enabled", False) if sufficiency is None else sufficiency
        threshold = next(v for v in (sufficiency_threshold, gate_cfg.get("threshold"),
                                     STAGE3_THRESHOLD) if v is not None)
        common = dict(metadata_filter=metadata_filter, reranker_model=reranker_model,
                      decision_model=decision_model)
        items = self.recall([question], limit=limit, depth=depth, date_from=date_from,
                            date_to=date_to, excluded_sources=excluded_sources, **common)
        gate, sufficient, retried = None, None, False
        if gate_on and items:
            gate, sufficient = self._judge_sufficiency(
                question, items, threshold, decision_model, bank_decision_model)
            if sufficient is False:
                retried = True
                wider = self.recall([question], limit=limit * 2, depth="deep",
                                    excluded_sources=excluded_sources, **common)
                seen: set[str] = set()
                items = [r for r in items + wider
                         if not (r.document_id in seen or seen.add(r.document_id))]
                gate, sufficient = self._judge_sufficiency(
                    question, items, threshold, decision_model, bank_decision_model)
        if not items:
            return AskResult(answer="not in memory", citations=[], items=[], sufficient=sufficient,
                             retried=retried, gate=gate, rendered="not in memory")
        prompt = render_prompt("ask-cited", {"question": question,
                                             "evidence": format_cited_rag_block(items, heading=None)})
        t0 = _time.monotonic()
        raw = llm(messages=[{"role": "user", "content": prompt}])
        answer = llm_text(raw).strip()
        self._emit_calls([{
            **llm_call_record(raw, purpose="ask_cited"), "json_mode": False, "messages_count": 1,
            "duration_ms": int((_time.monotonic() - t0) * 1000), "prompt_text": prompt,
            "response_text": answer, "error": None}])
        citations = build_citations(answer, items)
        return AskResult(answer=answer, citations=citations, items=items, sufficient=sufficient,
                         retried=retried, gate=gate, rendered=render_answer(answer, citations))

    def _judge_sufficiency(self, question: str, items: "list[RecalledMemory]", threshold: float,
                           call_model: str | None, bank_model: str | None) -> "tuple[dict, bool | None]":
        """Stage 3 verdict for `items`: ({score, threshold, model} or {error, threshold},
        sufficient or None when the decision model could not be asked)."""
        from prospecta.stages import JevSufficiency, _with_decision_model
        calls: list[dict] = []
        gate: dict = {"threshold": threshold}
        sufficient = None
        try:
            jev = _with_decision_model(self._jev, call_model or self._decision_model, bank_model)
            if jev is None:
                raise RuntimeError("the stage 3 gate needs Memory(jev=...)")
            excerpts = "\n\n".join(
                f"[{n}] {it.source} | {it.created_on or 'unknown'}\n{it.original_chunk}"
                for n, it in enumerate(items, start=1))
            score = JevSufficiency(jev.transport, jev.model, jev.timeout).score(
                question, excerpts, calls)
            gate.update(score=score, model=jev.model)
            sufficient = score >= threshold
        except Exception as exc:
            logger.warning("stage 3 sufficiency gate failed (%s: %s); answering from the "
                           "evidence found", type(exc).__name__, exc)
            gate["error"] = f"{type(exc).__name__}: {exc}"
        self._emit_calls(calls)
        return gate, sufficient

    # ------------------------------------------------------------------
    # The background sweeper (P14 filesystem drift safety net)
    # ------------------------------------------------------------------

    def start_sweeper(
        self,
        *,
        corpus_paths,
        sweep_interval_seconds: float = 86400.0,
        file_extensions: tuple = (".md", ".markdown"),
        ignore_patterns: list | None = None,
    ) -> None:
        """Start the background sweeper. Raises RuntimeError if already running.

        Sweeper is the implicit-file-change write path (P14 safety net). Per
        the substrate-opacity rule, sweeper is the legitimate home for
        filesystem awareness inside the library.
        """
        from pathlib import Path
        from prospecta._sweeper import SweeperConfig, SweeperThread

        if self._sweeper_thread is not None and self._sweeper_thread.is_alive():
            raise RuntimeError("sweeper is already running; call stop_sweeper first")

        config = SweeperConfig(
            corpus_paths=[Path(p) for p in corpus_paths],
            sweep_interval_seconds=float(sweep_interval_seconds),
            file_extensions=tuple(file_extensions),
            ignore_patterns=list(ignore_patterns) if ignore_patterns else None,
        )
        self._sweeper_config = config
        thread = SweeperThread(self, config)
        self._sweeper_thread = thread
        thread.start()

    def stop_sweeper(self, *, timeout: float = 5.0) -> bool:
        """Stop the background sweeper. Returns True iff thread exited cleanly.

        Idempotent: if no sweeper is running, returns True.
        """
        thread = self._sweeper_thread
        if thread is None:
            return True
        ok = thread.stop(timeout=timeout)
        if ok:
            self._sweeper_thread = None
        return ok

    def shutdown(self) -> None:
        """Stop sweeper (if running) + close tracer + pool. Idempotent."""
        try:
            if self._sweeper_thread is not None:
                self.stop_sweeper(timeout=5.0)
        finally:
            close_fn = getattr(self._tracer, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:  # pragma: no cover
                    logger.exception("tracer.close() raised; ignoring")
            self.close()
