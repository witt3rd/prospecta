"""Mode 1 of docs/postgres-hybrid.md: automatic recall by query expansion (the run).

`run_expansion` is what `Memory.expand_and_recall` does, for a chat message and for a task alike:

  gate (chat only)  ->  writer: the prompt becomes N structured searches  ->  every search through
  Stage 1 (hybrid_chunk) and Stage 2 (cross_encoder, against its OWN query), all in parallel  ->
  one merge  ->  one cited block.

Everything runs inside the caller's deadline and nothing is raised into the caller: a step that
fails or runs late degrades (the reason goes into `ExpandedRecall.fallbacks`), the rest goes on.

A search is one thread end to end: embed its query (no database connection held), Stage 1 on a
borrowed connection (a semaphore keeps at most min(max_concurrency, pool size) searches in the
database at once), then its rerank with the connection already given back. So the slow remote
steps (the embedder, the reranker) never occupy the pool, and 20 searches cost about the slowest
one, not a queue of them."""
from __future__ import annotations

import dataclasses
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import date
from typing import Any, Callable

from prospecta import _expand as ex
from prospecta._filters import today as _today
from prospecta._llmutil import llm_call_record, llm_text
from prospecta._scorecut import check_depth, estimate_tokens, pool_min_rel, rel_cut, resolve_depth
from prospecta._types import (
    ExpandedRecall, GateDecision, RecalledMemory, SearchRun, SearchSpec, TaskBrief,
)
from prospecta.channels import QueryPlan, RecallState, read_channel_config, run_channels
from prospecta.cited import HEADING, TASK_HEADING, document_citations, format_cited_rag_block
from prospecta.stages import (
    CrossEncoderReranker, RERANK_MIN_SCORE, RERANK_TIMEOUT_S, _with_decision_model, _with_scores,
    build_items, read_recall_config, resolve_reranker_model, run_reranker, totals,
)

logger = logging.getLogger("prospecta.expand")


def _ms(since: float) -> int:
    return int((time.monotonic() - since) * 1000)


def _call_within(fn: Callable, timeout: float):
    """fn() on a worker thread, waited for at most `timeout` seconds (TimeoutError after that;
    the thread is left to finish and its result is dropped)."""
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prospecta-expand-call")
    fut = pool.submit(fn)
    try:
        return fut.result(timeout=max(timeout, 0.0))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _day(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value.strip()[:10]).isoformat()
        except ValueError:
            pass
    raise ValueError(f"{name} must be a date or a YYYY-MM-DD string")


def _pick(call: Any, bank: Any, default: Any) -> Any:
    """The per-call argument, else the bank's recall_config value, else the default."""
    return call if call is not None else (bank if bank is not None else default)


def _clean(text: str) -> str:
    return " ".join(text.split())


@dataclasses.dataclass
class _Out:
    """What one search thread hands back (the coordinator applies it to the SearchRun)."""
    items: list[RecalledMemory] = dataclasses.field(default_factory=list)
    trace: dict | None = None
    calls: list[dict] = dataclasses.field(default_factory=list)
    ms: dict = dataclasses.field(default_factory=dict)
    n_candidates: int = 0
    reranked: bool = False
    fallback_reason: str | None = None
    error: str | None = None


@dataclasses.dataclass
class _Plan:
    """The settings of one run, resolved: per-call argument, else bank, else default."""
    bank_id: str
    cfg: dict
    entry: dict                      # the bank's hybrid_chunk channel entry
    deadline: float
    deadline_at: float
    max_searches: int
    max_concurrency: int
    max_tokens: int
    fallback_rel: float
    rerank_on: bool
    rr_model: str
    rr_min_score: float
    rr_timeout: float
    rr_dedupe: bool
    depth: str | None
    excluded: list[str] | None
    meta: dict | None
    window: tuple[str | None, str | None]


def run_expansion(memory, prompt, kind: str = "chat", *, agent_last: str | None = None,
                  project: str | None = None, date_from=None, date_to=None,
                  excluded_sources: list[str] | None = None, metadata_filter: dict | None = None,
                  deadline: float | None = None, max_searches: int | None = None,
                  max_concurrency: int | None = None, max_tokens: int | None = None,
                  query_writer_model: str | None = None, decision_model: str | None = None,
                  reranker_model: str | None = None, decision: Callable | None = None,
                  gate: bool | None = None, needs_memory_threshold: float | None = None,
                  min_score: float | None = None, depth: str | None = None,
                  record: bool | None = None, now: date | None = None) -> ExpandedRecall:
    """See Memory.expand_and_recall (the arguments are its own). Never raises."""
    run = _Run(memory, kind)
    try:
        return run.go(prompt, agent_last=agent_last, project=project, date_from=date_from,
                      date_to=date_to, excluded_sources=excluded_sources,
                      metadata_filter=metadata_filter, deadline=deadline, max_searches=max_searches,
                      max_concurrency=max_concurrency, max_tokens=max_tokens,
                      query_writer_model=query_writer_model, decision_model=decision_model,
                      reranker_model=reranker_model, decision=decision, gate=gate,
                      needs_memory_threshold=needs_memory_threshold, min_score=min_score,
                      depth=depth, record=record, now=now)
    except Exception as exc:        # never into the caller
        logger.warning("query expansion failed (%s: %s); nothing recalled", type(exc).__name__, exc,
                       exc_info=True)
        return run.result(error=f"{type(exc).__name__}: {exc}")


class _Run:
    def __init__(self, memory, kind: str):
        self.memory, self.requested = memory, kind
        self.kind = kind if kind in ("chat", "task") else "chat"
        self.t0 = time.monotonic()
        self.fallbacks: list[str] = []
        self.calls: list[dict] = []
        self.timings: dict[str, int] = {}
        self.searches: list[SearchSpec] = []
        self.runs: list[SearchRun] = []
        self.gate: dict | None = None
        self.deadline_hit = False

    def result(self, block: str = "", items=(), *, skipped: bool = False,
               skipped_reason: str | None = None, error: str | None = None,
               dropped: int = 0) -> ExpandedRecall:
        self.timings["total"] = _ms(self.t0)
        return ExpandedRecall(
            kind=self.kind, block=block, items=list(items), searches=list(self.searches),
            runs=list(self.runs), skipped=skipped, skipped_reason=skipped_reason, gate=self.gate,
            fallbacks=list(self.fallbacks), timings_ms=dict(self.timings),
            accounting=totals(self.calls), calls=list(self.calls), deadline_hit=self.deadline_hit,
            dropped_for_tokens=dropped, error=error)

    # ------------------------------------------------------------------ the run

    def go(self, prompt, *, agent_last, project, date_from, date_to, excluded_sources,
           metadata_filter, deadline, max_searches, max_concurrency, max_tokens, query_writer_model,
           decision_model, reranker_model, decision, gate, needs_memory_threshold, min_score,
           depth, record, now) -> ExpandedRecall:
        memory = self.memory
        if self.requested not in ("chat", "task"):
            return self.result(error="kind must be 'chat' or 'task'")
        check_depth(depth)
        brief = prompt if isinstance(prompt, TaskBrief) else None
        if brief is not None:
            self.kind = "task"
            text = ex.brief_text(brief)
            own_text = guard_text = "\n".join(p for p in (brief.title, brief.ask, brief.done) if p)
            project = project or brief.project
        else:
            text = own_text = guard_text = prompt if isinstance(prompt, str) else ""
        if not text.strip():
            return self.result(skipped=True, skipped_reason="empty prompt")
        kind = self.kind
        today = now or _today()

        # ---- settings: per call, else the bank's recall_config, else the default
        bank_id = memory._default_bank_id
        t = time.monotonic()
        with memory._pool.connection() as conn:
            cfg = read_recall_config(conn, bank_id)
            channel_config = read_channel_config(conn, bank_id)
            conn.rollback()
        self.timings["config"] = _ms(t)
        xcfg, g0, rr = cfg.get("expansion") or {}, cfg.get("stage0_gate") or {}, cfg.get("rerank") or {}
        models = cfg.get("models") or {}
        deadline = float(_pick(deadline, xcfg.get("deadline_s"), ex.DEFAULT_DEADLINE_S))
        max_searches = int(_pick(max_searches, xcfg.get("max_searches"), ex.DEFAULT_MAX_SEARCHES))
        max_concurrency = int(_pick(max_concurrency, xcfg.get("max_concurrency"), ex.DEFAULT_MAX_CONCURRENCY))
        max_tokens = int(_pick(max_tokens, xcfg.get("max_tokens"), 0))
        if deadline <= 0 or max_searches < 1 or max_concurrency < 1 or max_tokens < 0:
            raise ValueError("deadline must be > 0, max_searches and max_concurrency >= 1, max_tokens >= 0")
        threshold = float(_pick(needs_memory_threshold, g0.get("needs_memory_threshold"),
                                ex.NEEDS_MEMORY_THRESHOLD))
        use_gate = kind == "chat" and bool(_pick(gate, g0.get("enabled"), True))
        record = (kind == "task") if record is None else bool(record)
        deadline_at = self.t0 + deadline
        entry = next((e for e in channel_config
                      if e.get("name") == "hybrid_chunk" and e.get("enabled", True)), None)
        if entry is None:
            return self.result(error="the bank's channel_config has no hybrid_chunk channel: query "
                                     "expansion runs on Stage 1 (add {\"name\": \"hybrid_chunk\", "
                                     "\"weight\": 1})")
        if memory._embed is None:
            return self.result(error="query expansion needs an `embed` callable; construct "
                                     "Memory(embed=...)")
        plan = _Plan(
            bank_id=bank_id, cfg=cfg, entry=entry, deadline=deadline, deadline_at=deadline_at,
            max_searches=max_searches, max_concurrency=max_concurrency, max_tokens=max_tokens,
            fallback_rel=float(_pick(None, xcfg.get("fallback_min_rel"), ex.FALLBACK_MIN_REL)),
            rerank_on=bool(rr.get("enabled") and rr.get("stage") == "cross_encoder"
                           and memory._rerank_transport is not None),
            rr_model=resolve_reranker_model(reranker_model or memory._reranker_model, rr.get("model")),
            rr_min_score=float(_pick(min_score, rr.get("min_score"), RERANK_MIN_SCORE)),
            rr_timeout=float(rr.get("timeout_s", RERANK_TIMEOUT_S)),
            rr_dedupe=bool(rr.get("deduplicate_near_copies", True)), depth=depth,
            excluded=excluded_sources, meta=metadata_filter,
            window=(_day(date_from, "date_from"), _day(date_to, "date_to")))

        # ---- stage 0 gate (chat messages only)
        if use_gate:
            skip = self._gate(text, agent_last, decision, decision_model, models, threshold, plan)
            if skip is not None:
                if record:
                    memory._emit_calls(self.calls)
                return self.result(skipped=True, skipped_reason=skip)

        # ---- the writer: the prompt becomes the searches
        writer_model = ex.resolve_query_writer_model(query_writer_model, models.get("query_writer_model"))
        specs, why = self._write(text, kind, agent_last, today, writer_model, plan)
        if why is None:
            specs, cut = ex.apply_maximum(specs, max_searches)
            if cut is not None:
                self.fallbacks.append(f"writer wrote {cut} searches: cut to the maximum of {max_searches}")
        else:
            self.fallbacks.append(f"writer: {why}: one search with the prompt itself")
            specs = [SearchSpec(query=_clean(own_text))]
        specs, put_back = ex.guard_terms(specs, guard_text)
        if put_back:
            self.fallbacks.append("verbatim terms missing from every search put back into search 1: "
                                  + ", ".join(put_back))
        self.searches.extend(specs)
        self.runs.extend(SearchRun(index=i, spec=s) for i, s in enumerate(specs, start=1))
        if not plan.rerank_on:
            self.fallbacks.append("stage 2 (cross_encoder) is not configured for this bank: Stage 1 "
                                  "order stands for every search")

        # ---- stage 1 + stage 2, every search in parallel, inside the deadline
        outs = self._search_all(plan)

        # ---- merge, scope, valve, block
        t = time.monotonic()
        ok = [r for r in self.runs if r.status == "ok"]
        merged = ex.scope_first(ex.merge_records({r.index: outs[r.index].items for r in ok}), project)
        dropped = 0
        if plan.max_tokens > 0:
            kept: list[RecalledMemory] = []
            used = 0
            for it in merged:
                cost = estimate_tokens(it.original_chunk) + 40      # the chunk and its header
                if kept and used + cost > plan.max_tokens:
                    break
                kept.append(it)
                used += cost
            dropped, merged = len(merged) - len(kept), kept
        block = format_cited_rag_block(
            merged, heading=TASK_HEADING if kind == "task" else HEADING, own_project=project,
            searches=[(r.index, r.spec) for r in ok], cite=True, omitted=dropped)
        self.timings["merge"] = _ms(t)
        if record:
            self._record(merged, [outs[r.index].trace for r in ok if outs[r.index].trace])
        return self.result(block, merged, dropped=dropped)

    # ------------------------------------------------------------------ stage 0

    def _gate(self, text, agent_last, decision, decision_model, models, threshold, plan) -> str | None:
        """None to go on, else why the whole expansion is skipped."""
        memory = self.memory
        decide = decision
        if decide is None and memory._jev is not None:
            jev = _with_decision_model(memory._jev, decision_model or memory._decision_model,
                                       models.get("decision_model"))
            decide = ex.jev_gate(jev.transport, jev.model, jev.timeout)
        if decide is None:
            self.fallbacks.append("gate: no decision callable and no Memory(jev=...): expanded "
                                  "without the stage 0 gate")
            return None
        t = time.monotonic()
        rec = {"purpose": "expand_gate", "model": None, "tokens_in": None, "tokens_out": None,
               "cost_usd": None, "json_mode": False, "messages_count": 1, "prompt_text": None,
               "response_text": None, "error": None}
        verdict: GateDecision | None = None
        try:
            verdict = ex.normalize_decision(_call_within(
                lambda: decide(text, agent_last),
                min(ex.GATE_SHARE * plan.deadline, plan.deadline_at - time.monotonic())))
            rec.update(model=verdict.model, tokens_in=verdict.tokens_in,
                       tokens_out=verdict.tokens_out, cost_usd=verdict.cost_usd)
        except Exception as exc:
            name = type(exc).__name__
            rec["error"] = f"{name}: {exc}" if str(exc) else name
            self.fallbacks.append(f"gate failed ({rec['error']}): expanded without it")
            self.gate = {"error": rec["error"], "threshold": threshold}
        rec["duration_ms"] = self.timings["gate"] = _ms(t)
        self.calls.append(rec)
        if verdict is None:
            return None
        self.gate = {"needs_memory": verdict.needs_memory, "threshold": threshold, "skip": verdict.skip,
                     "model": verdict.model, "reason": verdict.reason}
        if verdict.skip or verdict.needs_memory < threshold:
            return verdict.reason or f"the gate: needs_memory {verdict.needs_memory:.2f} < {threshold:g}"
        return None

    # ------------------------------------------------------------------ the writer

    def _write(self, text, kind, agent_last, today, model, plan) -> tuple[list[SearchSpec], str | None]:
        """(the writer's searches, None) or ([], why the writer gave none)."""
        llm = self.memory._llm
        if llm is None:
            return [], "no query-writer model (construct Memory(llm=...))"
        from prospecta.channels.extract import _accepts_model
        prompt = ex.render_writer_prompt(text, kind=kind, agent_last=agent_last if kind == "chat" else None,
                                         max_searches=plan.max_searches, today=today)
        budget = max(0.1, min(ex.WRITER_SHARE * plan.deadline,
                              plan.deadline_at - time.monotonic() - ex.SEARCH_RESERVE * plan.deadline))
        t = time.monotonic()
        rec: dict = {"purpose": "expand_searches", "model": model, "tokens_in": None, "tokens_out": None,
                     "cost_usd": None, "json_mode": True, "messages_count": 1, "prompt_text": prompt,
                     "response_text": None, "error": None}
        kw = {"model": model} if _accepts_model(llm) else {}
        try:
            raw = _call_within(lambda: llm(messages=[{"role": "user", "content": prompt}],
                                           json_mode=True, **kw), budget)
            rec.update({k: v for k, v in llm_call_record(raw, purpose="expand_searches", model=model).items()
                        if k != "purpose"})
            rec["response_text"] = llm_text(raw)
            specs, why = ex.parse_searches(rec["response_text"], today)
        except Exception as exc:
            name = type(exc).__name__
            rec["error"] = f"{name}: {exc}" if str(exc) else name
            specs, why = [], f"the writer failed ({rec['error']})"
        rec["duration_ms"] = self.timings["writer"] = _ms(t)
        self.calls.append(rec)
        return specs, why

    # ------------------------------------------------------------------ stages 1 and 2

    def _search_all(self, plan: _Plan) -> dict[int, _Out]:
        memory = self.memory
        cancelled = threading.Event()
        db = threading.BoundedSemaphore(max(1, min(plan.max_concurrency, memory._pool.max_size)))
        t = time.monotonic()
        pool = ThreadPoolExecutor(max_workers=max(1, len(self.runs)), thread_name_prefix="prospecta-expand")
        futs = {pool.submit(_search, memory, r.spec, plan, cancelled, db): r for r in self.runs}
        done, pending = wait(futs, timeout=max(0.0, plan.deadline_at - time.monotonic()))
        cancelled.set()
        pool.shutdown(wait=False, cancel_futures=True)
        outs: dict[int, _Out] = {}
        for fut in done:
            run = futs[fut]
            try:
                outs[run.index] = fut.result()
            except Exception as exc:        # _search catches its own: a bug, not a crash
                outs[run.index] = _Out(error=f"{type(exc).__name__}: {exc}")
        for fut in pending:
            run = futs[fut]
            run.status = "late"
            self.deadline_hit = True
            self.fallbacks.append(f"search {run.index} ({run.spec.query!r}) was still running at the "
                                  f"{plan.deadline:g} s deadline: dropped")
        for run in self.runs:
            out = outs.get(run.index)
            if out is None:
                continue
            run.n_candidates, run.reranked, run.ms = out.n_candidates, out.reranked, out.ms
            run.fallback_reason, run.error = out.fallback_reason, out.error
            run.status = "error" if out.error else "ok"
            run.n_kept = 0 if out.error else len(out.items)
            self.calls.extend(out.calls)
            if out.error:
                self.fallbacks.append(f"search {run.index} ({run.spec.query!r}) failed: {out.error}")
            elif out.fallback_reason and plan.rerank_on:
                self.fallbacks.append(f"search {run.index} ({run.spec.query!r}) stage 2: "
                                      f"{out.fallback_reason}: Stage 1 order")
        self.timings["searches"] = _ms(t)
        for stage in ("embed", "stage1", "rerank"):
            self.timings[stage] = max((o.ms.get(stage, 0) for o in outs.values()), default=0)
        self.timings["slowest_search"] = max((o.ms.get("total", 0) for o in outs.values()), default=0)
        return outs

    # ------------------------------------------------------------------ events

    def _record(self, merged: list[RecalledMemory], traces: list[dict]) -> None:
        """The llm_call events and one `recall` event (queries = the searches' queries)."""
        from prospecta.memory import _channel_trace_payload, _serialize_recall_results
        memory = self.memory
        memory._emit_calls(self.calls)
        if memory._shadow_bank_id is not None:
            return
        try:
            memory._tracer("recall", {
                "bank_id": memory._default_bank_id, "queries": [s.query for s in self.searches],
                "mode": "hybrid", "n_results": len(merged), "duration_ms": _ms(self.t0), "trace": None,
                "results": _serialize_recall_results(merged), "synthesis": None,
                **_channel_trace_payload(traces)})
        except Exception:  # pragma: no cover
            logger.exception("tracer raised on recall; ignoring")


def _search(memory, spec: SearchSpec, p: _Plan, cancelled: threading.Event, db) -> _Out:
    """One search, end to end, on its own thread. Never raises (a failure is `out.error`)."""
    from prospecta._index import recalled_from_fused
    t_start = time.monotonic()
    out = _Out()
    try:
        # ---- embed: no connection is held while the embedder is asked
        t = time.monotonic()
        qvec = list(memory._embed([spec.query])[0])
        out.ms["embed"] = _ms(t)
        if cancelled.is_set():
            return out
        # ---- stage 1: a connection for the database work only
        window = (spec.date_from, spec.date_to) if (spec.date_from or spec.date_to) else p.window
        with db:
            if cancelled.is_set():
                return out
            t = time.monotonic()
            with memory._pool.connection() as conn:
                left_ms = max(100, int((p.deadline_at - time.monotonic()) * 1000))
                conn.execute(f"SET LOCAL statement_timeout = {left_ms}")     # ends with the transaction
                state = RecallState(conn=conn, bank_id=p.bank_id, embed=memory._embed,
                                    metadata_filter=p.meta, date_from=window[0], date_to=window[1],
                                    excluded_sources=p.excluded)
                state._embeddings[spec.query] = qvec
                fused, tr = run_channels(
                    state, QueryPlan(text=spec.query, exact_text=ex.exact_query(spec.terms)), [p.entry])
                chan_error = tr["channels"][0]["error"] if tr["channels"] else None
                pool = rel_cut(fused, lambda d: d.score, pool_min_rel(p.depth, p.cfg))
                items = build_items(conn, pool) if pool else []
                cited = document_citations(conn, [d.document_id for d in pool])
                conn.rollback()
            out.ms["stage1"] = _ms(t)
        out.n_candidates = len(fused)
        if chan_error:
            out.error = f"stage 1: {chan_error}"
            return out
        # ---- stage 2: against this search's own query, no connection held
        calls: list[dict] = []
        rec, outcome = None, None
        if items and p.rerank_on and not cancelled.is_set():
            t = time.monotonic()
            reranker = CrossEncoderReranker(
                memory._rerank_transport, p.rr_model, top_n=len(items), min_score=p.rr_min_score,
                timeout=max(0.2, min(p.rr_timeout, p.deadline_at - time.monotonic())),
                deduplicate=p.rr_dedupe)
            ordered, outcome, rec = run_reranker(reranker, spec.query, items, calls)
            out.ms["rerank"] = _ms(t)
            if outcome is None:
                out.fallback_reason = rec["fallback_reason"]
        elif items and not p.rerank_on:
            out.fallback_reason = "stage 2 is not configured"
        if outcome is not None:
            out.reranked = True
            docs = [_with_scores(it.doc, outcome) for it in ordered]
        else:       # Stage 1 order stands, cut by relevance to its best, never by count
            docs = [it.doc for it in rel_cut(items, lambda it: it.doc.score, p.fallback_rel)]
        recalled = recalled_from_fused(
            docs, p.bank_id, state.channel_lists, [p.entry["name"]],
            {"rerank": {"enabled": True, "stage": "cross_encoder"}} if out.reranked else None)
        out.items = [dataclasses.replace(
            r, created_on=cited.get(r.document_id, (None, None))[0],
            kind=cited.get(r.document_id, (None, None))[1],
            relevance_score=r.score if r.relevance_score is None else r.relevance_score)
            for r in recalled]
        tr.update(rerank=rec, hops=None, calls=calls, accounting=totals(calls),
                  depth=resolve_depth(p.depth, p.cfg))
        out.trace, out.calls = tr, calls
    except Exception as exc:
        out.error = f"{type(exc).__name__}: {exc}"
        logger.warning("search %r failed (%s: %s)", spec.query, type(exc).__name__, exc)
    finally:
        out.ms["total"] = _ms(t_start)
    return out
