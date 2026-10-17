"""Mode 1B of docs/postgres-hybrid.md: recall for a non-interactive task.

A task brief (title, ask, done criteria, project) has no conversation to
resolve: it is split into 2-4 facet queries by the query-writer model, every
facet runs Stage 1 + Stage 2, and the notes merge into one cited block, the
task's own project first and other projects labeled."""
from __future__ import annotations

import os
import re
import time

from prospecta._llmutil import llm_call_record, llm_text
from prospecta._template import render_prompt

DEFAULT_QUERY_WRITER_MODEL = "openrouter/google/gemini-2.5-flash-lite"
MIN_FACETS = 2
MAX_FACETS = 4

_QUOTED = re.compile(r'"[^"\n]+"')


def resolve_query_writer_model(call: str | None = None, bank: str | None = None) -> str:
    """The query-writer model: per-call, else the bank's recall_config.models.query_writer_model,
    else PROSPECTA_QUERY_WRITER_MODEL, else the default."""
    return (call or bank or os.environ.get("PROSPECTA_QUERY_WRITER_MODEL")
            or DEFAULT_QUERY_WRITER_MODEL)


def brief_queries(title: str, ask: str, done: str | None) -> list[str]:
    """The brief's own facets, verbatim: its title, its ask, its done criteria. What is
    searched when no query-writer model is available or its answer is unusable."""
    return list(dict.fromkeys(q.strip() for q in (title, ask, done) if q and q.strip()))


def decompose_brief(title: str, ask: str, done: str | None, project: str | None, *,
                    llm, model: str, max_facets: int = MAX_FACETS) -> tuple[list[str], list[dict]]:
    """(the facet queries, the model call records). The model's answer is cleaned: duplicates
    dropped, cut to `max_facets`, topped up from the brief when it gives fewer than
    MIN_FACETS, and every "quoted string" of the brief is kept verbatim in some query
    (appended to the first one when the model dropped it). Without a usable answer the
    queries are the brief's own (brief_queries)."""
    from prospecta.channels.extract import _accepts_model
    from prospecta.stages import parse_json_object
    own = brief_queries(title, ask, done)
    if llm is None:
        return own[:max_facets], []
    prompt = render_prompt("task-facets", {"title": title, "ask": ask, "done": done,
                                           "project": project, "max_facets": max_facets})
    t0 = time.monotonic()
    rec: dict = {"purpose": "task_facets", "model": model, "tokens_in": None, "tokens_out": None,
                 "cost_usd": None, "json_mode": True, "messages_count": 1,
                 "prompt_text": prompt, "response_text": None, "error": None}
    try:
        raw = llm(messages=[{"role": "user", "content": prompt}], json_mode=True,
                  **({"model": model} if _accepts_model(llm) else {}))
        rec.update({k: v for k, v in llm_call_record(raw, purpose="task_facets", model=model).items()
                    if k != "purpose"})
        rec["response_text"] = llm_text(raw)
        asked = parse_json_object(rec["response_text"]).get("queries")
        queries = list(dict.fromkeys(q.strip() for q in asked or [] if isinstance(q, str) and q.strip()))
    except Exception as exc:
        rec["error"] = f"{type(exc).__name__}: {exc}"
        queries = []
    finally:
        rec["duration_ms"] = int((time.monotonic() - t0) * 1000)
    queries = queries[:max_facets]
    if not queries:
        return own[:max_facets], [rec]
    for q in own:
        if len(queries) >= MIN_FACETS:
            break
        if q not in queries:
            queries.append(q)
    for quoted in _QUOTED.findall(f"{title}\n{ask}\n{done or ''}"):
        if not any(quoted in q for q in queries):
            queries[0] += f" {quoted}"
    return queries, [rec]


def scope_first(items: list, project: str | None) -> list:
    """The notes of the task's own project first, then those without a project, then
    other projects' notes; each group keeps its relevance order."""
    if not project:
        return list(items)

    def group(item) -> int:
        p = (item.metadata or {}).get("project")
        return 0 if p == project else (1 if not p else 2)
    return sorted(items, key=group)
