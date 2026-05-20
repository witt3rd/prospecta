"""Read-side bilateral spine: formulate_queries — pure function over LLM.

LLM-mediated multi-query expansion (JSON-mode). Library NEVER raises on
malformed JSON from the LLM; falls back to a single-query echo of the
input message. Parse-fallback is canonical observability (schema.md §13).

P-principles enforced here:
  P1 — read-half of bilateral spine: message → LLM → list[Query].
  P3 — zero provider imports; uses caller-supplied LLMCallable.
  P4 — caller can override the default prompt entirely.
  P5 — full message preserved on fallback (becomes single Query.text).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from prospecta._template import render_prompt, render_prompt_text
from prospecta._types import Query

if TYPE_CHECKING:
    from prospecta._types import LLMCallable


logger = logging.getLogger("prospecta._formulate")


ErrorKind = Literal["malformed_json", "schema_mismatch"]


@dataclass(frozen=True)
class FormulateOutcome:
    """Observability payload for the caller (Memory) to log on the event row.

    Attributes:
      parse_fallback — True if JSON parse or schema validation failed and
        the library fell back to [Query(text=message)].
      raw_response — verbatim LLM output (P5: preserved exactly as returned).
      error_kind — 'malformed_json' | 'schema_mismatch' | None.
    """

    parse_fallback: bool
    raw_response: str
    error_kind: ErrorKind | None
    prompt: str = ""
    """The rendered prompt text sent to the LLM (verbatim). Captured here
    so the Memory wrapper can thread it through to the llm_calls tracer
    payload (migration 0003)."""


def formulate_queries(
    message: str,
    *,
    llm: "LLMCallable",
    context: str | None = None,
    prompt_override: str | None = None,
    extra_context: dict | None = None,
) -> tuple[list[Query], FormulateOutcome]:
    """Generate multi-query expansion via a JSON-mode LLM call.

    Returns (queries, outcome). On parse failure returns
    ([Query(text=message)], outcome with parse_fallback=True). The library
    NEVER raises on malformed LLM JSON — robustness is load-bearing.
    """
    variables: dict = {
        "message": message,
        "context": context,
    }
    if extra_context:
        variables.update(extra_context)

    if prompt_override is not None:
        rendered = render_prompt_text(prompt_override, variables)
    else:
        rendered = render_prompt("formulate-queries", variables)

    raw = llm(messages=[{"role": "user", "content": rendered}], json_mode=True)
    if not isinstance(raw, str):
        # Defensive: LLMCallable contract says str; if a stub returns
        # something else, treat it as malformed.
        raw = str(raw)

    queries, error_kind = _parse(raw)
    if error_kind is not None:
        logger.warning(
            "formulate_queries: parse fallback (%s); using original message",
            error_kind,
        )
        return [Query(text=message)], FormulateOutcome(
            parse_fallback=True, raw_response=raw, error_kind=error_kind,
            prompt=rendered,
        )

    return queries, FormulateOutcome(
        parse_fallback=False, raw_response=raw, error_kind=None,
        prompt=rendered,
    )


def _parse(raw: str) -> tuple[list[Query], ErrorKind | None]:
    """Parse JSON-mode LLM output → list[Query].

    Returns (queries, error_kind). error_kind is None on success.
    On any structural mismatch returns ([], 'schema_mismatch').
    On malformed JSON returns ([], 'malformed_json').
    Empty `queries` arrays (after filtering) are treated as schema_mismatch.
    """
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return [], "malformed_json"

    if not isinstance(parsed, dict):
        return [], "schema_mismatch"
    items = parsed.get("queries")
    if not isinstance(items, list):
        return [], "schema_mismatch"

    queries: list[Query] = []
    for entry in items:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if not stripped:
            continue
        queries.append(Query(text=stripped))

    if not queries:
        return [], "schema_mismatch"

    return queries, None
