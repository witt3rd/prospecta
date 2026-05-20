"""Read-side bilateral spine: synthesize() — pure function over chunks.

Composes RAG synthesis from retrieved chunks via the caller-supplied LLM.

P-principles enforced here:
  P1 — read-half of bilateral spine: chunks → LLM → synthesis.
  P3 — zero provider imports; uses caller-supplied LLMCallable.
  P4 — caller can override the default prompt.
  P5 — chunks passed through FULL (no truncation).

Memory-free by design; the Memory wrapper composes formulate→recall→synthesize.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from prospecta._template import render_prompt, render_prompt_text

if TYPE_CHECKING:
    from prospecta._types import LLMCallable, RecalledMemory


def _format_chunks_as_context(chunks: "list[RecalledMemory]") -> str:
    """Render chunks into a string suitable for the default prompt's
    {{ context }} variable. Preserves FULL content (P5)."""
    if not chunks:
        return "(no chunks retrieved)"
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        header = f"--- chunk {i} (source={c.source!r}, score={c.score:.4f}) ---"
        parts.append(f"{header}\n{c.original_chunk}")
    return "\n\n".join(parts)


def synthesize(
    query: str,
    chunks: "list[RecalledMemory]",
    llm: "LLMCallable",
    *,
    prompt_override: str | None = None,
    context: dict | None = None,
) -> tuple[str, str]:
    """Compose RAG synthesis from retrieved chunks via the LLM.

    Returns ``(synthesis, rendered_prompt)``. The prompt is returned so the
    Memory wrapper can thread it through to the llm_calls tracer payload
    (migration 0003).

    Uses prospecta/prompts/rag-synthesize.md by default. Caller can override
    via prompt_override (rendered inline as Jinja2 with the same vars).

    Both forms receive:
      - query: str
      - chunks: list[RecalledMemory]
      - context: rendered chunk-context string (default-prompt variable)
      - any caller-supplied keys in `context` dict

    The LLM call is NOT json_mode (synthesis is free text).
    """
    chunk_context = _format_chunks_as_context(chunks)
    variables: dict = {
        "query": query,
        "chunks": chunks,
        "context": chunk_context,
    }
    if context:
        # Caller-supplied context dict can override defaults (e.g., custom
        # 'context' replacing the auto-rendered string). Caller wins.
        variables.update(context)

    if prompt_override is not None:
        rendered = render_prompt_text(prompt_override, variables)
    else:
        rendered = render_prompt("rag-synthesize", variables)

    synthesis = llm(messages=[{"role": "user", "content": rendered}])
    return synthesis, rendered
