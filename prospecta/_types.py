"""Public type definitions for prospecta.

All Protocols and dataclasses defined here are the public API surface.
See plan-v2.md §3.4 for the canonical contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol


class LLMCallable(Protocol):
    """Caller-supplied LLM. Library never imports providers directly.

    json_mode is keyword-only. When True, the return value MUST be valid JSON.
    The caller wires the provider's JSON-mode flag (OpenAI response_format,
    Anthropic tool-use, Hermes ctx.llm.complete_structured, etc.).
    """

    def __call__(
        self, messages: list[dict], *, json_mode: bool = False
    ) -> str: ...


class EmbedCallable(Protocol):
    """Caller-supplied embedder. Library never imports embedding providers.

    Returns a list of vectors (list[float]) parallel to the input. Vector
    dimensionality must match the bank's embedding_dim.
    """

    def __call__(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class Query:
    """A single recall query."""

    text: str
    metadata_filter: dict[str, Any] | None = None
    tags: list[str] | None = None
    tags_match: Literal["any", "all", "any_strict", "all_strict"] = "any"


@dataclass(frozen=True)
class RecalledMemory:
    """A single retrieved memory_item with full audit trail (P5: no truncation)."""

    content: str            # spine write-side: LLM-anticipated index_text
    original_chunk: str     # source text (P5)
    source: str             # documents.source (file path / URL / conv ID)
    score: float            # RRF score (or single-mode score)
    scores: dict[str, float]  # {"semantic": x, "lexical": y, "rrf": z}
    metadata: dict[str, Any]
    bank_id: str
    document_id: str


@dataclass(frozen=True)
class RAGResult:
    """Result of recall_synth. queries_to_results enables per-query trace
    (A9 fold from prospecta-impl ralplan Round 2).
    """

    synthesis: str
    sources: list[RecalledMemory]
    queries: list[Query]
    queries_to_results: dict[str, list[RecalledMemory]]


Tracer = Callable[[str, dict[str, Any]], None]
"""Observability primitive. Default no-op.

Fires on six named events:
  retain, recall, formulate_queries, index_single_file, sweep_pass, llm_call
"""
