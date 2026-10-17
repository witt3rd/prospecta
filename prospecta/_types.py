"""Public type definitions for prospecta.

All Protocols and dataclasses defined here are the public API surface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Protocol


class LLMCallable(Protocol):
    """Caller-supplied LLM. Library never imports providers directly.

    json_mode is keyword-only. When True, the return value MUST be valid JSON.
    The caller wires the provider's JSON-mode flag (OpenAI response_format,
    Anthropic tool-use, Hermes ctx.llm.complete_structured, etc.).

    A callable may also accept an optional `model` keyword hint; callers pass
    it only to callables whose signature accepts it.
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
    scores: dict[str, float]
    """Four-key scores dict, always-populated, all numeric (A6 invariant):
      - "semantic": cosine similarity from HNSW channel (1 - distance).
        0.0 when result didn't surface via semantic channel.
      - "lexical": ts_rank_cd over content_tsv (LLM-generated index_text /
        questions channel). Preserved name for backward compat per v0.1
        option γ; semantically equivalent to `lexical_content`.
      - "lexical_body": ts_rank_cd over body_tsv (original_chunk / source
        body channel). The P14 body-fallback safety net — rescues docs
        whose LLM-generated index_text drifted from query language.
        0.0 unless surfaced via mode='hybrid' three-channel fusion.
      - "rrf": fused Reciprocal Rank Fusion score (semantic +
        lexical_content + lexical_body in hybrid mode; identical to
        single-channel score in semantic/lexical modes).
    """
    metadata: dict[str, Any]
    bank_id: str
    document_id: str
    # The cited-evidence contract (docs/postgres-hybrid.md): where and when the hit
    # is from and why it matched. `source` and `original_chunk` above complete it.
    created_on: str | None = None         # the note's date, YYYY-MM-DD
    kind: str | None = None               # documents.source_kind: note, conversation, ...
    match_kind: str | None = None         # exact+semantic | exact | lexical | semantic | date
    relevance_score: float | None = None  # cross-encoder score (0..1) when stage 2 graded it, else `score`
    exact_snippet: str | None = None      # ts_headline of the matched words, <<marked>>
    found_by: tuple[int, ...] | None = None   # query expansion: the numbers of the searches that found it
    item_id: str | None = None            # the chunk (memory_items.id) whose text `original_chunk` is
    # query expansion's merge: the records that collapsed into this one as near-copies (by wording or
    # by meaning), best scored first: their document ids, and their sources (each once)
    collapsed_ids: tuple[str, ...] | None = None
    collapsed_sources: tuple[str, ...] | None = None


@dataclass(frozen=True)
class TaskBrief:
    """A non-interactive task: what Memory.expand_and_recall(kind='task') expands."""

    title: str
    ask: str
    done: str | None = None       # the done criteria
    project: str | None = None    # the task's own project: its notes come first, others are labeled


@dataclass(frozen=True)
class SearchSpec:
    """One structured search of a query expansion (docs/postgres-hybrid.md, Mode 1, point 3)."""

    query: str                    # what to find: for the semantic and stemmed retrievers and the reranker
    terms: tuple[str, ...] = ()   # names, identifiers, "quoted phrases", verbatim from the prompt: the exact retriever
    date_from: str | None = None  # YYYY-MM-DD: the period the prompt states or implies
    date_to: str | None = None


@dataclass
class SearchRun:
    """What became of one search: `index` is its 1-based number in the writer's order (the
    block's numbering and RecalledMemory.found_by)."""

    index: int
    spec: SearchSpec
    status: str = "pending"       # ok | late (past the deadline, dropped) | error | skipped
    n_candidates: int = 0         # Stage 1 candidates (notes)
    n_kept: int = 0               # what the search contributed after stage 2 and its floor
    reranked: bool = False        # stage 2 graded them (False: Stage 1 order stood)
    fallback_reason: str | None = None
    error: str | None = None
    ms: dict = field(default_factory=dict)   # embed, stage1, rerank, total


@dataclass(frozen=True)
class GateDecision:
    """The stage 0 gate's verdict for a chat message. `needs_memory` is the decision model's
    noul in 0..1 (the plugin's contract: at or above the threshold, 0.5, the message goes on).
    `skip` forces a skip whatever the number (e.g. the message is a greeting)."""

    needs_memory: float
    skip: bool = False
    reason: str | None = None
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None


@dataclass(frozen=True)
class ExpandedRecall:
    """Result of Memory.expand_and_recall. Never raised into the caller: a failure is in
    `error`, `fallbacks` and an empty or partial `block`."""

    kind: str                         # chat | task
    block: str                        # the cited block ("" when gated, empty or failed)
    items: list[RecalledMemory]       # the merged records, in block order; relevance_score is the best score
    searches: list[SearchSpec]        # the searches written (after the verbatim guard and the maximum)
    runs: list[SearchRun]             # one per search: status, counts, per-search ms
    skipped: bool = False             # the gate skipped the whole expansion
    skipped_reason: str | None = None
    gate: dict | None = None          # {needs_memory, threshold, model, ...} or {error}
    fallbacks: list[str] = field(default_factory=list)   # every degraded step, in words
    timings_ms: dict = field(default_factory=dict)       # gate, writer, stage1, rerank, merge, total
    accounting: dict = field(default_factory=dict)       # n_llm_calls, tokens_in, tokens_out, cost_usd
    calls: list[dict] = field(default_factory=list)      # every model call record
    deadline_hit: bool = False
    dropped_for_tokens: int = 0       # records left out by the max_tokens valve (0 = valve off or not hit)
    error: str | None = None

    @property
    def queries(self) -> list[str]:
        """The searches' queries (what the old TaskRecall.queries held)."""
        return [s.query for s in self.searches]


@dataclass(frozen=True)
class AskResult:
    """Result of Memory.ask: a grounded answer with inline [n] citations, where
    [n] is items[n-1]."""

    answer: str                   # the model's answer, [n] markers inline
    citations: list[dict]         # [{n, known, source, created_on, snippet, exact_snippet}]
    items: list[RecalledMemory]   # the evidence the answer may cite
    sufficient: bool | None       # the stage 3 verdict; None when the gate is off or failed
    retried: bool                 # a broadened retrieval ran because the evidence was not sufficient
    gate: dict | None             # {score, threshold, model} or {error}; None when off
    rendered: str                 # the answer followed by its numbered [source | date] sources


@dataclass(frozen=True)
class IndexStats:
    """Result of Memory.index_directory(). Sibling to RAGResult shape."""

    documents_added: int = 0
    documents_replaced: int = 0
    memory_items_added: int = 0
    files_scanned: int = 0
    files_skipped: int = 0
    files_unchanged: int = 0
    errors: int = 0


@dataclass(frozen=True)
class RAGResult:
    """Result of recall_synth. queries_to_results enables per-query trace
    (A9 fold from prospecta-impl ralplan Round 2).
    """

    synthesis: str
    sources: list[RecalledMemory]
    queries: list[Query]
    queries_to_results: dict[str, list[RecalledMemory]]
    # Grounded synthesis only (recall_synth(grounded=True)); empty otherwise.
    citations: list[dict] = field(default_factory=list)   # [{note, document_id, known}]
    evidence: list[dict] = field(default_factory=list)    # [{note, document_id, chunks}]
    synth_call: dict = field(default_factory=dict)        # model, tokens, cost_usd, duration_ms


class DocumentSourceConflictError(Exception):
    """Raised when retain() finds same content_hash already stored under a different source.

    Per docs/schema.md ("Re-retain", replace-on-source-match): identical content
    (same content_hash) under a *different* caller-supplied source is treated
    as an error rather than silently re-binding the document. The library
    treats `source` as opaque (substrate-opacity); the caller is the only
    party that knows what conflict-routing should happen.
    """

    def __init__(
        self,
        *,
        content_hash: str,
        existing_source: str | None,
        attempted_source: str | None,
    ) -> None:
        self.content_hash = content_hash
        self.existing_source = existing_source
        self.attempted_source = attempted_source
        super().__init__(
            f"Document with content_hash={content_hash[:12]}... already retained under "
            f"source={existing_source!r}; refusing to store under source={attempted_source!r}"
        )


class IndexTextGenerationError(Exception):
    """Raised when LLM returns no usable index_text strings.

    Carries the raw LLM response so the caller (or T14 sweeper) can decide
    whether to log + skip, retry with a different prompt, or surface.
    """

    def __init__(self, *, raw_response: str) -> None:
        self.raw_response = raw_response
        super().__init__(
            f"LLM produced no usable index_text strings; raw response: {raw_response[:200]!r}"
        )


Tracer = Callable[[str, dict[str, Any]], None]
"""Observability primitive. Default no-op.

Fires on six named events:
  retain, recall, formulate_queries, index_single_file, sweep_pass, llm_call
"""
