# Prospecta

**Bilateral prospective synthesis for memory retrieval.** Both ends of the exchange — the writer storing a memory and the reader querying it — speak through the same LLM-mediated abstraction: anticipated questions. The match happens in question-space, not content-space.

When a document is **stored**, an LLM writes the `index_text` that says *"this is what someone would ask to retrieve this."* The embedded vector encodes the anticipated question, not the body of the document.

When a message **arrives**, a second LLM call writes the questions this exchange's memory should service. The query vector also encodes a question, not the user's verbatim text.

Two anticipations, written in the same register, by similar prompts, aimed at meeting each other. The embedding similarity is the matching mechanism. The body of the source document is preserved verbatim and indexed via a third channel (Postgres BM25 over the original text) so that when the LLM-synthesized index_text drifts from the query, the body still rescues the document. The hybrid retrieval is the safety net under a spine that lives in question-space.

This is **not** classical RAG over chunks. Classical RAG embeds content and hopes the user's question lives nearby in embedding space. Prospecta makes both ends LLM-synthesized in anticipation language, then matches question-to-question — closer to how human memory actually works: you don't remember a transcript, you remember "what was this about."

---

## Status

**v0.1, ship-ready.** 241 tests pass. Bilateral spine validated empirically (see `tests/integration/test_bilateral_spine.py`). Hybrid safety net validated empirically (see `tests/integration/test_hybrid_safety_net.py`).

Substrate: Postgres 14+ with the `pgvector` extension. The library has zero provider imports — embedding model and LLM are injected callables; pick your own, or use the LiteLLM-backed defaults.

---

## Install

```bash
# Core only — bring your own embedder and LLM callable
pip install prospecta

# Capable defaults via LiteLLM (any provider it supports)
pip install 'prospecta[defaults]'

# Or single-provider minimal-deps paths
pip install 'prospecta[embed-sentence-transformers]'   # local, no API key
pip install 'prospecta[embed-openai]'                  # direct OpenAI HTTP
pip install 'prospecta[embed-openai-compatible]'       # infinity-emb, vLLM, etc.
```

## Quickstart

Three commands from `pip install` to first useful recall:

```bash
export OPENAI_API_KEY="sk-..."
prospecta init                    # docker compose up + migrate + create default bank
prospecta retain "Kelly was born March 4th 1990" \
    --source kelly.md \
    --index-text "When was Kelly born?"
prospecta search "Kelly birthday"
```

`prospecta init` is idempotent — re-runs detect already-up containers, already-applied migrations, and the already-created default bank. See `docs/deployments/{local-direct,neon,azure}.md` for non-Docker substrates.

---

## API at three layers

### Library — embedding-injected, host-agnostic

```python
from prospecta import Memory
from prospecta.defaults import make_default_embedder, make_default_llm

m = Memory(
    database_url="postgres://localhost:5432/prospecta",
    bank_id="default",
    embed=make_default_embedder(),   # or your own (list[str]) -> list[list[float]]
    llm=make_default_llm(),          # or your own (messages, *, json_mode) -> str
)

# Write side: LLM generates the anticipated questions
document_id = m.retain(
    "Kelly was born March 4th 1990. Her birthday party in 2026 was epic.",
    source="kelly.md",
    tags=["family", "kelly"],
)

# Read side: LLM generates the multi-query expansion; RAG synthesizes the answer
result = m.recall_synth("when's kelly's birthday again?")
print(result.synthesis)              # the answer
for query, chunks in result.queries_to_results.items():
    print(query, len(chunks))        # which sub-question retrieved what
```

### Caller-supplied index_text — the spine override

When the caller knows the right anticipated questions (or has refined them via a tighter prompt elsewhere), pass them directly. The library uses them verbatim and skips the LLM call. **Caller wins on every override.**

```python
m.retain(
    "Kelly was born March 4th 1990.",
    source="kelly.md",
    index_text=[
        "When was Kelly born?",
        "What year was Kelly born in?",
        "Kelly's date of birth?",
    ],
)
```

Same for read-side: pass `formulate_prompt_override=` to replace the default formulate prompt, or `synth_prompt_override=` for the synthesis prompt.

### CLI — full surface

```bash
prospecta migrate                        # apply pending migrations
prospecta create-bank --id alice --embedding-dim 384
prospecta index ./corpus                 # walk a directory; chunk + retain
prospecta search "query" --mode hybrid   # or semantic / lexical
prospecta retain "content..." --source X --tags a,b
prospecta sweep ./corpus --watch         # background sweeper for filesystem drift
prospecta stats                          # bank/document/event counters
prospecta config                         # show loaded config (passwords redacted)
```

---

## Observability

Every load-bearing event of the bilateral spine is recorded in Postgres event tables — a durable audit trail, not ephemeral stderr. Six tables capture the full flow:

- **`retain_events`** — what was retained, with the verbatim LLM-generated `index_text` (the anticipated questions) when not caller-supplied.
- **`recall_events`** — what was queried, the formulated sub-questions, the surfaced results (with per-channel scores), and the synthesized answer.
- **`formulate_events`** — the user's verbatim message, the LLM's JSON expansion, and parse-fallback diagnostics (`error_kind` discriminates `malformed_json` from `schema_mismatch`).
- **`llm_calls`** — every LLM call's verbatim prompt and response (opt-out via `PostgresSink(persist_llm_text=False)`).
- **`sweep_passes`** + **`sweeper_state`** — background filesystem sweeper timing and outcomes.

Inspect a session end-to-end with a few SQL queries from `pgcli` or any Postgres client. See [`docs/observability.md`](docs/observability.md) for the inspection guide, the cheat-sheet of queries, custom-tracer patterns, and the opt-outs.

---

## Where prospecta sits in the agent-memory landscape

There is no one-size-fits-all for agentic memory. Different systems make different bets about what's worth remembering and how to find it later. A quick survey of the 2025-2026 landscape, organized by primary retrieval mechanism. Where [Hermes Agent](https://hermes-agent.nousresearch.com) ships a bundled provider in a category, it's named inline.

**Buffer + summarize** *(e.g., LangChain `ConversationBufferMemory` / `ConversationSummaryMemory`).* A sliding window of recent turns, optionally LLM-compressed when the buffer grows. Cheap, simple, conversation-centric. Hermes's built-in `MEMORY.md` / `USER.md` files cover this lightweight always-in-context need at the platform level. Good fit when "memory" means "the last N exchanges" and recall is implicit.

**Content-space RAG** *(LlamaIndex, Khoj).* Documents are chunked, embedded as-is, and retrieved by embedding the user's verbatim query against the chunk vectors. Excellent composable primitives, broad ecosystem, mature tooling. Works best when query language matches document language closely — technical docs, code search, well-curated knowledge bases.

**Extract-then-store** *(mem0; Hermes-bundled providers: `mem0`, `supermemory`, `retaindb`).* An LLM extracts discrete facts or memories at write time; storage is then retrieved via semantic search, often with reranking. **mem0** is the canonical pattern — server-side fact extraction + dedup. **supermemory** adds profile recall + session-end conversation ingest. **retaindb** adds explicit memory types (preference, fact, event, etc.) and hybrid vector+BM25+rerank. Write-side is LLM-touched; the read-side embeds the query verbatim. Good fit for assistant memory where the unit is "a fact about the user."

**Hierarchical / tiered retrieval** *(Hermes-bundled providers: `byterover`, `openviking`).* Knowledge organized as a tree or filesystem-style hierarchy; retrieval walks the hierarchy with tiered escalation (fuzzy match → semantic → LLM-driven deep search). **byterover** is local-first with optional cloud sync; **openviking** (Volcengine) treats memory as a `viking://` URI namespace with abstract/overview/full read modes. Good fit when memory has inherent structure worth navigating, not just searching.

**Knowledge-graph memory** *(Zep, Graphiti, Cognee; Hermes-bundled provider: `hindsight`).* Entities and relations extracted into a graph, often with temporal awareness. Retrieval blends graph traversal with semantic search. **hindsight** supports three deployment modes (cloud, local-embedded with built-in Postgres, local-external Docker) and adds entity resolution + multi-strategy retrieval on top of the graph. Excellent when relationships between entities are the primary query shape ("who works with whom," "what changed between then and now") and when investment in ontology pays back across many queries.

**Compositional / vector-symbolic memory** *(Hermes-bundled provider: `holographic`).* Holographic Reduced Representations (HRR) — fixed-dimensional vectors that compose via circular convolution, allowing structured-fact storage with trust scoring and entity resolution. Local SQLite + FTS5. Good fit when facts have role-filler structure ("Kelly's mother is Janet"; "the meeting on Tuesday was about pricing") that should compose at recall time rather than being flattened to text.

**User-model dialectic** *(Hermes-bundled provider: `honcho`).* Cross-session user modeling via multi-pass dialectic reasoning. The unit isn't an individual fact, it's the evolving model of who the user is — refined turn-over-turn through bidirectional peer tools and persistent conclusions. Strong fit when the assistant needs to *understand* the user across time, not just retrieve facts about them.

**Hierarchical agent memory** *(MemGPT / Letta).* Tiered memory (core / recall / archival) with the agent itself managing eviction and promotion via tool calls — OS-style memory management for LLMs. Strong fit when memory is mutable working state under agent control, less so when memory is a stable corpus.

**Bilateral prospective synthesis** *(prospecta).* Both ends of the exchange are LLM-mediated in the same register — anticipated questions. The write-side LLM authors `index_text` as "what someone would ask to retrieve this"; the read-side LLM expands the message into the questions the memory should service. The match happens in question-space. A three-channel hybrid index (semantic + content-stem + body-stem, fused via RRF) catches the cases where the spine's anticipations drift from each other.

### Where prospecta is a strong fit

- The corpus is markdown-shaped documents that will be queried in language often quite different from how they were written. (Personal memory; project notes; conversational substrate; anything where the writer and reader weren't thinking the same words.)
- You can afford an LLM call at both index time AND query time. The cost buys robustness to phrasing drift that content-space RAG can't deliver — and the body channel rescues the document if the LLM mis-anticipates.
- You want a hot path through your own Postgres (local, Neon, Azure Flexible Server, etc.) rather than a managed service or a graph database.
- The caller-wins override surface matters: you'll sometimes know the right anticipated questions yourself and want the library to honor that verbatim.

### Where something else is likely better

- **Conversation-as-memory** where the unit is "the last N turns" — Hermes's built-in `MEMORY.md` or LangChain buffer memory is sufficient.
- **High-volume, low-margin retrieval** where two LLM calls per exchange exceeds the budget — content-space RAG with a good embedder is cheaper.
- **Cross-session user modeling** where the goal is understanding the user, not retrieving documents — reach for `honcho`.
- **Entity-relationship queries** that need graph traversal as the primary mechanism — reach for `hindsight`, Zep, or Graphiti.
- **Role-filler structured facts** that should compose at recall — reach for `holographic`.
- **Per-user fact storage** with a managed cloud backend — reach for `mem0`, `supermemory`, or `retaindb`.
- **Sliding agent-state memory** under tool-call eviction — Letta is purpose-built.

Prospecta is the spine + the safety net. Not the universal answer; a specific bet about where document retrieval breaks down (phrasing drift between writer and reader) and what fixes it (matching in question-space, with a body-channel safety net).

### The three retrieval channels

| Channel | What it embeds | What it rescues |
|---|---|---|
| Semantic (HNSW over `embedding`) | The anticipated question — LLM-authored at retain time | The default match path. Question-on-question similarity in vector space. |
| Lexical content (BM25 over `content_tsv`) | The same anticipated question, stemmed via Postgres `to_tsvector('english', ...)` | Phrasing drift the embedder misses — plurals vs singulars, derivational variants. |
| Lexical body (BM25 over `body_tsv`) | The full source body verbatim, stemmed | The hard case: when the LLM-synthesized `index_text` drifts semantically AND lexically from the query, the body channel still surfaces the document. |

Three channels, fused via Reciprocal Rank Fusion (k=60 default, configurable per call). The spine carries when it works; the safety net carries when it doesn't.

---

## What's load-bearing

Read `PRINCIPLES.md` for the full set of 18 principles that govern this codebase. The four that matter most for users:

- **P1 — Bilateral synthesis is the spine, not a feature.** Both write-side `index_text` generation and read-side multi-query formulation are LLM-mediated by design. The embedding match happens in question-space. Designs that let the spine become optional collapse prospecta back into a worse LlamaIndex.
- **P3 — LLM and embedder are injected callables.** The library imports neither openai nor anthropic nor litellm. You inject what you want. `prospecta.defaults` and `prospecta.embed.*` are extras modules that *also* don't ship in core.
- **P4 — Caller wins on every override.** `retain(index_text=...)` skips the LLM. `recall(queries=[...])` skips formulation. Any prompt can be overridden per-call.
- **P5 — Full content, no truncation.** `RecalledMemory.content` carries the matched index_text. `RecalledMemory.original_chunk` carries the source body. The library does not pre-judge what the caller will need.

---

## Caveats (v0.1)

Honest about what isn't there yet:

- **Sync only.** All `retain` / `recall` / `recall_synth` calls are blocking. Async surface is post-v0.1.
- **English-only stemming.** The `content_tsv` and `body_tsv` channels use `to_tsvector('english', ...)`. Other languages will lose stemming benefit (semantic + body-raw channels still work).
- **Embedding-dim change requires bank rebuild.** A bank's `embedding_dim` is fixed at creation. Switching embedder dimensions means a new bank + re-index.
- **HNSW build cost on large banks.** Postgres builds the per-bank HNSW index incrementally; first-bulk-index can be slow. Configurable via standard pgvector tuning (`m`, `ef_construction`).
- **Single-language FTS.** No multilingual tsvector support in v0.1.
- **No retrieval re-ranking model.** RRF fusion is the only post-retrieval step. A cross-encoder rerank layer would be additive but is post-v0.1.

---

## Deployment guides

- [`docs/deployments/local-direct.md`](docs/deployments/local-direct.md) — bare Postgres + pgvector locally, no Docker.
- [`docs/deployments/neon.md`](docs/deployments/neon.md) — Neon serverless Postgres (free tier exists).
- [`docs/deployments/azure.md`](docs/deployments/azure.md) — Azure Database for PostgreSQL Flexible Server with pgvector.

---

## Architecture references

- [`PRINCIPLES.md`](PRINCIPLES.md) — 18 load-bearing principles.
- [`docs/design/prospecta/schema.md`](docs/design/prospecta/schema.md) — Postgres DDL + RRF SQL canonical.
- [`docs/design/prospecta/decision-record-1.md`](docs/design/prospecta/decision-record-1.md) — locked architectural decisions.

---

## License

See `LICENSE`. ⚒️
