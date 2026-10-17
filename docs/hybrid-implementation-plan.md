# Plan: Bringing Prospecta to the PostgreSQL Hybrid Search & RAG Architecture

This plan maps Prospecta (`witt3rd/prospecta` and `witt3rd/hermes-prospecta`) from its current state to the target architecture in [`docs/postgres-hybrid.md`](./postgres-hybrid.md).

We already have ~80% of the pieces in place. The goal is to **simplify, unify, and wire the missing 20%** cleanly—keeping the **decision model** and **reranker model** configurable at every layer—before touching any live Augur or Forge config.

---

## Status

| Slice | What | State | PR |
| --- | --- | --- | --- |
| 1 | Chunk-only write path; `documents.created_on` always stamped | **done, merged** | #79 |
| 2 | Native Postgres Stage 1 (`hybrid_chunk`: reserved slots, dual `websearch_to_tsquery`, `ts_headline` snippets) | **done, merged** | #80; the stored exact vector (migration 0020), the plain-question OR and date-only questions in #83 |
| 3 | Configurable cross-encoder Stage 2 (`stage = cross_encoder`) | **done, merged** | #78 |
| 4 | Cited-evidence contract, `recall_for_task`, `ask` with the optional Stage 3 gate (library half) | **done, merged** | #81; Mode 1 automatic recall by query expansion (`expand_and_recall`, which replaced the task-facet decomposition) in #85; its design in #84 |
| 4b | A bank created fresh runs the architecture with no configuration (new-bank defaults); `import hindsight` writes chunk items | **done, merged** | #86 |
| 4c | Mode 1 recall fixes: `terms` are only names and identifiers; near-copies collapse by meaning too | **done, merged** | #87 |
| 5 | Offline verification on `augur-v2` / `forge-v2` | **verified read-only**; no gateway was switched by this repository's changes. The live migration `0020` and any gateway restart happen only with the captain's word (the `docs/exact-tsv.md` runbook) | none (no repository change) |
| 6 | `rung-memory-prospecta` and Spire follow the contract | **pending** (other repositories) | |
| 7 | Documentation overhaul in this repository | **in progress** | #88 |

This file stays until slices 5 and 6 finish; then it is deleted or reduced to a short record of what was done.

---

## 1. Where We Are (what the slices left behind)

`docs/postgres-hybrid.md` is the architecture and it is built; the table below is what is true now, against the plan's original target.

| Area | State |
| --- | --- |
| **Write path** | `retain()` writes `kind='chunk'` items (`content = original_chunk = the chunk`, at most 1000 characters, 100 of overlap) with **no model call**, and always stamps `documents.created_on`. Question items exist only when the caller passes `index_text=` (instead of the chunks) or `generate_questions=True`. |
| **Schema and indexes** | `body_tsv` (stemmed) and the stored `exact_tsv` (migration 0020, unstemmed, rewrote `memory_items`) with GIN indexes, per-bank partial HNSW indexes, `hnsw.iterative_scan = relaxed_order` in Stage 1. `docs/schema.md`. |
| **Stage 1** | The `hybrid_chunk` channel: semantic top 25, stemmed top 20, exact / phrase top 20, fused into 50 in one query with `exact_snippet`; filters (`date_from` / `date_to`, `excluded_sources`, metadata / audience) pushed into every retriever; plain questions read as an OR of their stemmed words; date-only questions list the window. **A bank created fresh has it** (#86); older banks keep the config they stored, and the legacy three-channel recall (`channel_config = []`, `docs/legacy.md`) still serves them. |
| **Stage 2** | `cross_encoder` (`qwen/qwen3-reranker-8b`, floor 0.15, `top_n` 10, 2.5 s, near-copy collapse, fallback to the Stage 1 order), configurable by call, bank and environment. A bank created fresh has it. The Sonnet and Jev rerank stages were removed from `stages.py` (`docs/legacy.md`). |
| **Stage 0 and Stage 3** | The decision model (`typesafe/jev-1.14`, System One) answers the `needs_memory` noul (Stage 0, chat only) and the `evidence_sufficient` score question (Stage 3, off by default). The gate decides only `needs_memory`; the writer proposes the period (the plan's `time_scope` decision was never built). |
| **Mode 1** | `Memory.expand_and_recall` / `recall_for_task`: query expansion, N searches in parallel, one cited block, a 7 s deadline. |
| **Citations** | `RecalledMemory` carries `source`, `created_on`, `kind`, `match_kind`, `relevance_score`, `exact_snippet`; `format_cited_rag_block`; `/lookup` returns them. |

---

## 2. Configuration Contract (Models & Thresholds Configurable Everywhere)

The plan's single JSON contract was not built as a file or a profile layer; the settings are the per-call argument, then the bank's `recall_config` / `channel_config`, then the environment, then the default. The shipped keys (`docs/recall-stages.md`):

```python
recall_config = {
  "rerank":  {"enabled": True, "stage": "cross_encoder", "model": "qwen/qwen3-reranker-8b",
              "top_n": 10, "min_score": 0.15, "timeout_s": 2.5, "deduplicate_near_copies": True},
  "models":  {"decision_model": "typesafe/jev-1.14",
              "query_writer_model": "openrouter/google/gemini-2.5-flash-lite"},
  "stage0_gate": {"enabled": True, "needs_memory_threshold": 0.5},
  "expansion": {"max_searches": 20, "max_concurrency": 8, "deadline_s": 7},
  "stage3_sufficiency": {"enabled": False, "threshold": 0.70},
}
channel_config = [{"name": "hybrid_chunk", "weight": 1,
                   "params": {"semantic_limit": 25, "stemmed_limit": 20, "exact_limit": 20,
                              "fused_limit": 50, "hnsw_ef_search": 100}}]
```

**Environment variable defaults:**
* `PROSPECTA_DECISION_MODEL` (default: `typesafe/jev-1.14`, alias `PROSPECTA_JEV_MODEL` kept for backward compatibility)
* `PROSPECTA_RERANKER_MODEL` (default: `qwen/qwen3-reranker-8b`)
* `PROSPECTA_QUERY_WRITER_MODEL` (default: `openrouter/google/gemini-2.5-flash-lite`)

---

## 3. Implementation Slices (In Order, All in `prospecta` First): the record

The text below is the plan as written; the Status table above says what happened.

### Slice 1: Stop Write-Time Question Generation (`_retain.py`, `_index.py`) — done (#79)
* **Change:**
  * In `Memory.retain()`, make chunk-only indexing the default (`kind='chunk'`, `content = original_chunk = chunk.content`), with no LLM call to `generate_index_text()`.
  * Keep `index_text=...` only when a caller explicitly passes manual index strings; remove automatic LLM question generation from the default write path.
  * Ensure `documents.created_on` is always populated at write time (defaulting to the retain date when frontmatter has no date) so citations are never `undated`.
* **Verification:** Unit + Postgres tests confirming `retain()` makes **0 LLM calls**, writes `kind='chunk'` rows with `body_tsv` and `embedding`, and stamps `created_on`.

### Slice 2: Native Postgres Stage 1 — Dual `websearch_to_tsquery` + Reserved-Slot Hybrid + `ts_headline` Snippets (`db/queries.py`, `channels/`) — done (#80, #83)
* **Change:**
  * Use the existing stored `body_tsv` (`english`) and the stored `exact_tsv` = `word_vector('simple', original_chunk)` (migration `0020`, with its own GIN index; the same rule `memory_items_keyword_gin` of migration `0018` indexes) with **`websearch_to_tsquery('english', query)`** and **`websearch_to_tsquery('simple', query)`**.
  * Support `"exact phrase"` adjacency (`<->`, case-insensitive, unstemmed in `simple`), `OR`, and `-exclude` natively.
  * Push `bank_id`, `date_from`/`date_to` (`COALESCE(d.created_on, d.created_at::date)`), `excluded_sources` (anti-echo), and `metadata_filter` / `_allow` down into all three CTEs (`semantic` top 25, `stemmed` top 20, `exact` top 20 over `kind='chunk'`).
  * Fuse via `FULL OUTER JOIN` into top 50 candidates with `rrf_score` fallback, and compute `exact_snippet` via `ts_headline()` on the winning rows whenever `exact_rank` or `stem_rank` is present.
  * Retire `InProcessBm25` (the in-memory Python BM25 index rebuild that caused multi-second cold spikes on 73K notes) from the default bank config.
* **Verification:** Real Postgres tests for:
  1. Quoted phrase `"electric dreams"` matching adjacent words case-insensitively and ranking above scattered word matches.
  2. Identifier / version / flag strings (`v1.14`, `0x80070005`) matching via the `simple` GIN index.
  3. Boolean `OR` and `-exclude`.
  4. `exact_snippet` returning verbatim `<<...>>` highlights.
  5. Filter pushdown (`date_from`/`date_to`, `excluded_sources`) with `SET LOCAL hnsw.iterative_scan = relaxed_order`.

### Slice 3: Configurable Cross-Encoder Reranker in Stage 2 (`stages.py`) — done (#78)
* **Change:**
  * Add `CrossEncoderReranker` (`stage: "cross_encoder"`) to `prospecta/stages.py` alongside `jev_score` and `sonnet_listwise`.
  * Transport: `openrouter_rerank_transport(api_key=..., url="https://openrouter.ai/api/v1/rerank")` sending `{model, query: "Instruct: ...\nQuery: ...", documents, top_n}` with configurable `model` (`PROSPECTA_RERANKER_MODEL` / `recall_config.rerank.model`, default `qwen/qwen3-reranker-8b`), `timeout_s` (default `2.5s`), and `min_rel_score` (default `0.15`).
  * Add near-duplicate note collapse (`deduplicate_near_copies`: cosine/token-jaccard > 0.88 collapses to the highest-scored note).
  * Record full accounting (`model`, `tokens_in`, `cost_usd`, `duration_ms`, `fallback_reason`) in `calls` / `recall_events`.
  * On any HTTP/timeout error, fall back cleanly to Stage 1's RRF order.
* **Verification:** Offline transport tests (mock + error fallback + near-duplicate collapse) + one gated live OpenRouter integration check verifying `qwen/qwen3-reranker-8b` and a swapped model name.

### Slice 4: Mode 1 (Automatic Recall for Chats & Tasks) and Mode 2 (Interactive Search & Cited RAG Output) (`memory.py`, `service.py`, `hermes-prospecta`) — library half done (#81, #85); the `hermes-prospecta` half is slice 6
* **Change:**
  * **Cited evidence contract (`RecalledMemory` & `format_cited_rag_block`):** Every returned item includes `source`, `created_on` (`YYYY-MM-DD`), `category`/`kind`, `match_kind`, `relevance_score`, `exact_snippet`, and `original_chunk`.
  * **Mode 1A (Conversational automatic recall in `hermes-prospecta`):**
    * Stage 0 (`decision_model`, default `typesafe/jev-1.14`) decides `needs_memory` and `time_scope`.
    * Query writer (`query_writer_model`, default `openrouter/google/gemini-2.5-flash-lite`) resolves follow-ups against `agent_last` only, preserving `"quoted phrases"` verbatim.
    * Passes `excluded_sources` (last 2 turns of current session) into Stage 1 + Stage 2 (`reranker_model`, default `qwen/qwen3-reranker-8b`).
    * Formats the injected `<memory-context>` block with citation headers and `Exact match snippet:` lines.
  * **Mode 1B (Non-interactive task/commission recall):**
    * Add `memory.recall_for_task(title=..., ask=..., done=..., project=...)` (and `POST /lookup` batch support) that returns a cited RAG context block. (Its first version decomposed the brief into 2–4 facets; that is replaced by Mode 1 query expansion, `Memory.expand_and_recall`, used by chat and task alike: see `docs/postgres-hybrid.md` section 1.)
  * **Mode 2A & 2B (Interactive Search & Interactive RAG):**
    * `memory.search()` and `POST /lookup` skip Stage 0 Jev, run Stage 1 + Stage 2, and return `exact_snippet` + citation metadata on every hit.
    * `memory.ask()` (Mode 2B interactive RAG) optionally runs Stage 3 (`decision_model` sufficiency gate when `stage3_sufficiency.enabled=True`) and synthesizes a grounded answer with `[1] [source | date]` citations and verbatim snippets.
* **Verification:** End-to-end tests for Mode 1A (chat turn), Mode 1B (task brief), Mode 2A (direct keyword/phrase search with snippets), and Mode 2B (cited RAG answer).

### Slice 5: Offline Verification on `augur-v2` and `forge-v2` (Read-Only, Before Any Live Switch) — verified read-only
* **Change:** None to live gateways.
* **Verification (read-only against the existing database):**
  1. Confirm `augur-v2` and `forge-v2` chunk coverage (Note: `augur-v2` has 9,987 `chunk` items and `forge-v2` has 76,636 `chunk` items already in `memory_items`, alongside the old `question` items; verify 100% of `documents` in both banks have at least one `kind='chunk'` item before ignoring `kind='question'`).
  2. Run a 20-query offline check (exact quotes like `"electric dreams"`, proper names, dates, conversational follow-ups, and technical queries) measuring Stage 1 latency, Stage 2 Qwen latency, citation/snippet formatting, and total time.
  3. Present the 20-query output table to the captain. **Only after your explicit word** do we update `augur-v2` / `forge-v2` `channel_config` & `recall_config` and restart the gateways.

### Slice 6 (Follow-On): Update `rung-memory-prospecta` (and Spire) to Match — pending
* **Change (`witt3rd/rung-memory-prospecta`):**
  * Bump the pinned `prospecta` dependency to the new hybrid `main`.
  * Wire the Rung MCP memory provider hooks (`recall` before turn, `retain` after turn, and agent-facing search/ask tools) to the new contract:
    * `retain`: chunk-only default (0 LLM calls at write time, `created_on` stamped).
    * `recall`: Stage 0 (`decision_model`) + Stage 1 (`hybrid_chunk` with `excluded_sources` anti-echo and `date_from`/`date_to`) + Stage 2 (`cross_encoder` `reranker_model`), returning the cited RAG block (`format_cited_rag_block` with `exact_snippet`).
    * Expose `PROSPECTA_DECISION_MODEL`, `PROSPECTA_RERANKER_MODEL`, and `PROSPECTA_QUERY_WRITER_MODEL` in the provider's container/env config.
* **Change (Spire):** Apply the same Stage 0 $\rightarrow$ Stage 1 $\rightarrow$ Stage 2 (`qwen/qwen3-reranker-8b`) + cited snippet pattern once Augur and Forge are verified.

### Slice 7 (Follow-On): Full Documentation Overhaul in `witt3rd/prospecta` — in progress
* **Why:** The repo carries a lot of legacy documents (the question-indexing spine, the graph/linker/entity era, the map-reduce and bake-off era, BM25 seams, and design history) that describe approaches we no longer run and add confusion.
* **Change:**
  * Make `docs/postgres-hybrid.md` the one normative architecture, and rewrite `README.md` to describe Prospecta as it is: chunk-only write path, Stage 0 decision gate, native Postgres Stage 1, Stage 2 cross-encoder, optional Stage 3, the cited-evidence contract, and the modes (chat recall, task recall, search, ask), with the configurable models.
  * Inventory every document under `docs/` (including `docs/design/`, `docs/explainer/`, `docs/deployments/`) and the top-level files, and for each decide: keep and rewrite to the current truth, fold into another doc, or delete. Delete anything describing retired approaches (question-indexing as the spine, graph/linker/entities, map-reduce recall, the in-process BM25 seam, superseded plans); git history keeps them.
  * Keep only operational docs that are still true (schema/migrations, backup and restore, source contract, limits, testing), each corrected to the current code.
  * Retire `docs/hybrid-implementation-plan.md` itself once all slices are done (or reduce it to a short record of what was done).
* **Review:** The captain sees the keep / rewrite / delete list before any deletion lands.

---

## 4. What Stays Untouched Until You Say Go

* **Augur and Forge live gateways and configs** stay frozen as they are right now.
* **Existing database rows** in `augur-v2` and `forge-v2` are not deleted or modified during Slices 1–4.
* **Spire (`spire-venue`)** waits until Prospecta's implementation and offline verification are complete and you are happy with how Augur and Forge behave.
