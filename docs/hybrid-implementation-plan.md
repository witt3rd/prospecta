# Plan: Bringing Prospecta to the PostgreSQL Hybrid Search & RAG Architecture

This plan maps Prospecta (`witt3rd/prospecta` and `witt3rd/hermes-prospecta`) from its current state to the target architecture in [`docs/postgres-hybrid.md`](./postgres-hybrid.md).

We already have ~80% of the pieces in place. The goal is to **simplify, unify, and wire the missing 20%** cleanly—keeping the **decision model** and **reranker model** configurable at every layer—before touching any live Augur or Forge config.

---

## 1. Where We Are vs. Target (`docs/postgres-hybrid.md`)

| Area | What Prospecta Has Today | Gap to Target |
| --- | --- | --- |
| **Write Path (`retain`, `index_single_file`, `source.py`)** | `retain()` calls an LLM (`generate_index_text`) to generate ~5 model-written questions (`kind='question'`) per note and only writes `kind='chunk'` if `child_chunks=True`. | **Drop write-time LLM question generation.** Write `kind='chunk'` items directly (`content = original_chunk = chunk.content`), deterministic and fast ($0 LLM write cost). Existing `question` rows in `augur-v2`/`forge-v2` stay harmlessly in the table (ignored by retrieval) or can be dropped later. |
| **PostgreSQL Schema & Indexes (`memory_items`)** | Has `embedding` (with per-bank HNSW indexes), `body_tsv` (`GENERATED ALWAYS AS to_tsvector('english', original_chunk) STORED` + GIN), and `memory_items_keyword_gin` (`simple` expression GIN index from migration `0018`). `semantic.py` already sets `SET LOCAL hnsw.iterative_scan = 'relaxed_order'`. | Almost complete. Today `keyword.py`, `bm25.py` (in-memory Python BM25), and `semantic.py` run as separate Python round-trips, and `keyword.py` builds manual `OR` `tsquery` strings instead of `websearch_to_tsquery` (so `"electric dreams"` phrase adjacency and `-exclude` don't work). Unify Stage 1 into Postgres native `websearch_to_tsquery('english', ...)` + `websearch_to_tsquery('simple', ...)` + `ts_headline` exact-match snippets, and retire the slow in-process Python BM25 rebuild. |
| **Stage 1 Candidate Generation & Fusion** | `fusion.py` has `floor_union` and `rrf`. | Wire the 3 native Postgres retrievers (**Semantic `dense_chunk`**, **Stemmed Lexical `fts_en`**, **Exact/Phrase Lexical `fts_exact`**) with pushed-down SQL filters (`bank_id`, date range, `excluded_source_ids` anti-echo, metadata/audience) and reserved candidate quotas ($\rightarrow$ ~50 candidates) + `rrf_score` fallback + `exact_snippet` (`ts_headline` on the top 50). |
| **Stage 2 Cross-Encoder Reranker (`stages.py`)** | `stages.py` supports `sonnet_listwise` and `jev_score` (`STAGES = ("sonnet_listwise", "jev_score")`). | Add **`cross_encoder`** stage (`CrossEncoderReranker`) calling `POST https://openrouter.ai/api/v1/rerank` with configurable model (default `qwen/qwen3-reranker-8b`), relevance floor, near-duplicate collapse, and automatic fallback to Stage 1 RRF order on timeout/error. |
| **Stage 0 Gate & Stage 3 Sufficiency (`stages.py`, `prefetch_contract.py`)** | `prefetch_contract.py` has Stage 0 Jev gate + query writer + anti-echo, but calls `jev_score` for Stage 2 ranking; `stages.py` has Jev gate tied only to `sonnet_listwise`. | Make **decision model** configurable (`decision_model`, default `typesafe/jev-1.14`) and **reranker model** configurable (`reranker_model`, default `qwen/qwen3-reranker-8b`). Let `prefetch_contract.py` (Mode 1A chat recall) and task/service lookup (Mode 1B & Mode 2) use Stage 1 + Stage 2 (`cross_encoder`) instead of a second Jev ranking call. |
| **Citations & Exact-Match Snippets (`RecalledMemory`, `/lookup`, RAG block)** | `RecalledMemory` has `source`, `document_id`, `scores`, `metadata`. `build_items` formats `[source | date | person]`. | Add `created_at` / `date`, `exact_snippet` (`ts_headline`), and `match_kind` (`exact`, `lexical`, `semantic`, `exact+semantic`) to `RecalledMemory`, `/lookup` JSON responses, and the formatted RAG evidence block (`format_cited_rag_block`). |

---

## 2. Configuration Contract (Models & Thresholds Configurable Everywhere)

Both models and their thresholds are configurable at three levels (per-call override $\rightarrow$ per-bank `recall_config` / profile `prospecta.json` $\rightarrow$ environment variable $\rightarrow$ default):

```json
{
  "models": {
    "decision_model": "typesafe/jev-1.14",
    "reranker_model": "qwen/qwen3-reranker-8b",
    "query_writer_model": "openrouter/google/gemini-2.5-flash-lite",
    "embedding_model": "openai/text-embedding-3-small"
  },
  "stage0_gate": {
    "enabled": true,
    "needs_memory_threshold": 0.5
  },
  "stage1_postgres": {
    "semantic_limit": 25,
    "stemmed_limit": 20,
    "exact_limit": 20,
    "fused_limit": 50,
    "hnsw_ef_search": 100
  },
  "stage2_rerank": {
    "enabled": true,
    "stage": "cross_encoder",
    "model": "qwen/qwen3-reranker-8b",
    "top_n": 10,
    "min_rel_score": 0.15,
    "timeout_s": 2.5,
    "deduplicate_near_copies": true
  },
  "stage3_sufficiency": {
    "enabled": false,
    "threshold": 0.70
  }
}
```

**Environment variable defaults:**
* `PROSPECTA_DECISION_MODEL` (default: `typesafe/jev-1.14`, alias `PROSPECTA_JEV_MODEL` kept for backward compatibility)
* `PROSPECTA_RERANKER_MODEL` (default: `qwen/qwen3-reranker-8b`)
* `PROSPECTA_QUERY_WRITER_MODEL` (default: `openrouter/google/gemini-2.5-flash-lite`)

---

## 3. Implementation Slices (In Order, All in `prospecta` First)

### Slice 1: Stop Write-Time Question Generation (`_retain.py`, `_index.py`)
* **Change:**
  * In `Memory.retain()`, make chunk-only indexing the default (`kind='chunk'`, `content = original_chunk = chunk.content`), with no LLM call to `generate_index_text()`.
  * Keep `index_text=...` only when a caller explicitly passes manual index strings; remove automatic LLM question generation from the default write path.
  * Ensure `documents.created_on` is always populated at write time (defaulting to the retain date when frontmatter has no date) so citations are never `undated`.
* **Verification:** Unit + Postgres tests confirming `retain()` makes **0 LLM calls**, writes `kind='chunk'` rows with `body_tsv` and `embedding`, and stamps `created_on`.

### Slice 2: Native Postgres Stage 1 — Dual `websearch_to_tsquery` + Reserved-Slot Hybrid + `ts_headline` Snippets (`db/queries.py`, `channels/`)
* **Change:**
  * Use the existing stored `body_tsv` (`english`) and `word_vector('simple', original_chunk)` (backed by `memory_items_keyword_gin` from migration `0018`) with **`websearch_to_tsquery('english', query)`** and **`websearch_to_tsquery('simple', query)`**.
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

### Slice 3: Configurable Cross-Encoder Reranker in Stage 2 (`stages.py`)
* **Change:**
  * Add `CrossEncoderReranker` (`stage: "cross_encoder"`) to `prospecta/stages.py` alongside `jev_score` and `sonnet_listwise`.
  * Transport: `openrouter_rerank_transport(api_key=..., url="https://openrouter.ai/api/v1/rerank")` sending `{model, query: "Instruct: ...\nQuery: ...", documents, top_n}` with configurable `model` (`PROSPECTA_RERANKER_MODEL` / `recall_config.rerank.model`, default `qwen/qwen3-reranker-8b`), `timeout_s` (default `2.5s`), and `min_rel_score` (default `0.15`).
  * Add near-duplicate note collapse (`deduplicate_near_copies`: cosine/token-jaccard > 0.88 collapses to the highest-scored note).
  * Record full accounting (`model`, `tokens_in`, `cost_usd`, `duration_ms`, `fallback_reason`) in `calls` / `recall_events`.
  * On any HTTP/timeout error, fall back cleanly to Stage 1's RRF order.
* **Verification:** Offline transport tests (mock + error fallback + near-duplicate collapse) + one gated live OpenRouter integration check verifying `qwen/qwen3-reranker-8b` and a swapped model name.

### Slice 4: Mode 1 (Automatic Recall for Chats & Tasks) and Mode 2 (Interactive Search & Cited RAG Output) (`memory.py`, `service.py`, `hermes-prospecta`)
* **Change:**
  * **Cited evidence contract (`RecalledMemory` & `format_cited_rag_block`):** Every returned item includes `source`, `created_on` (`YYYY-MM-DD`), `category`/`kind`, `match_kind`, `relevance_score`, `exact_snippet`, and `original_chunk`.
  * **Mode 1A (Conversational automatic recall in `hermes-prospecta`):**
    * Stage 0 (`decision_model`, default `typesafe/jev-1.14`) decides `needs_memory` and `time_scope`.
    * Query writer (`query_writer_model`, default `openrouter/google/gemini-2.5-flash-lite`) resolves follow-ups against `agent_last` only, preserving `"quoted phrases"` verbatim.
    * Passes `excluded_sources` (last 2 turns of current session) into Stage 1 + Stage 2 (`reranker_model`, default `qwen/qwen3-reranker-8b`).
    * Formats the injected `<memory-context>` block with citation headers and `Exact match snippet:` lines.
  * **Mode 1B (Non-interactive task/commission recall):**
    * Add `memory.recall_for_task(title=..., ask=..., done=..., project=...)` (and `POST /lookup` batch support) that decomposes a task brief into 2–4 facet queries, runs batched Stage 1 + Stage 2, and returns a cited RAG context block.
  * **Mode 2A & 2B (Interactive Search & Interactive RAG):**
    * `memory.search()` and `POST /lookup` skip Stage 0 Jev, run Stage 1 + Stage 2, and return `exact_snippet` + citation metadata on every hit.
    * `memory.ask()` (Mode 2B interactive RAG) optionally runs Stage 3 (`decision_model` sufficiency gate when `stage3_sufficiency.enabled=True`) and synthesizes a grounded answer with `[1] [source | date]` citations and verbatim snippets.
* **Verification:** End-to-end tests for Mode 1A (chat turn), Mode 1B (task brief), Mode 2A (direct keyword/phrase search with snippets), and Mode 2B (cited RAG answer).

### Slice 5: Offline Verification on `augur-v2` and `forge-v2` (Read-Only, Before Any Live Switch)
* **Change:** None to live gateways.
* **Verification (read-only against the existing database):**
  1. Confirm `augur-v2` and `forge-v2` chunk coverage (Note: `augur-v2` has 9,987 `chunk` items and `forge-v2` has 76,636 `chunk` items already in `memory_items`, alongside the old `question` items; verify 100% of `documents` in both banks have at least one `kind='chunk'` item before ignoring `kind='question'`).
  2. Run a 20-query offline check (exact quotes like `"electric dreams"`, proper names, dates, conversational follow-ups, and technical queries) measuring Stage 1 latency, Stage 2 Qwen latency, citation/snippet formatting, and total time.
  3. Present the 20-query output table to the captain. **Only after your explicit word** do we update `augur-v2` / `forge-v2` `channel_config` & `recall_config` and restart the gateways.

---

## 4. What Stays Untouched Until You Say Go

* **Augur and Forge live gateways and configs** stay frozen as they are right now.
* **Existing database rows** in `augur-v2` and `forge-v2` are not deleted or modified duringSlices 1–4.
* **Spire (`spire-venue`)** waits until Prospecta's implementation and offline verification are complete and you are happy with how Augur and Forge behave.
