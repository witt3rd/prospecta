# Numeric limits (audit)

Standing rule: **no caps**. A limit is kept only if it is physical (a provider, database or
model window forces it) or a tuning / design constant that does not clip content or evidence.
Everything else is removed or score-based. Where a physical limit forces clipping, the full
untruncated text is logged in a warning.

Kind: **physical** = imposed from outside; **design** = a parameter chosen for a physical
window, not a clip; **tuning** = a constant of an algorithm, not a cap; **ours-config** = our
bound on work or cost, configurable, never clips evidence; **caller** = the caller's own
argument.

This page lists the limits of the current design (`docs/postgres-hybrid.md`). The limits of the
removed graph, linker, reader / hop and BM25 code are in git history.

## The write path

| Limit | Where | Value | Why | Kind |
|---|---|---|---|---|
| Chunk size / overlap | `_retain.py` `CHUNK_MAX_CHARS` 1000, `CHUNK_OVERLAP_CHARS` 100; `index_directory(chunk_size=1000, chunk_overlap=200)` | | sized for the embedder window and precise retrieval; **not a clip**: the whole text is kept as chunks | design |
| Embedding dimension | `banks.embedding_dim` | 1536 for the default embedder | `text-embedding-3-large` cut to 1536 (pgvector's HNSW limit is 2000); fixed per bank at creation | physical |
| Embedding batch | `_embed_migrate.py` `BATCH_SIZE` | 128 | provider batch efficiency; no effect on content | ours-config |
| Import batch | `prospecta import hindsight --batch` | 1000 | rows per transaction | ours-config |
| Date parse `[:10]` | `_filters.py` | 10 characters | an ISO date is 10 characters | format |

## Stage 1 (the `hybrid_chunk` channel)

| Limit | Where | Value | Why | Kind |
|---|---|---|---|---|
| Reserved slots and HNSW `ef_search` | `db/queries.py` `STAGE1_*`; channel params `semantic_limit` / `stemmed_limit` / `exact_limit` / `fused_limit` / `hnsw_ef_search` | 25 / 20 / 20 / 50, 100 | each retriever keeps its own top slots so an exact hit always reaches the reranker, which reads the fused 50; none clips evidence | design, ours-config |
| RRF weights and `k` | `_STAGE1_SQL` | 0.45 / 0.25 / 0.30, k = 60 | the fallback score inside the fused 50 and the order when Stage 2 does not run; the cross-encoder re-scores every pair, so only membership of the 50 matters | tuning |
| Plain-question OR | `docs/exact-tsv.md` | **no cap** on words | a cap dropped words silently; what a long question costs is measured in `docs/exact-tsv.md` | measured |
| Date-only listing | `fused_limit` | 50 | the newest chunks inside the window | design |

## Stage 2 (the cross-encoder) and the rerank pool

| Limit | Where | Value | Why | Kind |
|---|---|---|---|---|
| Rerank pool | `recall_config.rerank.min_rel_score`, depths `standard` 0.15 / `deep` 0.05 (`depth=`, `recall_config.depth`, `docs/recall-stages.md`) | fused score >= **0.15 x the best fused score** (`POOL_MIN_REL`, `_scorecut.py`), 0.05 for `deep` | the pool decides only the size of the RERANK input: it is every fused note at or above the relative score, never a count. With Stage 1's reserved slots it admits the whole fused 50 in practice | tuning |
| Cross-encoder request size | `stages.py` `RERANK_BATCH_TOKENS` | 32,000 tokens (est. 4 chars / token) | `qwen/qwen3-reranker-8b` reads 41K tokens; the rest is margin for tokenizer variance, the instruction and the query. A larger pool goes out as several requests (scores are calibrated, so they merge); a lone note is never cut | design |
| Score floor | `rerank.min_score` | 0.15 | an absolute cut on the calibrated 0..1 score; the only thing that empties a result | tuning |
| `top_n` | `rerank.top_n` | 10 | the notes kept after the floor and the near-copy collapse (Mode 1 keeps every note above the floor) | ours-config |
| Timeout | `rerank.timeout_s` | 4 s | latency bound on an optional stage; failure keeps the Stage 1 order | ours-config |
| Retry | `rerank.retries` | 1 (0 turns it off) | one more request when the provider answers 503 or 429, inside the same `timeout_s` (the retry gets what is left, only if at least 0.5 s remain; a timeout, any other status or a malformed reply is never retried). Both requests are logged | ours-config |
| Near-copy collapse inside Stage 2 | `NEAR_COPY_JACCARD` | token Jaccard 0.88 | two notes this alike are one answer (`deduplicate_near_copies`) | tuning |
| Evidence clip for the reranker | `stages.py` | **none** | the best chunk goes in full; a chunk is already bounded by chunking | removed |

## Mode 1 (query expansion)

| Limit | Where | Value | Why | Kind |
|---|---|---|---|---|
| Deadline | `expansion.deadline_s` / `deadline=` | 7 s | one clock for the gate, the writer, every search and every rerank; a search still running at the deadline is dropped and recorded | ours-config |
| Gate share | `_expand.py` `GATE_SHARE` | 30% of the deadline | a gate that is late lets the message through | ours-config |
| Writer share | `WRITER_SHARE`, `SEARCH_RESERVE` | 50% of the deadline, and the searches always keep 25% | the writer cannot starve the searches | ours-config |
| Searches | `expansion.max_searches` | 20 | the hard maximum: a bad model reply cannot explode the fan-out; a reply over it is cut (first searches kept) and recorded | ours-config |
| Concurrency | `expansion.max_concurrency` | 8, and never more than the pool | searches in the database at once; a search's embed and rerank hold no connection | ours-config |
| Token valve | `expansion.max_tokens` | 0 (off) | the block is sized by relevance, never by a count; the valve keeps records best first until the estimate is reached (always at least one) and says how many it left out | ours-config |
| Stage 1 order cut without a rerank | `expansion.fallback_min_rel` | 0.5 x the search's best RRF score | never a count | ours-config |
| Near-copy collapse by meaning | `stages.py` `NEAR_COPY_JACCARD` / `NEAR_COPY_COSINE`; call argument `near_copy_cosine=`, `recall_config.expansion.near_copy_cosine`, `PROSPECTA_NEAR_COPY_COSINE` | 0.88 token Jaccard, then 0.90 chunk-embedding cosine | a record at or above the cosine to a better-scored kept record is a copy of it and collapses (it is listed in the kept one); 0.90 was measured on the live banks (`docs/postgres-hybrid.md`, Mode 1 point 6): the same note in several wordings is at 0.91 and above, distinct notes at 0.876 and below | ours-config (measured) |
| Gate threshold | `stage0_gate.needs_memory_threshold` | 0.5 | a decision on the 0..1 `needs_memory` noul, not a clip | tuning |
| Suggested searches | `suggest_searches` | chat: 3 up to 30 words, 5 up to 80, 8 up to 200, 12 beyond; task: one per 15 words, 8 to 20 | the writer is told the number; the maximum above is the only limit | tuning |
| Period sanity | `MIN_DATE`, a year ahead | 1990-01-01 .. +1 year | a period the writer names outside it, or reversed, is dropped (no window, never a wrong one) | design |
| Term length | `TERM_MAX_CHARS` | 200 characters | a quoted phrase longer than this is not a verbatim term | design |

Measured latency of 20 searches (`tests/test_expand_perf.py`) is in `docs/postgres-hybrid.md`,
section 1, point 9.

## Stage 3 (the sufficiency gate)

| Limit | Where | Value | Why | Kind |
|---|---|---|---|---|
| Threshold | `stage3_sufficiency.threshold`, `STAGE3_THRESHOLD` | 0.70 | `evidence_sufficient` / 3 under this runs one broadened retry; a decision, not a clip | tuning |
| Decision-model request size | `stages.py` `JEV_INPUT_BYTES` | 60,000 B | the provider's request limit; a lone passage over `JEV_PASSAGE_MAX_BYTES` is cut with a full-text warning | physical |
| Decision-model timeout | `JEV_TIMEOUT_S` | 10 s | latency bound on an optional stage | ours-config |

## Calls and display

| Limit | Where | Value | Why | Kind |
|---|---|---|---|---|
| `limit` on `search` / `recall` / `ask` | `Memory.search(limit=10)`, `ask(limit=8)` | 10 / 8 | the caller's request for how many results | caller |
| `/lookup` | `service.py` `MAX_LIMIT`, questions per call | 100, 50 | request bounds of the service; a task takes no `limit` | ours-config |
| `content_preview` | `memory.py` `PREVIEW_LEN` | 200 chars | inspection-only audit column in `recall_events`; full content stays in `memory_items` | display only |

No `max_tokens` is passed to any model call. Not audited here: CLI / TUI display widths.
