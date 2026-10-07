# Numeric limits (audit)

Standing rule: **no caps**. A limit is kept only if it is physical (a provider,
database or model window forces it) or a tuning/design constant that does not
clip content or evidence. Everything else is removed or score-based. Where a
physical limit forces clipping, the full untruncated text is logged in a warning.

Kind: **physical** = imposed from outside; **design** = a parameter chosen for a
physical window, not a clip; **tuning** = a constant of an algorithm, not a cap;
**ours-config** = our bound on work or cost, configurable, never clips evidence.

| Limit | Where | Value | Why | Kind |
|---|---|---|---|---|
| Scope promotion count (`PROMOTE_MAX`) | `channels/meta.py` | **removed** | replaced by a score boost: `promote_weight` (meta param, 1.0) × top score × cosine; no count | was ours |
| Scope-set size for grounded synthesis (`SCOPE_MAX`) | `_synth.py` | **removed** | an explicit `scope=` is always the whole set | was ours |
| Evidence clip for the reranker (`EVIDENCE_CHARS`, 1000) | `stages.py` | **removed** | the best chunk goes in full; a chunk is already bounded by chunking | was ours |
| Jev passage clip (`JEV_PASSAGE_CHARS`, 2400) | `stages.py`, `_linker.py` | **removed** | replaced by `fit_passage`: cut only if one passage cannot fit a request (below), with a full-text warning | was ours |
| Jev request size | `stages.py` `JEV_INPUT_BYTES` | 60,000 B | Jev provider request limit; items are split across requests, a lone passage over `JEV_PASSAGE_MAX_BYTES` is cut with a warning | physical |
| Jev candidates per request | `JEV_BATCH` / `JEV_MAX_BATCH` | 15 / 16 | Jev accepts at most 16 candidates per request; larger pools use several requests | physical |
| Jev timeout | `JEV_TIMEOUT_S` | 10 s | latency bound on an optional stage; failure keeps the fused order | ours-config (constructor arg) |
| Fallback chunk (`_matching_chunk`) | `_retain.py` | **removed clip** | the text is returned whole when a note has no chunk items | was ours |
| Graph hop clamp (`MAX_HOPS` min) | `channels/graph.py` | **removed clamp**; default 2 | `max_hops` is a walk-depth param; default stays 2 | ours-config |
| Graph node cap (`node_cap`) | `channels/graph.py` | 60 (param) | bounds the walk's work (Jev-Mem's value); best-first by score, so it drops only the weakest | ours-config |
| Rerank pool | `stages.py` `RERANK_POOL`, `recall_config.rerank.pool` | 30 | LLM cost/latency; the pool is the fused top by score; raise or set per bank | ours-config |
| Reader hops | `recall_config.reader` `max_follow_ups`, `max_new` | 2, 10 | bound LLM calls and pool growth; candidates join best-rank first; per bank | ours-config |
| Reader `top` / `join_top` | `recall_config.reader` | 8 / 15 | how many ranked notes the reader sees / are kept ahead of new candidates | ours-config |
| Channel candidate `limit`, `POOL` | `channels/recall.py`, registry params | 50 per channel, pool 30 | candidates per channel before fusion; `pool = max(POOL, limit)`; per-channel param | ours-config |
| Seeds | graph `seeds` | 10 | graph walk starting set | ours-config |
| Linker caps | `_linker.py` `ENTITY_CAP` 10, `TEMPORAL_CLOSE_CAP` 5, `VECTOR_TOP` 5 | | bound links created per note at retain (not recall evidence) | ours-config |
| `top` / `limit` on `search` | `Memory.search(limit=10)` | 10 | the caller's request for how many results | caller's argument |
| Grounded evidence notes (was `TOP_NOTES` 6) and chunks per note (was `CHUNKS_PER_NOTE` 3) | `_synth.py` | **removed** | notes and chunks are kept by score: `>= recall_config.evidence.min_rel_score` (0.5) x the best; an explicit `scope=` is the whole set; `evidence_top` is an optional per-call note count, `None` by default | was ours |
| Synthesizer context window | `_synth.py` `DEFAULT_CONTEXT_TOKENS` | 200,000 tokens (`recall_config.evidence.context_tokens`) | physical model window (4 chars/token estimate, 4,000 tokens reserved for template/question/answer); only this can drop evidence, lowest-ranked first, with one full warning naming what was dropped | physical |
| Chunk size / overlap | `_retain.py` `CHUNK_MAX_CHARS` 1000, `CHUNK_OVERLAP_CHARS` 100 | | **design parameter** sized for the embedder window and precise retrieval; not a clip: the whole text is kept as chunks | design |
| Embedding dimension | schema | 1536 | `text-embedding-3-large` cut to 1536 (pgvector index limit is 2000) | physical |
| Embedding batch | `_embed_migrate.py` `BATCH_SIZE` | 128 | provider batch efficiency; no effect on content | ours-config |
| Import batch | `prospecta import-hindsight --batch` | 1000 | rows per transaction | ours-config |
| RRF `k` | `rrf_k`, `Memory(rrf_k=60)` | 60 | **fusion constant** (dampens rank differences), not a cap | tuning |
| Graph `decay`, type weights, `promote_weight` | channel params | 0.5, .., 1.0 | score weights, not caps | tuning |
| Gate threshold | `recall_config.gate.threshold` | 2.95 | score cut on Jev's 0..3 scale; a decision, not a clip | tuning |
| `content_preview` | `memory.py` `PREVIEW_LEN` | 200 chars | inspection-only audit column in `recall_events`; full content stays in `memory_items` | display only |
| Regex/filter date parse `[:10]` | `_filters.py` | 10 chars | an ISO date is 10 characters | format |

No `max_tokens` is passed to any LLM call. Not audited here: CLI/TUI display widths.

## Scope promotion: cover@10 (Q096-style synthetic set)

60 non-members (fused 1.0 down), 40 members (fused ~0.3), 3 gold members buried in the set.
Printed by `tests/test_filters.py` (`pytest -s -k q096`):

| Variant | cover@10 |
|---|---|
| old `PROMOTE_MAX` cap (set of 40 > 12: no promotion) | 0.00 |
| boost, gold cosine 0.9 | 1.00 |
| boost, weak-cosine non-gold members (gold 0.85, others 0.05) | 1.00 |
| boost, all member cosines 0 (no boost on a zero score) | 0.00 |
