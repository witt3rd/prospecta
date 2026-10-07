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
| Jev candidates per request | `JEV_BATCH` / `JEV_MAX_BATCH` | 15 / 16 | Jev accepts at most 16 candidates per request; larger pools use several requests (the Linker batches by `JEV_MAX_BATCH` candidates, all relation questions each, within `JEV_INPUT_BYTES`) | physical |
| Jev timeout | `JEV_TIMEOUT_S` | 10 s | latency bound on an optional stage; failure keeps the fused order | ours-config (constructor arg) |
| Fallback chunk (`_matching_chunk`) | `_retain.py` | **removed clip** | the text is returned whole when a note has no chunk items | was ours |
| Graph hop clamp (`MAX_HOPS` min) | `channels/graph.py` | **removed clamp**; default 2 | `max_hops` is a walk-depth param; default stays 2 | ours-config |
| Graph node cap (was `node_cap` 60) and hop-2 `frontier` (200) | `channels/graph.py` `node_min_rel` | **removed**; every reached neighbour with score >= 0.4 x the best reached | scores only fall along a path, so the walk expands a hop-1 item only if one more step (decay x the largest type weight) could still reach `node_min_rel` x the best hop-1 score, and keeps what clears the relative cut. A stored `node_cap`/`frontier`/`seeds` param is ignored | was ours |
| Rerank pool (was `RERANK_POOL` 30) | `stages.py`, `recall_config.rerank.min_rel_score`; named depths `standard` 0.15 / `deep` 0.05 (`recall(..., depth=)`, `recall_config.depth`, see `recall-stages.md`) | **removed**; fused score >= 0.15 x the best fused score (`POOL_MIN_REL`, the scout's knee) | the pool decides only the size of the RERANK input and is the quality/cost knob: configurable down to 0.05 (about 1,083 notes per question on the scout's corpus; 0.15 dropped 27 of 273 strict-gold notes only together with the old channel cuts). The pool is every fused note at or above the relative score; the reranker sees all of it. Bounded only by the reader model's window (below) | was ours |
| Hop follow-ups (was `max_follow_ups` 2) and new notes (was `max_new` 10) | `stages.py` `_hop` | **removed** | the reader gives one follow-up per distinct missing fact (cheap searches, no LLM); every new note at or above `reader.hop_min_rel_score` (0.4) x the best of its follow-up run joins and the joined set is reranked again | was ours |
| Reader `top` 8 / `join_top` 15 | `stages.py` `_hop`, `reader.min_rel_score` | **removed**; rerank grade (0..3, else fused score) >= 0.6 x the best | the reader sees every qualifying note; the whole reranked set stays ahead of new candidates | was ours |
| Channel candidate `limit` 50, fusion `POOL` 30 | `channels/semantic.py`, `bm25.py`, `meta.py`, `recall.py`, `fusion.py` | **removed**; channels do not cut: `min_rel` defaults to 0 (cosine and BM25 alike; override per bank via the `min_rel` param), meta = whole filter set | a channel returns every candidate it can fetch (`fetch_until_cut`: fetches in pages of 64 (throughput) and doubles until the cut is reached). Fusion keeps every document; the pool cut is the stage's. A stored `limit`/`recall_config` count key is ignored | was ours |
| Graph seeds (was `seeds` 10) | `channels/graph.py` `seed_min_rel` | **removed**; every pool document with RRF >= 0.5 x the best | the walk's starting set | was ours |
| Linker caps (was `NEIGHBOURS` 10, `VECTOR_TOP` 5, `TEMPORAL_CLOSE_CAP` 5) | `_linker.py` `neighbour_min_rel` 0.9, `neighbour_min_cos` 0.5, `temporal_min_rel` 0.5 | **removed** | candidates per anchor: the pgvector neighbours with cosine >= 0.9 x the nearest, ending at the first marginal drop > 0.1 x the nearest, and never below the absolute cosine `neighbour_min_cos` 0.5 (HNSW top-n, doubling only until the stop is decided: cost is linear in notes; per-document candidates examined/judged and LLM cost are in the link state stats); the model-free fallback also keeps the absolute `vector_floor` 0.75; TEMPORALLY_CLOSE: every note within `TEMPORAL_DAYS` (3) whose closeness 1/(1+days) >= 0.5 x the closest. Jev judges candidates in batches of its request limit (physical); each unordered pair is judged once (`memory_link_pairs`). Not counts: `ENTITY_HUB` 30 and graph `hub_cap` 200 (hub-entity definition and walk work bound, design) | was ours |
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

## Score-based pools (what a stage sees)

Rule: a stage sees everything at or above a score relative to the best; only the reading
model's physical window can split it. Window: Claude Sonnet 5.5, 200,000 tokens
(`SONNET_CONTEXT_TOKENS`, 4 chars/token estimate, 4,000 reserved). A qualifying set that
exceeds it is split into consecutive batches and map-reduced, never truncated:

- **Rerank**: each batch is graded by its own listwise call; grades merge into one order
  (grade first, then the fused order). One batch (the usual case) is the unchanged call.
- **Reader**: each batch is read; sufficient only if every batch is, follow-ups are the union.
- **Failure**: a failed batch is retried once (`BATCH_ATTEMPTS`), then the stage fails loudly
  with a full untruncated warning (query and every note id) and the fused order stands: nothing is dropped.
- Defaults (`_scorecut.py`): pool 0.4, reader 0.6, hop 0.4, channels as above. Chosen from the
  design's RRF arithmetic (single-channel rank 30 ~0.6 of a top note) and tuned only on synthetic
  data: still to be tuned on the real 120 questions + the Greg question against a real bank.
  `min_rel 0` means no cut.
- **Thresholds are unmeasured starting defaults chosen by the author**: pool 0.4 of the best
  fused RRF score, channel cuts 0.6 cosine / 0.15 BM25, reader 0.6, hop 0.4. Config keys:
  channel param `min_rel`; `recall_config.rerank.min_rel_score`, `recall_config.reader.min_rel_score`,
  `recall_config.reader.hop_min_rel_score`. The rung-caretaker scout tunes them on roger with the
  real question set. `max_follow_ups` no longer applies (the cut is score-based).
- **Reranker cost on the synthetic bank goes down**: the pool (fused notes >= 0.4 x best) was 6-27
  notes over 8 probe queries (mean ~14) against the old fixed 30. On a real bank, where scores
  are flatter, it may be larger: that is what the tuning on real data decides.
- **Semantic channels never stop at an index's reach**: an HNSW scan returns about `ef_search`
  (40) rows, so a short page would look like an exhausted source. Each page fetch runs in its
  own transaction with `hnsw.iterative_scan = relaxed_order` (pgvector >= 0.8) or, on older
  pgvector, with index scans off (exact), logging one warning. BM25 backends (pg_search
  `LIMIT n`, in-process ranking of every match) return exactly `n` rows when more exist, so a
  short page is true exhaustion there.
- `prospecta eval` no longer slices to 30 (`pool=None` scores the whole selected pool).

Synthetic before/after (`/tmp/e11eval.py`: 320 notes, 8 set questions, stub embedder, no LLM):
hit@1 .875, hit@10 1.0, MRR .9375, cover@10 .906 both before and after; candidates per
question from the question channel 50 -> about 10. Cost (LLM) not measurable offline.

No `max_tokens` is passed to any LLM call. Not audited here: CLI/TUI display widths.

## Recall switches that are OFF by default

| Switch | Where | Default | Why |
|---|---|---|---|
| Rerank blend (`rerank.blend.enabled`, 0.7/0.3) | `stages.py` | **off** | with score-based pools it regressed hit@1 (V3 re-run: gold1 0.72 vs 0.86 off, gold2 0.85 vs 0.97 off) |
| Scope-promotion re-sort (meta param `promote`) | `channels/meta.py` | **off** | same V3 regression; opt in until it beats the v2 numbers |

`prospecta eval --rerank-blend/--no-rerank-blend --scope-promote/--no-scope-promote` force each per run. The promotion table below describes the code when `promote` is on.

## Scope promotion: cover@10 (Q096-style synthetic set)

60 non-members (fused 1.0 down), 40 members (fused ~0.3), 3 gold members buried in the set.
Printed by `tests/test_filters.py` (`pytest -s -k q096`):

| Variant | cover@10 |
|---|---|
| old `PROMOTE_MAX` cap (set of 40 > 12: no promotion) | 0.00 |
| boost, gold cosine 0.9 | 1.00 |
| boost, weak-cosine non-gold members (gold 0.85, others 0.05) | 1.00 |
| boost, all member cosines 0 (no boost on a zero score) | 0.00 |

### Graph and linker thresholds (second PR)

Config names and defaults, all relative to the best score (0 = no cut): graph channel params
`seed_min_rel` 0.5, `node_min_rel` 0.4; Linker `neighbour_min_rel` 0.9, `temporal_min_rel` 0.5.
Like the first PR's, these are unmeasured starting defaults chosen by the author (no links
exist on the synthetic bank, so the synthetic eval is unchanged); tuning on the real question
set is for the rung-caretaker scout on roger. Hub entities (`hub_cap` 200 holders) remain a
work bound of the walk; at the default cut hub neighbours (0.5 x best) pass, so a graph walk
over a bank with huge hubs can return many notes: raise `node_min_rel` per bank if so.
