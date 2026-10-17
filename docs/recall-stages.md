# Recall configuration: channels, stages, models, cited evidence

A bank's recall is configured by two JSON columns: `banks.channel_config` (which channels
find the candidates: Stage 1) and `banks.recall_config` (what runs after them: Stage 2, the
Mode 1 settings, Stage 3, the models, the depth). `docs/postgres-hybrid.md` is the design; this
page is the reference for the settings.

**A bank created fresh stores** (`prospecta/bank_defaults.py`):

```python
channel_config = [{"name": "hybrid_chunk", "enabled": True, "weight": 1, "params": {}}]
recall_config  = {"rerank": {"enabled": True, "stage": "cross_encoder"}}
```

Everything not named is the code default below: the model, the floor, `top_n` and the timeout
are not stored, so a changed default (or `PROSPECTA_RERANKER_MODEL`) reaches every bank that
does not pin its own. Change a bank with `Memory.set_channel_config([...])` and
`Memory.set_recall_config({...})` (they replace the whole value; `{}` turns every stage off).

**One channel, and what a stored config may hold.** `hybrid_chunk` is the only channel, and the
keys below are the only `recall_config` keys. A **new** config (`set_channel_config`,
`set_recall_config`) is checked: another channel name or another key raises `ValueError`. A
**stored** one is never rewritten and never an error: a `channel_config` that is empty, or names
only channels the library no longer has, runs `hybrid_chunk` with its defaults; an entry that
names a removed channel (`dense_chunk`, `meta`, `bm25`, `keyword`, `question`, `graph`) is
ignored beside it and listed in the recall trace's `skipped`; a `recall_config` key the library
does not read (`gate`, `reader`, `link_completeness`, `fusion`, `evidence`, `rerank.blend`, a
`rerank.stage` other than `cross_encoder`, ...) is ignored, silently. A bank that stored a rerank
stage the library does not have therefore keeps the Stage 1 order. (`hybrid_chunk` with
`"enabled": false` is an explicit choice and stays off: nothing is retrieved, and
`expand_and_recall` returns an `error`.)

| `recall_config` key | what | page section |
|---|---|---|
| `rerank` | Stage 2: `enabled`, `stage` (`cross_encoder`), `model`, `top_n`, `min_score`, `timeout_s`, `deduplicate_near_copies`; `min_rel_score` (the pool cut) | Stage 2 |
| `expansion` | Mode 1: `max_searches`, `max_concurrency`, `deadline_s`, `max_tokens`, `fallback_min_rel`, `near_copy_cosine` | Mode 1 settings |
| `stage0_gate` | Mode 1 gate: `enabled`, `needs_memory_threshold` | Mode 1 settings |
| `stage3_sufficiency` | Stage 3: `enabled`, `threshold` | Stage 3 |
| `models` | `decision_model`, `query_writer_model` | Models |
| `depth` | the bank's default rerank-pool cut, `standard` or `deep` | Recall depth |

## Stage 1: the `hybrid_chunk` channel

One query over the bank's `kind = 'chunk'` items with three retrievers, each keeping its own
reserved slots (`docs/postgres-hybrid.md`, sections 3 to 6). Channel `params`, all optional
integers >= 1:

| param | default | |
|---|---|---|
| `semantic_limit` | 25 | slots of the semantic (HNSW) retriever |
| `stemmed_limit` | 20 | slots of the stemmed lexical retriever (`body_tsv`) |
| `exact_limit` | 20 | slots of the exact / phrase retriever (`exact_tsv`) |
| `fused_limit` | 50 | candidates the fusion keeps |
| `hnsw_ef_search` | 100 | candidates the HNSW scan reads before it keeps scanning |

Per-call filters of `search`, `recall`, `ask`, `expand_and_recall` and `POST /lookup`, pushed into
all three retrievers: `date_from` / `date_to` (the note's date, inclusive), `excluded_sources`
(the anti-echo list of `documents.source` values) and `metadata_filter`; they apply to every bank.
Needs pgvector 0.8 or newer (`docs/deployment.md`).

## Stage 2: the cross-encoder

`rerank = {"enabled": true, "stage": "cross_encoder", ...}` (`CrossEncoderReranker`). Every
pooled note (header plus best chunk) is scored against the query by a cross-encoder through
OpenRouter's rerank endpoint, which returns a calibrated `relevance_score` per note. Wire shape,
verified against <https://openrouter.ai/docs/api/api-reference/rerank/submit-a-rerank-request.md>
(read 2026-10-17) and one live call: `POST https://openrouter.ai/api/v1/rerank`, body
`{model, query, documents, top_n}` (the query is `Instruct: ...\nQuery: <query>`), reply
`{model, provider, results: [{index, relevance_score, document}], usage: {total_tokens, cost}}`.
Default model `qwen/qwen3-reranker-8b` (served by Fireworks, $0.20 per M tokens). `top_n` sent is
the number of documents in the request, because the cuts below need every score.

After scoring, in this order: notes under `min_score` (default 0.15) are cut; near-copies
collapse into the highest-scored one (`deduplicate_near_copies`, default true: token Jaccard of
the evidence text above 0.88; Jaccard because no embedding is at hand at this stage, it is
deterministic and costs no call); the best `top_n` (default 10) stay. These are the only ways the
stage removes a note and it never adds one; the record lists what each cut removed
(`n_below_min_score`, `collapsed` as dropped -> kept document, `n_over_top_n`). The grade of a
kept note (0..1) lands in `RecalledMemory.scores["rerank"]` and is its `relevance_score`. If the
provider answers and nothing reaches the floor, the result is empty: that is the answer.

Any provider error, timeout (`timeout_s`, default 4 s), HTTP 429/503, or reply that is not
exactly one finite score per note keeps the Stage 1 order for the whole pool and records
`fallback_reason` (the call is still logged with its error); **the stage never raises into
recall.** With no rerank transport on the Memory (no `OPENROUTER_API_KEY`) the stage cannot run
and the Stage 1 order stands, with the reason recorded. Requests carry at most 32K estimated
tokens (the model reads 41K; see `docs/limits.md`); a larger pool goes out as parallel requests
whose scores merge. Accounting: each request is one `llm_calls` row (`model` as the provider
reports it, `tokens_in` = `usage.total_tokens`, `cost_usd` = `usage.cost`, `duration_ms`,
`error`); the stage record carries the configured `model`, the totals, `latency_ms` and
`fallback_reason`.

Wire the transport: `Memory(rerank_transport=openrouter_rerank_transport())` (key from
`OPENROUTER_API_KEY`; the CLI and the service build it when the key is set).

```python
mem.set_recall_config({
  "rerank": {"enabled": True, "stage": "cross_encoder", "model": "qwen/qwen3-reranker-8b",
             "top_n": 10, "min_score": 0.15, "timeout_s": 4.0, "deduplicate_near_copies": True},
  "models": {"decision_model": "typesafe/jev-1.14"},
})
```

`rerank.min_score` is the absolute floor on the cross-encoder's score; it is not
`rerank.min_rel_score`, which cuts the rerank *pool* relative to the best fused score (see Recall
depth).

## Models

All configurable; the first that applies:

- **Reranker model**: the per-call `reranker_model=` (`search`, `recall`, `recall_for_task`, `ask`,
  `/lookup`), then the `Memory(reranker_model=)` argument, then `recall_config.rerank.model`, then
  `PROSPECTA_RERANKER_MODEL`, then `qwen/qwen3-reranker-8b`.
- **Decision model** (System One: the Stage 0 gate and the Stage 3 sufficiency gate): the per-call
  `decision_model=`, then `Memory(decision_model=)`, then `recall_config.models.decision_model`,
  then `PROSPECTA_DECISION_MODEL` (alias `PROSPECTA_JEV_MODEL`, read at import), then
  `typesafe/jev-1.14`. It needs `Memory(jev=JevScore(openrouter_jev_transport()))`
  (`POST https://openrouter.ai/api/v1/systemone`).
- **Query-writer model** of `expand_and_recall` (and so of `recall_for_task`): the per-call
  `query_writer_model=`, then `recall_config.models.query_writer_model`, then
  `PROSPECTA_QUERY_WRITER_MODEL`, then `openrouter/google/gemini-2.5-flash-lite`; it is passed to
  the `llm` callable as a `model=` hint when the callable accepts one.

## Mode 1 settings (`expansion`, `stage0_gate`)

`Memory.expand_and_recall(prompt, kind="chat" | "task", ...)` and `Memory.recall_for_task(...)`
(`docs/postgres-hybrid.md`, section 1, is the design; `docs/limits.md` has the numbers). Per
call, else the bank's `recall_config`, else the default:

| setting | call argument | bank `recall_config` | default |
|---|---|---|---|
| deadline for everything | `deadline=` | `expansion.deadline_s` | 7 s |
| hard maximum of searches | `max_searches=` | `expansion.max_searches` | 20 |
| searches in the database at once | `max_concurrency=` | `expansion.max_concurrency` | 8 (and the pool size) |
| token valve | `max_tokens=` | `expansion.max_tokens` | 0 (off) |
| Stage 1 order cut when no rerank | | `expansion.fallback_min_rel` | 0.5 x the best |
| near-copies collapsed by meaning (chunk-embedding cosine at or above; a number in (0, 1]) | `near_copy_cosine=` | `expansion.near_copy_cosine` (env `PROSPECTA_NEAR_COPY_COSINE`) | 0.90 |
| gate threshold / switch | `needs_memory_threshold=`, `gate=` | `stage0_gate.needs_memory_threshold`, `.enabled` | 0.5 / on |
| score floor of Stage 2 | `min_score=` | `rerank.min_score` | 0.15 |
| models | `query_writer_model=`, `decision_model=`, `reranker_model=` | `models.*`, `rerank.model` | see Models |

`decision=callable` injects the caller's own gate (a number in 0..1, a bool, a `GateDecision` or a
mapping with `needs_memory`); `gate=False` switches it off; with neither a callable nor a
`Memory(jev=)` the gate does not run (recorded in `fallbacks`). Mode 1 reranks with the
cross-encoder only, and only when the bank's `rerank.stage` is `cross_encoder` **and** the Memory
has a rerank transport; otherwise each search keeps its Stage 1 order cut to the candidates within
`fallback_min_rel` of its best RRF score. `POST /lookup {"task": ...}` returns the same block
plus `searches`, `fallbacks` and `error`, and takes no `limit`.

## Cited evidence and the modes

Every `RecalledMemory` carries the cited-evidence contract next to `source` and `original_chunk`:
`created_on` (`YYYY-MM-DD`), `kind` (`documents.source_kind`), `match_kind` (`exact+semantic`,
`exact`, `lexical`, `semantic`, `date`; from the retrievers that found the note; `date` is a note
listed only because it lies in a wordless question's date window), `relevance_score` (the cross-encoder's 0..1 score when it graded the note, else the
fused `score`) and `exact_snippet` (`ts_headline` of the matched words, `<<marked>>`, from the
`hybrid_chunk` channel); Mode 1 adds `found_by`. `prospecta.cited.format_cited_rag_block(items)`
renders items as the cited, non-instructional block (`[n] source=... | date=... | kind=... |
match=...`, an `Exact match snippet:` line when one fired, then the full chunk).

- **Mode 2A, search** (`Memory.search`, `recall`, `POST /lookup`): Stage 1 + Stage 2, no Stage 0
  gate. `date_from`, `date_to` and `excluded_sources` are pushed into the channel's three
  retrievers.
- **Mode 1, automatic recall by query expansion** (`expand_and_recall`, `recall_for_task`): the gate
  (chat only), the query writer (the prompt becomes N structured searches), every search through
  Stage 1 and Stage 2 against its own query in parallel, one merge, one cited block, inside a
  deadline. It returns an `ExpandedRecall` (`block`, `items`, `searches`, `runs`, `fallbacks`,
  `timings_ms`, `accounting`; `.queries`) and never raises. Calls and one `recall` event are
  recorded for a task by default (`record=`), not for a chat.
- **Mode 2B, ask** (`Memory.ask(question)`): Stage 1 + Stage 2, then the synthesis model answers
  from the top notes alone and cites `[n]` per statement. `AskResult.citations` ties each `[n]` to
  its source, date and the verbatim sentence of the chunk that supports it; `rendered` appends
  them as `[n] [source | date] "snippet"`. With no evidence the answer is `not in memory` and no
  model is called.

### Stage 3: the sufficiency gate

Off by default. `recall_config.stage3_sufficiency = {"enabled": true, "threshold": 0.70}` (or
`ask(sufficiency=True, sufficiency_threshold=...)`): the decision model answers the System One
`evidence_sufficient` score question (0..3) over the excerpts; the verdict is that score / 3.
Under the threshold ONE broadened retrieval runs (rerank pool at `deep`, twice the limit, the date
window dropped; `excluded_sources` and the metadata filter stay), its notes join the evidence, and
the verdict is taken again; the model answers either way. A gate that fails (no `Memory(jev=)`,
provider error) never blocks the answer: `AskResult.gate` holds the error and `sufficient` is
`None`.

## Recall depth

Two depths set the rerank pool cut (the size of the reranker's input):

| depth | pool cut (x best fused score) | use |
|---|---|---|
| `standard` (default) | 0.15 | automatic per-turn recall |
| `deep` | 0.05 | explicit recall, set / discovery questions, Stage 3's broadened retry |

The parameter is named exactly **`depth`**, with the values `"standard"` and `"deep"` (anything
else raises `ValueError`; `None` / omitted = not set). Precedence, first that applies:

1. the per-call `depth`;
2. an explicit `recall_config.rerank.min_rel_score` (a raw cut, kept for back-compat);
3. the bank default `recall_config.depth` (`mem.set_recall_config({"depth": "deep", ...})`);
4. `standard`.

Where it is accepted: `Memory.recall(queries, ..., depth="deep")`, `Memory.search(text, ...,
depth="deep")`, `expand_and_recall(..., depth=)`, `ask(..., depth=)`; the CLI
`prospecta search QUERY --depth {standard,deep}` and `prospecta eval QUESTIONS.md --depth ...`.
The trace is `recall_events.depth` (migration 0015; NULL on rows written before it).

## Accounting

`recall_events.cost_usd`, `tokens_in`, `tokens_out`, `n_llm_calls` sum the recall's model calls;
each call also lands in `llm_calls` with `model`, `tokens_in`, `tokens_out`, `cost_usd`
(`docs/observability.md`). An LLM callable reports usage by returning
`prospecta.stages.LLMResult` (`make_accounted_llm()` does); a bare string works and leaves the
usage NULL.
