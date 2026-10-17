# Recall stages: rerank, Jev gate, reader and hop

After the channels fuse (`banks.channel_config`), a bank can run more stages,
all configured by `banks.recall_config` (migration 0009; `{}` = none, recall is
unchanged). Set it with `Memory.set_recall_config(...)`.

```python
mem.set_recall_config({
  "rerank": {"enabled": True, "stage": "sonnet_listwise", "min_rel_score": 0.15,
             "blend": {"enabled": False, "weight_rerank": 0.7, "weight_fused": 0.3, "floor": False}},
  "gate":   {"enabled": False, "threshold": 2.95},          # Jev, needs Memory(jev=...)
  "reader": {"enabled": False, "min_rel_score": 0.6, "hop_min_rel_score": 0.4},
})
```

- **Depth** (`standard` | `deep`): the named rerank pool cut. `standard` = 0.15 x the
  best fused score (the default; automatic per-turn recall), `deep` = 0.05 (explicit
  recall, set/discovery questions, map-reduce). See "Recall depth" below.
- **Sonnet rerank** (`SonnetListwise`): every fused note scoring >= `min_rel_score` x the best fused score (batched by the model window, see `limits.md`), one best chunk (in
  full, never clipped) per note under a header of note name, date and person; the model replies
  JSON `{grades, ranking}`; order = ranking, then grade, then fused order. Any
  failure (provider error, unparseable reply) keeps the fused order and records
  `fallback_reason` in `recall_events.rerank`.
- **Blend** (`rerank.blend`, **off by default**, opt in with `enabled: true`; see `limits.md`): when on, the reranker's order
  is blended with the fused order, not substituted for it. score =
  `weight_rerank` * rank score (1.0 first .. 0.0 last) + `weight_fused` * fused
  RRF / best fused RRF. Defaults `weight_rerank` 0.7, `weight_fused` 0.3. With
  `floor: true` the fused top `keep` (3) stay inside the first `within` (10).
  The floor is off by default. With blend off (the default) the order is the pure reranker order.
  Applies to either stage; the parameters and `moved` land in
  `recall_events.rerank.blend`.
- **Jev** (`JevScore`, `rerank.stage = "jev_score"`) in Spire's call shape: one
  `score` question per candidate, criteria 0 to 3, 15 candidates per request (requests are split by this size), 60 KB per request, 10 s timeout, Jev only reorders. With
  `gate.enabled`, Jev scores first; if its top score >= `threshold` the Jev order
  stands, else (or if Jev fails) Sonnet reranks. Jev is built dark: pass
  `Memory(jev=JevScore(openrouter_jev_transport()))` (key from `OPENROUTER_API_KEY`).
  Wire shape (verified against Spire's `library-rank.ts`, rung's `jev.rs` and
  Jev-Mem's recorded live exchanges): `POST https://openrouter.ai/api/v1/systemone`,
  model `typesafe/jev-1.14` (override: see Models below; previous default `typesafe/jev-1.13`), body exactly `{model, state:{query}, questions:{p0..:
  {type:"score", instructions:{task, kind, title, text}, criteria}}}` with the
  four-string 0..3 legend and the record `task` copied from `library-rank.ts`;
  the reply's `answers` must hold exactly the asked ids, each a finite score in
  0..3, else there is no ranking and the fused order stands. Cost and tokens come
  from `usage.cost`, `usage.input_tokens`, `usage.output_tokens`. Tests use an
  offline transport of the same shape.
- **Cross-encoder** (`CrossEncoderReranker`, `rerank.stage = "cross_encoder"`): Stage 2 of
  `postgres-hybrid.md`. Every pooled note (header + best chunk) is scored against the query by a
  cross-encoder through OpenRouter's rerank endpoint, which returns a calibrated
  `relevance_score` per note. Wire shape, verified against
  <https://openrouter.ai/docs/api/api-reference/rerank/submit-a-rerank-request.md> (read
  2026-10-17) and one live call: `POST https://openrouter.ai/api/v1/rerank`, body
  `{model, query, documents, top_n}` (the query is `Instruct: ...\nQuery: <query>`), reply
  `{model, provider, results:[{index, relevance_score, document}], usage:{total_tokens, cost}}`.
  Default model `qwen/qwen3-reranker-8b` (served by Fireworks, $0.20 per M tokens). We send
  `top_n` = the number of documents in the request, because the cuts below need every score.
  After scoring, in this order: notes under `min_score` (default 0.15) are cut; near-copies
  collapse into the highest-scored one (`deduplicate_near_copies`, default true: token Jaccard of
  the evidence text above 0.88; Jaccard because no embedding is at hand at this stage, it is
  deterministic and costs no call); the best `top_n` (default 10) stay. These are the only ways
  the stage removes a note and it never adds one; the record lists what each cut removed
  (`n_below_min_score`, `collapsed` as dropped -> kept document, `n_over_top_n`). The grade of a kept note
  (0..1) lands in `RecalledMemory.scores["rerank"]`. If the provider answers and nothing reaches
  the floor, the result is empty: that is the answer. Any provider error, timeout
  (`timeout_s`, default 2.5 s), HTTP 429/503, or reply that is not exactly one finite score per
  note keeps the fused order for the whole pool and records `fallback_reason` (the call is still
  logged with its error); the stage never raises into recall. Requests carry at most 32K estimated
  tokens (the model reads 41K; see `limits.md`); a larger pool goes out as parallel requests whose
  scores merge. Accounting: each request is one `llm_calls` row (`model` as the provider reports it,
  `tokens_in` = `usage.total_tokens`, `cost_usd` = `usage.cost`, `duration_ms`, `error`); the
  stage record carries the configured `model`, the totals, `latency_ms` and `fallback_reason`.
  Wire it like Jev: `Memory(rerank_transport=openrouter_rerank_transport())` (key from
  `OPENROUTER_API_KEY`; the CLI builds it when the key is set).

  ```python
  mem.set_recall_config({
    "rerank": {"enabled": True, "stage": "cross_encoder", "model": "qwen/qwen3-reranker-8b",
               "top_n": 10, "min_score": 0.15, "timeout_s": 2.5, "deduplicate_near_copies": True},
    "models": {"decision_model": "typesafe/jev-1.14"},
  })
  ```

  `rerank.min_score` is the absolute floor on the cross-encoder's score; it is not
  `rerank.min_rel_score`, which keeps cutting the rerank *pool* relative to the best fused score.
  The gate needs `sonnet_listwise`, so it does not combine with this stage.
- **Models** (all configurable, first that applies): the reranker model is the per-call
  `reranker_model=` (`search`, `recall`, `recall_for_task`, `ask`, `/lookup`), then the
  `Memory(reranker_model=)` argument, then `recall_config.rerank.model`, then
  `PROSPECTA_RERANKER_MODEL`, then `qwen/qwen3-reranker-8b`. The decision model (Jev: `jev_score`,
  the gate, the Jev reader, the stage 3 sufficiency gate) is the per-call `decision_model=`, then the
  `Memory(decision_model=)` argument, then `recall_config.models.decision_model`, then
  `PROSPECTA_DECISION_MODEL` (alias `PROSPECTA_JEV_MODEL`, read at import), then `typesafe/jev-1.14`.
  The query-writer model of `expand_and_recall` (and so of `recall_for_task`) is the per-call `query_writer_model=`, then
  `recall_config.models.query_writer_model`, then `PROSPECTA_QUERY_WRITER_MODEL`, then
  `openrouter/google/gemini-2.5-flash-lite`; it is passed to the `llm` callable as a `model=` hint
  when the callable accepts one.
- **Reader and hop**: the reader sees every excerpt graded >= `reader.min_rel_score` x the best and answers sufficient or
  gives one follow-up query per missing fact; these run only the cheap channels (`dense_chunk`,
  `bm25`, `question`, `hybrid_chunk`, whichever the bank enables); new notes scoring >= `hop_min_rel_score` x the best of their run join the
  reranked set and the joined set is reranked once. At most one hop; all in `recall_events.hops`.
- **Accounting**: `recall_events.cost_usd`, `tokens_in`, `tokens_out`,
  `n_llm_calls` sum the recall's model calls; each call also lands in `llm_calls`
  with `model`, `tokens_in`, `tokens_out`, `cost_usd`. An LLM callable reports
  usage by returning `prospecta.stages.LLMResult` (`make_accounted_llm()` does);
  a bare string works and leaves the usage NULL.
- `RecalledMemory.scores` gains `rerank` (the Sonnet grade) and `jev` (the Jev
  score), numeric (0.0 when not graded), on banks with a `recall_config`.

The model is `Memory(rerank_llm=..., rerank_model=...)`, falling back to `llm`;
the intended one is `anthropic/claude-sonnet-5.5` through OpenRouter.

## Cited evidence and the modes of `docs/postgres-hybrid.md`

Every `RecalledMemory` carries the cited-evidence contract next to `source` and `original_chunk`:
`created_on` (`YYYY-MM-DD`), `kind` (`documents.source_kind`), `match_kind` (`exact+semantic`,
`exact`, `lexical`, `semantic`, `date`; from the channels that found the note, `None` for a bank on the
legacy path; `date` is a note listed only because it lies in a wordless question's date window), `relevance_score` (the cross-encoder's 0..1 score when `rerank.stage` is
`cross_encoder` and graded the note, else the fused `score`) and `exact_snippet` (`ts_headline`
of the matched words, `<<marked>>`, from the `hybrid_chunk` channel). `prospecta.cited.format_cited_rag_block(items)`
renders items as the cited, non-instructional block (`[n] source=... | date=... | kind=... | match=...` with the `match_kind` above, an
`Exact match snippet:` line when one fired, then the full chunk).

- **Mode 2A, search** (`Memory.search`, `recall`, `POST /lookup`): Stage 1 + Stage 2, no stage 0 gate. Per-call
  `date_from`, `date_to` (the note's date, inclusive) and `excluded_sources` (the anti-echo list) are pushed into the
  `hybrid_chunk` channel's three retrievers and raise `ValueError` where that channel does not run.
- **Mode 1, automatic recall by query expansion** (`Memory.expand_and_recall(prompt, kind="chat" | "task")`; task
  recall is `Memory.recall_for_task(title=, ask=, done=, project=)`, a thin wrapper of it): the gate (chat only), the
  query writer (the prompt becomes N structured searches), every search through Stage 1 and Stage 2 against its
  own query in parallel, one merge, one cited block, inside a deadline. The README section "Automatic recall by
  query expansion" has the settings; `docs/postgres-hybrid.md` section 1 is the design. It returns an
  `ExpandedRecall` (`block`, `items`, `searches`, `runs`, `fallbacks`, `timings_ms`, `accounting`; `.queries`).
  The `/lookup` task form no longer takes `limit`: the block is sized by relevance (`max_tokens` is its valve).
- **Mode 2B, ask** (`Memory.ask(question)`): Stage 1 + Stage 2, then the synthesis model answers from the top
  notes alone and cites `[n]` per statement. `AskResult.citations` ties each `[n]` to its source, date and the
  verbatim sentence of the chunk that supports it; `rendered` appends them as `[n] [source | date] "snippet"`.
  With no evidence the answer is `not in memory` and no model is called.

### Stage 3: the sufficiency gate

Off by default. `recall_config.stage3_sufficiency = {"enabled": true, "threshold": 0.70}` (or
`ask(sufficiency=True, sufficiency_threshold=...)`): the decision model answers the System One
`evidence_sufficient` score question (0..3, the stopping question of the Jev reader) over the excerpts; the verdict
is that score / 3. Under the threshold ONE broadened retrieval runs (rerank pool at `deep`, twice the limit, the
date window dropped; `excluded_sources` and the metadata filter stay), its notes join the evidence, and the
verdict is taken again; the model answers either way. A gate that fails (no `Memory(jev=)`, provider
error) never blocks the answer: `AskResult.gate` holds the error and `sufficient` is `None`. The wire
shape is the verified `score` request above; the plan's `noul` / alpha decisions endpoint is not specified in the repo.

## Recall depth

Two depths set the rerank pool cut (the size of the reranker's input):

| depth | pool cut (x best fused score) | use |
|---|---|---|
| `standard` (default) | 0.15 | automatic per-turn recall |
| `deep` | 0.05 | explicit recall, set/discovery questions, map-reduce |

The parameter is named exactly **`depth`**, with the values `"standard"` and `"deep"`
(anything else raises `ValueError`; `None`/omitted = not set). Precedence, first that applies:

1. the per-call `depth`;
2. an explicit `recall_config.rerank.min_rel_score` (a raw cut, kept for back-compat);
3. the bank default `recall_config.depth` (`mem.set_recall_config({"depth": "deep", ...})`);
4. `standard`.

Where it is accepted (providers pass it straight through):

- `Memory.recall(queries, ..., depth="deep")`, `Memory.search(text, ..., depth="deep")`,
  `Memory.recall_synth(message, ..., depth="deep")`, `Memory.recall_mapreduce(message, ..., depth=...)`.
  `recall_synth(grounded=True, scope=[...])` (a set question) defaults to `deep`, and
  `recall_mapreduce` defaults to `deep`, unless `depth` is given. (Prospecta has no MCP/HTTP
  surface; the Memory API is the surface.)
- CLI: `prospecta search QUERY --depth {standard,deep}` and
  `prospecta eval QUESTIONS.md --depth {standard,deep}`.
- `prospecta eval` records the depth used: `report["depth"]` in `--json`, and the report line
  `cuts: ... rerank pool >= 0.05 x best fused score (depth: deep)`.
- Trace: `recall_events.depth` (migration 0015; NULL on legacy rows) holds the depth of each
  recall, including shadow-pair events.
