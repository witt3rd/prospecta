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
- **Models** (both configurable, first that applies): the reranker model is the `Memory(reranker_model=)`
  argument, then `recall_config.rerank.model`, then `PROSPECTA_RERANKER_MODEL`, then
  `qwen/qwen3-reranker-8b`. The decision model (Jev: `jev_score`, the gate, the Jev reader) is the
  `Memory(decision_model=)` argument, then `recall_config.models.decision_model`, then
  `PROSPECTA_DECISION_MODEL` (alias `PROSPECTA_JEV_MODEL`, read at import), then `typesafe/jev-1.14`.
- **Reader and hop**: the reader sees every excerpt graded >= `reader.min_rel_score` x the best and answers sufficient or
  gives one follow-up query per missing fact; these run only the cheap channels (`dense_chunk`,
  `bm25`, `question`, whichever the bank enables); new notes scoring >= `hop_min_rel_score` x the best of their run join the
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
