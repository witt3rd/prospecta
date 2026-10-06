# Recall stages: rerank, Jev gate, reader and hop

After the channels fuse (`banks.channel_config`), a bank can run more stages,
all configured by `banks.recall_config` (migration 0009; `{}` = none, recall is
unchanged). Set it with `Memory.set_recall_config(...)`.

```python
mem.set_recall_config({
  "rerank": {"enabled": True, "stage": "sonnet_listwise", "pool": 30},
  "gate":   {"enabled": False, "threshold": 2.95},          # Jev, needs Memory(jev=...)
  "reader": {"enabled": False, "top": 8, "join_top": 15, "max_follow_ups": 2, "max_new": 10},
})
```

- **Sonnet rerank** (`SonnetListwise`): the fused top 30, one best chunk (<= 1,000
  chars) per note under a header of note name, date and person; the model replies
  JSON `{grades, ranking}`; order = ranking, then grade, then fused order. Any
  failure (provider error, unparseable reply) keeps the fused order and records
  `fallback_reason` in `recall_events.rerank`.
- **Jev** (`JevScore`, `rerank.stage = "jev_score"`) in Spire's call shape: one
  `score` question per candidate, criteria 0 to 3, 15 candidates per request (two
  requests for 30), 60 KB per request, 10 s timeout, Jev only reorders. With
  `gate.enabled`, Jev scores first; if its top score >= `threshold` the Jev order
  stands, else (or if Jev fails) Sonnet reranks. Jev is built dark: pass
  `Memory(jev=JevScore(openrouter_jev_transport()))`. The request/response wire
  shape in `prospecta/stages.py` is written from the design's description of
  Spire's `library-rank.ts`, not verified against the live endpoint; the transport
  is one injectable function so it is easy to correct.
- **Reader and hop**: the reader sees the top 8 excerpts and answers sufficient or
  <= 2 follow-up queries; these run only the cheap channels (`dense_chunk`,
  `bm25`, `question`, whichever the bank enables); <= 10 new notes join the top 15
  and the joined set is reranked once. At most one hop; all in `recall_events.hops`.
- **Accounting**: `recall_events.cost_usd`, `tokens_in`, `tokens_out`,
  `n_llm_calls` sum the recall's model calls; each call also lands in `llm_calls`
  with `model`, `tokens_in`, `tokens_out`, `cost_usd`. An LLM callable reports
  usage by returning `prospecta.stages.LLMResult` (`make_accounted_llm()` does);
  a bare string works and leaves the usage NULL.
- `RecalledMemory.scores` gains `rerank` (the Sonnet grade) and `jev` (the Jev
  score), numeric (0.0 when not graded), on banks with a `recall_config`.

The model is `Memory(rerank_llm=..., rerank_model=...)`, falling back to `llm`;
the intended one is `anthropic/claude-sonnet-5.5` through OpenRouter.
