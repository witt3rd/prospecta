# Grounded synthesis and `prospecta eval`

## Grounded synthesis (opt-in)

`recall_synth(message, grounded=True)` replaces the plain RAG step with the
cited synthesis of the hybrid design (8.8). Evidence: the best chunk of each of
the notes of the blended recall scoring at least `evidence.min_rel_score` of
the best, each with its chunks kept by score (see `docs/limits.md`; `evidence_top`
is an optional per-call note count, `None` by default), or, with `scope=[note names or
document ids]` that resolves to any number of notes, the whole scope set (one recall, the question as the only query; `scope` defaults to the extracted filter set). The model
cites `[note name]` for every claim and answers `not in memory` when the
evidence lacks it. `RAGResult.citations` (`note`, `document_id`, `known`; unknown
= the model cited a name not in the evidence), `.evidence` and `.synth_call`
(model, tokens, cost, latency) come back. The synthesis LLM is `Memory(synth_llm=...)`, falling back to the general `llm` (not the rerank LLM); `synth_call['model']` comes only from the `LLMResult` and is `None` when it reports none. The eval report names which LLM ran (`synth_llm` / `general llm`) and its reported model(s), or `unreported`. The citation list is stored in
`recall_events.citations` (migration 0011, one nullable column). Default
`recall_synth` is unchanged.

## `prospecta eval QUESTIONS.md [--ablate] [--synth] [--judge] [--json]`

Library: `prospecta.evaluation` (`load_questions`, `run_eval`, `format_report`).
Questions file:

```
## Q001 | multinote
What did Kelly say about the trip, and where did we stay?
gold: kelly-trip.md, hotel.md
gold2: kelly-trip.md
answer: She said it was too expensive; we stayed at the Harbour Inn.
```

The project's `questions.md` format is also read directly: `### Qnnn` blocks with
`class:`, `question:` (may continue on following lines), `gold:` and optional
`gold2:` / `answer:` lines (used when the file has `###` blocks and no `##`).

`gold` (note names = `documents.source`; folder and `.md` ignored) and the
optional second gold `gold2` give hit@1, hit@10, MRR and cover@10 each; `answer`
is for `--judge` (answer-correct by a Sonnet judge). `--ablate` reruns with
each live channel off, each enabled stage (rerank, gate, reader) off, and each
registered channel the bank lacks on at weight 1; a channel **earns** its
weight when removing it costs hit@10, hit@1 or MRR ≥ 0.03 on either gold, or
cover@10 ≥ 0.03 on the multi-note questions (a channel not yet run **gains** by
the same margin). The report gives latency and cost per stage (each channel,
each LLM stage, total). Recall runs through the production path (`_search_channels`: filter extraction,
promotion, rerank, hop), and a channel or stage error is shown in the report
(`ERROR xN`). Nothing is written to the bank (no config change, no
`recall_events` rows from the retrieval runs); `--synth` runs real
`recall_synth(grounded=True)` and so does log those events. Eval never changes
channel weights: it tells the owner which to set.
