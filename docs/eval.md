# `prospecta eval`

`prospecta eval QUESTIONS.md [--ablate] [--synth] [--judge] [--json] [--depth standard|deep]`
scores a bank's recall on a questions file and reports latency and cost per stage. Library:
`prospecta.evaluation` (`load_questions`, `run_eval`, `format_report`).

## What it measures, and what it does not

It runs each question through the same pipeline as `Memory.search`: the bank's own
`channel_config` (for a bank created fresh, Stage 1 `hybrid_chunk`) and its `recall_config`
stages (Stage 2, the cross-encoder). A bank with no channel config (a legacy bank) is scored with
`prospecta.channels.DEFAULT_CHANNEL_CONFIG` and the report says so.

It does **not** run Mode 1 (`expand_and_recall`, the query writer and the gate) or Mode 2B (`ask`):
there is no measurement of the automatic recall's expansion or of the cited answer here. `--synth`
runs the legacy grounded synthesis (`recall_synth(grounded=True)`, `docs/legacy.md`), not `ask`.
Judging Mode 1 ("did the agent have what it needed") is the `ExpandedRecall` record of
`docs/postgres-hybrid.md`, section 1, point 10, on real prompts.

## Questions file

```
## Q001 | multinote
What did Kelly say about the trip, and where did we stay?
gold: kelly-trip.md, hotel.md
gold2: kelly-trip.md
answer: She said it was too expensive; we stayed at the Harbour Inn.
```

The project's `questions.md` format is also read directly: `### Qnnn` blocks with `class:`,
`question:` (may continue on following lines), `gold:` and optional `gold2:` / `answer:` lines
(used when the file has `###` blocks and no `##`).

## Metrics

`gold` (note names = `documents.source`; folder and `.md` ignored) and the optional second gold
`gold2` give hit@1, hit@10, MRR and cover@10 each; `answer` is for `--judge` (answer-correct by a
Sonnet judge, with `--synth`).

`--ablate` reruns with each live channel off, each enabled stage (rerank, gate, reader) off, and
each registered channel the bank lacks on at weight 1; a channel **earns** its weight when
removing it costs hit@10, hit@1 or MRR >= 0.03 on either gold, or cover@10 >= 0.03 on the
multi-note questions (a channel not yet run **gains** by the same margin). The channels it tries
on include the retired ones (`docs/legacy.md`). The report gives latency and cost per stage (each
channel, each model stage, total), and a channel or stage error is shown in it (`ERROR xN`).
Nothing is written to the bank (no config change, no `recall_events` rows from the retrieval
runs); `--synth` runs real `recall_synth(grounded=True)` and so does log those events. Eval never
changes channel weights: it tells the owner which to set. `--rerank-blend/--no-rerank-blend` and
`--scope-promote/--no-scope-promote` force the two legacy switches per run;
`--depth` sets the rerank pool cut for the run (`docs/recall-stages.md`).
