# `prospecta eval`

`prospecta eval QUESTIONS.md [--ablate] [--synth] [--judge] [--json] [--depth standard|deep]`
scores a bank's recall on a questions file and reports latency and cost per stage. Library:
`prospecta.evaluation` (`load_questions`, `run_eval`, `format_report`).

## What it measures, and what it does not

It runs each question through the same pipeline as `Memory.search`: the bank's own
`channel_config` (Stage 1 `hybrid_chunk`; an empty one is its defaults) and its `recall_config`
stages (Stage 2, the cross-encoder).

It does **not** run Mode 1 (`expand_and_recall`, the query writer and the gate): there is no
measurement of the automatic recall's expansion here. `--synth` runs `ask` (Mode 2B) on every
question and reports whether its citations name a gold note, how often it answered `not in
memory`, and the latency; `--judge` adds an answer-correct verdict against the `answer:` lines.
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
model judge, with `--synth`).

`--ablate` reruns with Stage 2 (the rerank) off, when the bank has it enabled; the stage
**earns** its place when removing it costs hit@10, hit@1 or MRR >= 0.03 on either gold, or
cover@10 >= 0.03 on the multi-note questions. (`hybrid_chunk` is the one channel: there is no
other to ablate.) The report gives latency and cost per stage (the channel, each model stage,
total), and a channel or stage error is shown in it (`ERROR xN`). Nothing is written to the bank
(no config change, no `recall_events` rows from the retrieval runs); `--synth` runs real `ask`
calls and so does log those events. Eval never changes a bank's config: it tells the owner what
to set. `--depth` sets the rerank pool cut for the run (`docs/recall-stages.md`).
