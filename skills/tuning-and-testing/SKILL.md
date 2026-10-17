---
name: tuning-and-testing
description: How to change anything in recall without guessing. Use before editing a prompt, a threshold, a default or a stage, and when writing a report about quality.
---

# tuning-and-testing

A change to recall is a claim about quality, and a claim needs a measurement with a size, not a hunch.
*Why:* in the bake-off six plausible designs, some richer and costlier, did not beat plain search, and the
failures that were found (a keyword leg that matched nothing, a fusion that demoted the best channel) were
invisible until counted. *Evidence:* [`references/bake-off.md`](references/bake-off.md).

## The loop

1. **Write the panel first, from invented text.** No real note text and no name from a private note, in the
   panel, the repo, a report or an issue. Invent names (`HARBOR-7`, a lighthouse committee). *Why:* a panel
   is committed and shared; private text must not travel with it. *Evidence:* prospecta PR 87 and issue 66
   keep numbers only; hermes-prospecta PR 38's panel is 28 invented lines.
2. **Shape the panel.** Balanced classes (the lines that should trigger the behaviour and the lines that
   must not, equal in number), an expected answer per line, and the lines of exactly the shape that failed
   live. **Run each line 3 times** to see the variance of a model that is not deterministic.
   *Why:* an unbalanced panel rewards a change that always answers one way; one run hides a coin flip.
   *Evidence:* PR 38: 28 lines x 3 runs per contract on the real decision model, 168 calls, before against
   after, per class (`recall/references/date-window.md`). PR 87 notes the query writer alone spans 83 to 262
   items on one prompt, so a single run proves nothing.
3. **Red first.** Write the tests or the panel, watch them fail on main for the reason you expect, then
   change the code. *Evidence:* PR 87: 62 new tests, red on main (the new API absent).
4. **Measure before and after on the real model**, same panel, same day, and report both with their sizes
   (n / N), including what got worse. PR 38 lists the two lines that moved to a wider scope.
5. **Score on live data before it goes live, read-only through the live path.** Replay real prompts through
   the same code against a copy or a read-only session of the live bank: `default_transaction_read_only=on`,
   a statement timeout, a small pool, nothing recorded (`record=False`), nothing written, nothing restarted.
   Where the model is not deterministic, **replay the same writer replies** into both versions so the
   difference is the change. *Evidence:* PR 87's method (24 real recalls; the same writer replies replayed
   into the branch; live banks of 10,032 and 76,689 chunks).
6. **Report counts only.** Counts, ranks, scores, class names, ids; never memory text, never a prompt that
   came from a real conversation. A stand-in prompt of the same shape is written fresh. *Evidence:* issue 66
   and PR 87, each stating "numbers only, no note text kept".
7. **Say the limits.** The sample, single runs, what the panel cannot see. `prospecta eval` states what it
   does not measure (`docs/eval.md`); the bake-off summary lists its own validity limits.

Independent check: Spire's decision model reproduced the plugin's panel result: 18 of 42 wrongly windowed before the wording
fix; after it 1, 0 and 1 of 42 on three runs, real windows kept 42 of 42; the one residual is a single line (an artistic
flattery plus "this week's theme"). So measure **three runs**, not one. Slice order that worked in Spire's hybrid build, each
slice small and red first: pool -> cited block -> reranker -> query expansion -> anti-echo -> time window -> one door.

## Tools that already exist

- [`docs/eval.md`](../../docs/eval.md): `prospecta eval QUESTIONS.md [--ablate]` scores Stage 1 + 2 (hit@1,
  hit@10, MRR, cover@10) and shows whether the rerank earns its place (it must move a metric by 0.03).
  It never writes to the bank. It does **not** run Mode 1.
- `tests/test_expand_perf.py` (latency of 20 searches), `tests/test_expand_terms.py`,
  `tests/test_expand_meaning.py` (terms filter and near-copy collapse), `tests/test_docs_links.py` (docs).
  How to run them: [`docs/testing.md`](../../docs/testing.md). Never point `PROSPECTA_TEST_PG_URL` at a live bank.
- The pattern for a model-answered question: hermes-prospecta `tools/time_scope_panel.py` with
  `tests/data/time_scope_panel.json` (`python tools/time_scope_panel.py BEFORE AFTER --runs 3`; it prints
  per-class agreement, never the key). Copy the pattern for a new question rather than eyeballing replies.

## Pitfalls

- A threshold tuned on the panel it was scored on is overfit: the bake-off's keyword weight was tuned on
  the same 40 questions and its 30 of 40 did not reproduce in the fused arm.
- A default changes only banks that do not pin their own value ([`operating`](../operating/SKILL.md)).
- A docs-only or wording change to a prompt is still a behaviour change: measure it.
