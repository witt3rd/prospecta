---
name: recall
description: Automatic recall (Mode 1) for a chat message or a task. Use when deciding whether to search, how to build the searches from a prompt, or when to put a date window on them.
---

# recall

Mode 1 turns a prompt into a few searches and hands the agent one cited block of what memory holds about it.
The design is [`docs/postgres-hybrid.md`](../../docs/postgres-hybrid.md) section 1; the settings are
[`docs/recall-stages.md`](../../docs/recall-stages.md); every number is in [`docs/limits.md`](../../docs/limits.md).
Each practice below carries its reason and its evidence. Where the plugin and the library differ, the
paragraph says which side owns the rule.

## 1. When to search, and when to skip

- **Skip acknowledgements, go-aheads, greetings and live-state questions** (`ok`, `ship it`, "what is
  running right now?"). *Why:* an expansion costs a writer call and a block of tokens, and a live-state
  answer must come from a live tool, never from an old note. The library does it with the Stage 0 gate:
  the decision model answers `needs_memory`; at or above 0.5 the message goes on.
  *Evidence:* `docs/postgres-hybrid.md` section 1, point 1; `docs/limits.md` (gate threshold).
- **A task is never gated; it always expands.** *Why:* a brief is never an acknowledgement. *Evidence:* same section.
- **A skill turn or a cron turn gets no recall at all, and its text is never expanded or searched.**
  *Why:* the text is instructions the skill already follows; the writer copied that vocabulary into the
  exact-match terms and a block of noise came back (one skill-wake prompt: 83 to 262 items from 18 to 56
  different days, none about anything the wake asked). *Evidence:* the plugin's `skill_turn_shape` and
  `gate` in `prefetch_contract.py` (hermes-prospecta); prospecta PR 87 (the measurement above). This rule
  lives in the plugin; the library has no notion of a skill turn.
- **A gate that fails lets the message through.** *Why:* one expansion costs less than one unremembered
  turn. *Evidence:* `docs/postgres-hybrid.md`, point 1; [`PRINCIPLES.md`](../../PRINCIPLES.md) P22.

## 2. Build the searches

- **About 3 searches for a short chat message, up to 10 to 20 for an elaborate message or a task**
  (never more than `max_searches`, 20). Each search covers one piece of surrounding context: the names
  and subsystems, earlier decisions that govern them, earlier work, the people, the dates. *Why:* a
  search is cheap and runs in parallel; one broad search loses the specific ones. *Evidence:* 20
  searches took 1.55 s against 1.24 s for one (`tests/test_expand_perf.py`, `docs/postgres-hybrid.md`
  point 9).
- **Each search is a `query` for meaning plus `terms`.** The `query` feeds the semantic and stemmed
  retrievers and the reranker. `terms` are **only** verbatim names, identifiers, acronyms, versions,
  paths, error codes and "quoted phrases", never common words. *Why:* terms go to the exact retriever;
  a common word there matches half the bank. A deterministic filter (`keep_term`) enforces it after the
  writer replies. *Evidence:* prospecta PR 87: 62 new tests; replaying the same writer replies, terms
  fell 34 to 51 percent on two real skill-wake prompts.
- **A name, identifier or quoted phrase the prompt has that no search carries is put back into search 1.**
  *Why:* the writer may drop the one string that finds the note. Common words and all-caps emphasis
  words are not put back. *Evidence:* `docs/postgres-hybrid.md`, point 3.
- **A writer that fails falls back to ONE search: the prompt itself.** If the filter drops every search,
  nothing is asked: the writer found nothing to look up. *Evidence:* point 2 of the same section.

## 3. The date-window rule

A search gets a date window **only when the line asks about what happened, was said or was done within a
period**, so that notes from before it would be worthless. A relative period that only modifies the
request (this week's theme, today's plan, tonight, next week, a deadline) is **no window**. **When in
doubt, no window.**

*Why:* a search with no window loses nothing; a wrong window can lose everything, because Stage 1 pushes
the window into every retriever. *Evidence and the wording:* [`references/date-window.md`](references/date-window.md).

## 4. What comes back

Near-copies merge by wording, then by meaning; the block is sized by relevance, never by a count; what is
already in the agent's context is excluded in SQL; every item is cited.
[`references/merge-and-block.md`](references/merge-and-block.md).

## Related

[`search-and-ask`](../search-and-ask/SKILL.md) (what each search does),
[`tuning-and-testing`](../tuning-and-testing/SKILL.md) (changing any of this without guessing).
