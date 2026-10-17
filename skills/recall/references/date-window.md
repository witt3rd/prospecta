# The date-window rule: wording and evidence

## The rule

Give a search a date window only when the line asks about what happened, was said or was done within a
period, so that notes from before it would be worthless. A relative period that only modifies the request
(this week's theme, today's plan, tonight, next week, a deadline) is no window, and so is any line where
older notes could also answer. When in doubt, no window.

## Why

Stage 1 pushes `date_from` / `date_to` into all three retrievers before their slots are cut
(`docs/postgres-hybrid.md` section 5), so a window the request did not need hides every older note. The
library already treats a period it cannot trust as no period: `clean_window` in `prospecta/_expand.py`
drops a date that is not real, before 1990, more than a year ahead or reversed ("a wrong window would hide
the notes, a missing one only widens the search"). The rule extends that stance to a window that is
well-formed but not wanted.

## The failure that prompted it

A live message about a recurring theme said "this week" twice. The decision model answered `past_week`,
so every search carried a past-week window and the block said nothing answers this. Replaying the same
searches read-only: without the window they found 40 notes spanning five months (mostly exact matches,
relevance 0.6 to 1.0); with it each exact search kept about one.

## Evidence: hermes-prospecta PR 38

Measured 2026-10-17 on the plugin's `time_scope` question, real decision model `typesafe/jev-1.14`,
28 invented lines (14 per class) times 3 runs, the contract before against after the rewording:

| class | answer | before | after |
|---|---|---|---|
| a period only modifies the request (42 runs) | no window (expected) | 24 / 42 | 42 / 42 |
| | wrongly windowed | 18 / 42 | 0 / 42 |
| the line truly limits memory to a past period (42 runs) | some window | 42 / 42 | 42 / 42 |
| | exact scope as expected | 39 / 42 | 37 / 42 |

The three lines of exactly the evidence shape (two "this week" phrases, a proper name, an acronym; three
variants, 9 runs): `past_week` 9 / 9 before, none 9 / 9 after. Two lines changed scope (one got a wider window, one is month-sized either way) and none lost
its window (the cost: exact scope 39 to 37 of 42). The panel is `tests/data/time_scope_panel.json` and the
script `tools/time_scope_panel.py` in hermes-prospecta; `tuning-and-testing` describes the pattern.

## Who owns the wording

The plugin's recall contract (`contract/recall-contract-v1.md`, the `time_scope` question) owns it, because
the plugin passes the window to every search. On prospecta main the writer prompt
(`prospecta/prompts/expand-searches.md`) still says `date_from` / `date_to` are set "only when the prompt
states or implies a period" and has not been given this rule: an adopter that lets the library's writer
choose windows should port the wording through a PR to this skill and the prompt, with a panel.

## Invented examples

| line | window? |
|---|---|
| "choose the lighthouse theme for THIS WEEK's HARBOR-7 newsletter" | none: the week says when the request happens |
| "what is on today's plan for the dock repair?" | none |
| "what did we say about the dock repair last Tuesday?" | yes: asks what was said in a period |
| "any older notes on the dock repair, or recent ones?" | none: older notes also answer |
