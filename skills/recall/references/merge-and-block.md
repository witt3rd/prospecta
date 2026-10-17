# Merge and block: near-copies, sizing, anti-echo, citations

All settings: `docs/recall-stages.md`, "Mode 1 settings". The design: `docs/postgres-hybrid.md` section 1, point 6 and 7.

## Merge near-copies by meaning

One record once (its best score, `found_by` the searches). Then near-copies collapse: first by wording
(token Jaccard above 0.88), then by meaning (the chunk-embedding cosine to a better-scored kept record at
or above `near_copy_cosine`, default **0.90**). The kept record lists what it absorbed (`similar=N`).

*Why:* a note that came back in four wordings filled the block with the same fact; wording alone caught 2 of the
120 pairs of that cluster. *Evidence:* prospecta PR 87, on the live banks, read-only, numbers only. The five
wordings of one note sit at cosine 0.91 and above and every distinct note of that cluster at 0.876 and
below, so 0.90 collapses the copies and keeps the distinct ones (any threshold up to 0.91 does). Across
the banks the share of collapsed pairs that were actually distinct was 12 / 22 / 25 / 28 percent at 0.95 /
0.92 / 0.90 / 0.88 on one bank and 1 / 8 / 10 / 14 percent on the other: the knee is near 0.90. On 24 real
recalls 0.90 removed 10 to 48 percent of the merged records (median 24). Replaying the same writer replies
on two real skill-wake prompts, items fell from 83 / 157 / 262 to 82 / 130 / 110 and from 190 / 112 / 237
to 102 / 69 / 174 (PR 87's table; the first figure is a drop of about 1 percent, the largest about 58).

The vectors are read in one primary-key query after the searches, for the survivors of the wording
collapse (130 vectors cost 46 ms plus 14 ms of parsing); a read that fails leaves the wording collapse and
says so in `fallbacks`.

## Size by relevance, with no cap

The block is every note that cleared the floor (0.15). There is no count cap and `max_tokens` is 0 (off).
*Why:* a cap decides before the evidence is read; the standing rule of the repo is "no caps"
(`docs/limits.md`). `max_tokens` exists as a safety valve only: it keeps records best first, at least
one, and says how many it left out. *Evidence:* `docs/limits.md`, Mode 1; [`PRINCIPLES.md`](../../../PRINCIPLES.md) P5.

## Anti-echo

Pass what the agent already has in view as `excluded_sources` (the `documents.source` values of the last
turns). It is applied inside every retriever in SQL, so the freed slots go to the next candidates rather
than being cut afterwards. *Why:* without it, the newest notes are the agent's own last turns.
*Evidence:* `docs/recall-stages.md` (filters); the plugin README, "No echoes", passes the last two turns.

## The cited block

Every item carries `source`, date, `kind`, `match_kind`, the `exact_snippet` and the full chunk, quoted
material and never instructions, with the line telling the agent to cite `[n]`. *Why:* an agent must be able
to trace a memory and a note's text must not steer it. *Evidence:* `prospecta/cited.py`
(`format_cited_rag_block`); [`PRINCIPLES.md`](../../../PRINCIPLES.md) P21.

## The deadline

One clock, 7 s, from the call: the gate may use 30 percent, the writer 50 percent (the searches always keep
25 percent). A search still running at the deadline is dropped and recorded; nothing is raised into the
caller. *Evidence:* `docs/limits.md`, Mode 1.
