# The memory bake-off, in one paragraph, and what it taught

**One paragraph.** The bake-off (summary dated 2026-10-15; stopped there; Gate 1a spend $37.7 of a total of
about $67 against a $450 cap) scored several memory systems and plain-search baselines on 180 questions
over three real histories and the 160 questions of the public LoCoMo benchmark, with one answering model,
one set of judges and one 8,000-token evidence budget. **Nothing beat plain keyword or plain fused search,
in six tries, on either set.** On real histories plain keyword search scored 57.8 percent against 43.3 for
the old Prospecta recipe (B1). With a working keyword leg, rank-consensus fusion **demoted the keyword
leg's top hit**: the fused top 10 held 48 of 114 answerable evidence passages while the keyword leg alone
held 67, and 24 of the 26 lost hits were out-voted by the other channels' agreement, not by weights. A
keyword-first union ("keyword hits first, others fill the rest") was then measured on retrieval alone and
reached 73.1 percent evidence-hit@10 against 65.4 percent for the best single channel (these two figures are
from the bake-off follow-up and were not re-derived here). **B1 had no working keyword leg**: its two
lexical legs required every query word to match and returned nothing for any of 164 queries, so recall was
dense search alone (prospecta issue 66, measured 2026-10-14: evidence hit 7 of 40 against 24 of 40 for plain
BM25).

## What it taught the design

| finding | what the design does |
|---|---|
| plain search is the bar | Stage 1 is plain Postgres retrieval; nothing model-shaped is written (`retain`) |
| fusion that votes can bury the best channel | each retriever keeps reserved slots, so an exact hit always reaches the cross-encoder; RRF order inside the 50 does not matter (`search-and-ask`) |
| a formed question recalls 10 to 18 points fewer needed notes than the raw turn | the reranker reads the search `query` plus verbatim `terms` from the prompt; the writer's job is surrounding context, and the verbatim guard puts missed names back |
| a block of raw text beat 10 short notes | the block is sized by relevance with no count cap (`recall`) |
| abstention is not a strength | "not in memory" only when nothing clears the floor; the false-abstain rate belongs beside any abstention number |

## Validity limits (the bake-off's own)

Real-history questions were code-built templates that favour exact-token search and are not a benchmark;
systems were compared whole; B2 ran on half samples; almost every cell is a single run; the questions are
not what the agents actually ask. The report lives outside this repo (aggregates only, no private text).
Judge by real recall traffic: the `ExpandedRecall` record (`docs/postgres-hybrid.md`, section 1, point 10).
