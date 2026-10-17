# Why the write path is chunks only

The 1.0.0 release removed everything model-shaped from the write path and the read path
([`CHANGELOG.md`](../../../CHANGELOG.md), "Removed since 0.1"): the question-indexing write path and the
question lane, the graph, linker and entities, map-reduce, the in-process BM25 and keyword channels, the
legacy Sonnet / Jev rerank, gate and reader.

## The reason

The memory bake-off scored its systems and plain-search baselines on the same questions (summary
dated 2026-10-15; 180 questions on three real histories, 160 on the public LoCoMo benchmark; same
answering model and judges; same 8,000-token evidence budget). **Nothing beat plain keyword or plain fused
search, in six tries, on either set.** The richer Prospecta recipe (B2) cost $0.13 to $0.16 and 25 to 80 s a recall for no proven gain: +7.5
points on its own 90-question half sample, under the +8 point margin the bake-off had set. A formed question recalled 10 to 18
points fewer needed notes than the raw turn.

The failures that were found were retrieval wiring, not missing write-time intelligence:

- The legacy hybrid's two lexical legs used a query that needs every word to match, so they returned
  nothing for any of 164 queries and recall was dense search alone (prospecta issue 66, measured 2026-10-14:
  7 of 40 evidence hits against 24 of 40 for plain BM25).
- A rank-consensus fusion demoted the best channel's top hit (see `tuning-and-testing`).

So the design keeps the note's own text, indexes it three ways (embedding, stemmed words, exact words) and
spends the model budget at read time on a cross-encoder, where a failure degrades to the Stage 1 order
instead of corrupting a stored note.

## Limits of this evidence

The real-history questions were code-built templates (time, set, never-mentioned) that favour exact-token
search; they are not a benchmark. Most cells were single runs. The bake-off ran on the previous recipe; it
did not measure today's `hybrid_chunk` + cross-encoder. Its report lives outside this repo.
