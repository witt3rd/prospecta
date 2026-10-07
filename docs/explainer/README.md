# Prospecta, explained: a deck

`prospecta-explainer-v1.pdf` (54 slides, 1920x1080) and `prospecta-explainer-v1.pptx` (the same slides as pictures, with
speaker notes on every slide; not editable) explain Prospecta to a reader with no background in memory systems: what it
is, how a note is retained and recalled, every recall technique with what it is good and bad at, how scores are built,
four kinds of question walked through, why a hybrid beats any single technique, the tradeoffs, the honest weak spots, and
how Rung and Hermes plug in. Every example (Ines, her diary, her notes) is invented; no real note is quoted. The charts
carry real measurements.

## The short version

- Prospecta is long-term memory for beings and agents on Postgres with pgvector: one bank per being; retain what
  happened, recall what matters. It replaced Hindsight; Forge and Augur moved onto it.
- At write time it keeps each note whole and adds chunks (at most 1,000 characters, parent kept), LLM-written anticipated
  questions, 1,536-dimension embeddings, entities, aliases and links, and date and person metadata.
- At read time: a plan, five channels (dense chunks, BM25, anticipated questions, metadata, graph), weighted reciprocal
  rank fusion, a Jev gate and a Sonnet reranker, an optional second hop, then a grounded cited answer; map-reduce and a
  name resolver for set questions.
- hit@10 on 100 questions climbs 0.72 (small model) to 0.92 (strong embedder), 0.93 (fusion), 0.97 (rerank), 0.98
  (second hop), 0.99 (full blend). Set cover@10 went from 0.60 to 0.76 over the rounds.
- A standard recall costs about $0.22 (a set question about $0.38) at a median of about 19 s. The blend of rerank and
  fused order and the promotion re-sort were tried and turned off because they hurt; the linker was once quadratic.
- Honest limits: model-written questions (one of the owner's own so far), n = 100, set questions not solved, latency.

## Where the numbers come from

Internal evaluation rounds on a real corpus of 1,845 notes (100 to 121 questions; strict and pooled "any note that
holds the answer" golds): the first test (60 questions: Jev-Mem 0.32 at Rung's budget, 0.63 at 1,000-character records,
plain small model 0.62, Prospecta alone 0.67, Jev-Mem plus Prospecta 0.73), the analysis round (the ladder, per-class and
synthesis tables), and rounds 2 to 6 (ablations, map-reduce, depth, cost and latency, linker cost, the name resolver).
Each slide's small print names its source. Where this deck and a round differ, the round wins. The reports are not in
this repository.

## Rebuild

```sh
docs/explainer/build.sh            # needs the prerequisites listed in build.sh
```

`build.sh` fetches the three builder files from the pinned commit, checks their sha256, and runs `build.py`, which
writes `prospecta-explainer-v1.pdf` and `.pptx` here and publishes nothing unless its gate passes (layout lint,
a source line and speaker notes on every slide, PDF pages = PPTX slides, and a scan of the slide text, the PDF text and
the notes for local paths, e-mail addresses, hosts, our own tool names and day words).

The builder is the house explainer builder of **janus-infra/spire-venue, commit
`a69d1d79425583525cef616cf98afc3252b74453`** (`docs/explainers`: `theme.css`, `render.mjs`, and the figure primitives
from `spire/figures.py`, lines 1 to 116; `build.py` here is `spire/build.py` from the same commit, adapted). The
repository must be readable to rebuild (`SPIRE_VENUE_REPO` overrides its address). Nothing from it is copied into this
repository.

## What differs from the Spire builder

No screenshots (the `shot` kind is gone); no dependency on spire-project's privacy and day-word modules (their word lists
sit in `build.py`); an agenda that takes any number of parts and an optional legend; a diagram with an optional
takeaway band; local paths. Slide kinds, theme, renderer, layout lint, PPTX wrap and gate are the builder's own.
`figures.py` sets its type a little larger than the primitives' default and wraps text more conservatively; it does not
change the primitives.

## Files

- `slides.py`: every slide's words, figure, source line and speaker notes.
- `figures.py`: the drawn figures (invented diary of Ines, charts of real numbers), written to `diagrams/` by the build.
- `build.py`, `build.sh`: the recipe. `.gitignore` keeps the fetched builder files and the build output out of git.
