# Links, entities and the graph channel (migration 0006)

The Jev-Mem techniques, Postgres-native (hybrid retrieval design 8.7).

**Tables** (additive; `memory_entities`, `memory_item_entities`, `memory_links`,
`memory_link_state`; links and entity rows cascade on item delete).

**Linker** (`prospecta._linker.Linker`, `JevRelationJudge`; `Memory(linker=Linker(llm=..., judge=...))`) runs after
`retain` / `index_single_file` on one worker thread (`asynchronous=False` runs inline;
`Memory.wait_for_links()` drains; `Memory.link_document(id)` and `Memory.link_pending()`
are the on-demand and safety-net paths). Each link run uses its own dedicated connection
(`ConnectionPool.dedicated()`), never the shared one; a step that deadlocks is retried with
backoff, and a step that still fails is recorded in `memory_link_state` as `error` and picked
up again by `link_pending`. Per document:

- `NEXT` between consecutive chunks, and `PRECEDES` / `SUCCEEDS` / `TEMPORALLY_CLOSE`
  (within 3 days, closeness >= `temporal_min_rel` 0.5 x the closest) to the chronological neighbours of the same `person`, in SQL
  from `prospecta_doc_date(document_metadata, created_at, created_on)` (created_on first); no model call.
- Entities by the injected `llm` (Sonnet; prompt `extract-entities`), then
  `ENTITY/SHARED_ENTITY` links by a join. An entity held by more than 30 items (`ENTITY_HUB`) is a hub
  and gets no per-anchor links; GraphExpand reaches its holders through the entity table.
  The same pass emits aliases (nicknames, pen names, aka) per person (migration 0014):
  `memory_entity_aliases` rows resolve an alias to the person entity, so a note using any alias
  is held by the person; mentions already extracted under an alias move onto the person.
  `Linker.backfill_aliases(conn, bank_id, limit, after)` is the resumable backfill by document id
  (progress in `memory_alias_state`; errors recorded and skipped).
- Semantic links: every pgvector neighbour per anchor item with cosine >= `neighbour_min_rel` (0.9) x the nearest, ending at the first marginal drop > (1 - `neighbour_min_rel`) x the nearest, and never below the absolute cosine floor `neighbour_min_cos` (0.5; see `limits.md`). With a
  `JevRelationJudge(transport)` Jev answers `semantic`, `causes`, `caused_by` as System
  One `score` questions (score/3 >= 0.6 makes a link, `origin='jev'`). Each unordered pair is judged once and cached in `memory_link_pairs` (migration 0016; A->B and B->A share it, SEMANTIC writes both directions; re-runs ask nothing), and one request carries up to 96 questions (32 candidates x 3 relations) within an estimated 45,000 input tokens (`JevRelationJudge(max_questions_per_call, max_input_tokens, relations_per_call)`; a 400 `max_tokens_exceeded` halves and retries); without one, or if
  Jev fails, neighbours with cosine >= 0.75 become `RELATED_TO` (`origin='pgvector'`).

**GraphExpand** (`graph`, kind `expand`) seeds from every pool document with RRF >= `seed_min_rel` (0.5) x the best,
walks two bounded, indexed hops over links in both directions (plus the entity-table join for hub
entities), multiplying by
decay 0.5, the link type weight (SEMANTIC 1.0, CAUSAL 0.8, TEMPORAL 0.5, ENTITY 0.5) and
the link confidence; every reached neighbour scoring >= `node_min_rel` (0.4) x the best reached, reported per document. It is **enabled
in `DEFAULT_CHANNEL_CONFIG` at weight 1** (captain's decision; the design had it dark);
set `weight` (0 silences it) and `params` (`seed_min_rel`, `node_min_rel`, `max_hops`, `decay`, `hub`, `hub_cap`, `type_weights`) per bank. Its contribution is to be measured on real questions.

**JevReader** (`recall_config.reader.type = "jev"`, needs `Memory(jev=...)`): Jev's
`evidence_sufficient`, `continue_useful`, `missing_evidence` as one request decide
sufficiency; the follow-up probe is the best excerpt itself.

## Link completeness: all-pairs (default) or connected

`link_completeness` sets how many qualifying neighbours (cosine floor + relative stop, as
before) are judged. A missing setting means `all-pairs` everywhere (linker, bank
`recall_config` validation). Decision (captain): quality over speed, cost is not the limit,
so `all-pairs` is the default knowingly and `connected` is the explicit cheaper setting.

- `all-pairs` (DEFAULT): judged at CHUNK level: every anchor (chunk, else question) of a note
  against its neighbours from the per-anchor HNSW fetch; each unordered chunk pair once through
  the pair cache (keyed by the text-hash pair), links written between the judged chunks. 96
  questions per call, the pair cache and the similarity floor keep it affordable. No
  per-note collapse, no top-K, no mutual test.
- `connected`: the reduced-pairs rule, and ONLY here the per-note collapse (the first anchor of
  each note represents it: one pair per note pair, A->B and B->A one pair; links between a
  note's other chunks and other notes are NOT judged): at most `judge_top_k` (32 = one call) neighbours per
  note, judged only if MUTUAL (each in the other's top 32) or among the note's
  `judge_nearest` (3). SEMANTIC (same subject) is treated as transitive inside a dense
  cluster (members beyond the top 32 are reached through cluster neighbours; this
  transitivity assumption for SEMANTIC links is UNPROVEN); CAUSAL /
  LEADS_TO is not transitive and is written only for judged pairs. It DROPS some direct
  semantic pair links in dense clusters (connectivity is preserved) and chunk-level links
  of non-first chunks; the quality/cost knob. The 3-5 calls/note target is this mode's goal
  only, not all-pairs'.
- both: a pair whose normalised text (modulo frontmatter, whitespace, case) is identical
  is RELATED_TO at 1.0 with no call; the cache `memory_link_pairs` is keyed by the
  text-hash pair, so a re-run asks nothing (rows cascade with their items: re-indexing a
  note re-judges its pairs).

Set it per bank: `Memory.set_recall_config({"link_completeness": "connected"})` (key of
`banks.recall_config`, no migration); per process/import: `Linker(link_completeness=...)`
or `PROSPECTA_LINK_COMPLETENESS=connected|all-pairs` for the CLI (wins over the bank).

### Upgrading later: `link-pass` (resumable, background)

A bank can import with `connected` and be upgraded without a re-import:

    prospecta link-pass --mode all-pairs [--resume] [--limit N]
    memory.link_pass("all-pairs", limit=None, background=True)   # a Future of the progress dict

Moving to `all-pairs` re-runs at chunk level. It re-runs only the semantic step of every linked document whose link state does not
record `completeness = all-pairs`, on its own connection (never blocks retain, own thread
when background). Progress lines/dicts: documents, pairs_judged (`judged`), pairs_cached,
failed, remaining. Resumable: a finished document records its completeness, and the pair
cache means no pair is asked twice, so an interrupted pass loses nothing; failed documents
stay for the next run.

### Benchmark (`scripts/bench_linker.py`)

STUB Jev with a per-call token cost model (nothing leaves the process); Postgres+pgvector
real. Dense synthetic corpus: clusters of 226/170/130/110/100 notes (~40 percent of 1,845),
a third of notes multi-chunk, exact duplicates, a planted causal partner every 10th note.
Before = main at 6f0e54c. All-pairs figures are chunk-level (re-run). Figures are modelled; the scout re-measures on the real corpus.

| notes | judged pairs/note: before / all-pairs / connected | calls/note: before / all-pairs / connected | cost USD: before / all-pairs / connected |
|---|---|---|---|
| 500  | 10.2 / 10.2 / 6.0  | 0.94 / 0.93 / 0.81 | 12.4 / 12.4 / 7.6 |
| 1000 | 18.9 / 18.9 / 6.5  | 1.21 / 1.20 / 0.84 | 45.7 / 45.6 / 16.2 |
| 1845 | 32.2 / 32.1 / 6.6  | 1.63 / 1.62 / 0.85 | 144.5 / 144.4 / 30.8 |
| 3690 | 21.9 / 21.9 / 5.9  | 1.31 / 1.29 / 0.86 | 195.6 / 195.5 / 54.4 |

Recall (same corpus, ground truth from the generator): causal 100 percent in every mode and
size; cluster connectivity 1.0 in every mode; semantic DIRECT pair recall (pairs of one
cluster with a link): before 0.78; all-pairs 0.84/0.78/0.78/0.78 (500/1000/1845/3690; the
relative stop, not the cap, bounds it); connected 0.58/0.27/0.14/0.13.

Scaling of all-pairs calls/note. Fixed absolute cluster sizes (above; growth past 1845
adds unrelated notes): 0.93 / 1.20 / 1.62 / 1.29 calls/note: a plateau. Worst case, every
cluster growing with the bank (`BENCH_CLUSTERS=scaled`): 0.93 / 1.20 / 1.62 / 2.41 calls/note,
judged pairs/note 10 / 19 / 32 / 58, cost USD 12.4 / 45.6 / 144.4 / 522.6, direct recall at
3690 0.73. Calls/note grows about 2.6x over 7.4x notes in the worst case: mildly
super-linear, not quadratic in calls/note; judged pairs/note grows with cluster size
(judging all pairs of a cluster is inherently so); `connected` stays flat at ~6 judged
pairs/note in both. The stub's calls/note is below the real 6-20 because synthetic notes
have few anchors; the real driver was chunk-level pairs (all-pairs is chunk-level again;
connected is one pair per note pair).

Recall note: the benchmark's ground truth is note-level and cannot see a lost chunk-level
link; `tests/test_graph_links.py` has a chunk-level test (a link from a note's non-first chunk):
found in all-pairs, lost in connected.
