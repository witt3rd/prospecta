# Legacy: retired recall machinery that is still in the code

**New banks do not use anything on this page**, and it is not how Prospecta is meant to be
run. A bank created fresh runs the design of `docs/postgres-hybrid.md`: chunk items, the
`hybrid_chunk` channel, a cross-encoder rerank, no linker, no entity extraction
(`prospecta/bank_defaults.py`). The machinery below was the previous design; it is off for a
new bank, nothing has been removed from the code, and it keeps working for the banks that
already use it. This page says what exists and where it still runs, so that nobody mistakes
it for the architecture.

## Which banks are legacy

A bank whose stored `channel_config` is `[]` (the column default; every bank created before
the new-bank defaults, unless it was configured by hand) is a **legacy bank**:

```sql
SELECT bank_id, channel_config = '[]'::jsonb AS legacy, recall_config, link_config FROM banks;
```

The library does not change a bank's stored config. To move a legacy bank to the current
design, create `<bank>-v2` with `prospecta migrate-bank` (it writes chunk items from the
documents and gets the new-bank defaults, `docs/embedding-migration.md`), or set the config by
hand on a bank that already has chunk items:

```python
mem.set_channel_config([{"name": "hybrid_chunk", "weight": 1}])
mem.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder"}})
```

`hybrid_chunk` reads `kind = 'chunk'` items only. A bank written with `index_text=` holds only
question items and has nothing for it to find.

## What exists

| machinery | where in the code | new banks | where existing banks still use it |
|---|---|---|---|
| **Questions as a stored index.** `kind = 'question'` items: an anticipated question, embedded, with the matching chunk as its `original_chunk` | write: `retain(index_text=...)`, `retain(generate_questions=True)`, `prospecta retain --index-text`, frontmatter `index_text:` of `index_directory`, `SourceIndexer(generate_questions=True)`; read: the `question` channel and the legacy recall | not written unless the caller asks; never read | legacy banks (the semantic and `content_tsv` retrievers match them); `migrate-bank` copies the chunks and re-embeds the existing questions, no model call |
| **The three-channel recall** (`channel_config = []`): semantic over `content`, `content_tsv`, `body_tsv`, fused by RRF | `db/queries.hybrid_search`, `Memory.search` | not used | every `search`, `recall`, `recall_synth`, `ask` on a legacy bank. It takes no stages and no Stage 1 filters (`date_from`, `date_to`, `excluded_sources` raise `ValueError`); `expand_and_recall` returns an `error` |
| **Other channels** `dense_chunk`, `question`, `meta`, `bm25`, `keyword`, `graph`; `DEFAULT_CHANNEL_CONFIG` (`dense_chunk` 4, `question` 1, `graph` 0, `meta` 3); `fusion.method = floor_union`; scope promotion | `prospecta/channels/` | not configured | a bank with a non-empty config that names them. `bm25` needs the `pg_search` index of migration 0012 (or an explicit in-process backend, whose index files live under `PROSPECTA_BM25_DIR`, else `$PROSPECTA_DATA_DIR/bm25`, else memory); `keyword` scans or uses the index of migration 0018 (`prospecta keyword-index`, `docs/exact-tsv.md`) |
| **Rerank stages** `sonnet_listwise` (the default `stage` of a `rerank` block), `jev_score`, the Jev gate, the reader and its second hop, the rerank blend | `prospecta/stages.py`, keys `rerank`, `gate`, `reader` of `recall_config` | `cross_encoder` only | a bank whose `recall_config` names them. They run only through a non-empty `channel_config` |
| **Question-space read path**: `formulate_queries`, `recall_synth` (also `grounded=True`), `recall_mapreduce`, the synthesis prompts | `Memory.formulate_queries`, `recall_synth`, `recall_mapreduce`; `_formulate.py`, `_synth.py`, `_mapreduce.py` | not used by Mode 1 or Mode 2 | callers of those methods; `prospecta eval --synth` |
| **Links, entities, aliases, the graph channel, the Linker** | `_linker.py`, `_entities.py`, `channels/graph.py`; tables of migrations 0006, 0014, 0016 | off: `link_config = {"links": false, "entities": false}`; the CLI does not build the linker | a bank that stored `{}` keeps the defaults: links **off**, entity extraction **on** (`get_link_config()` shows it) |
| **`prospecta eval`** | `evaluation.py`, `docs/eval.md` | measures the legacy channel set | any bank; it does not measure Mode 1 or `ask` |

## Legacy `recall_config` keys (reference)

They apply only to a bank with a non-empty `channel_config` that names them. The code is
`prospecta/stages.py`; `prospecta.stages.DEFAULT_RECALL_CONFIG` is the reference constant (it is
not applied to any bank: `stage` defaults to `sonnet_listwise` when a `rerank` block names none).

```python
mem.set_recall_config({
  "rerank": {"enabled": True, "stage": "sonnet_listwise", "min_rel_score": 0.15,
             "blend": {"enabled": False, "weight_rerank": 0.7, "weight_fused": 0.3, "floor": False}},
  "gate":   {"enabled": False, "threshold": 2.95},          # Jev, needs Memory(jev=...)
  "reader": {"enabled": False, "min_rel_score": 0.6, "hop_min_rel_score": 0.4},
})
```

- **Sonnet listwise rerank** (`rerank.stage = "sonnet_listwise"`): every fused note scoring at least
  `min_rel_score` x the best fused score, batched by the model window, one best chunk per note under
  a header of name, date and person; the model replies JSON `{grades, ranking}`; the order is the
  ranking, then the grade, then the fused order. The model is `Memory(rerank_llm=, rerank_model=)`,
  else `llm` (the CLI builds `anthropic/claude-sonnet-5.5` through OpenRouter). Any failure keeps the
  fused order and records `fallback_reason`.
- **Jev score** (`rerank.stage = "jev_score"`): one System One `score` question per candidate
  (criteria 0 to 3), 15 candidates and at most 60 KB per request, 10 s timeout; it only reorders.
  `gate.enabled` makes Jev score first and keeps its order when the best score reaches `threshold`
  (2.95), else Sonnet reranks; the gate needs `sonnet_listwise`, so it does not combine with the
  cross-encoder.
- **Reader and hop** (`reader`): the reader (`type` `sonnet` or `jev`) reads every excerpt graded at
  least `min_rel_score` x the best and answers sufficient, or gives follow-up queries that run the
  cheap channels (`dense_chunk`, `bm25`, `question`, `hybrid_chunk`); new notes at least
  `hop_min_rel_score` x the best of their run join and the joined set is reranked once. At most one
  hop; recorded in `recall_events.hops`.
- **Blend** (`rerank.blend`, off): blends the reranker's order with the fused order
  (`weight_rerank` 0.7, `weight_fused` 0.3; `floor: true` keeps the fused top 3 inside the first 10).
  It regressed hit@1 in the last measurement, so it is off.
- **`fusion`** (`{"method": "floor_union", "floor": 3}`): puts first every document in the top
  `floor` of any live channel, then the rest in fused order. Absent = weighted RRF.
- **Grounded synthesis** (`recall_synth(message, grounded=True)`, opt-in) and **`evidence`**
  (`min_rel_score` 0.5, `context_tokens` 200,000): the evidence is the best chunk of each note of the
  blended recall scoring at least `evidence.min_rel_score` x the best (or, with `scope=[...]`, the
  whole scope set: one recall, the question as the only query); the model cites `[note name]` for
  every claim and answers `not in memory` when the evidence lacks it. `RAGResult.citations` is
  stored in `recall_events.citations` (migration 0011). The model is `Memory(synth_llm=...)`, else
  `llm`. The replacement is `Memory.ask`.
- **`link_completeness`** (`all-pairs` or `connected`): how the linker judges pairs.
- Scores: `RecalledMemory.scores` gains `rerank` (the Sonnet grade) and `jev` (the Jev score) on
  banks with these stages.

## The linker and entity extraction

- Links are off unless `link_config.links` is true; entity extraction runs after a write when a
  linker is present and `link_config.entities` is true. A bank created fresh stores both false,
  so it does nothing; a bank that stored `{}` still extracts entities (one model call per
  document written).
- The CLI builds the linker only when the bank's `link_config` enables links or entities, or
  when `PROSPECTA_LINKER=on`; `PROSPECTA_LINKER=off` never builds it (it needs
  `OPENROUTER_API_KEY` in every case). Library callers pass `Memory(linker=...)` themselves.
- Switch a bank with `Memory.set_link_config({"links": ..., "entities": ...})`; there is no CLI
  for it.
- `prospecta link-pass [--mode all-pairs|connected] [--limit N]` builds links for documents
  already written (resumable); `prospecta drop-links [--yes]` removes the link rows of a bank in
  small transactions, keeping entities, aliases, link state and the pair cache.
- The graph channel is silent (weight 0) in `DEFAULT_CHANNEL_CONFIG` and returns nothing on a
  bank with no link rows.

Measured at the time: links moved recall scores by 0.00 to 0.02 and grew `memory_links` to about
300 million rows on a 73,000-note bank, which is why they are off by default. The benchmark
tables of the retired `graph-links` doc are in the git history (`git log -- docs/graph-links.md`).

## Rules of thumb

- Do not build on any row above for a new bank. Anything the table says new banks do not run
  has a replacement in `docs/postgres-hybrid.md`.
- The retired material is documented here only to the depth needed to operate a bank that
  still uses it; the code (`prospecta/stages.py`, `prospecta/channels/`, `prospecta/_linker.py`)
  is its reference. Removing it is a separate change from the documentation overhaul.
