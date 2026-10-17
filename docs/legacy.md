# Legacy: what was removed, and what existing banks still do

**New banks do not use anything on this page.** A bank created fresh runs the design of
`docs/postgres-hybrid.md`: chunk items, the `hybrid_chunk` channel, a cross-encoder rerank, no
linker, no entity extraction (`prospecta/bank_defaults.py`). The machinery that design replaced
has been **removed from the code**. This page says what was removed, how to get it back, what a
bank that still names it does now, and what was kept on purpose.

## Removed, and how to get it back

It was removed in commit **`8d4593d`**, the change that follows **`67c0706`** (main before the
removal): `67c0706` is the last commit that has all of it. Nothing was tagged (a tag is a remote
object; none was created). To read or restore a file:

```
git show 67c0706:prospecta/_linker.py
git checkout 67c0706 -- prospecta/channels/bm25.py          # into your working tree
git log --diff-filter=D --stat -- prospecta/_linker.py       # the commit that removed it
```

| removed | where it was | what a bank that still names it does now |
|---|---|---|
| **The linker, entity extraction, aliases, links** (`Linker`, `Memory(linker=)`, `link_document`, `link_pending`, `link_pass`, `drop_links`, `set_link_config`, `get_link_config`; `PROSPECTA_LINKER`, `PROSPECTA_LINK_COMPLETENESS`, `PROSPECTA_JUDGE_FLOOR_FRAC`) | `_linker.py`, `_entities.py`, `_linkdrop.py` | Nothing writes links or entities for any bank, a stored `link_config` of `{}` (which used to mean "entities on") included. The rows already written stay. |
| **CLI** `link-pass`, `drop-links`, `keyword-index` | `cli/link_pass.py`, `cli/drop_links.py`, `cli/migrate.py` | The commands do not exist. |
| **The graph channel** | `channels/graph.py` | Skipped with one warning per process. |
| **The `bm25` channel** (`PgSearchBm25`, `InProcessBm25`, `AutoBm25`; `PROSPECTA_BM25_DIR`) | `channels/bm25.py` | Skipped, as above. |
| **The `keyword` channel** | `channels/keyword.py` | Skipped, as above. |
| **The `question` channel** (the dense search over `kind = 'question'` items) | `channels/semantic.py` | Skipped, as above. |
| **`recall_mapreduce`** (set questions about one entity) | `_mapreduce.py` | The method does not exist. |
| **Model-written questions at write time** (`retain(generate_questions=True)`, `index_text_prompt_override`, `SourceIndexer(generate_questions=True)`) | `_index_text.py` | The parameters do not exist. A caller's own `index_text=` still works. |
| **The Sonnet listwise rerank and the Jev rerank** (`rerank.stage = sonnet_listwise` / `jev_score`; `rerank.stage` unset, which meant `sonnet_listwise`), **the Jev gate** (`gate`), **the reader and its second hop** (`reader`, `recall_events.hops`), **the rerank blend** (`rerank.blend`), `link_completeness`, `stages.DEFAULT_RECALL_CONFIG` | `stages.py` | Skipped with one warning per process: the **Stage 1 order stands**, no model is called for them. `set_recall_config` refuses them. |
| **The previous design's prompts** `extract-entities`, `mapreduce-map`, `mapreduce-reduce`, `generate-index-text`, `rerank-listwise`, `read-sufficiency`, and the benchmark scripts of the linker and the graph channel | `prompts/`, `scripts/` | |

A removed channel or stage in a **stored** config is never an error: `build_channels` and the
stages skip it and log one warning per process per name (`prospecta.retired`), and the recall
goes on with what remains (a config of retired channels only returns nothing). A **new** config
cannot name them: `set_channel_config` and `set_recall_config` raise `ValueError`. No bank row is
ever rewritten: the library does not change a bank's stored `channel_config`, `recall_config` or
`link_config`.

## Which banks are legacy

A bank whose stored `channel_config` is `[]` (the column default; every bank created before the
new-bank defaults, unless it was configured by hand) is a **legacy bank**:

```sql
SELECT bank_id, channel_config = '[]'::jsonb AS legacy, recall_config, link_config FROM banks;
```

To move a legacy bank to the current design, create `<bank>-v2` with `prospecta migrate-bank` (it
writes chunk items from the documents and gets the new-bank defaults,
`docs/embedding-migration.md`), or set the config by hand on a bank that already has chunk items:

```python
mem.set_channel_config([{"name": "hybrid_chunk", "weight": 1}])
mem.set_recall_config({"rerank": {"enabled": True, "stage": "cross_encoder"}})
```

`hybrid_chunk` reads `kind = 'chunk'` items only. A bank written with `index_text=` holds only
question items and has nothing for it to find.

## Kept on purpose

| kept | why |
|---|---|
| **The three-channel recall** (`channel_config = []`: semantic over `content`, `content_tsv`, `body_tsv`, fused by RRF; `db/queries.hybrid_search`, `Memory.search`) | It is the read path of the banks that stored `[]` (`augur` and `forge` are such banks: backups, no longer served by a gateway). It takes no stages and no Stage 1 filters (`date_from`, `date_to`, `excluded_sources` raise `ValueError`); `expand_and_recall` returns an `error`. `search(mode="semantic" / "lexical")` and `retain --index-text` belong with it. |
| **`kind = 'question'` items** in the database, and `migrate-bank`'s copy of them (re-embedded, no model call; `_embed_migrate.py`) | Existing banks' question items stay untouched; the legacy path matches them. |
| **Every migration and table** (0006, 0012, 0014, 0016, 0018, 0019 and the tables, functions and columns they made) | An applied migration is never edited or deleted; existing banks hold rows in those tables. The tables are empty on a new bank and nothing reads or writes them now. |
| **`memory_items_keyword_gin`** (migration 0018) | No code uses this index any more. It is **not dropped**: a database that has it keeps it; a new database no longer builds it (`prospecta migrate` stopped calling the builder; the advisory key `MIGRATE_LOCK_KEY + 3` stays reserved). Dropping it is the operator's call (`docs/exact-tsv.md`). |
| **`banks.link_config`**, written `{"links": false, "entities": false}` by `create_bank` | This library reads nothing from it. An older library reading the bank treats a stored `{}` as "entities on", so the column is still written to keep a new bank unlinked by any version. |
| **The channels `dense_chunk` and `meta`** with `fusion.method = floor_union` and scope promotion, `extract_filters` | Not named by the removal, and the plugin's date-cue gate wraps `prospecta._index.extract_filters`. `meta` matches a person's name to `documents.person` (honorifics and case fold) without the entity tables. |
| **`recall_synth`, `formulate_queries`** (also `grounded=True`, `recall_config.evidence`) and `prospecta eval` | `hermes-prospecta` calls `recall_synth` (the `prospecta_recall` tool and the old prefetch mode); `eval` measures the channel set that remains. |
| **The `cross_encoder` stage and the decision model** (`JevScore`: a holder of the transport and model; `JevSufficiency`) | Stage 2, Mode 1's Stage 0 gate and Stage 3 of `ask` use them. |

## Measured at the time

Links moved recall scores by 0.00 to 0.02 and grew `memory_links` to about 300 million rows on a
73,000-note bank, which is why they were off by default and then removed. The benchmark tables of
the retired `graph-links` doc are in the git history (`git log -- docs/graph-links.md`).
