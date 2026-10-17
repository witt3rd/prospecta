# Schema

The Postgres schema as the migrations `0001` to `0020` build it
(`prospecta/db/migrations/`). It is the reference for anyone who queries or operates the
database; the design that uses it is `docs/postgres-hybrid.md`. Everything below was
read from a database migrated from nothing to 0020 (pgvector 0.8.7), not from the design
notes.

## Layout in one screen

```
banks ──< documents ──< memory_items            kind = 'chunk' (the retrieval unit) | 'question' (legacy)
  │            │              ├──< memory_item_entities >── memory_entities ──< memory_entity_aliases   (legacy)
  │            │              ├──< memory_links, memory_link_pairs                                       (legacy)
  │            ├──< source_refs                  (records indexed in place from host views)
  │            ├──< memory_link_state, memory_alias_state                                                (legacy)
  ├──< retain_events, recall_events ──< recall_event_candidates, llm_calls, formulate_events
  └──< sweep_passes, sweeper_state
prospecta_schema_version                         (the migration ledger)
```

Every table is keyed by `bank_id` (directly, or through its `document_id` / `item_id`) and
deleting a bank cascades to everything it owns, except `llm_calls.bank_id` and
`retain_events.document_id`, which are set to NULL.

## The migration runner

`prospecta migrate` (`run_migrations`, `prospecta/db/migrate.py`) applies the numbered SQL
files in order, once each, recording them in `prospecta_schema_version`. Concurrent runners
serialise on one advisory lock. Hand-written SQL, no ORM; a migration is additive unless it
says otherwise (only `0020` rewrites a table).

Four migrations have a second half that cannot run inside a transaction
(`CREATE INDEX CONCURRENTLY`); `run_migrations` does it right after the commit and on every
later run, so re-running `prospecta migrate` repairs an interrupted build: `0004` (backfill of
`kind`, the `(bank_id, kind)` index and the per-kind HNSW indexes), `0005` (the
`created_on` / `person` indexes), `0018` (the keyword index) and `0020` (the `exact_tsv` GIN
index). `0020` also has a placeholder, `{{EXACT_TSV_EXPRESSION}}`, that the runner fills from
`prospecta/db/keyword_index.py:indexed_expression()` so the word rule is defined once.

**Shared-database mode.** With `PROSPECTA_SCHEMA` (or `Memory(schema=...)`) every connection
pins `search_path` to that schema, every object lands in it, the migration ledger included,
and no extension is created (`vector` must exist; `pg_search` is used only if already
installed). See `docs/source-contract.md`.

## Tables

### banks

One row per bank. A bank is the unit of isolation: its own embedding width, its own recall
config, its own indexes.

| column | type | meaning |
|---|---|---|
| `bank_id` | text PK | `[A-Za-z0-9_-]{1,63}` |
| `embedding_dim` | integer, not null | vector width, **fixed at creation** (changing it is a new bank: `docs/embedding-migration.md`) |
| `embedding_model_id` | text | informational |
| `mission`, `retain_mission` | text | free text the caller keeps with the bank |
| `config` | jsonb `'{}'` | unused by the library |
| `channel_config` | jsonb `'[]'` (0008) | the recall channels (`docs/recall-stages.md`). `[]` is a **legacy bank** (`docs/legacy.md`) |
| `recall_config` | jsonb `'{}'` (0009) | rerank, expansion, gate, Stage 3, models, depth (`docs/recall-stages.md`) |
| `link_config` | jsonb `'{}'` (0019) | linker switches `{links, entities}` (`docs/legacy.md`) |
| `created_at`, `updated_at` | timestamptz | |

The three config columns keep the legacy values as their column `DEFAULT`s. A bank created by
`Memory.create_bank` (so by `prospecta init`, `create-bank`, the service, `migrate-bank` and
`import hindsight`) gets explicit values in the `INSERT` (`prospecta/bank_defaults.py`):
`channel_config = [hybrid_chunk]`, `recall_config = {"rerank": {"enabled": true, "stage":
"cross_encoder"}}`, `link_config = {"links": false, "entities": false}`. No migration writes
bank config, so a bank that exists keeps what it stored; a bank inserted by anything but
`create_bank` is a legacy bank.

### documents

The source of truth for what was retained: the verbatim note.

| column | type | meaning |
|---|---|---|
| `id` | uuid PK | |
| `bank_id` | text → banks | |
| `source` | text | the caller's identifier of the origin (a path, a URL, `turn:41`); opaque to the library; the anti-echo filter and re-retain key on it |
| `original_text` | text, not null | the full note, never truncated; `''` for the empty anchor of a source-view record |
| `content_hash` | text | sha256 of the text; **UNIQUE (bank_id, content_hash)** |
| `tags` | text[] | GIN-indexed |
| `document_metadata`, `retain_params` | jsonb | the caller's metadata over the frontmatter-derived `created`, `created_on`, `person` |
| `created_on` | date (0005, 0013) | the note's own date (the `metadata=` argument, else frontmatter; keys `created_on`, `created`, `date`), else **today in `PROSPECTA_TIMEZONE`** (UTC when unset): `retain` always stamps it |
| `person`, `source_kind` | text (0005) | from frontmatter `person`, `source_kind` / `type`; `source_kind` is the cited `kind` |
| `created_at`, `updated_at` | timestamptz | when the row was written |

Indexes: `(bank_id, source)`, `(bank_id, created_at DESC)`, `(bank_id, created_on)`,
`(bank_id, person)`, GIN on `tags`, and the unique `(bank_id, content_hash)`.

The **note date** every date filter and every citation uses is
`COALESCE(documents.created_on, documents.created_at::date)`.

### memory_items

What is searched. One row per chunk (or, on a legacy bank, per anticipated question).

| column | type | meaning |
|---|---|---|
| `id` | uuid PK | |
| `bank_id`, `document_id` | → banks, → documents | |
| `kind` | text `'question'` (0004) | **`'chunk'`** (a slice of its document, written by `retain` and the directory index) or `'question'` (an anticipated question, only when the caller supplies `index_text` or asks for `generate_questions`; the column default is `'question'` for rows that predate the column) |
| `content` | text | for a chunk, the chunk text; for a question, the question |
| `original_chunk` | text | the source text of the item: the chunk itself, or for a question the chunk of the note that matches it. What Stage 2 reads and what is cited |
| `embedding` | vector (no fixed width) | the embedding of `content`; a trigger checks its width against `banks.embedding_dim` |
| `ordinal`, `char_start`, `char_end` | integer (0004) | the chunk's position and its offsets in `documents.original_text`; NULL for questions |
| `content_tsv` | tsvector, generated | `to_tsvector('english', content)`; read only by the legacy path |
| `body_tsv` | tsvector, generated (0002) | `to_tsvector('english', COALESCE(original_chunk, ''))`: **the stemmed retriever of Stage 1** |
| `exact_tsv` | tsvector, generated (0020) | the keyword word rule over `original_chunk` in the `simple` configuration (unstemmed; a path is also split at its slashes): **the exact / phrase retriever of Stage 1** |
| `metadata` | jsonb | the document's metadata (a question item adds `index_text_caller_supplied`); `record_id`, `kind`, `revision`, `project`, `audience_key` for source-view records; filtered with `@>` |
| `tags`, `context`, `update_mode`, `llm_generated` | | provenance: `llm_generated` is true only for model-written questions; `update_mode` is `'append'` |
| `created_at`, `updated_at` | timestamptz | |

Chunks: at most **1000 characters**, cut at paragraph boundaries, with **100 characters** of
overlap from `retain` and **200** by default from `index_directory` (`chunk_size`,
`chunk_overlap`; `prospecta/_chunker.py`). No model is called to write them.

Indexes: `(bank_id, created_at DESC)`, `(bank_id, kind)`, `(document_id)`, GIN on `tags`,
`content_tsv`, `body_tsv`, `exact_tsv` (`memory_items_exact_tsv_gin`, built concurrently), the
legacy keyword expression index `memory_items_keyword_gin` (0018), and, **per bank, three
partial HNSW indexes** on `(embedding::vector(N)) vector_cosine_ops` with `m = 16`,
`ef_construction = 64`: `memory_items_embedding_<bank>_idx` (`WHERE bank_id = '<bank>'`),
`…_<bank>_chunk_idx` and `…_<bank>_question_idx` (`… AND kind = '<kind>'`), where `N` is the
bank's `embedding_dim` and a hyphen in the bank id becomes an underscore (a name over 63 bytes
is shortened with an md5 suffix). `create_bank` builds them (`CREATE INDEX CONCURRENTLY`, so it
cannot run in a transaction). A query uses them only when its text repeats the same
`embedding::vector(N)` cast and the literal `bank_id` / `kind` predicates, which is why Stage 1
is planned per call (`prepare=False`). **There is no `halfvec` and no partitioning.**

Trigger: `memory_items_embedding_dim_check` (before insert or update of `embedding` /
`bank_id`) raises if `vector_dims(embedding)` differs from the bank's `embedding_dim`.

### source_refs (0017)

The ledger of records indexed **in place** from host-provided views
(`docs/source-contract.md`): one row per `(bank_id, kind, record_id)` with the indexed
`revision`, the filter columns the view supplied (`person`, `project`, `created_date`,
`audience_key`) and the `document_id` of an empty anchor document. It holds no record text.

### Event tables

All append-only except `sweeper_state`; see `docs/observability.md` for what each carries and
how to read it.

| table | one row per |
|---|---|
| `retain_events` | `retain` call: `document_id`, `items_count`, `index_text_caller_supplied`, `index_text_generated` (0003, only for model-written questions), `duration_ms`, `error` |
| `recall_events` | recall: `queries`, `mode`, `n_results`, `results` (0003; a 200-character preview per hit), `synthesis`, and the trace columns `plan`, `channels`, `fusion`, `rerank`, `hops` (0007), `cost_usd`, `tokens_in`, `tokens_out`, `n_llm_calls` (0009), `citations` (0011), `depth` (0015) |
| `recall_event_candidates` (0007) | candidate a channel returned for a recall: `recall_event_id`, `query_index`, `document_id`, `item_id`, `channel`, `rank`, `score` |
| `llm_calls` | model call: `prompt_name` (the purpose), `messages_count`, `json_mode`, `duration_ms`, `error`, `prompt_text`, `response_text` (0003, opt-out), `model`, `tokens_in`, `tokens_out`, `cost_usd` (0007) |
| `formulate_events` | `formulate_queries` call (legacy): `message`, `n_queries_out`, `parse_fallback`, `error_kind` (0003), `raw_response` |
| `sweep_passes`, `sweeper_state` | background sweep: files seen / indexed / pruned, errors, duration; `sweeper_state` is the last pass per `(bank_id, corpus_path)` |

### Legacy tables

Created by `0006`, `0014`, `0016`; used only by the entity / link / graph machinery that new
banks do not run (`docs/legacy.md`): `memory_entities`, `memory_item_entities`,
`memory_entity_aliases`, `memory_links`, `memory_link_state`, `memory_link_pairs`,
`memory_alias_state`, and the SQL function `prospecta_doc_date` (0006, 0013). They are empty for a bank that never
extracted entities or built links.

## Write semantics

**`retain(content, source=...)`** (`prospecta/_retain.py`), in one transaction:

1. `content_hash = sha256(content)`. A transaction advisory lock on `(bank, hash)` serialises
   writers of the same text.
2. A document with the same hash and the **same source** exists: **replace**. The document row
   is kept (`updated_at`, tags and metadata refreshed), its `memory_items` are deleted and
   written again.
3. The same hash under a **different source**: `DocumentSourceConflictError`. The library
   treats `source` as opaque and does not re-bind a document; the caller routes.
4. A new hash: a new document row, whatever the source. (`index_directory` goes further: a file
   whose content changed replaces the earlier document of that source.)
5. The note is cut into chunks (frontmatter excluded), embedded in one call, and written as
   `kind = 'chunk'` items. **No model is called.** With `index_text=` the caller's questions are
   written *instead of* the chunks (`child_chunks=True` writes both); `generate_questions=True`
   has the model write them. A directory file whose frontmatter has `index_text:` gets its
   questions and its body chunks.
6. `documents.created_on` is always set (see above); a `retain_events` row is written.

**Deleting** a document (`remove_documents`, a source-view `remove`) deletes its items, their
entity and link rows and its `source_refs` row by cascade.

## Gotchas

1. **HNSW is per bank, per kind.** The partial indexes are why Stage 1 needs the literal bank
   and kind and the `embedding::vector(N)` cast, and why creating a bank is not instant on a
   big table. Dropping a bank leaves its indexes to be dropped by hand.
2. **`exact_tsv` is stored, and adding it rewrote the table.** Migration `0020` computes it for
   every row (`ACCESS EXCLUSIVE`, every index rebuilt). Before applying it to a live database
   read `docs/exact-tsv.md`.
3. **`body_tsv`, `exact_tsv` and `content_tsv` are generated.** They are never written to;
   changing `original_chunk` or `content` recomputes them. Their word rules are fixed in the
   migration: changing the rule is a new migration and a rewrite.
4. **`created_on` is always set**, so a note without a date of its own carries the day it was
   retained; a date filter then selects by retention day for such notes.
5. **`embedding_dim` is immutable.** A bank cannot be re-embedded in place: `migrate-bank`
   writes `<bank>-v2` beside it.

## Migrations

| # | what it adds |
|---|---|
| 0001 | `banks`, `documents`, `memory_items` (+ `content_tsv`, the dimension trigger), the event tables, the ledger |
| 0002 | `memory_items.body_tsv` + GIN |
| 0003 | durable event detail: `retain_events.index_text_generated`, `recall_events.results` / `synthesis`, `llm_calls.prompt_text` / `response_text`, `formulate_events.error_kind` |
| 0004 | `memory_items.kind`, `ordinal`, `char_start`, `char_end`; `(bank_id, kind)`; per-kind HNSW indexes (post-commit) |
| 0005 | `documents.created_on`, `person`, `source_kind` + indexes (post-commit) |
| 0006 | entities, entity-item join, typed links, link state, `prospecta_doc_date` (legacy) |
| 0007 | recall trace: `plan`, `channels`, `fusion`, `rerank`, `hops`; `recall_event_candidates`; `llm_calls.model` / tokens / cost |
| 0008 | `banks.channel_config` |
| 0009 | `banks.recall_config`; `recall_events` cost and token totals |
| 0010 | comments only |
| 0011 | `recall_events.citations` |
| 0012 | optional `pg_search` BM25 index on chunk items; a no-op where `pg_search` is not installable (legacy) |
| 0013 | `prospecta_doc_date` reads `documents.created_on` |
| 0014 | `memory_entity_aliases`, `memory_alias_state` (legacy) |
| 0015 | `recall_events.depth` |
| 0016 | `memory_link_pairs` (legacy) |
| 0017 | `source_refs` |
| 0018 | the keyword expression index `memory_items_keyword_gin` (post-commit; legacy) |
| 0019 | `banks.link_config` |
| 0020 | `memory_items.exact_tsv` + GIN (**rewrites `memory_items`**) |
