# Importing a Hindsight bank

```bash
# restore a Hindsight pg_dump / cold copy into a scratch Postgres first (read-only use), then:
prospecta --bank mybank import hindsight postgres://.../hindsight --hindsight-bank hermes
prospecta import hindsight postgres://.../hindsight --all-banks     # each lands in a same-named bank
```

Options: `--batch N` (rows per batch, default 1000), `--json` (machine report). Exit 0 = no failures, 3 = some rows failed (listed; the rest imported), 1/2 = bad arguments / source unreadable.

The source is only read. The format is the real Hindsight schema (Postgres, pgvector, alembic `b3e8d1c6f4a9`); the tests use a synthetic database generated with that table/column layout, generic content, scaled-down counts.

## Where each of the 17 Hindsight tables goes

| Hindsight table | Lands in prospecta as |
|---|---|
| `memory_units` | one `documents` row (`source = hindsight:<unit id>`, `original_text` = text verbatim, `created_at`/`updated_at` kept) + one `memory_items` row (content = text). Whole unit row (fact_type, context, event_date, occurred_*, mentioned_at, proof_count, source_memory_ids = observation provenance, consolidated_at, metadata, ids…) kept in `documents.document_metadata.hindsight`. Tags: `hindsight`, `fact_type:<t>`, `entity:<slug>`, unit tags. |
| `unit_entities` | `imported_edges` kind `unit_entity` |
| `entity_cooccurrences` | `imported_edges` kind `cooccurrence` (count, last_cooccurred) |
| `memory_links` (3.37M in the real dump) | `imported_edges` kind `memory_link` (type, entity, weight, created_at) |
| `entities`, `documents`, `chunks`, `observation_history`, `mental_models`, `mental_model_history`, `knowledge_pages`, `invalidated_memory_units`, `directives`, `file_storage`, `audit_log`, `llm_requests`, `banks` | `imported_records` (kind per table), one row each, the full source row as JSON (primary key as `source_key`). Nothing is interpreted; it is preserved so nothing is lost. Hindsight `documents`/`chunks` keep their original text here; the units reference them by id. |

`imported_records` and `imported_edges` are created on first import (`CREATE TABLE IF NOT EXISTS`, cascade-deleted with the bank). Edges and records are inserted in batches (`INSERT … SELECT unnest(…) ON CONFLICT DO NOTHING`), so the 3.37M links stream through a server-side cursor in `--batch`-sized chunks.

### Deliberately not carried (counted and reported as `not carried`)

- `alembic_version` — Hindsight's migration marker.
- `async_operations`, `graph_maintenance_queue` — transient job queues; their products are imported.
- `bank_stats_cache` — derived cache.
- `webhooks` — outbound endpoint config with signing secrets; secrets are not copied.
- Derived columns not copied: `memory_units.search_vector` (prospecta regenerates its own tsvector) and the `memory_units_bm25` view.

## Embeddings (dim-384 decision)

Hindsight embeds each unit's text with MiniLM (384-dim) — the same as prospecta's offline default `all-MiniLM-L6-v2`. So each unit's vector is **carried verbatim** into `memory_items.embedding` when its dimension equals the target bank's. If the target bank doesn't exist it is created with the source's dimension (384). Units with no embedding, or a different dimension than an existing bank, are **re-embedded** with the configured embedder (`PROSPECTA_EMBEDDER`); if a batch embed call fails it is retried text by text, so one bad text never sinks the others. Only a unit that cannot be embedded at all (or is unreadable) lands in the loss report as failed, with the reason; with no embedder configured nothing can be re-embedded and those units are reported failed, never dropped. Re-run once the embedder works: already-imported units are skipped. The report shows `embeddings: carried=… re_embedded=…`.

Consequence: imported items are indexed on the fact text (content-space), not on LLM-authored anticipated questions. Recall queries against a carried bank must be embedded with the same 384-dim MiniLM model (`PROSPECTA_EMBEDDER=sentence-transformers`).

## Idempotency

- Units: a document with source `hindsight:<id>` already in the bank → skipped as duplicate. `content_hash` is `sha256("hindsight:<id>\n<text>")`, so two units with identical text are both kept (the unique content hash would otherwise collapse them).
- Records/edges: primary-keyed, `ON CONFLICT DO NOTHING`. Existing rows are never updated; re-running the same dump adds nothing.

## No-loss report

Per Hindsight table: `in = imported + skipped_duplicate + failed` (or `in = not_carried`). Failed rows are listed with table, key and reason; a failing batch is retried row by row so one bad row doesn't hide the rest. Also reported: embeddings carried vs re-embedded, and notes (bank created, tables missing from the source).
