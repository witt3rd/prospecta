# Embedding dimension migration (384 → 1,536): runbook

A bank's `embedding_dim` is fixed, so changing it means a **new bank**,
`<bank>-v2`. The old bank is only read, never modified; the old 384-d path
keeps serving until you switch the callers' bank id.

**Never run this against a live database before a verified dump, and rehearse
on a restored scratch copy first.**

## 1. Dump and verify

```
pg_dump -Fc -f /path/to/backup/prospecta-$(date +%F).dump "$DATABASE_URL"
pg_restore --list /path/to/backup/prospecta-*.dump >/dev/null     # readable
createdb prospecta_scratch                                         # then:
psql prospecta_scratch -c 'CREATE EXTENSION IF NOT EXISTS vector'
pg_restore --no-owner -d prospecta_scratch /path/to/backup/prospecta-*.dump
```

Compare `documents` and `memory_items` counts per bank between live and the
scratch restore. Nothing proceeds without a verified dump. (`prospecta backup`
/ `restore` wrap the dump and restore steps.)

## 2. Rehearse on the scratch restore

Point `DATABASE_URL` at the **scratch** database, apply additive migrations
(`prospecta migrate`), then run the backfill below. Check the report and
counts. Only then repeat on live, in a window with writers paused.

## 3. Backfill

```
prospecta --database-url "$DATABASE_URL" migrate-bank --source-bank forge
# defaults: target forge-v2, 1536 dims, model id openai/text-embedding-3-large@1536,
#           embedder litellm openai/text-embedding-3-large cut to 1536, batches of 128
```

This creates `<bank>-v2` (`embedding_dim=1536`,
`embedding_model_id=openai/text-embedding-3-large@1536`, plus its HNSW
indexes) and, per source document, in one transaction: copies the document
(recording `migrated_from_document_id` in `document_metadata`), inserts its
`kind='chunk'` items (≤1,000 chars at paragraph boundaries, small overlap) and
re-embeds each existing `kind='question'` item's text (no LLM call).

**Resumable by document id.** A source document is done when the target holds a
document carrying its id; re-running the same command skips those. Embedder
errors (rate limits) retry with exponential backoff and, if they persist, abort
cleanly (exit 4) - just run the command again. Exit 5 means documents remain
(e.g. `--max-documents` staging). `--max-documents N` bounds a run.

Library: `prospecta._embed_migrate.migrate_bank(memory, source_bank, embed=...,
embedding_dim=...)` takes any embedder callable (tests use stubs).
Cost: roughly $0.3 for ~3k documents, ~$7 for ~71k (design 8.4).

## 4. Shadow reads

```python
Memory(database_url=url, bank_id="forge", embed=embed384,
       shadow_bank_id="forge-v2", shadow_embed=embed1536)
```

Every `recall()` then also queries the new bank; callers still receive only
the old bank's results. Both runs are stored in `recall_events` (one row per
bank, same `trace->'shadow'->>'id'`, `role` old/new, plus both result sets
mapped to original document ids and their `jaccard` overlap). A shadow failure
is recorded in the trace and never fails the recall. Compare on the logged
queries (`recall_synth` is not shadowed). The single-event behaviour is
unchanged when no shadow bank is set.

## 5. Cut over and rollback

Change the bank id in the callers' config to `<bank>-v2` (and its embedder).
Rollback is the same flip back. Keep the old bank for a rollback period, then
remove it (the delete cascades its items). Those are separate, human-approved
steps; this tool does none of them.
