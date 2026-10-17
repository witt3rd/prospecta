# The stored exact vector (`memory_items.exact_tsv`, migration 0020)

The Stage 1 exact / phrase retriever (`docs/postgres-hybrid.md`, `hybrid_chunk` channel) matches and
ranks the `simple` word vector of `original_chunk`. Until migration 0020 that vector was computed
per row at query time, so a question of only very common words (`the`) re-tokenized every chunk that
holds them: 6 to 7 seconds on a synthetic 73,000-chunk bank, 2.5 s on forge-v2 in the slice 5
verification, growing linearly with the bank. Migration 0020 stores the vector:

- `memory_items.exact_tsv tsvector GENERATED ALWAYS AS (<word rule>) STORED` — the database fills it on
  every insert and update; no application code can forget it.
- `memory_items_exact_tsv_gin`, a GIN index on it, built `CREATE INDEX CONCURRENTLY` (writes are never
  blocked) right after the migration commits.

**One definition of the word rule.** `indexed_expression()` in `prospecta/db/exact_tsv.py` (the
`simple` configuration plus the pieces of a path split at slashes) is the expression migration 0018 once
indexed for the removed `keyword` channel. The migration file carries only the placeholder
`{{EXACT_TSV_EXPRESSION}}`, which `migrate.run_migrations` fills from that function; the query names only
the column. `tests/test_exact_tsv.py` checks that every row's column equals the expression, and that the
Stage 1 results equal the old expression form row for row.

`memory_items_keyword_gin` (migration 0018) is no longer used by any code (the `keyword` channel was
removed; see "The retired keyword index" at the end of this page). A database that has it keeps it.

## What the migration costs (read before applying it to a live database)

**Adding the column REWRITES `memory_items`.** PostgreSQL computes the new column for every row of every
bank and kind, writes a new copy of the table and rebuilds every index on it (the HNSW indexes
included), all under an **`ACCESS EXCLUSIVE` lock held for the whole rewrite (172 s on the synthetic
73,000-chunk bank with its HNSW indexes, table below): reads and writes of `memory_items` wait until it
commits.** Nothing can query the table while it runs, so on augur-v2 and forge-v2 the migration runs
**with the gateways STOPPED, inside the restart window**, never against a serving gateway: a recall
that arrives during the rewrite hangs until its request timeout. Every process that calls
`run_migrations` with this release (`prospecta migrate`, `prospecta init`, `python -m prospecta.service`
at start-up) runs it, so a gateway must not start on this release before the migration has been run by
hand, and the live window length is a **measurement on a copy** (pre-flight below), not an estimate.

Measured on a synthetic 73,000-chunk bank (1536-dim vectors, 120-word chunks, two HNSW indexes of
about 600 MB each, container Postgres 16.15 with pgvector 0.8.7 on a 22-core host):

| step | time |
|---|---|
| `ALTER TABLE ... ADD COLUMN exact_tsv ... STORED`, table + non-HNSW indexes only | 30 s |
| the same with the two HNSW indexes present (rebuilt by the rewrite) | **172 s** |
| `CREATE INDEX CONCURRENTLY memory_items_exact_tsv_gin` (no lock) | 2.6 s |

| size (73,000 chunks) | before | after |
|---|---|---|
| `memory_items`, table + TOAST + all indexes | 1,998 MiB | 2,143 MiB (+145 MiB, +7%) |
| the new column (stored, compressed) | | 109 MiB (about 1.5 KB a row: every word is in it twice, see the rule) |
| `memory_items_exact_tsv_gin` | | 18 MiB |

The other indexes shrink a little (the rewrite removes their bloat), so the net growth is smaller than
column plus index. Real chunks are shorter than the synthetic ones (p50 326 characters), so the column
and its time per row are smaller; the HNSW rebuild, which grows with the vector indexes, dominates.
The live figures are an estimate, not a measurement: the slice 5 report lists about 2.2 GB of HNSW
indexes for forge-v2 and 0.5 GB for augur-v2 (against 1.2 GB in the synthetic bank), plus the old
banks' indexes, so expect **roughly 6 to 10 minutes** for the rewrite on x-class hardware and more on
slower hardware: the pre-flight below measures it on a copy. The rewrite also writes the whole table and its
indexes to WAL and needs free disk of about the current size of `memory_items` plus the new column.

## Operator runbook (augur-v2 and forge-v2: at go-live, with the captain's word only)

One database holds both banks in one `memory_items` table, so this is one migration, not one per bank.

### Pre-flight (REQUIRED, before the window): measure the migration on a COPY of the live database

The time and size figures above are for a synthetic bank. The live table's rewrite time and size delta
depend on its row count, its chunk text and above all its HNSW indexes, so **no go-live is scheduled
until the two numbers below have been measured on a copy of the live data and written into the go-live
notes.** The copy is a scratch database on host x or chef; **the live database is never migrated for a
rehearsal and never connected to by anything but the dump.**

1. **Copy.** The rewrite scales with the whole table and its indexes, not with one bank, so copy the
   whole database (it holds every bank's `memory_items`, documents and the HNSW indexes): the step 1
   backup is that copy (`pg_dump -Fc`, read-only: it takes no lock that blocks a reader or writer). On
   x or chef start a throwaway Postgres of the **same major version and pgvector version as live**
   (`SELECT version();` and `SELECT extversion FROM pg_extension WHERE extname = 'vector';` on live;
   settings that matter for the timing: `shared_buffers`, `maintenance_work_mem`, `max_wal_size`; use the
   live values), then restore the dump into it (`docs/embedding-migration.md` section 1 has the
   commands; `prospecta backup` / `restore` wrap them):
   ```
   pg_restore --no-owner -d "$SCRATCH_DATABASE_URL" prospecta-YYYY-MM-DD.dump
   ```
   Compare `documents` and `memory_items` counts per bank between live and the scratch.
2. **Measure before.** On the scratch:
   ```sql
   SELECT pg_size_pretty(pg_total_relation_size('memory_items')) AS table_toast_and_indexes,
          pg_size_pretty(pg_relation_size('memory_items'))       AS heap,
          pg_size_pretty(pg_indexes_size('memory_items'))        AS indexes,
          pg_size_pretty(pg_database_size(current_database()))   AS database;
   ```
   and note the free disk on the scratch volume (`df -h`).
3. **Migrate the scratch and time it.** With the new release's CLI against the **scratch** URL only:
   `time prospecta --database-url "$SCRATCH_DATABASE_URL" migrate` (it prints `Applied: [20]`). Record the
   wall-clock **seconds** (the `ACCESS EXCLUSIVE` rewrite, then the concurrent index build), the peak
   free-disk dip and the WAL written (`SELECT pg_current_wal_lsn()` before and after; `pg_wal_lsn_diff`).
4. **Measure after**: the same size query; record the **size delta** (table, indexes, database).
5. **Write the numbers into the go-live notes**: rewrite seconds, index-build seconds, size delta, WAL, and
   the free disk the live volume needs (at least the current `pg_total_relation_size('memory_items')`
   again, plus the delta). The restart window is those seconds with a margin (at least twice) plus the
   restart; if that window is not acceptable, the alternative (a plain nullable column kept by a trigger
   and backfilled in batches, which avoids the rewrite and its lock but is not the stored generated
   column `docs/postgres-hybrid.md` specifies) is reconsidered before go-live, not during it.

The scratch can be repeated as often as needed (drop the scratch database and restore again); only the
numbers are carried to the go-live. Remove the scratch and the dump when done (they hold real data).

### The go-live window

1. **Back up first.** `docs/backup-restore.md` (a `pg_dump` of the database; the generated column is
   recomputed on restore, it is not dumped). Check free disk against the pre-flight's figure.
2. **Stop the gateways** (augur and forge, and anything else that reads or writes `memory_items`: the
   rewrite's `ACCESS EXCLUSIVE` lock makes every query wait) for the whole window. **No gateway or
   service starts on this release before step 3 has finished.**
3. **Migrate by hand**, with the new release's CLI and the database URL in the environment:
   `prospecta migrate`. It prints `Applied: [20]`. Expect the pre-flight's seconds: the rewrite first (the
   lock), then the index build (no lock). Optionally `ANALYZE memory_items;` afterwards (autovacuum does
   it in due course; the Stage 1 plan does not depend on it).
4. **Verify.**
   ```sql
   SELECT attgenerated FROM pg_attribute
    WHERE attrelid = 'memory_items'::regclass AND attname = 'exact_tsv';   -- 's'
   SELECT indisvalid, indisready FROM pg_index
    WHERE indexrelid = 'memory_items_exact_tsv_gin'::regclass;             -- t, t
   SELECT count(*) FROM memory_items WHERE exact_tsv IS NULL AND original_chunk IS NOT NULL;   -- 0
   ```
   and one `the` question through Stage 1 (about 0.1 to 0.7 s, not seconds).
5. **Start the gateways** on the new release (the end of the window).
6. **Interrupted index build** (the migration was killed after the rewrite): run `prospecta migrate`
   again. It sees the invalid or missing index, drops the leftover and rebuilds it concurrently.
   The rewrite is never repeated: the migration is recorded as applied, and the column add is
   `IF NOT EXISTS`.

**Rollback** (before or after step 5; the old release answers on the old schema):

```sql
DROP INDEX IF EXISTS memory_items_exact_tsv_gin;
ALTER TABLE memory_items DROP COLUMN IF EXISTS exact_tsv;      -- instant: no rewrite
DELETE FROM prospecta_schema_version WHERE version = 20;
```

then run the old release. Rolling back and migrating again round-trips (`tests/test_exact_tsv.py`).
If the rewrite itself fails, its transaction rolls back and nothing changed.

## What is left after the stored column

A question of only very common words no longer tokenizes anything, but ranking by `ts_rank_cd` still
scores every chunk that holds a query word (a few microseconds each; the GIN index returns candidates
in no rank order, so a `LIMIT` before the rank would change which chunks win and is not done). On the
synthetic bank: `the` (44,498 matching chunks) 6.2 s to 0.25 s, `the and a of` 6.7 s to 0.67 s, a word
in 12% of the chunks 1.1 s to 0.09 s, one in 40% of them 3.9 s to 0.32 s; selective queries stay at
5 to 20 ms. The cost still grows with the bank and with the number of matching chunks.

## The plain-question OR (stemmed retriever) and its cost

A plain question is read by the stemmed retriever as the OR of **all** its stemmed words
(`docs/postgres-hybrid.md` section 3). There is no cap: an earlier version kept the 8 longest words and
dropped the rest without a word, and that silent loss is gone. The OR ranks every chunk holding any of
the words (`ts_rank_cd`, a few microseconds a matching chunk, and more with every extra word), so the
cost depends on how common the words are and how many there are.

Measured on the 73,000-chunk synthetic bank (Zipf words, container Postgres 16 on host x; p50 and max
of 9 warm runs, server-side `EXPLAIN ANALYZE` execution time; the client-side time is within 10%):

| plain question | server p50 | server max |
|---|---|---|
| 8 mid-frequency words (what the cap allowed) | 0.32 s | 0.37 s |
| 20 rare words (each in under 1% of the chunks) | 0.31 s | 0.40 s |
| 20 words: 14 of mixed frequency and 6 stop words | 0.29 s | 0.35 s |
| **20 mid-frequency words** (each in a few percent of the chunks) | **1.41 s** | 2.66 s |
| 20 very common non-stop words (the worst case) | 3.16 s | 3.58 s |
| 40 mid-frequency words | 4.22 s | 5.91 s |

So a 20-word question is fine when its words are mixed or rare (about 0.3 s, like a short one), and slow,
over a second, when most of its words are common content words; a long list of common words costs
seconds. This is not capped here: whether to bound it (and how) is a product decision. The automatic
recall of `docs/postgres-hybrid.md` section 1 feeds the stemmed retriever a short `query` per search, so
the words per OR stay few there.

## The retired keyword index (`memory_items_keyword_gin`, migration 0018)

Not part of Stage 1, and **no longer used by any code**. Migration 0018 owed a GIN expression
index, `memory_items_keyword_gin`, over the word rule of the `keyword` channel (the `simple`
text-search configuration, stemming off, plus path pieces split at slashes): the same rule
`exact_tsv` is generated from (`indexed_expression()` in `prospecta/db/exact_tsv.py`). The
`keyword` channel and `prospecta keyword-index` were removed (`docs/legacy.md`), and
`prospecta migrate` no longer builds the index.

- **A database that has it keeps it.** It is not dropped by any code. It costs disk and write time
  on every row written, shared by all banks of the database; the operator may drop it when
  nothing of the previous library still runs (`DROP INDEX CONCURRENTLY memory_items_keyword_gin`):
  that is a decision for the operator, not a step of any upgrade.
- **A new database does not get it.**
- Its advisory key (`MIGRATE_LOCK_KEY + 3`) stays reserved (`prospecta/db/builder_locks.py`), so a
  process still running the previous library, which builds the index, takes turns with this one.
