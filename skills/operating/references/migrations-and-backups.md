# Migrations and backups: the exact steps

Sources: [`docs/exact-tsv.md`](../../../docs/exact-tsv.md) (runbook and pre-flight),
[`docs/embedding-migration.md`](../../../docs/embedding-migration.md),
[`docs/backup-restore.md`](../../../docs/backup-restore.md), [`CHANGELOG.md`](../../../CHANGELOG.md).

## The verified dump

```
pg_dump -Fc -f /path/to/backup/prospecta-$(date +%F).dump "$DATABASE_URL"
sha256sum /path/to/backup/prospecta-*.dump          # record it
pg_restore --list /path/to/backup/prospecta-*.dump | wc -l   # record the entry count
```

Then restore into a scratch database (`createdb`, `CREATE EXTENSION vector`, `pg_restore --no-owner -d`) and
compare `documents` and `memory_items` counts per bank with live. The commands are from
`docs/embedding-migration.md`; the checksum and the entry count are the house record that this dump is the
verified one. `prospecta backup` / `restore` are the plain-SQL variant (`docs/backup-restore.md`): restore only into an
**empty** database, never a live one. Keep the latest verified dump of each kind (pre-migration, pre-upgrade,
...) and remove the older ones.

## Migration 0020 (schema 20)

1. Pre-flight, on a scratch restore of the whole database, same Postgres and pgvector versions as live:
   `time prospecta --database-url "$SCRATCH" migrate` (prints `Applied: [20]`); record rewrite seconds,
   index-build seconds, size delta, WAL, free disk. The window is those seconds times at least two, plus the restart.
2. Back up (above). Check free disk against the pre-flight (about the current size of `memory_items`
   plus the new column).
3. Stop every gateway and service that touches `memory_items`.
4. `prospecta migrate` by hand. The rewrite holds `ACCESS EXCLUSIVE` first, then the GIN index builds
   concurrently with no lock. An interrupted index build is repaired by running `prospecta migrate` again.
5. Verify (`attgenerated = 's'`, the index valid and ready, no NULL `exact_tsv`), then start the gateways.

Rollback of the migration: drop the index and the column, delete version 20 from `prospecta_schema_version`
(exact SQL in `docs/exact-tsv.md`). Rollback of an upgrade: restore the pre-upgrade dump; a bank created by
1.0 stores `link_config = {}`, which an older library reads as "entities on", so it would start extracting
entities (`CHANGELOG.md`).

## Re-embedding

`prospecta migrate-bank --source-bank X` writes `X-v2` beside the source and only reads the source; switch the
callers' bank id afterward. A bank's `embedding_dim` is fixed at creation.
