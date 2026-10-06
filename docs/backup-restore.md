# Backup and restore

```bash
prospecta backup /path/to/prospecta.sql      # pg_dump of the whole database
prospecta restore /path/to/prospecta.sql     # replay into an EMPTY database
```

Both use the CLI connection setting (`--database-url` or `DATABASE_URL`) and
need `pg_dump` / `psql` on PATH.

- The dump is plain SQL of the entire database: every bank, documents, memory
  items, events, and the pgvector data (the `vector` extension is created by the
  dump itself).
- Restore runs in a single transaction and stops on the first error, so a
  failed restore leaves nothing half-applied. Target must be a fresh database
  (create it first: `createdb prospecta_new`); it does not drop or merge.
- A newer `pg_dump` client than the server emits `SET transaction_timeout`;
  restore strips that line, so version skew is harmless.
- Restore into a different database than the source to verify a backup; never
  point it at a live database that already holds data.

Tested end to end in `tests/test_backup_restore.py`: retain, backup, restore
into a fresh database, recall.
