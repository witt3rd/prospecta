---
name: operating
description: Running a live Prospecta bank and the gateways that use it. Use before a backup, a migration, a restart, a plugin or library change going live, a per-bank setting change, or any cleanup.
---

# operating

The operator rules, each with its reason. Where the evidence is a document in this repo it is linked; where it
is house practice from the go-live runbooks (kept outside this repo), it says so. Migrations and backup
commands are exact because they are fragile: [`references/migrations-and-backups.md`](references/migrations-and-backups.md).
Restart and live-plugin detail: [`references/restarts-and-live-plugin.md`](references/restarts-and-live-plugin.md).

## Before any change: a verified dump

`pg_dump -Fc`, record its `sha256`, run `pg_restore --list` and note the entry count; for a migration also
restore into a scratch database and compare per-bank `documents` / `memory_items` counts. Keep only the
latest verified dump of each kind. *Why:* a migration is not undone and nothing proceeds without a dump you
know reads back. *Evidence:* [`docs/embedding-migration.md`](../../docs/embedding-migration.md) section 1,
[`docs/exact-tsv.md`](../../docs/exact-tsv.md) (pre-flight), [`docs/backup-restore.md`](../../docs/backup-restore.md);
the sha256 and keep-latest habit is house practice. A dump holds real data: remove scratch copies when done.

## Migrations

Schema version **20**. Migration `0020` adds `memory_items.exact_tsv` and **rewrites the table under an
`ACCESS EXCLUSIVE` lock** (172 s on a synthetic 73,000-chunk bank; expect minutes live). Both gateways are
**stopped**, the rewrite is **timed on a copy first**, and `prospecta migrate` is run **by hand with nothing
else starting**: any process that calls `run_migrations` (the service at start-up, `prospecta init`) applies
it, so a gateway must never be the first to start on a new release. *Why:* every query on the table waits;
a recall that arrives mid-rewrite hangs until its timeout. *Evidence:* `docs/exact-tsv.md` (runbook), `CHANGELOG.md` 1.0.0.

## Restarts

A library or plugin **code** change loads only at a gateway restart. Restart only after a quiet check, twice,
20 seconds apart: no turn in the log for a few minutes, no child processes, the state file says running.
Never at a mail-check minute. One gateway at a time. Keep **a venv per library pin**, so rollback is
repointing a drop-in at the previous venv. *Why:* a restart kills a turn in flight. *Evidence:* house practice
(go-live runbook); the plugin README says what needs no restart. Detail in the reference.

## The live-plugin rule

**The live plugin directory the gateways load must not follow main.** Keep it **detached** at a recorded
commit, moved only by the go-live runbook. The plugin re-reads its recall contract file by modification
time on every recall, so a contract or other runtime file change goes live at pull time **without a restart**,
while code waits for one. A project clone that syncs with main must be a **different directory**.
*Why:* a merge to main would otherwise change what a running gateway reads, with no decision and no record.
*Evidence:* `load_contract` in hermes-prospecta `prefetch_contract.py` keys its cache on the file's mtime;
the rule itself is house practice. [`references/restarts-and-live-plugin.md`](references/restarts-and-live-plugin.md).

## Per-bank settings

A value stored in the bank's `recall_config` / `channel_config` **wins over a code default**, so changing a
default changes only banks that do not pin their own. A new bank stores only `hybrid_chunk` and
`rerank = cross_encoder`; the model, floor, `top_n` and timeout come from code and environment. A change to
one bank is a one-row update (`Memory.set_recall_config`; it replaces the whole value) and is read on every
recall, so it needs no restart. *Evidence:* `docs/recall-stages.md` (top), `read_recall_config` in
`prospecta/stages.py`, `docs/schema.md` (banks). Resolution order: per call, bank, environment, default.

## Connections

About 10 per gateway (the plugin's default is 4 per Memory and it can build several Memorys (main, shadow, dual write); the
service pool defaults to 10). Check `SELECT count(*), state FROM pg_stat_activity GROUP BY state;` against
`max_connections` before adding a process. *Why:* running out stops every recall. *Evidence:* hermes-prospecta
README (`max_connections`, `connection_watch`); [`docs/source-contract.md`](../../docs/source-contract.md) ("Database connections").

## What never to do

- **Never delete the old bank rows** after a migration or re-embed (`<bank>` stays beside `<bank>-v2`);
  old tables and rows stay as history. *Evidence:* `CHANGELOG.md` ("Old tables and rows ... are never read" and never rewritten);
  `docs/embedding-migration.md` ("the source bank is only read, never modified").
- **Never prune session history.** *Why:* it is the only copy of what was said. House rule.
- **No caps on content.** No truncation, no count limits on evidence ([`PRINCIPLES.md`](../../PRINCIPLES.md) P5; `docs/limits.md`).
- **Never drive a live database to rehearse**; use a scratch restore. *Evidence:* `docs/exact-tsv.md`.

## Related

[`tuning-and-testing`](../tuning-and-testing/SKILL.md) before changing a default; [`retain`](../retain/SKILL.md) for `exact_tsv`.
