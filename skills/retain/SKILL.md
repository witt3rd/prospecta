---
name: retain
description: Writing notes into a Prospecta bank. Use when deciding what to retain, how to date and source it, or what the write path does and does not do.
---

# retain

`Memory.retain(content, source=..., tags=..., metadata=...)` and `prospecta retain` write one note. The
reference is [`docs/schema.md`](../../docs/schema.md), "Write semantics"; the code is
`prospecta/_retain.py`.

## What a write is

- **Chunks, no model.** The note is stored verbatim in `documents.original_text`, cut into chunks of at
  most 1000 characters at paragraph boundaries (100 characters of overlap), embedded in one call and
  written as `kind = 'chunk'` rows, in one transaction. Nothing calls a model; there is no question
  indexing and no linking. *Why:* a write that calls a model can fail, cost money and drift with the model
  version; a match on the note's own words stays true. *Evidence:* [`PRINCIPLES.md`](../../PRINCIPLES.md)
  P7 and P20; `docs/observability.md` (`retain_events`: no model call is logged at write time).
- **The write needs only the embedder.** It works offline with local embeddings and no key. *Evidence:*
  [`README.md`](../../README.md), "Offline / keyless retain".
- **Why the linker and extraction are gone.** The bake-off found nothing that beat plain keyword or
  fused search, and the richer recipe cost seconds and dollars per recall. See
  [`references/why-chunks-only.md`](references/why-chunks-only.md).

## Give every note a date

`documents.created_on` is the **note date**: the note's own date (frontmatter `created`, `created_on` or
`date`, or the `metadata=` argument; the caller's keys win), else today in `PROSPECTA_TIMEZONE` (UTC when
unset). It is always set. *Why:* every date window, the date-only search (a query with no words and a
window lists the newest notes in it) and the date shown in every citation read it; a note without its own
date carries its retention day, so a window then selects by when it was written down, not by when it
happened. Pass the real date whenever you know it (a backfill, an import). *Evidence:* `docs/schema.md`
(documents; gotcha 4); `docs/postgres-hybrid.md` section 3 (date-only questions).

## Source, tags, metadata, idempotence

- **`source`** is an opaque string the caller owns (a path, a URL, `turn:41`); the library never parses
  it. It is the citation target and the key of the anti-echo filter, so choose one you can later name in
  `excluded_sources`. With none given, `retain://<hash prefix>` is used.
- **Same content, same source: replace** (the document row stays; its chunks are rewritten). **Same
  content, different source: `DocumentSourceConflictError`**; the caller routes, the library never
  re-binds silently. **New content: a new document.** *Why:* a retry or a re-run must not duplicate
  notes. *Evidence:* `docs/schema.md`, write semantics steps 1 to 4; `tests/test_retain_roundtrip.py`.
- **`tags`** and **`metadata`** are stored with the document. `metadata` is filterable with `@>` through
  `metadata_filter`, and `person` / `source_kind` (frontmatter or metadata) become the filters and the
  cited `kind`.
- **`index_directory`** walks a corpus with the same write path and frontmatter rules (chunk overlap 200
  there by default); the sweeper is the drift safety net, never the hot path (`PRINCIPLES.md` P14).

## The `exact_tsv` column

The exact / phrase retriever reads `memory_items.exact_tsv`, a stored generated column (the unstemmed
`simple` words of the chunk plus the pieces of a path split at slashes). The database fills it on every
insert, so no write path can forget it, and a new database needs nothing. *Why stored:* computed per row
at query time, a question of very common words took 6 to 7 seconds on a 73,000-chunk synthetic bank; stored
it takes 0.25 s. Adding it to an existing database rewrites the table: see the `operating` skill.
*Evidence:* [`docs/exact-tsv.md`](../../docs/exact-tsv.md).

## Importers

`prospecta import hindsight` carries each unit as one chunk row (its vector verbatim when the width
matches, else re-embedded) and keeps everything else as inert records; re-running adds nothing. A bank an
import creates gets the new-bank defaults. *Evidence:* [`docs/import-hindsight.md`](../../docs/import-hindsight.md).
`SourceIndexer` indexes a host's records in place from views and keeps no text beyond the chunks
([`docs/source-contract.md`](../../docs/source-contract.md)).

## Related

[`search-and-ask`](../search-and-ask/SKILL.md) reads what this writes; [`operating`](../operating/SKILL.md)
covers migrations and backups.
