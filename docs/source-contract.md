# Source contract, shared-database mode and service

Prospecta stays a standalone library. These three features let it run inside a
database another application already owns, and index that application's
records where they live. Prospecta holds no knowledge of the host; it only
reads the contract below.

## Shared-database mode

Set `PROSPECTA_SCHEMA` (or `Memory(schema=...)`, `run_migrations(url, schema=...)`).
Every table, index, function and the `prospecta_schema_version` table live in
that one schema; every connection runs with `search_path = "<schema>", public`.
Nothing is created elsewhere. The `vector` extension must already be installed by
the database owner; in this mode Prospecta never creates an extension (migration
0012's optional `pg_search` index, which no code reads any more, is skipped unless that
extension is installed).

- `run_migrations` creates the schema when missing (an existing one is left alone).
- A configuration check runs before any start (`Memory(...)`, migrations,
  `prospecta health`, the service): the schema must exist, the role needs
  USAGE and CREATE on it, a probe `CREATE TABLE`/`INSERT` (rolled back) must
  succeed, and `vector` must be installed. Otherwise it raises `SchemaError`
  and nothing starts.
- Names: lower-case `[a-z_][a-z0-9_]*`, not `public`, `pg_*` or `information_schema`.
- Unset = the standalone behaviour, unchanged.

Minimum grants for the role: `CONNECT` on the database, `USAGE, CREATE` on the
Prospecta schema, `USAGE` on `public` (for the `vector` type), and `SELECT`
on the source views (below).

## Source contract

Config: `PROSPECTA_SOURCE_SCHEMA` (the schema the host provides) and
`PROSPECTA_SOURCE_VIEWS` (comma list, default `records`). Each view must have
exactly these columns (any type that casts to the one shown; ids may be uuid or int):

| column | type | meaning |
|---|---|---|
| `kind` | text | record family, e.g. note, message |
| `record_id` | text | id, unique within `kind` |
| `revision` | text | opaque string; must change whenever the record's text, title or audience changes (see below) |
| `title` | text | optional |
| `text` | text | the content to index |
| `person` | text | optional filter |
| `project` | text | optional filter |
| `created_date` | date | optional |
| `audience_key` | text | optional access scope; lookups can filter on it |
| `deleted_at` | timestamptz | non-NULL = the record is deleted |

`revision` is compared only for equality; Prospecta never orders or parses it.
Host obligation: change it whenever the record's `text`, `title` or
`audience_key` changes. A record whose revision is unchanged is skipped
(`unchanged`), so a change that does not move the revision is never seen. When
the revision changes, `audience_key` (and `project`, `person`) are re-read from
the view; a changed audience key retires the record's old items and re-derives
them, and the new key takes effect for the next lookup after that reindex.
Until then lookups use the old key (`/coverage` lists the record as `stale`).

The service refuses to start if a view is missing, lacks a column or is not readable.

What Prospecta keeps: its own derived items only (chunks and their embeddings), each stamped with
`kind`, `record_id`, `revision` (plus `person`, `project`, `audience_key`,
`created_date` for filters), and one `source_refs` row per record
(kind, record id, revision, filter columns; **no text**). The `documents` row
that anchors the items has `original_text = ''`. Chunks are derived items and do
contain pieces of the text, as they do for any Prospecta index; they are
rebuilt from the view and removed with the record.

### Allowed-audience filter

`allowed_audience_keys` (list of strings) and `allowed_projects` (list of
strings) are optional. When given, a record whose audience key (project) is not
in the list is never a candidate: it cannot appear in a result or a rerank
input. The condition is part of each retriever's query in Stage 1, applied
before its slots are cut. An empty list
allows nothing. A record with an empty or missing audience key
(project) is visible only if the list contains `""`. When a list is absent
there is no restriction (unchanged behavior). In the library the same filter is
`metadata_filter={"_allow": {"audience_key": [...], "project": [...]}}`.

## Indexing interface (`prospecta.source.SourceIndexer`)

All idempotent; they take the record's reference, never its text.

- `index_record(kind, record_id)`: new -> `indexed`; same revision -> `unchanged`;
  changed revision -> the old derived items are retired and rebuilt (`indexed`);
  deleted or gone from the view -> retired (`removed`, or `absent` if never indexed).
- `reindex_kind(kind)`: runs the above for every record of the kind in the views and for every record Prospecta holds.
- `remove_record(kind, record_id)`: retire now.
- `reconcile(references)`: for each reference, re-read the views and apply the
  rules of `index_record`; see "Reconcile" below.
- `coverage(kind=None, project=None)`: one entry per (kind, project) with `live`,
  `indexed`, and lists of exact references: `missing` (live in the view, nothing
  derived), `stale` (derived revision differs from the view's), `orphaned`
  (derived, but deleted or gone in the view).

## Service

`python -m prospecta.service` (or the `Dockerfile`). Stdlib HTTP, JSON.

| call | body / query | answer |
|---|---|---|
| `GET /healthz` | | `{"status":"ok"}` (no auth) |
| `POST /lookup` | `{"questions":[...], "limit":10, "kind"?, "project"?, "person"?, "audience_key"?, "allowed_audience_keys"?, "allowed_projects"?, "date_from"?, "date_to"?, "excluded_sources"?, "reranker_model"?, "decision_model"?}` | `{"results":[{"question", "results":[{"kind","record_id","revision","score","created_on","match_kind","exact_snippet","relevance_score"}]}]}` |
| `POST /lookup` (a task brief) | `{"task":{"title","ask","done"?,"project"?}, "max_tokens"?, "date_from"?, "date_to"?, "excluded_sources"?, "reranker_model"?, "decision_model"?}` (no `limit`) | `{"task":{"queries":[...], "context":"## Recalled Project Context ...", "searches":[{"query","terms","date_from","date_to"}], "results":[{"source","created_on","kind","record_id","revision","project","match_kind","exact_snippet","relevance_score","found_by"}], "fallbacks":[...], "error":null}}` |
| `POST /retain` | `{"content","source","bank"?,"metadata"?:{"created_on","project","person","kind"},"tags"?,"create"?}` | `{"document_id","status":"created"|"replaced"|"unchanged"}` |
| `POST /lung-memory/1` | lung's wire: `{"op":"recall"|"search"|"retain", ...}` (see "The `lung-memory/1` adapter") | `{"status":"ok","records":[...]}`, `{"status":"retained","id"}` or `{"status":"unavailable","reason"}`, each with `"retention"` |
| `POST /index` | `{"kind","record_id"}` or `{"kind","reindex":true,"concurrency"?}` | status per above |
| `POST /remove` | `{"kind","record_id"}` | status |
| `POST /reconcile` | `{"references":[{"kind","record_id"} or {"kind","project","title"}], "concurrency"?}` (any number, at least 1) | `{"indexed","unchanged","removed","absent","unreadable":[{"reference","error"}]}` |
| `GET /coverage?kind=&project=` | | `{"coverage":[...]}` |

Every hit says where and when it is from and why it matched (the cited-evidence contract of
`docs/postgres-hybrid.md`): `created_on` (the record's date, `YYYY-MM-DD`), `match_kind`
(`exact+semantic`, `exact`, `lexical`, `semantic`, `date` (a wordless question's date-window listing), or null), `exact_snippet` (the matched words in
context, `<<marked>>`, or null) and `relevance_score` (the cross-encoder's 0..1 score when the bank
runs stage 2 with it, else the fused `score`). `date_from` / `date_to` (`YYYY-MM-DD`, the record's
date, inclusive) and `excluded_sources` (document sources to leave out, the anti-echo list) apply
inside Stage 1 and need the `hybrid_chunk` channel in the bank's config (else 400; every bank created
fresh has it);
`reranker_model` / `decision_model` override the stage 2 / decision model for the call.

**The questions form** runs each question through Stage 1 + Stage 2 (no Stage 0 gate) and returns the
best hit per record.

**The task form** is Mode 1 (`docs/postgres-hybrid.md`, section 1): `Memory.recall_for_task`. The
query writer expands the brief into searches (up to `expansion.max_searches`, 20), every search runs
through Stage 1 + Stage 2 in parallel against its own query inside the deadline (7 s), and one cited
context block comes back: the task's `project` first, other projects' notes labeled with theirs.
`searches` says what ran (`query`, verbatim `terms`, the period), `results` the items of the block
with `found_by` (the searches that found each: their numbers in the block), `fallbacks` every
degraded step and `error` a failure that stopped the recall (the call itself still answers 200). The
block is sized by relevance, not by count, so the task form takes **no `limit`** (400);
`max_tokens` (default 0: off) is only a safety valve. The task is never gated and is recorded
(`recall_events`, `llm_calls`).

`/lookup` accepts `"explain": true` (default false; must be a boolean) on the questions form. The
default response is otherwise unchanged. With it, each question's object gains `channels_run` (channels
that ran and did not fail) and `channels_failed` (`[{"channel","reason"}]`, a reason in
words, starting `failed:` for an error or `skipped:` for a channel disabled in the bank's
config or that the library no longer has), and each hit gains `channels`:
`[{"channel","rank","score","weight"}]` for every channel that returned it (rank within the
channel, the channel's raw score, the fusion weight applied). `hybrid_chunk` is the only channel.

### Banks and standalone notes

**A bank per call.** Every JSON body may carry `"bank"` (a `GET` takes `?bank=`). Absent, the call
is about the process's default bank (`PROSPECTA_BANK`), so a host that never sends it sees no
change. One process, one connection pool, any number of banks: a named bank is looked up on every
call. An unknown bank is `404`; `/retain` with `"create": true` makes it first (`Memory.create_bank`,
the embedder's dimension, or `PROSPECTA_EMBEDDING_DIM`, and the new-bank defaults). A bank whose
`embedding_dim` differs from the service's embedder is `409`; an invalid name is `400`. The service
has one bearer token for every bank; it does not authorize per bank.

**`POST /retain`** writes a note that no host view holds (the chunk-only write path of
`docs/schema.md`; no model is called). `source` is the key and is required:

| call | answer |
|---|---|
| a source never seen | `created` |
| the same `source`, new `content` | `replaced`: the old text under that key is deleted, the new one stored |
| the same `source`, the same `content` (and the same `tags` and `metadata`) | `unchanged`: nothing is written; with other `tags` or `metadata` the note is rewritten and the answer is `replaced` |
| the same `content` under another `source` | `409`, nothing changes |

A retry of a call whose answer was lost is therefore safe. `metadata` takes only `created_on`
(`YYYY-MM-DD`, the note's date, else today), `project`, `person` and `kind` (all filterable in
`/lookup`); `tags` is a list of strings. A note is never refused for its size: `/retain` has no
body limit (the other routes keep 1 MiB), and it is stored whole, as chunks.

**`/lookup` returns standalone notes.** In the questions form a note with no `record_id` (one from
`/retain`) comes back as `{"kind":"chunk","record_id":null,"revision":null,"source","document_id",
"content", ...}` with the same citation fields as a record. Nobody holds its text elsewhere, so
`content` is its best-matching chunk. The task form already returned them (`kind` is the note's
own kind, `record_id` null), and accepts `excluded_sources` as the questions form does.

### The `lung-memory/1` adapter

`POST /lung-memory/1` speaks the wire that lung (the Lean agent) specified and proved on its side
(`lung/spec/LungSpec/MemoryWire.lean`, PART B), so lung can use the service as its memory provider.
It sends the same `Authorization: Bearer <PROSPECTA_SERVICE_TOKEN>` as every other call (401
without it), JSON in, JSON out, stdlib only. Banks are named in the request; a retain makes its bank
on first use (a recall or search of a bank that does not exist is an error, below). Like `/retain` it has no body limit.

| request | what runs | reply |
|---|---|---|
| `{"op":"recall","task":"...","banks":["b1","b2"]}` | Mode 1 as the task form of `/lookup` (the query writer, Stage 1 + Stage 2, one merged set) over each bank, results concatenated in the request's bank order | `{"status":"ok","records":[...],"retention":"..."}` |
| `{"op":"search","query":"...","banks":["b1"]}` | the questions form of `/lookup` (best chunk per note, best first, 10 per bank) over each bank | the same |
| `{"op":"retain","bank":"b1","text":"...","source":"..."}` | `/retain` with `create`: `source` is the idempotency key (same key and text: unchanged; same key, new text: replaced) | `{"status":"retained","id":"<document id>","retention":"..."}` |
| `{"op":"delete_bank","bank":"lung-scratch-x"}` | this service's own op (not in the Lean spec): removes a scratch bank. Only an exact name starting `lung-scratch-` is allowed. Its per-bank partial indexes (hnsw and the two lexical, per kind) are dropped, then in one transaction its documents and the bank row are deleted (the schema's `ON DELETE CASCADE` takes the items, links, entities, events and source refs) | `{"status":"deleted","bank":"...","documents":N,"retention":"..."}`, `N` counted before the delete |

**Timing.** A `recall` takes about 5 s (the query-writer model call); a `search` has no model call and
is faster. Callers should use a 30 s timeout.

**Unknown bank.** Every bank a `recall` or `search` names, and the bank of a `delete_bank`, must
exist; otherwise the reply is `{"status":"unavailable","reason":"scope","detail":"unknown bank: <name>",
"retention":"..."}` (HTTP 200). All banks of a request are checked before anything runs, so one
unknown bank fails the whole request (the detail names the first, in request order). A mistyped bank
in a profile therefore fails loudly instead of reading as "nothing found". `retain` is unchanged: it
is the one op that creates a bank, on first use, so it has no unknown-bank case. `detail` is an extra
field for humans: lung's decoder (`decodeOutcome`, `decodeRetain`) reads `status`, `reason`, `id` and
`records` and `retention` by name and ignores any other field. `delete_bank` of a name that does not
start with `lung-scratch-` is refused with `{"status":"unavailable","reason":"scope","detail":
"delete_bank is only allowed for lung-scratch-* banks","retention":"..."}`, and the bank is untouched.

A record is `{"bank","id","text","source","date"}`: `bank` is the name the request used (never a
bank the request did not name), `id` the document id, `text` the best-matching chunk, verbatim,
`source` the note's stored source, `date` the note's date as `YYYY-MM-DD`. Ids are distinct in a
reply (a note of many chunks is one record). Nothing found is `{"status":"ok","records":[],...}`
(a bank that exists and has no matching note). Lung's decoder takes all of a
reply or none of it, so a hit that has no source or no date is **left out** (never defaulted).
Every note stored through `/retain` or `/lung-memory/1` has a source and a date, so this can only
happen to a document written another way with a NULL or empty `source` (a direct database write);
a date cannot be missing (the note's date, else its creation day). The service logs how many it left
out.

A failure is `{"status":"unavailable","reason":"<word>","retention":"..."}` (plus `detail` for
`scope`) with HTTP **200**. The
reasons this service gives: `provider` (a bank of another embedding dimension, the same text already
stored under another source, no embedder, a recall that failed or any unexpected error), `timeout`
(the database pool timed out, or the recall deadline was hit: a partial recall is never returned),
`malformed` (a retain with a bank name that is not valid, or with blank text), `scope` (an unknown bank,
or a `delete_bank` of a non-scratch name). `transport`, `cost` and `retention` are the caller's to say;
the service never gives them. If one of several banks
cannot be read, the whole reply is `unavailable`: lung would otherwise believe it consulted them all.
**400** is only for a body that parses as none of the three ops (not JSON, not an object, an unknown
`op`, a missing or wrongly typed field, an empty `bank` or `source` on a retain, an empty `bank` on a delete_bank); its body is still
`{"status":"unavailable","reason":"malformed","retention":"..."}`.

`retention` is `"attested"` only when the operator sets `PROSPECTA_ZDR_ATTESTED=1` (exactly `1`: the
operator asserts that the deployment's OpenRouter key or workspace is zero-data-retention);
otherwise `"unattested"`. Prospecta does not enforce zero data retention; the flag is a claim, said on every reply
(including a 400).

```
POST /lung-memory/1  {"op":"retain","bank":"lung-class-work","text":"Deploys go out on Tuesdays.","source":"lung/session-7"}
-> {"status":"retained","id":"6e664361-a6f0-40ad-b032-681928cfc6f4","retention":"unattested"}

POST /lung-memory/1  {"op":"search","query":"when do deploys go out","banks":["lung-class-work"]}
-> {"status":"ok","records":[{"bank":"lung-class-work","id":"6e664361-a6f0-40ad-b032-681928cfc6f4",
    "text":"Deploys go out on Tuesdays.","source":"lung/session-7","date":"2026-11-23"}],"retention":"unattested"}

POST /lung-memory/1  {"op":"recall","task":"plan this week's release","banks":["lung-class-work","lung-domain-shop"]}
-> {"status":"ok","records":[],"retention":"attested"}                       (PROSPECTA_ZDR_ATTESTED=1)

POST /lung-memory/1  {"op":"delete_bank","bank":"lung-scratch-run1"}
-> {"status":"deleted","bank":"lung-scratch-run1","documents":3,"retention":"unattested"}

POST /lung-memory/1  {"op":"search","query":"x","banks":["lung-class-wrk"]}          (a mistyped bank)
-> {"status":"unavailable","reason":"scope","detail":"unknown bank: lung-class-wrk","retention":"unattested"}

POST /lung-memory/1  {"op":"search","query":"x","banks":["narrow"]}          (a bank of another embedding dimension)
-> {"status":"unavailable","reason":"provider","retention":"unattested"}
```

### Reconcile

`POST /reconcile` takes references and makes Prospecta match the views for each:
present and changed -> indexed; same revision -> left alone (`unchanged`);
row absent or `deleted_at` set -> its derived items are retired (`removed`;
`absent` when nothing was ever indexed). Counts are per reference. A reference
Prospecta cannot read (malformed, or a view error) is listed under `unreadable`
with the reason and does not stop the others. Repeating a call changes nothing
once the first has been applied.

References are processed on a bounded pool of worker threads that borrow from the
service's shared connection pool (chunking, embedding and writes, and the
anticipated-question model call when enabled, all run in the worker). The size is `PROSPECTA_INDEX_CONCURRENCY`
(default 8, and never more than the connection pool's size); a request may lower or raise it with `"concurrency"` (1..32, else
400). Extra references wait in a queue; none is refused. Results are the same as
a sequential run (same counts, same `unreadable` list in request order, same
derived items). One record is only ever indexed by one worker at a time (a
per-record advisory lock), so overlapping requests for the same records never
duplicate derived items; the later one finds the record `unchanged`. `/index`
with `reindex` uses the same pool.

A reference is `{kind, record_id}` (one record) or `{kind, project, title}`
(document scope): every live row of that kind, project and title is one
document. Its text is the rows' texts joined in (`created_date`, `record_id`)
order, and its revision is a hash of the rows' (`record_id`, `revision`) pairs,
so adding, removing or changing any piece re-derives the whole document and
the new set of pieces replaces the stored set. Deleted rows do not count as
pieces; when none is left the document is retired. It is kept under the record id
`doc:` + JSON `[project, title]`, which is what `/lookup` and `/coverage` show.
Rows of one document must share an `audience_key`; if they differ the reference
is `unreadable` and nothing changes. Document scope reads `person` only when all
rows agree, and `created_date` is the earliest. `/coverage` does not know
document scope and lists its anchor as `orphaned`.

Environment: `DATABASE_URL`, `PROSPECTA_SCHEMA`, `PROSPECTA_SOURCE_SCHEMA`,
`PROSPECTA_SOURCE_VIEWS`, `PROSPECTA_BANK` (default `prospecta`),
`PROSPECTA_EMBEDDING_DIM`, `PROSPECTA_SERVICE_HOST` (default 127.0.0.1),
`PROSPECTA_SERVICE_PORT` (8080), `PROSPECTA_SERVICE_TOKEN` (bearer token;
required to listen on a non-loopback address), and the usual embedder/LLM
variables. Only chunk items are written (no model call at index time). The host decides when to call `/index` (for example after a
change) and can use `/coverage` to find what to re-send.

```
docker build -t prospecta-service .
docker run -p 8080:8080 -e DATABASE_URL=... -e PROSPECTA_SCHEMA=prospecta \
  -e PROSPECTA_SOURCE_SCHEMA=host_views -e PROSPECTA_SERVICE_TOKEN=... prospecta-service
```

## Database connections

The service holds one bounded connection pool for the whole process. Every
request (and every indexing worker) borrows a connection for the length of its
work and returns it, on success or error; no thread keeps one. A request that
finds none free waits, up to `PROSPECTA_REQUEST_TIMEOUT` seconds (default 30),
then gets `503`.

| Variable | Default | Meaning |
|---|---|---|
| `PROSPECTA_DB_POOL_MAX` | 10 | most connections the process opens |
| `PROSPECTA_DB_POOL_IDLE_SECONDS` | 60 | an idle connection is closed after this |
| `PROSPECTA_REQUEST_TIMEOUT` | 30 | seconds a request waits for a connection |

When 70 percent of the pool is in use the service logs one warning line naming
the pool size and the count in use, at most once a minute.
