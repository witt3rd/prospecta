# Source contract, shared-database mode and service

Prospecta stays a standalone library. These three features let it run inside a
database another application already owns, and index that application's
records where they live. Prospecta holds no knowledge of the host; it only
reads the contract below.

## Shared-database mode

Set `PROSPECTA_SCHEMA` (or `Memory(schema=...)`, `run_migrations(url, schema=...)`).
Every table, index, function and the `prospecta_schema_version` table live in
that one schema; every connection runs with `search_path = "<schema>", public`.
Nothing is created elsewhere. The `vector` extension (and `pg_search`, if you
want the BM25 index) must already be installed by the database owner; in this
mode Prospecta never creates an extension and migration 0012 skips `pg_search`
unless it is installed.

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

What Prospecta keeps: its own derived items only (chunks, anticipated
questions, embeddings, entities, links), each stamped with
`kind`, `record_id`, `revision` (plus `person`, `project`, `audience_key`,
`created_date` for filters), and one `source_refs` row per record
(kind, record id, revision, filter columns; **no text**). The `documents` row
that anchors the items has `original_text = ''`. Chunks are derived items and do
contain pieces of the text, as they do for any Prospecta index; they are
rebuilt from the view and removed with the record. The entity linker reads
its text back from those chunks.

Limits: the map-reduce recall (`recall_mapreduce`) scans `documents.original_text`,
so in-place records are not covered by it.

### Allowed-audience filter

`allowed_audience_keys` (list of strings) and `allowed_projects` (list of
strings) are optional. When given, a record whose audience key (project) is not
in the list is never a candidate in any channel (keyword, meaning, questions,
links, graph): it cannot appear in a result, a rerank input or a follow-up
question. The condition is part of each channel's query, applied before
ranking, and the graph walk never reaches or crosses a disallowed item. An
empty list allows nothing. A record with an empty or missing audience key
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
| `POST /lookup` | `{"questions":[...], "limit":10, "kind"?, "project"?, "person"?, "audience_key"?, "allowed_audience_keys"?, "allowed_projects"?}` | `{"results":[{"question", "results":[{"kind","record_id","revision","score"}]}]}` |

`/lookup` accepts `"explain": true` (default false; must be a boolean). The default
response is unchanged. With it, each question's object gains `channels_run` (channels
that ran and did not fail) and `channels_failed` (`[{"channel","reason"}]`, a reason in
words, starting `failed:` for an error or `skipped:` for a channel disabled in the bank's
config), and each hit gains `channels`: `[{"channel","rank","score","weight"}]` for every
channel that returned it (rank within the channel, the channel's raw score, the fusion
weight applied). Banks without a channel config use the fixed legacy fusion: `channels_run`
names its three channels and a hit's `rank` and `weight` are null.
| `POST /index` | `{"kind","record_id"}` or `{"kind","reindex":true,"concurrency"?}` | status per above |
| `POST /remove` | `{"kind","record_id"}` | status |
| `POST /reconcile` | `{"references":[{"kind","record_id"} or {"kind","project","title"}], "concurrency"?}` (any number, at least 1) | `{"indexed","unchanged","removed","absent","unreadable":[{"reference","error"}]}` |
| `GET /coverage?kind=&project=` | | `{"coverage":[...]}` |

### Reconcile

`POST /reconcile` takes references and makes Prospecta match the views for each:
present and changed -> indexed; same revision -> left alone (`unchanged`);
row absent or `deleted_at` set -> its derived items are retired (`removed`;
`absent` when nothing was ever indexed). Counts are per reference. A reference
Prospecta cannot read (malformed, or a view error) is listed under `unreadable`
with the reason and does not stop the others. Repeating a call changes nothing
once the first has been applied.

References are processed on a bounded pool of worker threads that borrow from the
service's shared connection pool (chunking, the anticipated-question model call, embedding
and writes all run in the worker). The size is `PROSPECTA_INDEX_CONCURRENCY`
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
variables. Without an LLM only chunk items are written; with one, anticipated
questions too. The host decides when to call `/index` (for example after a
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
