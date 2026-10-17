# Optional pg_search BM25 seam

The shipped BM25 channel (`bm25`) has an in-library bm25s-style backend (`InProcessBm25`), now an explicit opt-in.
Migration `0012_pg_search_bm25.sql` is an optional, guarded addition: only if the
`pg_search` extension is installable (`pg_available_extensions`) it creates the BM25
index `memory_items_original_chunk_bm25` on `memory_items(id, original_chunk)` for
`kind='chunk'`; otherwise it is a no-op. Nothing installs the extension or changes a
Postgres image. The index is a plain `CREATE INDEX` (inside a DO block); on a big bank
create it by hand under that name first.

`Bm25Chunks` uses `AutoBm25` by default behind the `Bm25Backend` interface: `PgSearchBm25`
(`original_chunk ||| query`, ordered by `pdb.score(id)`) when that index exists; otherwise the
channel fails (recorded in the recall trace) and never builds the in-process index on its own.
Assign `Bm25Chunks.backend` to choose: `AutoBm25(fallback=InProcessBm25())` falls back to the
in-process index. The native `hybrid_chunk` channel (see the README) replaces both.
