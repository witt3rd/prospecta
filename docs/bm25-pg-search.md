# Optional pg_search BM25 seam

The shipped BM25 channel (`bm25`) is the in-library bm25s-style one (`InProcessBm25`).
Migration `0012_pg_search_bm25.sql` is an optional, guarded addition: only if the
`pg_search` extension is installable (`pg_available_extensions`) it creates the BM25
index `memory_items_original_chunk_bm25` on `memory_items(id, original_chunk)` for
`kind='chunk'`; otherwise it is a no-op. Nothing installs the extension or changes a
Postgres image. The index is a plain `CREATE INDEX` (inside a DO block); on a big bank
create it by hand under that name first.

`Bm25Chunks` uses `AutoBm25` by default behind the `Bm25Backend` interface: `PgSearchBm25`
(`original_chunk ||| query`, ordered by `pdb.score(id)`) when that index exists, else
`InProcessBm25`. Assign `Bm25Chunks.backend` to force one.
