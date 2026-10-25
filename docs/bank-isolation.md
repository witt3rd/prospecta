# Bank isolation: one bank's size never slows another

Stage 1 (`docs/postgres-hybrid.md`) runs three retrievers over `memory_items`, a table every bank shares.
Each is served by an index:

| retriever | index | per bank? |
|---|---|---|
| semantic | `memory_items_embedding_<bank>_chunk_idx` (HNSW, partial on bank and kind) | yes, since 0004 |
| stemmed words | `memory_items_body_tsv_<bank>_chunk_gin` (GIN, partial on bank and kind) | yes, since the bank-isolation change |
| exact words | `memory_items_exact_tsv_<bank>_chunk_gin` (GIN, partial on bank and kind) | yes, same |

Before the change the two word retrievers read the one shared GIN (`memory_items_body_tsv_gin`,
`memory_items_exact_tsv_gin`): a query for a small bank walked the posting lists of every bank, built a bitmap of
all their matching rows, then threw almost all of it away on the `bank_id` / `kind` test. Measured on a copy of the
live database (a 495,368-chunk bank beside two banks of 11,746 and 78,831 chunks), the small banks' Stage 1 took
1.8 to 2.1 times as long as without the large bank; with the per-bank indexes it is the same as without it
(`data/prospecta-bank-isolation/report.md` of the firstmate repository holds the numbers).

`create_bank` builds the bank's two GIN indexes after its HNSW index (`CREATE INDEX CONCURRENTLY`, so writes are
never blocked); `run_migrations` builds any that are missing and rebuilds an invalid leftover, so a bank created
by raw SQL, or before this change, is covered by the next `prospecta migrate`. The shared GIN indexes stay: a bank
without its own indexes still searches, slower in proportion to the other banks' size. Dropping a bank leaves its
indexes to be dropped by hand, like its HNSW index.
