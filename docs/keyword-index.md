# Stored keyword index

The `keyword` channel matches the question's words against `memory_items.original_chunk`
(chunk text; for an unchunked document that is the document text). Migration 0018 adds a
GIN expression index, `memory_items_keyword_gin`, over the channel's word rule: the
'simple' text-search configuration (stemming off) plus path pieces split at slashes. The
channel's query uses the same expression (defined once in `prospecta/db/keyword_index.py`),
so Postgres uses the index and keeps it current on every write.

- **New and migrated databases:** `prospecta migrate` builds it after migration 0018.
- **Existing bank, one command:** `prospecta keyword-index` (idempotent; also repairs an
  interrupted build). `prospecta keyword-index --rebuild` rebuilds a valid one; the old
  index serves until the swap.
- **No lock on writes:** built with `CREATE INDEX CONCURRENTLY`. Reads and writes continue;
  the build takes longer and waits for transactions open when it starts.
- **Fallback:** without a valid index, and for `stemming: true` queries, the channel runs the
  same expression as a scan of the bank (correct, slow). `KeywordChunks.used_index` reports
  which path ran.
- **Cost:** disk and write cost grow with text size. The index is shared by all banks in the
  database; the bank filter is applied after the index lookup.
