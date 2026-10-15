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

## Phrases and adjacency

- `"double quoted text"` is an exact phrase (words adjacent, in order; `phraseto_tsquery`
  under the same `simple` configuration and word rule, so `v1.2.3` and `src/main.rs`
  work inside it). With quotes present only passages holding every phrase are returned;
  unquoted words still add to the rank. An unbalanced quote or an empty `""` is plain text.
- An unquoted multi-word question still returns every any-word hit; passages that hold
  more adjacent pairs of its words (in the question's order) rank above ones with the
  words scattered, and the score includes one point per adjacent pair (at most 16 pairs).
- The boost is computed only for the top N passages by the any-word score
  (`"params": {"boost_top": N}`, default 300; 0 turns it off). Those are ordered by
  boost, then score; the remaining hits follow in any-word order, none dropped. The
  boost checks the word-rule vector (built once per boosted row) against the pair, so
  on a 60,000-passage bank it costs at most about 1.1x the unboosted query.
- The stored index narrows candidates for phrases too; adjacency is then checked on
  the two parts of the word rule separately, so with or without the index the results
  are identical.
