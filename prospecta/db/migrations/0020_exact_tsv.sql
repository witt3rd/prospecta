-- 0020: memory_items.exact_tsv, the stored exact vector of the Stage 1 exact retriever.
--
-- Until now the exact / phrase retriever computed the 'simple' word vector of
-- original_chunk per row at query time (the expression migration 0018 indexes), so a
-- question made only of very common words re-tokenized every chunk that holds them. The
-- vector is now a STORED generated column; the query matches and ranks the column.
--
-- The generation expression is the one word rule of prospecta/db/keyword_index.py
-- (indexed_expression), filled in below by migrate.run_migrations so that no second copy
-- of the rule exists to drift from it.
--
-- THIS REWRITES memory_items: PostgreSQL computes the new column for every row, writing a
-- new copy of the table and rebuilding every index on it (HNSW included), under an ACCESS
-- EXCLUSIVE lock, in this transaction. Seconds at 73,000 rows, minutes on a big bank with
-- vector indexes. See docs/exact-tsv.md before applying this to a live database.
--
-- The GIN index on the column is built CONCURRENTLY (it cannot run inside this
-- transaction) by run_migrations right after this file commits. Idempotent: a column that
-- exists is left alone (no second rewrite). Rollback: DROP INDEX memory_items_exact_tsv_gin;
-- ALTER TABLE memory_items DROP COLUMN exact_tsv; DELETE FROM prospecta_schema_version
-- WHERE version = 20.
ALTER TABLE memory_items
    ADD COLUMN IF NOT EXISTS exact_tsv tsvector
    GENERATED ALWAYS AS ({{EXACT_TSV_EXPRESSION}}) STORED;

COMMENT ON COLUMN memory_items.exact_tsv IS
    'STORED generated column: the keyword word rule (prospecta/db/keyword_index.py, '
    'indexed_expression) over original_chunk, unstemmed ''simple'' configuration. Matched and '
    'ranked by the Stage 1 exact retriever through memory_items_exact_tsv_gin (migration 0020).';
