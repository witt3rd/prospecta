-- 0004: memory_items kind + child-chunk position (hybrid retrieval design 8.3).
-- Additive only: every ADD COLUMN is metadata-only (constant default or NULL).
-- The parent of a chunk is the existing document_id; no new foreign key.
-- The index (bank_id, kind), the batched backfill and the per-kind HNSW indexes
-- need CREATE INDEX CONCURRENTLY / autocommit batches, so run_migrations does
-- them after this transaction commits (prospecta.db.migrate.finish_0004).

ALTER TABLE memory_items ADD COLUMN kind       TEXT NOT NULL DEFAULT 'question';
ALTER TABLE memory_items ADD COLUMN ordinal    INTEGER;
ALTER TABLE memory_items ADD COLUMN char_start INTEGER;
ALTER TABLE memory_items ADD COLUMN char_end   INTEGER;

COMMENT ON COLUMN memory_items.kind IS
    '''question'' (anticipated-question item, the default) or ''chunk'' (child chunk of at most 1,000 chars of the parent document).';
COMMENT ON COLUMN memory_items.ordinal IS
    'Position of a chunk item within its parent document (0-based); NULL for question items.';
COMMENT ON COLUMN memory_items.char_start IS
    'Start offset of a chunk item in documents.original_text; NULL for question items.';
COMMENT ON COLUMN memory_items.char_end IS
    'End offset (exclusive) of a chunk item in documents.original_text; NULL for question items.';
