-- 0002: Add body_tsv generated column for body-content lexical search.
-- Implements P14 (hybrid as true safety net): when LLM-generated index_text
-- drifts from query language, body content (original_chunk) still rescues
-- the document via BM25 fusion through the body channel.
--
-- The existing content_tsv channel indexes memory_items.content (the
-- LLM-anticipated index_text / questions). This second channel indexes
-- the source body (original_chunk). Three-channel RRF fusion
-- (semantic + lexical_content + lexical_body) ships in queries.hybrid_search.
--
-- GENERATED column on existing table populates retroactively for all
-- existing rows. Idempotent via IF NOT EXISTS on the index; column-add
-- guarded by a DO block.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'memory_items' AND column_name = 'body_tsv'
    ) THEN
        ALTER TABLE memory_items
        ADD COLUMN body_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('english', COALESCE(original_chunk, ''))) STORED;
    END IF;
END$$;

CREATE INDEX IF NOT EXISTS memory_items_body_tsv_gin
    ON memory_items USING GIN(body_tsv);

COMMENT ON COLUMN memory_items.body_tsv IS
    'STORED generated column: to_tsvector(''english'', COALESCE(original_chunk, '''')). '
    'Indexed via GIN for the body-channel of hybrid retrieval — when LLM-generated '
    'index_text drifts hard from query language, BM25 on body content still surfaces '
    'the doc via RRF fusion (P14 honest safety net, body-fallback axis).';
