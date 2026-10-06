-- 0012: OPTIONAL pg_search BM25 index on memory_items(original_chunk), chunk items only.
-- Guarded: if the pg_search extension is not installable on this server the
-- whole migration is a no-op (the in-library bm25s channel keeps serving).
-- Additive only; nothing existing is altered. Plain CREATE INDEX (a DO block
-- runs in a transaction, so CONCURRENTLY is not possible); on a large bank
-- create the index by hand beforehand under the same name to skip the build.
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_available_extensions WHERE name = 'pg_search') THEN
        RAISE NOTICE 'pg_search not available: skipping BM25 index (bm25s channel stays in use)';
        RETURN;
    END IF;
    CREATE EXTENSION IF NOT EXISTS pg_search;
    EXECUTE 'CREATE INDEX IF NOT EXISTS memory_items_original_chunk_bm25
             ON memory_items USING bm25 (id, original_chunk)
             WITH (key_field = ''id'')
             WHERE kind = ''chunk''';
EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'pg_search BM25 index not created (%): skipping', SQLERRM;
END$$;
