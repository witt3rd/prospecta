-- 0005: documents filter columns (hybrid retrieval design 8.3 / 8.5).
-- Additive only: every ADD COLUMN is nullable with no default (metadata-only).
-- Filled at retain from the parsed frontmatter; existing rows are backfilled
-- from document_metadata and the (bank_id, created_on) / (bank_id, person)
-- indexes are built CONCURRENTLY after this transaction commits
-- (prospecta.db.migrate.finish_0005).

ALTER TABLE documents ADD COLUMN created_on  DATE;
ALTER TABLE documents ADD COLUMN person      TEXT;
ALTER TABLE documents ADD COLUMN source_kind TEXT;

COMMENT ON COLUMN documents.created_on IS
    'Note date (frontmatter created / created_on / date), for the MetadataScope filter channel; NULL when absent.';
COMMENT ON COLUMN documents.person IS
    'Person the note is about (frontmatter person); the bank''s vocabulary of people is its distinct values.';
COMMENT ON COLUMN documents.source_kind IS
    'Kind of note (frontmatter source_kind / type); NULL when absent.';
