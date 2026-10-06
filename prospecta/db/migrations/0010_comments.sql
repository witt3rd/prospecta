-- 0010: correct column comments. content_tsv / body_tsv are ranked with
-- ts_rank_cd (cover density), not BM25. Comment-only; no behaviour change.
-- 0001/0002 are applied history and are not edited.

COMMENT ON COLUMN memory_items.content_tsv IS
    'STORED generated column: to_tsvector(''english'', content). Indexed via GIN for the lexical channel of hybrid retrieval, ranked with ts_rank_cd (full-text cover-density rank, not BM25). Queries use websearch_to_tsquery, so every query term must match.';

COMMENT ON COLUMN memory_items.body_tsv IS
    'STORED generated column: to_tsvector(''english'', COALESCE(original_chunk, '''')). '
    'Indexed via GIN for the body channel of hybrid retrieval, ranked with ts_rank_cd '
    '(full-text cover-density rank, not BM25). Queries use websearch_to_tsquery, so every '
    'query term must match; the channel fires on short queries only (P14 safety net, body-fallback axis).';
