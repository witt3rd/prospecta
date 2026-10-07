-- 0015: recall depth ('standard' | 'deep') on the recall trace.
-- Additive only: one nullable TEXT column (metadata-only ADD COLUMN).
ALTER TABLE recall_events ADD COLUMN IF NOT EXISTS depth TEXT;
COMMENT ON COLUMN recall_events.depth IS
    'Recall depth: standard (rerank pool cut 0.15) or deep (0.05). NULL on legacy rows.';
