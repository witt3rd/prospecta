-- 0008: per-bank recall channel registry (hybrid retrieval design 8.2/8.3).
-- ADD COLUMN with a constant default is metadata-only. '[]' means "no channel
-- config": the bank keeps the legacy three-channel hybrid recall unchanged.
ALTER TABLE banks ADD COLUMN IF NOT EXISTS channel_config JSONB NOT NULL DEFAULT '[]'::jsonb;

COMMENT ON COLUMN banks.channel_config IS
    'Ordered list of recall channels: [{"name","enabled","weight","params"}]. Empty = legacy hybrid recall.';
