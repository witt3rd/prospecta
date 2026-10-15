-- 0019: per-bank linking switches. Additive: ADD COLUMN with a constant default.
-- '{}' means the defaults: link_kinds [next, causal], extract_entities off (see docs/graph-links.md).
-- No populated column is dropped, rewritten or retyped; no row is deleted.
ALTER TABLE banks ADD COLUMN IF NOT EXISTS link_config JSONB NOT NULL DEFAULT '{}'::jsonb;

COMMENT ON COLUMN banks.link_config IS
    'Linker switches: {"link_kinds": list of next, causal, semantic, temporal, entity, related_to (default [next, causal]: the relations a search cannot answer), "extract_entities": bool (default false)}.';
