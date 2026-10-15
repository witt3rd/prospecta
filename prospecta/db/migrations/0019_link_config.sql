-- 0019: per-bank linking switches. Additive: ADD COLUMN with a constant default.
-- '{}' means the defaults: links off, entities on (see docs/graph-links.md).
-- No populated column is dropped, rewritten or retyped; no row is deleted.
ALTER TABLE banks ADD COLUMN IF NOT EXISTS link_config JSONB NOT NULL DEFAULT '{}'::jsonb;

COMMENT ON COLUMN banks.link_config IS
    'Linker switches: {"links": bool (default false: write SEMANTIC/TEMPORAL/NEXT/ENTITY/RELATED_TO link rows), "entities": bool (default true: extract entities and aliases)}.';
