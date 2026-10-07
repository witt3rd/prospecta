-- One judgment per UNORDERED item pair: the Linker's Jev answer for (src, dst)
-- answers both directions (semantic, causes, caused_by), so A->B and B->A share
-- it and a re-run asks nothing. Additive: a new table, no populated column touched.
CREATE TABLE IF NOT EXISTS memory_link_pairs (
    pair_hash  TEXT PRIMARY KEY,                 -- sha256 of the sorted item id pair
    bank_id    TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    src        UUID NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    dst        UUID NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    hits       JSONB NOT NULL DEFAULT '[]'::jsonb, -- [[link_type, subtype, confidence, forward]] relative to src -> dst
    judged_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS memory_link_pairs_dst ON memory_link_pairs (dst);
CREATE INDEX IF NOT EXISTS memory_link_pairs_src ON memory_link_pairs (src);
COMMENT ON TABLE memory_link_pairs IS 'Cached Linker relation judgments, one row per unordered item pair; cascades away with either item.';
