-- 0006: entities, typed links and the linker's state (hybrid retrieval design 8.7).
-- Additive only: three new tables (+ one state table and one SQL helper function);
-- no existing table or column is touched. The tables are empty on creation, so
-- plain CREATE INDEX (no CONCURRENTLY) is safe. Links and entity rows cascade on
-- memory_items / banks delete.

CREATE TABLE IF NOT EXISTS memory_entities (
    id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bank_id TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    name    TEXT NOT NULL,
    norm    TEXT NOT NULL,
    etype   TEXT NOT NULL DEFAULT 'other',
    UNIQUE (bank_id, norm, etype)
);

CREATE TABLE IF NOT EXISTS memory_item_entities (
    item_id   UUID NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    entity_id UUID NOT NULL REFERENCES memory_entities(id) ON DELETE CASCADE,
    n         INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (item_id, entity_id)
);
CREATE INDEX IF NOT EXISTS memory_item_entities_entity_idx ON memory_item_entities (entity_id);

CREATE TABLE IF NOT EXISTS memory_links (
    bank_id    TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    src        UUID NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    dst        UUID NOT NULL REFERENCES memory_items(id) ON DELETE CASCADE,
    link_type  TEXT NOT NULL,                 -- SEMANTIC | CAUSAL | TEMPORAL | ENTITY
    subtype    TEXT NOT NULL,                 -- RELATED_TO | LEADS_TO | PRECEDES | SUCCEEDS | TEMPORALLY_CLOSE | NEXT | SHARED_ENTITY
    confidence REAL NOT NULL DEFAULT 1.0,
    origin     TEXT NOT NULL DEFAULT 'sql',   -- 'jev' | 'pgvector' | 'sql' | 'import'
    evidence   JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (src, dst, link_type, subtype)
);
CREATE INDEX IF NOT EXISTS memory_links_src_idx ON memory_links (src);
CREATE INDEX IF NOT EXISTS memory_links_dst_idx ON memory_links (dst);
CREATE INDEX IF NOT EXISTS memory_links_bank_idx ON memory_links (bank_id);

-- One row per document the Linker has finished (or failed): lets the async
-- worker and `link_pending` find documents still to link. Reset when a
-- document's items are replaced (delete_memory_items_for_document).
CREATE TABLE IF NOT EXISTS memory_link_state (
    document_id UUID PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    bank_id     TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    status      TEXT NOT NULL,                -- 'linked' | 'error'
    stats       JSONB NOT NULL DEFAULT '{}'::jsonb,
    error       TEXT,
    linked_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The date a note is about: its frontmatter `created` when a valid date, else
-- when it was retained. Used by the Linker's temporal links (no model call).
CREATE OR REPLACE FUNCTION prospecta_doc_date(meta JSONB, created TIMESTAMPTZ)
RETURNS DATE LANGUAGE plpgsql STABLE AS $$
BEGIN
    IF meta ? 'created' AND (meta->>'created') ~ '^\d{4}-\d{2}-\d{2}' THEN
        RETURN substring(meta->>'created' FROM 1 FOR 10)::date;
    END IF;
    RETURN created::date;
EXCEPTION WHEN others THEN
    RETURN created::date;
END $$;

COMMENT ON TABLE memory_entities IS 'Entities (people, orgs, places, projects, events) extracted at retain by the Linker.';
COMMENT ON TABLE memory_item_entities IS 'Which entity a memory item mentions, and how often (n).';
COMMENT ON TABLE memory_links IS 'Typed links between memory items (GraphExpand walks them in both directions). origin: jev | pgvector | sql | import.';
COMMENT ON TABLE memory_link_state IS 'Documents the Linker has processed; absent row = pending.';
