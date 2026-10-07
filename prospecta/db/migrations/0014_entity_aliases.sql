-- 0014: aliases (nicknames, pen names, "also known as") of entities.
-- Additive only: two new empty tables, nothing existing is touched, so plain
-- CREATE INDEX is safe. An alias row resolves a name to the canonical (person)
-- entity; the Linker attaches notes that use any alias to that entity.

CREATE TABLE IF NOT EXISTS memory_entity_aliases (
    bank_id   TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    norm      TEXT NOT NULL,                -- normalised alias
    alias     TEXT NOT NULL,                -- alias as written
    entity_id UUID NOT NULL REFERENCES memory_entities(id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (bank_id, norm)
);
CREATE INDEX IF NOT EXISTS memory_entity_aliases_entity_idx ON memory_entity_aliases (entity_id);

-- Documents the alias backfill has handled (resumable by document id).
CREATE TABLE IF NOT EXISTS memory_alias_state (
    document_id UUID PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    bank_id     TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    status      TEXT NOT NULL,              -- 'done' | 'error'
    stats       JSONB NOT NULL DEFAULT '{}'::jsonb,
    error       TEXT,
    done_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE memory_entity_aliases IS 'Alias (nickname, pen name, aka) -> canonical entity, per bank.';
COMMENT ON TABLE memory_alias_state IS 'Documents processed by the alias extraction; absent row = pending.';
