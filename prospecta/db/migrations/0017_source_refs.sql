-- 0017: reference ledger for records indexed IN PLACE from host-provided views
-- (docs/source-contract.md). One row per (bank, kind, record id): the revision
-- that is indexed and the filter columns the view supplied. It holds NO record
-- text; the documents row it points at is an empty anchor (original_text = '')
-- and the derived memory_items carry the same kind/record_id/revision in their
-- metadata. Additive; deleting the document cascades to this row.
CREATE TABLE IF NOT EXISTS source_refs (
    bank_id       TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    kind          TEXT NOT NULL,
    record_id     TEXT NOT NULL,
    revision      TEXT NOT NULL,
    document_id   UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    person        TEXT,
    project       TEXT,
    created_date  DATE,
    audience_key  TEXT,
    indexed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (bank_id, kind, record_id)
);
CREATE INDEX IF NOT EXISTS source_refs_document_idx ON source_refs (document_id);
CREATE INDEX IF NOT EXISTS source_refs_bank_project_idx ON source_refs (bank_id, project);
