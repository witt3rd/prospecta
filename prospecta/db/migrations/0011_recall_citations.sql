-- 0011: grounded synthesis (hybrid retrieval design 8.8): the list of notes a
-- grounded recall_synth cited. Additive only: one nullable column, no default
-- rewrite, no populated column touched.

ALTER TABLE recall_events ADD COLUMN IF NOT EXISTS citations JSONB;

COMMENT ON COLUMN recall_events.citations IS
    'Notes cited by a grounded synthesis: [{"note","document_id","known"}] in order of first citation; known=false when the model cited a name that was not in the evidence. NULL for plain recall and legacy synthesis.';
