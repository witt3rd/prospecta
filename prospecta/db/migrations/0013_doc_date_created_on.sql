-- 0013: temporal links follow the note's own date.
-- Additive: a 3-argument overload of prospecta_doc_date that reads the
-- documents.created_on column first, then metadata `created`, then created_at.
-- The 2-argument function from 0006 is left untouched.
CREATE OR REPLACE FUNCTION prospecta_doc_date(meta JSONB, created TIMESTAMPTZ, created_on DATE)
RETURNS DATE LANGUAGE sql STABLE AS $$
    SELECT COALESCE(created_on, prospecta_doc_date(meta, created))
$$;
