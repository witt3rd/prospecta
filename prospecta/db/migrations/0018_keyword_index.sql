-- Stored keyword index (see prospecta/db/keyword_index.py).
-- The index itself is built CONCURRENTLY (it cannot run inside this transaction)
-- by run_migrations right after this file commits, or by `prospecta keyword-index`.
-- Nothing to change in the schema here; this version marks that step as owed.
SELECT 1;
