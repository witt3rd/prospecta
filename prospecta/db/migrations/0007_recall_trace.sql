-- 0007: recall trace (hybrid retrieval design 8.3 / 8.9).
-- Additive only: nullable JSONB columns (metadata-only ADD COLUMN), one new
-- table, nullable accounting columns on llm_calls. No populated column is
-- dropped, rewritten or retyped.

ALTER TABLE recall_events ADD COLUMN IF NOT EXISTS plan     JSONB;
ALTER TABLE recall_events ADD COLUMN IF NOT EXISTS channels JSONB;
ALTER TABLE recall_events ADD COLUMN IF NOT EXISTS fusion   JSONB;
ALTER TABLE recall_events ADD COLUMN IF NOT EXISTS rerank   JSONB;
ALTER TABLE recall_events ADD COLUMN IF NOT EXISTS hops     JSONB;

COMMENT ON COLUMN recall_events.plan IS
    'Query plan of a channel-pipeline recall: query text(s), formulations, extracted filters. NULL on legacy rows.';
COMMENT ON COLUMN recall_events.channels IS
    'Per channel: name, kind, weight, n, latency_ms, cost_usd, error (and query_index). NULL on legacy rows.';
COMMENT ON COLUMN recall_events.fusion IS
    'Fusion record: method, k, weights, pool document ids. NULL on legacy rows.';
COMMENT ON COLUMN recall_events.rerank IS
    'Rerank stage record (reserved for the reranker; NULL until one runs).';
COMMENT ON COLUMN recall_events.hops IS
    'Reader / hop record (reserved; NULL until one runs).';

-- One row per candidate per channel: the durable form of per-channel scores.
CREATE TABLE IF NOT EXISTS recall_event_candidates (
    id               BIGSERIAL PRIMARY KEY,
    recall_event_id  BIGINT NOT NULL REFERENCES recall_events(id) ON DELETE CASCADE,
    query_index      INTEGER NOT NULL DEFAULT 0,
    document_id      UUID,
    item_id          UUID,
    channel          TEXT NOT NULL,
    rank             INTEGER NOT NULL,
    score            DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS recall_event_candidates_event_idx
    ON recall_event_candidates (recall_event_id);

ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS model      TEXT;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS tokens_in  INTEGER;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS tokens_out INTEGER;
ALTER TABLE llm_calls ADD COLUMN IF NOT EXISTS cost_usd   NUMERIC;
