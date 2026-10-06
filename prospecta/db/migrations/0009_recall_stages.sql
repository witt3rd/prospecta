-- 0009: recall stages (hybrid retrieval design 8.7 / 8.8 / 8.9): per-bank
-- rerank / Jev gate / reader-hop config and recall-level accounting.
-- Additive only: ADD COLUMN with a constant default (banks.recall_config, '{}'
-- means every stage is off and recall is unchanged) and nullable columns.
-- No populated column is dropped, rewritten or retyped.

ALTER TABLE banks ADD COLUMN IF NOT EXISTS recall_config JSONB NOT NULL DEFAULT '{}'::jsonb;

COMMENT ON COLUMN banks.recall_config IS
    'Recall stages after fusion: {"rerank":{"enabled","stage","pool"},"gate":{"enabled","threshold"},"reader":{"enabled",...}}. Empty = no stage runs.';

ALTER TABLE recall_events ADD COLUMN IF NOT EXISTS cost_usd    NUMERIC;
ALTER TABLE recall_events ADD COLUMN IF NOT EXISTS tokens_in   INTEGER;
ALTER TABLE recall_events ADD COLUMN IF NOT EXISTS tokens_out  INTEGER;
ALTER TABLE recall_events ADD COLUMN IF NOT EXISTS n_llm_calls INTEGER;

COMMENT ON COLUMN recall_events.cost_usd IS
    'Summed cost of the model calls (rerank, Jev, reader) made by this recall; NULL when no stage ran or the provider reported none.';
COMMENT ON COLUMN recall_events.tokens_in IS
    'Summed input tokens of the recall''s model calls; NULL when unreported.';
COMMENT ON COLUMN recall_events.tokens_out IS
    'Summed output tokens of the recall''s model calls; NULL when unreported.';
COMMENT ON COLUMN recall_events.n_llm_calls IS
    'Number of model calls this recall made (0 or NULL = none).';
COMMENT ON COLUMN recall_events.rerank IS
    'Rerank stage record: stage, model, gate decision, per-candidate grade / Jev score, final order, latency, cost, fallback reason. NULL when no stage ran.';
COMMENT ON COLUMN recall_events.hops IS
    'Reader / hop record: reader verdict, follow-up queries, new candidates, second rerank. NULL when no reader ran.';
