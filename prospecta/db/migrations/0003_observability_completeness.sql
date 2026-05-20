-- 0003: observability completeness pass.
--
-- Closes the durability gap in event logging:
--   retain_events:    persist verbatim generated index_text (when LLM-authored)
--   recall_events:    persist the results that came back + synthesis text
--   llm_calls:        persist verbatim prompt + response (opt-in via PostgresSink flag)
--   formulate_events: persist error_kind discriminator for parse failures
--
-- Idempotent via per-column existence checks (mirrors 0002's pattern). Re-running
-- this migration is a no-op against a database that already has the columns.

-- retain_events.index_text_generated TEXT[]
--   Verbatim LLM output for index_text generation. NULL when caller supplied
--   index_text (no LLM call) or when retain pre-T16 wrote rows without it.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'retain_events' AND column_name = 'index_text_generated'
    ) THEN
        ALTER TABLE retain_events ADD COLUMN index_text_generated TEXT[];
    END IF;
END$$;

COMMENT ON COLUMN retain_events.index_text_generated IS
    'Verbatim list of LLM-generated index_text strings (questions). NULL when '
    'index_text was caller-supplied (P4 caller-wins) or when retain pre-dated '
    'this column. One row per retain() call captures the full list.';

-- recall_events.results JSONB
--   Inspection record: which chunks came back for this recall. NOT a P5
--   violation — full content lives in memory_items.original_chunk; this is
--   a 200-char preview per element, explicitly inspection-only.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'recall_events' AND column_name = 'results'
    ) THEN
        ALTER TABLE recall_events ADD COLUMN results JSONB;
    END IF;
END$$;

COMMENT ON COLUMN recall_events.results IS
    'JSONB array of {source, document_id, rank, scores, content_preview}. '
    'content_preview is the first 200 chars of memory_items.content — '
    'inspection-only; full content_preview lives in memory_items. This is '
    'the audit trail for "which chunks surfaced", not the canonical content '
    'store. NULL on rows written before this column existed.';

-- recall_events.synthesis TEXT
--   The RAG synthesis output. NULL for plain recall() (no synthesis);
--   populated for recall_synth() calls.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'recall_events' AND column_name = 'synthesis'
    ) THEN
        ALTER TABLE recall_events ADD COLUMN synthesis TEXT;
    END IF;
END$$;

COMMENT ON COLUMN recall_events.synthesis IS
    'Verbatim RAG synthesis output. NULL for plain recall() (no synthesis); '
    'populated for recall_synth().';

-- llm_calls.prompt_text + llm_calls.response_text TEXT
--   Verbatim prompt rendered and response returned. Opt-in via PostgresSink
--   constructor flag persist_llm_text (default True for v0.1.x β). When the
--   flag is off, the columns are written NULL even when the payload carries
--   them.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'llm_calls' AND column_name = 'prompt_text'
    ) THEN
        ALTER TABLE llm_calls ADD COLUMN prompt_text TEXT;
    END IF;
END$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'llm_calls' AND column_name = 'response_text'
    ) THEN
        ALTER TABLE llm_calls ADD COLUMN response_text TEXT;
    END IF;
END$$;

COMMENT ON COLUMN llm_calls.prompt_text IS
    'Verbatim rendered prompt text. Captured at call site (index_text, '
    'formulate_queries, synthesize). NULL when PostgresSink was constructed '
    'with persist_llm_text=False, or on rows written before this column existed.';

COMMENT ON COLUMN llm_calls.response_text IS
    'Verbatim LLM response. Captured at call site. NULL when PostgresSink '
    'was constructed with persist_llm_text=False, or on rows written before '
    'this column existed.';

-- formulate_events.error_kind TEXT
--   T13 verifier proviso #2: distinguishes malformed_json vs schema_mismatch
--   without parsing raw_response.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'formulate_events' AND column_name = 'error_kind'
    ) THEN
        ALTER TABLE formulate_events ADD COLUMN error_kind TEXT;
    END IF;
END$$;

COMMENT ON COLUMN formulate_events.error_kind IS
    'Parse-failure discriminator: ''malformed_json'' | ''schema_mismatch'' | NULL. '
    'NULL on success (parse_fallback=False). Persisted from FormulateOutcome '
    '(prospecta._formulate). T13 verifier proviso #2.';
