# Observability

Prospecta records every load-bearing event of the bilateral spine in Postgres event tables. The persisted record is durable — you query it later with `pgcli` or any Postgres client — not ephemeral stderr that scrolls past. This is the audit trail and the debugging surface in one.

## What gets recorded

Six event types, one table each (plus the `documents` + `memory_items` tables, which are the canonical content store):

| Table | Fires on | Carries |
|---|---|---|
| `retain_events` | every `retain()` call | `document_id`, `items_count`, `index_text_caller_supplied`, **`index_text_generated`** (verbatim LLM-authored questions when not caller-supplied), `duration_ms`, `raw_llm_response`, `error` |
| `recall_events` | every `recall()` AND `recall_synth()` call | `queries` (formulated list), `mode`, `n_results`, **`results`** (JSONB array of per-chunk `{source, document_id, rank, scores, content_preview}`), **`synthesis`** (TEXT — populated by `recall_synth`, NULL for plain `recall`), `duration_ms`, `trace` |
| `formulate_events` | every `formulate_queries()` call | `message` (verbatim user input), `n_queries_out`, `json_mode_used`, `parse_fallback`, `raw_response` (verbatim LLM JSON), **`error_kind`** (`'malformed_json'` / `'schema_mismatch'` / NULL), `duration_ms` |
| `llm_calls` | every LLM call (3 sites: `index_text` generation, formulate, synthesize) | `prompt_name` (`'index_text'` / `'formulate'` / `'synthesize'`), `messages_count`, `json_mode`, `duration_ms`, **`prompt_text`** (verbatim rendered prompt), **`response_text`** (verbatim LLM response), `error` |
| `sweep_passes` | every background sweep cycle | `corpus_path`, `started_at` / `ended_at`, `files_seen` / `files_indexed` / `files_pruned` / `errors_count`, `duration_ms`, `pass_metadata` |
| `sweeper_state` | upserted on each sweep | `(bank_id, corpus_path)` → `last_pass_started_at`, `last_pass_ended_at`, `last_pass_duration_ms`, `last_pass_files_indexed`, `last_pass_errors` |

(Bolded fields are the columns added by migration 0003 — full durable trace beyond bare timing and counts.)

## The pgcli cheat sheet

Connect to your prospecta DB and run these. All assume `bank_id='default'` — adjust the `WHERE` if you're using multiple banks.

### Last 5 retains: what did the agent commit to memory?

Per-call summary of write activity: how many items each retain produced, whether the caller supplied `index_text` (P4 caller-wins) or the LLM authored it, the LLM-authored questions verbatim when applicable, and timing.

```sql
SELECT
    re.created_at,
    re.document_id,
    d.source,
    re.items_count,
    re.index_text_caller_supplied                AS caller_supplied,
    re.index_text_generated                      AS llm_questions,
    re.duration_ms,
    re.error
FROM retain_events re
LEFT JOIN documents d ON d.id = re.document_id
WHERE re.bank_id = 'default'
ORDER BY re.created_at DESC
LIMIT 5;
```

### Last 5 recalls: query → expansion → synthesis end-to-end

Each recall pairs with one preceding `formulate_events` row (the message expansion that produced its `queries`). Join by `bank_id` + proximity in time to walk a single inspection thread end-to-end.

```sql
WITH recent_recalls AS (
    SELECT
        id, bank_id, created_at, queries, mode, n_results,
        duration_ms, synthesis
    FROM recall_events
    WHERE bank_id = 'default'
    ORDER BY created_at DESC
    LIMIT 5
)
SELECT
    r.created_at                                 AS recall_at,
    f.message                                    AS user_message,
    f.n_queries_out,
    f.parse_fallback,
    f.error_kind,
    r.mode,
    r.n_results,
    r.duration_ms                                AS recall_ms,
    LEFT(r.synthesis, 200)                       AS synthesis_preview
FROM recent_recalls r
LEFT JOIN LATERAL (
    SELECT *
    FROM formulate_events fe
    WHERE fe.bank_id = r.bank_id
      AND fe.created_at <= r.created_at
    ORDER BY fe.created_at DESC
    LIMIT 1
) f ON TRUE
ORDER BY r.created_at DESC;
```

### Per-chunk inspection of a specific recall

`recall_events.results` is a JSONB array — one element per surfaced chunk. Unwrap with `jsonb_array_elements` to see source, document_id, rank, per-channel scores, and the 200-char preview for each chunk that came back.

```sql
SELECT
    elem->>'source'                              AS source,
    (elem->>'rank')::int                         AS rank,
    elem->>'document_id'                         AS document_id,
    elem->'scores'                               AS scores,
    elem->>'content_preview'                     AS preview
FROM recall_events,
     jsonb_array_elements(results) AS elem
WHERE id = <RECALL_EVENT_ID>
ORDER BY (elem->>'rank')::int;
```

(Pull `<RECALL_EVENT_ID>` from the "last 5 recalls" query above.)

### LLM call audit by prompt name

Each LLM site (`index_text`, `formulate`, `synthesize`) emits an `llm_calls` row. Group to see call volume and timing distribution; head the prompt + response to verify what's actually going over the wire.

```sql
-- Volume + timing distribution, last 24h
SELECT
    prompt_name,
    COUNT(*)                                     AS calls,
    ROUND(AVG(duration_ms))                      AS avg_ms,
    MAX(duration_ms)                             AS max_ms,
    COUNT(*) FILTER (WHERE error IS NOT NULL)    AS errors
FROM llm_calls
WHERE bank_id = 'default'
  AND created_at >= now() - INTERVAL '24 hours'
GROUP BY prompt_name
ORDER BY calls DESC;

-- Most recent prompt + response per site
SELECT
    created_at,
    prompt_name,
    duration_ms,
    LEFT(prompt_text,   400)                     AS prompt_head,
    LEFT(response_text, 400)                     AS response_head,
    error
FROM llm_calls
WHERE bank_id = 'default'
ORDER BY created_at DESC
LIMIT 10;
```

### Sweeper status

One row per `(bank_id, corpus_path)` reflecting the last completed sweep. Use this to see whether the background sweeper is keeping up and where errors are landing.

```sql
SELECT
    corpus_path,
    last_pass_ended_at,
    last_pass_duration_ms,
    last_pass_files_seen,
    last_pass_files_indexed,
    last_pass_files_pruned,
    last_pass_errors,
    last_error
FROM sweeper_state
WHERE bank_id = 'default'
ORDER BY last_pass_ended_at DESC NULLS LAST;
```

For historical sweep cadence (was it running every 5 minutes? did it stop?):

```sql
SELECT
    started_at, ended_at, duration_ms,
    files_seen, files_indexed, files_pruned, errors_count
FROM sweep_passes
WHERE bank_id = 'default'
ORDER BY started_at DESC
LIMIT 20;
```

### Bank stats one-liner

For document/item counts, the easiest path is the CLI:

```bash
prospecta stats
```

Or via SQL directly:

```sql
SELECT
    b.bank_id,
    b.embedding_dim,
    (SELECT COUNT(*) FROM documents     WHERE bank_id = b.bank_id) AS documents,
    (SELECT COUNT(*) FROM memory_items  WHERE bank_id = b.bank_id) AS items,
    (SELECT COUNT(*) FROM retain_events WHERE bank_id = b.bank_id) AS retains,
    (SELECT COUNT(*) FROM recall_events WHERE bank_id = b.bank_id) AS recalls
FROM banks b
ORDER BY b.bank_id;
```

## Reading a recall event end-to-end

A worked example: walk through what happens when an agent calls `m.recall_synth("when's kelly's birthday?")`, in order.

1. **`formulate_events` row inserted** — the LLM expanded the user's message into N sub-questions. `raw_response` has the verbatim JSON the LLM returned; `parse_fallback` tells you whether parsing succeeded; `error_kind` discriminates the failure mode (`malformed_json` vs `schema_mismatch` vs NULL). The parsed sub-questions become the input to recall.
2. **Per-channel retrieval runs** (HNSW + content_tsv BM25 + body_tsv BM25), three CTEs FULL OUTER JOIN'd via RRF. No event row per channel — this is in-flight SQL, not a logged event.
3. **`recall_events` row inserted** — `queries` lists the sub-questions, `mode='hybrid'` (or `semantic`/`lexical`), `n_results=N`, `results` is a JSONB array with one entry per surfaced chunk (source, document_id, rank, per-channel scores dict, 200-char content preview), `synthesis` is the RAG output text, `duration_ms` is total wall-clock for the recall+synth.
4. **`llm_calls` row inserted for the synthesize call** — `prompt_name='synthesize'`, `prompt_text` is the rendered RAG-synthesize template with the retrieved chunks substituted in, `response_text` is the verbatim synthesis.

So from one `recall_synth`, you get 1 `formulate_events` row + 1 `recall_events` row + 2 `llm_calls` rows (one for `formulate`, one for `synthesize`). The formulate `llm_call` shows you the JSON-mode prompt + the raw LLM JSON; the synthesize `llm_call` shows you the chunk-stuffed prompt + the answer. Plain `recall()` (no synthesis) drops the synthesize `llm_call` and leaves `recall_events.synthesis` NULL.

## Configuration

### Default behavior

When `Memory(database_url=..., bank_id=..., llm=..., embed=...)` is constructed without an explicit `tracer=` kwarg, the default tracer is `PostgresSink(pool=<memory's pool>, bank_id=<memory's bank>)`. All six event types persist automatically — no opt-in needed.

### Disable verbose LLM payloads

`prompt_text` and `response_text` on `llm_calls` capture the verbatim prompt and response. For high-volume production where the storage cost or the prompts themselves contain sensitive context, opt out at sink construction:

```python
from prospecta import Memory
from prospecta.observability import PostgresSink

sink = PostgresSink(
    database_url="postgres://...",
    bank_id="default",
    persist_llm_text=False,
)
m = Memory(
    database_url="postgres://...",
    bank_id="default",
    llm=...,
    embed=...,
    tracer=sink,
)
```

When `persist_llm_text=False`, the `prompt_text` and `response_text` columns are written as NULL even though the tracer payload carries them. The other event tables (`retain_events`, `recall_events`, `formulate_events`) are unaffected — the opt-out is scoped specifically to `llm_calls`.

### Custom tracers

The tracer is a `Tracer` Protocol callable — `(event: str, payload: dict) -> None`. The library ships four implementations and you can supply your own.

```python
from prospecta._tracer import NoOpTracer, RecordingTracer, CompositeTracer
from prospecta.observability import PostgresSink

# Drop every event (no persistence; useful for tests or read-only setups)
m = Memory(..., tracer=NoOpTracer())

# Capture in-memory (useful for tests + transient debugging)
rec = RecordingTracer()
m = Memory(..., tracer=rec)
# ... do stuff ...
for event_name, payload in rec.events:
    print(event_name, payload)

# Combine: persist to Postgres AND capture in memory
sink = PostgresSink(database_url="...", bank_id="default")
rec = RecordingTracer()
m = Memory(..., tracer=CompositeTracer(sink, rec))
```

For an entirely custom sink — say, dispatching events to an OpenTelemetry trace exporter or a metrics aggregator — implement the Protocol directly:

```python
def my_tracer(event: str, payload: dict) -> None:
    # ... your dispatch logic ...
    pass

m = Memory(..., tracer=my_tracer)
```

The six event names are stable: `retain`, `recall`, `formulate_queries`, `sweep_pass`, `index_single_file`, `llm_call`. The payload dicts for each are documented in `prospecta/observability/postgres_sink.py` — the `_handle_*` methods are the canonical reference for which payload keys land in which columns.

## What is NOT persisted

Honest about the boundaries:

- **Cosine distance / HNSW probe details.** Individual vector arithmetic isn't logged; you see the surfaced result + RRF score, not the per-channel internal scoring within the HNSW probe. To inspect at that level, query `memory_items` directly or use `Memory.search(mode='semantic')` / `mode='lexical'` to compare channels in isolation.
- **Chunker decisions.** Markdown chunking happens before retain; the chunked text lands as `memory_items.original_chunk`, but the chunker's reasoning (split here vs. there) isn't recorded.
- **Per-channel scores for items that did NOT surface in the top-K.** Only the returned chunks land in `recall_events.results`. Items that scored below the cutoff are not retained in the event row.

## Performance impact

Event-table writes are append-only INSERTs (`sweeper_state` is the one UPSERT). Each event is a single small row; the load is dominated by the LLM calls and the retrieval SQL themselves, not by the trace persistence. For comparison: a typical `recall_synth` runs 2 LLM calls (~1–3 s each) and ~10–100 ms of hybrid SQL — adding 3 event-table INSERTs (~1 ms each) is negligible.

If trace volume becomes a concern long-term, you have three levers:

1. Set `persist_llm_text=False` to drop the largest columns from `llm_calls`.
2. Implement a custom tracer that filters by event name (e.g., persist only `recall` + `retain`, drop `llm_call` + `index_single_file`).
3. Schedule periodic pruning of old event rows via SQL (`DELETE FROM recall_events WHERE created_at < now() - INTERVAL '90 days'`). Out of scope for v0.1.

## See also

- [`docs/design/prospecta/schema.md`](design/prospecta/schema.md) — canonical DDL for all tables.
- [`PRINCIPLES.md`](../PRINCIPLES.md) — load-bearing principles, including the observability principle.
- [`prospecta/observability/postgres_sink.py`](../prospecta/observability/postgres_sink.py) — payload contract reference for each event handler.
