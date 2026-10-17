# Observability

Prospecta records every load-bearing event in Postgres event tables. The record is durable: you query it later with `psql`, `pgcli` or any Postgres client, not ephemeral stderr that scrolls past. It is the audit trail and the debugging surface in one. `docs/schema.md` has the columns; this page says what each event is and how to read it.

## What gets recorded

| Table | Fires on | Carries |
|---|---|---|
| `retain_events` | every `retain()` call | `document_id`, `items_count` (the chunks written), `index_text_caller_supplied`, `duration_ms`, `error`. `index_text_generated` and `raw_llm_response` are filled only in rows written before the model-written questions were removed (`docs/legacy.md`) |
| `recall_events` | every `recall()` (so every `ask()`), every `expand_and_recall` / `recall_for_task` that records (a task by default, a chat message not), `recall_synth` | `queries` (the question, or the searches the writer wrote), `mode`, `n_results`, `results` (JSONB array of per-hit `{source, document_id, rank, scores, content_preview}`), `duration_ms`, `depth`, and the pipeline trace: `plan`, `channels`, `fusion`, `rerank` (`hops` is empty in a recall that ran after the reader was removed), plus `cost_usd`, `tokens_in`, `tokens_out`, `n_llm_calls`; `synthesis` and `citations` for the legacy synthesis |
| `recall_event_candidates` | one row per candidate per channel of a recall | `recall_event_id`, `query_index`, `document_id`, `item_id`, `channel` (`hybrid_chunk`), `rank`, `score` |
| `llm_calls` | every model call: the query writer, the Stage 0 gate, every Stage 2 rerank request, Stage 3, the `ask` answer | `prompt_name` (the **purpose**, below), `model`, `tokens_in`, `tokens_out`, `cost_usd`, `messages_count`, `json_mode`, `duration_ms`, `error`, **`prompt_text`** and **`response_text`** (verbatim; opt-out) |
| `formulate_events` | every `formulate_queries()` call (the legacy question-space path) | `message`, `n_queries_out`, `json_mode_used`, `parse_fallback`, `error_kind` (`malformed_json` / `schema_mismatch` / NULL), `raw_response`, `duration_ms` |
| `sweep_passes` | every background sweep cycle | `corpus_path`, `started_at` / `ended_at`, `files_seen` / `files_indexed` / `files_pruned` / `errors_count`, `duration_ms`, `pass_metadata` |
| `sweeper_state` | upserted on each sweep | `(bank_id, corpus_path)` → the last pass's start, end, duration, files indexed, errors |

`Memory.search` on its own writes no event; `recall`, `ask`, `expand_and_recall` and `recall_for_task` do (`record=`; a chat message is not recorded by default, a task is). With a shadow bank (`docs/embedding-migration.md`) a recall writes the old / new pair instead of the single event.

### `llm_calls.prompt_name` (the purposes)

| purpose | what | current design |
|---|---|---|
| `expand_gate` | the Stage 0 `needs_memory` decision | yes |
| `expand_searches` | the query writer | yes |
| `cross_encoder_rerank` | one Stage 2 rerank request (`model` as the provider reports it, `tokens_in` = `usage.total_tokens`, `cost_usd` = `usage.cost`) | yes |
| `stage3_sufficiency` | the Stage 3 `evidence_sufficient` question | yes |
| `ask_cited` | the `ask` answer | yes |
| `index_text` | model-written questions at write time (removed; rows written before it stay) | `docs/legacy.md` |
| `formulate_queries`, `synthesize_grounded`, `rerank_listwise`, `jev_score`, `reader_sufficiency`, `extract_filters`, `extract_entities`, `jev_relations`, `mapreduce_map`, `mapreduce_reduce` | the retired question-space, Sonnet / Jev, reader, entity, linker and map-reduce calls: no code writes them any more, rows already written stay | `docs/legacy.md` |

## Reading a recall event end to end

An agent calls `m.recall_for_task(title=..., ask=..., done=..., project=...)` (or `expand_and_recall` with `record=True`):

1. **`llm_calls`**: one `expand_searches` row (the writer; its `prompt_text` is the rendered prompt, `response_text` the searches), one `expand_gate` row for a chat message, then one `cross_encoder_rerank` row per rerank request (a search's pool goes out as several parallel requests when it is large). Each has `model`, tokens, cost and any `error`.
2. **One `recall_events` row**: `queries` is the list of the searches' `query` text, `n_results` the merged items, `duration_ms` the whole call, `results` one entry per item of the block. The trace columns hold, per search (`per_query`), the plan, `channels` (the `hybrid_chunk` count and latency), `fusion`, and `rerank`: `stage`, `model`, `n_candidates`, `n_below_min_score`, `collapsed`, `n_over_top_n`, `latency_ms`, `fallback_reason` (non-NULL when Stage 1's order stood), and the totals.
3. **`recall_event_candidates`**: every Stage 1 candidate of every search, with its rank and fused score.

On the legacy question-space path a recall also has a preceding `formulate_events` row (the model's expansion of the message into sub-questions, with `parse_fallback` and `error_kind`) and, for `recall_synth`, one more `llm_calls` row for the synthesis (`docs/legacy.md`); the TUI's recall thread joins them (`tui/README.md`).

What the caller sees of the same run is in the `ExpandedRecall`: `searches`, `runs` (what each found), `fallbacks` (every degraded step), `timings_ms` and `accounting`.

## The psql cheat sheet

Connect to your prospecta DB and run these. All assume `bank_id='default'`; adjust the `WHERE` for other banks.

### Last 5 retains: what did the agent commit to memory?

```sql
SELECT
    re.created_at,
    re.document_id,
    d.source,
    re.items_count,
    re.index_text_caller_supplied                AS caller_supplied,
    re.duration_ms,
    re.error
FROM retain_events re
LEFT JOIN documents d ON d.id = re.document_id
WHERE re.bank_id = 'default'
ORDER BY re.created_at DESC
LIMIT 5;
```

### Last 10 recalls: cost, latency, and whether the reranker ran

```sql
SELECT
    r.created_at,
    r.mode,
    r.n_results,
    r.duration_ms,
    r.depth,
    r.n_llm_calls,
    r.cost_usd,
    r.queries->>0                                AS first_query,
    r.rerank->'per_query'->0->>'fallback_reason' AS stage2_fallback
FROM recall_events r
WHERE r.bank_id = 'default'
ORDER BY r.created_at DESC
LIMIT 10;
```

`stage2_fallback` is NULL when the cross-encoder answered, and says why when the Stage 1 order stood (timeout, HTTP error, no transport).

### Per-hit inspection of a specific recall

`recall_events.results` is a JSONB array, one element per surfaced note:

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

(Pull `<RECALL_EVENT_ID>` from the query above.) The candidates a channel found, including those that did not make the result:

```sql
SELECT query_index, channel, rank, round(score::numeric, 5) AS score, document_id
FROM recall_event_candidates
WHERE recall_event_id = <RECALL_EVENT_ID>
ORDER BY query_index, rank;
```

### Model-call audit by purpose, with cost

```sql
-- Volume, timing, tokens and cost by purpose, last 24h
SELECT
    prompt_name,
    COUNT(*)                                     AS calls,
    ROUND(AVG(duration_ms))                      AS avg_ms,
    MAX(duration_ms)                             AS max_ms,
    COUNT(*) FILTER (WHERE error IS NOT NULL)    AS errors,
    SUM(tokens_in)                               AS tokens_in,
    SUM(tokens_out)                              AS tokens_out,
    ROUND(SUM(cost_usd)::numeric, 4)             AS cost_usd
FROM llm_calls
WHERE bank_id = 'default'
  AND created_at >= now() - INTERVAL '24 hours'
GROUP BY prompt_name
ORDER BY calls DESC;

-- Most recent prompt + response per purpose
SELECT created_at, prompt_name, model, duration_ms,
       LEFT(prompt_text, 400) AS prompt_head, LEFT(response_text, 400) AS response_head, error
FROM llm_calls
WHERE bank_id = 'default'
ORDER BY created_at DESC
LIMIT 10;
```

### Sweeper status

One row per `(bank_id, corpus_path)` reflecting the last completed sweep:

```sql
SELECT corpus_path, last_pass_ended_at, last_pass_duration_ms, last_pass_files_seen,
       last_pass_files_indexed, last_pass_files_pruned, last_pass_errors, last_error
FROM sweeper_state
WHERE bank_id = 'default'
ORDER BY last_pass_ended_at DESC NULLS LAST;
```

For the history (was it running every 5 minutes? did it stop?):

```sql
SELECT started_at, ended_at, duration_ms, files_seen, files_indexed, files_pruned, errors_count
FROM sweep_passes
WHERE bank_id = 'default'
ORDER BY started_at DESC
LIMIT 20;
```

### Bank stats one-liner

The CLI: `prospecta stats`. Or in SQL:

```sql
SELECT
    b.bank_id,
    b.embedding_dim,
    b.channel_config = '[]'::jsonb                              AS legacy_bank,
    (SELECT COUNT(*) FROM documents     WHERE bank_id = b.bank_id) AS documents,
    (SELECT COUNT(*) FROM memory_items  WHERE bank_id = b.bank_id) AS items,
    (SELECT COUNT(*) FROM retain_events WHERE bank_id = b.bank_id) AS retains,
    (SELECT COUNT(*) FROM recall_events WHERE bank_id = b.bank_id) AS recalls
FROM banks b
ORDER BY b.bank_id;
```

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

- **Cosine distance / HNSW probe details.** Individual vector arithmetic isn't logged; you see the surfaced result and each candidate's rank and fused score, not the scoring inside the HNSW probe. To inspect at that level, query `memory_items` directly.
- **Chunker decisions.** Markdown chunking happens before retain; the chunked text lands as `memory_items.original_chunk`, but the chunker's reasoning (split here vs. there) isn't recorded.
- **Items that did NOT surface.** Only the returned notes land in `recall_events.results`. A bank with a non-empty `channel_config` also persists every channel candidate in `recall_event_candidates` (migration 0007); a legacy bank (`channel_config = []`) does not.
- **Plain `Memory.search`.** It records nothing; `recall` and the Mode 1 calls do.

## Performance impact

Event-table writes are append-only INSERTs (`sweeper_state` is the one UPSERT). Each event is a single small row; the load is dominated by the LLM calls and the retrieval SQL themselves, not by the trace persistence. For comparison: a Mode 1 recall makes a writer call, up to 20 rerank requests (each a few hundred milliseconds) and tens of milliseconds of Stage 1 SQL; the event-table INSERTs (~1 ms each, one per model call plus the recall) are negligible.

If trace volume becomes a concern long-term, you have three levers:

1. Set `persist_llm_text=False` to drop the largest columns from `llm_calls`.
2. Implement a custom tracer that filters by event name (e.g., persist only `recall` + `retain`, drop `llm_call` + `index_single_file`).
3. Schedule periodic pruning of old event rows via SQL (`DELETE FROM recall_events WHERE created_at < now() - INTERVAL '90 days'`). Out of scope for v0.1.

## See also

- [`docs/schema.md`](schema.md) — the tables and columns.
- [`docs/recall-stages.md`](recall-stages.md) — the stages whose records these tables hold.
- [`PRINCIPLES.md`](../PRINCIPLES.md) — load-bearing principles, including the observability principle.
- [`prospecta/observability/postgres_sink.py`](../prospecta/observability/postgres_sink.py) — payload contract reference for each event handler.
