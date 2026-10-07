# Schema Planner — Prospecta v0.1 Postgres+pgvector Stance

**Domain:** `~/src/witt3rd/prospecta/` — schema slice
**Becomes:** §5 of `plan-v2.md`
**Authored:** 2026-05-18 by Forge ⚒️ (Planner role, R1)
**Inputs:** `decision-record-1.md`, `schema-context.md`, `PRINCIPLES.md`, hindsight-client v0.6.1 seed material
**Verdict (self-graded):** APPROVE — see end.

---

## Premise

The schema's load-bearing intent is **bilateral synthesis as first-class shape, with hybrid retrieval as the default safety net.** Everything else — multi-tenancy, observability, sweeper state, JSON contracts — is downstream of two facts: (1) the unit of retrieval is an LLM-anticipated question (`memory_items.content` = the `index_text`, per `decision-record-1.md` Decision 5), and (2) the default recall path runs semantic + lexical in parallel and fuses results via RRF, because the spine's match-in-question-space property must degrade gracefully when the LLM's anticipated question doesn't match the caller's actual query (P1 spine + P14 safety net, composed). The schema is shaped to make that the cheap path, not the heroic one.

## Inherited constraints (brief)

Postgres ≥ 14 + pgvector; direct connection via `database_url` (no daemon, Decision 1); `embed` injected as `Callable[[list[str]], list[list[float]]]` (Decision 2); `bank_id` on every row (Decision 4); hindsight's `banks` / `documents` / `memory_items` shape lifted, KG (entities, fact_types, mental_models, observation_scopes, nodes_by_fact_type) dropped (Decision 5, confirmed against `/tmp/hindsight-src/hindsight_client-0.6.1/hindsight_client_api/models/memory_item.py` and `recall_request.py`). Hybrid retrieval is v0.1, not deferred (Decision 5 + schema-context.md §3). Bilateral spine is locked five rounds back.

---

## Table set — full DDL outline

The canonical v0.1 tables. Eight tables total: 3 substrate (banks, documents, memory_items), 4 event logs (retain, recall, formulate, llm_calls), 1 sweeper state. No `sweep_passes` table separate from sweeper_state — see §"Event tables" for the rationale.

### Extensions + migrations bookkeeping

```sql
CREATE EXTENSION IF NOT EXISTS vector;          -- pgvector
CREATE EXTENSION IF NOT EXISTS pg_trgm;         -- trigram fallback for FTS edge cases (cheap, useful)

CREATE TABLE prospecta_schema_version (
    version       INTEGER PRIMARY KEY,
    applied_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    description   TEXT NOT NULL
);
```

### `banks`

```sql
CREATE TABLE banks (
    bank_id            TEXT PRIMARY KEY,
    config             JSONB NOT NULL DEFAULT '{}'::jsonb,
    mission            TEXT,
    retain_mission     TEXT,
    embedding_dim      INTEGER NOT NULL,        -- locked at bank creation (see §memory_items deep-dive)
    embedding_model_id TEXT,                    -- caller-provided label, opaque to library
    fts_config         REGCONFIG NOT NULL DEFAULT 'english',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX banks_created_at_idx ON banks(created_at);
```

`mission` / `retain_mission` lifted from hindsight's `BankConfigResponse` (`bank_config_response.py`) — opaque text the caller may use to prompt-engineer per-bank retention policy. `embedding_dim` and `fts_config` are prospecta additions; rationale in deep-dive below.

### `documents`

```sql
CREATE TABLE documents (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bank_id            TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    source             TEXT,                    -- file path, URL, or caller-supplied identifier
    original_text      TEXT NOT NULL,           -- P5: no truncation, source of truth
    content_hash       TEXT NOT NULL,           -- sha256 of original_text, dedup key
    tags               TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    document_metadata  JSONB NOT NULL DEFAULT '{}'::jsonb,
    retain_params      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (bank_id, content_hash)              -- dedup: same content within a bank = same document
);

CREATE INDEX documents_bank_source_idx ON documents(bank_id, source);
CREATE INDEX documents_tags_gin       ON documents USING GIN(tags);
CREATE INDEX documents_created_at_idx ON documents(bank_id, created_at DESC);
```

Field set tracks hindsight's `DocumentResponse` (`document_response.py`) minus `nodes_by_fact_type` (KG, dropped per Decision 5). UUIDs for ids — caller doesn't need to coordinate, and we avoid serial-leak (Decision 5's "load-bearing decisions" question).

### `memory_items` — the spine table

```sql
CREATE TABLE memory_items (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bank_id         TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Spine columns (P1)
    content         TEXT NOT NULL,              -- the LLM-anticipated question form (index_text)
    original_chunk  TEXT NOT NULL,              -- the document chunk this index_text covers (P5)
    embedding       vector NOT NULL,            -- typed at runtime per bank.embedding_dim
    content_tsv     tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,

    -- Provenance + filtering
    context         TEXT,                       -- optional caller-supplied context blob
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    tags            TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    update_mode     TEXT NOT NULL DEFAULT 'append',  -- 'append' | 'replace' (hindsight parity)
    llm_generated   BOOLEAN NOT NULL,           -- did the spine fire, or was index_text caller-supplied (P4)?

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

Indexes for memory_items live in §"Index strategy" — they are the load-bearing decisions and deserve their own section.

### Event tables (four, separate, typed columns)

```sql
CREATE TABLE retain_events (
    id                       BIGSERIAL PRIMARY KEY,
    bank_id                  TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    document_id              UUID REFERENCES documents(id) ON DELETE SET NULL,
    items_count              INTEGER NOT NULL,
    index_text_caller_supplied BOOLEAN NOT NULL,
    duration_ms              INTEGER NOT NULL,
    raw_llm_response         TEXT,               -- P5: full, never truncated
    error                    TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX retain_events_bank_time_idx ON retain_events(bank_id, created_at DESC);

CREATE TABLE recall_events (
    id                BIGSERIAL PRIMARY KEY,
    bank_id           TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    queries           JSONB NOT NULL,           -- the formulated queries (array of strings)
    mode              TEXT NOT NULL,            -- 'hybrid' | 'semantic' | 'lexical'
    n_results         INTEGER NOT NULL,
    duration_ms       INTEGER NOT NULL,
    query_timestamp   TIMESTAMPTZ,              -- caller-supplied "as of" timestamp (hindsight parity)
    trace             JSONB,                    -- per-side rankings + RRF fusion detail when trace=true
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX recall_events_bank_time_idx ON recall_events(bank_id, created_at DESC);

CREATE TABLE formulate_events (
    id                BIGSERIAL PRIMARY KEY,
    bank_id           TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    message           TEXT NOT NULL,            -- input to formulate_queries
    n_queries_out     INTEGER NOT NULL,
    json_mode_used    BOOLEAN NOT NULL,
    parse_fallback    BOOLEAN NOT NULL,
    raw_response      TEXT NOT NULL,            -- P5
    duration_ms       INTEGER NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX formulate_events_bank_time_idx ON formulate_events(bank_id, created_at DESC);

CREATE TABLE llm_calls (
    id              BIGSERIAL PRIMARY KEY,
    bank_id         TEXT REFERENCES banks(bank_id) ON DELETE SET NULL,
    prompt_name     TEXT NOT NULL,              -- 'rag-synthesize' | 'formulate-queries' | 'generate-index-text'
    messages_count  INTEGER NOT NULL,
    json_mode       BOOLEAN NOT NULL,
    duration_ms     INTEGER NOT NULL,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX llm_calls_bank_time_idx   ON llm_calls(bank_id, created_at DESC);
CREATE INDEX llm_calls_prompt_time_idx ON llm_calls(prompt_name, created_at DESC);
```

### `sweeper_state`

```sql
CREATE TABLE sweeper_state (
    bank_id              TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    corpus_path          TEXT NOT NULL,
    last_pass_started_at TIMESTAMPTZ,
    last_pass_ended_at   TIMESTAMPTZ,
    last_pass_files_seen INTEGER,
    last_pass_files_indexed INTEGER,
    last_pass_files_pruned  INTEGER,
    last_pass_errors     INTEGER,
    last_pass_duration_ms INTEGER,
    last_error           TEXT,
    PRIMARY KEY (bank_id, corpus_path)
);
```

One row per `(bank_id, corpus_path)`, upserted on each pass. **This subsumes `sweep_passes` from schema-context.md** — see "Event tables" decision below for why.

---

## `memory_items` deep-dive

### Granularity: one row per `index_text`, multiple rows per document

Locked. Each `memory_item` is one LLM-anticipated question form (per P1 spine) paired with the chunk of `documents.original_text` it covers via `original_chunk`. A single document with five `index_text` values in frontmatter produces five `memory_items` rows; a non-frontmatter document gets chunked first, then each chunk gets an LLM-generated `index_text`, producing N rows.

**Why not per-chunk-only:** the spine demands the row's vector be the embedding of a *question*, not content. Per-chunk-only would either (a) embed content (defeating P1) or (b) need a separate `index_texts` child table — adding a join on every recall for no payoff. Hindsight's `MemoryItem` (`memory_item.py`) is also per-content-unit; we converge.

**Why not per-document-with-array-of-questions:** would put the embedding column on a child table anyway, since pgvector needs one vector per row to index. Same shape, more ceremony.

**Dedup story:** `documents` is deduped by `(bank_id, content_hash)`. `memory_items` is NOT deduped — same content with different `index_text` is a different row by design. If a caller retains the same chunk twice with the same index_text, that's a caller bug; we surface via `update_mode='replace'` semantics on retain (delete-then-insert by `(document_id, content_hash_of_original_chunk)` — practical handling lives in the library, not the schema constraint).

### Vector dimensionality: per-bank, locked at bank creation

Locked. `banks.embedding_dim` records the dimension; `memory_items.embedding` is typed `vector` (untyped at DDL level) and the library asserts dim matches `banks.embedding_dim` on insert. Index is built with the exact dimension via `vector(N)` cast at index-create time (see §Index strategy).

**Why per-bank not per-row:** Donald's case is exactly the load: Donald uses BGE-large at home (1024-d), Cookie uses OpenAI text-embedding-3-large (3072-d) remotely, both write to one Postgres. They write to **different banks**. Cross-embedder within one bank is a non-feature — you cannot fuse vectors from different spaces. Locking per-bank makes that physically impossible (which is correct) and lets us build one HNSW index per bank with the right dimensionality. Per-row would force every query to filter by `embedding_dim` and use multiple HNSW indexes anyway — same complexity, more ways to corrupt the index.

**Embedding-model migration in v0.2:** rebuild the bank's vectors against a new embedder, alter `banks.embedding_dim`, drop/rebuild HNSW. This is a documented v0.2 operation, not a v0.1 hot path.

### FTS configuration: `english` default, per-bank override

`memory_items.content_tsv` is a STORED generated column using `to_tsvector('english', content)`. Bank-level override lives in `banks.fts_config` (`regconfig`), but **the generated column hardcodes `'english'` in v0.1** — Postgres generated columns cannot reference another table. v0.2 will route per-bank FTS via a trigger if multi-language users surface.

**Why `english` over `simple`:** stemming + stopword removal materially improves recall on Donald's substrate (English markdown). `simple` is a v0.2 escape hatch for callers whose content is non-English or whose match semantics benefit from exact tokens (code, identifiers). The lexical side of the hybrid is the safety net (P14) — `english` is the more forgiving default; `simple` is the more precise option. Default to the safety net's strength.

**Weighting:** v0.1 does not use `setweight` — `content` is the single FTS field. Tags and context are filtered via SQL predicates, not weighted into the tsvector. Adding weighted FTS (`setweight(to_tsvector(content), 'A') || setweight(to_tsvector(context), 'B')`) is a clean v0.2 extension.

### Spine columns (`content`, `original_chunk`, `embedding`, `content_tsv`)

These four are the load-bearing identity of the table. Removing any one of them breaks something:

- `content` → spine (P1)
- `original_chunk` → no-truncation (P5), what the caller actually gets to read
- `embedding` → semantic side of hybrid
- `content_tsv` → lexical side of hybrid

Everything else is metadata.

---

## Index strategy

### `memory_items` — the indexes that matter

```sql
-- HNSW on embedding, per-bank (built after bank creation with known dim)
-- Created by migration helper, not DDL bootstrap, because dim is per-bank.
-- Example for a 1024-d bank:
CREATE INDEX memory_items_emb_hnsw_bank_xyz
    ON memory_items
    USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE bank_id = 'xyz';

-- GIN on FTS
CREATE INDEX memory_items_tsv_gin
    ON memory_items USING GIN(content_tsv);

-- Composite B-tree for bank-scoped scans + temporal ordering
CREATE INDEX memory_items_bank_time_idx
    ON memory_items(bank_id, created_at DESC);

-- Document-scoped lookups (delete cascade, replace mode)
CREATE INDEX memory_items_doc_idx ON memory_items(document_id);

-- Tag filtering
CREATE INDEX memory_items_tags_gin ON memory_items USING GIN(tags);
```

### HNSW over IVFFlat — locked

v0.1 ships HNSW. Rationale:

1. **Workload shape.** Schema-context describes ~100K–1M memory_items, occasional reads, moderate writes. HNSW's query latency advantage matters; its build cost (15-30 min for 1M rows) is amortized across many recalls. IVFFlat needs `lists` tuned to row count and requires `ANALYZE` for the planner — operationally more fragile.
2. **Modern default.** pgvector ≥ 0.5 ships HNSW as the recommended index; community guidance through 2025 converges on HNSW for sub-million-row corpora with quality-sensitive recall. IVFFlat is the choice when you have >10M vectors and need fast rebuild.
3. **Concurrency.** HNSW supports concurrent inserts cleanly; IVFFlat's `lists` partitioning is sensitive to skew. Prospecta's write pattern (sweeper + retain on same bank) benefits.
4. **Recall tunability via `ef_search`.** Set per-query (`SET LOCAL hnsw.ef_search = 100`), trades latency for quality without rebuilding.

Defaults: `m = 16`, `ef_construction = 64`. Caller can override at `Memory(...)` init via `index_params` for the per-bank index migration.

### Per-bank HNSW index naming

Convention: `memory_items_emb_hnsw_<bank_id_safe>` where `bank_id_safe` is the bank_id with non-alphanumeric chars replaced by `_`. Partial index (`WHERE bank_id = ...`) so each bank's index is independent — drop one bank, drop one index, no global rebuild.

### Why partial indexes instead of one global HNSW

Two reasons:
1. **Dimensionality:** pgvector requires a fixed dim per index. Per-bank dim means per-bank index. Forced.
2. **Locality of access:** every recall query is bank-scoped. A partial index `WHERE bank_id = 'xyz'` is automatically used by the planner when the query has `bank_id = 'xyz'` predicate. No global scan.

### Event tables, banks, documents, sweeper_state

Indexes shown inline at table DDL. Pattern: composite `(bank_id, created_at DESC)` on event tables for the ops queries Donald named (schema-context §6: "recall events in last 7 days," "median formulate duration," etc.). GIN on tags arrays where applicable. No HNSW outside `memory_items`.

---

## Hybrid retrieval — SQL pattern, scoring, caller override

### Default mode: `hybrid` via Reciprocal Rank Fusion (RRF), `k = 60`

Locked. RRF is the canonical pgvector hybrid pattern (Supabase, Neon, AWS docs all converge on it through 2025). It's simple, robust, doesn't require score normalization, and works when one side returns zero results (the other side just wins).

**`k = 60` locked, not exposed.** Caller can override via library config (`hybrid_rrf_k`) but the default is fixed. Exposing it on every recall API call would be configuration noise (P12 honest config surface) — `k = 60` is the canonical value across the literature; tuning it requires evidence we won't have in v0.1.

### Per-side candidate limits

Each side returns 50 candidates before fusion. Fusion produces up to 50 final results (configurable via `limit` on recall). 50 is the lever the caller actually wants — `limit` exposed; `per_side` stays internal default.

### Example query — full hybrid recall pattern

```sql
WITH
  semantic AS (
    SELECT id, document_id, content, original_chunk, metadata, tags,
           1 - (embedding <=> $query_embedding::vector) AS sem_score,
           ROW_NUMBER() OVER (ORDER BY embedding <=> $query_embedding::vector) AS sem_rank
    FROM memory_items
    WHERE bank_id = $bank_id
      AND ($tags::TEXT[] IS NULL OR tags && $tags)
    ORDER BY embedding <=> $query_embedding::vector
    LIMIT 50
  ),
  lexical AS (
    SELECT id, document_id, content, original_chunk, metadata, tags,
           ts_rank_cd(content_tsv, plainto_tsquery('english', $query_text)) AS lex_score,
           ROW_NUMBER() OVER (
               ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', $query_text)) DESC
           ) AS lex_rank
    FROM memory_items
    WHERE bank_id = $bank_id
      AND content_tsv @@ plainto_tsquery('english', $query_text)
      AND ($tags::TEXT[] IS NULL OR tags && $tags)
    ORDER BY lex_score DESC
    LIMIT 50
  ),
  fused AS (
    SELECT
      COALESCE(s.id, l.id) AS id,
      COALESCE(s.document_id, l.document_id) AS document_id,
      COALESCE(s.content, l.content) AS content,
      COALESCE(s.original_chunk, l.original_chunk) AS original_chunk,
      COALESCE(s.metadata, l.metadata) AS metadata,
      COALESCE(s.tags, l.tags) AS tags,
      s.sem_score,
      l.lex_score,
      (COALESCE(1.0 / (60 + s.sem_rank), 0.0)
       + COALESCE(1.0 / (60 + l.lex_rank), 0.0)) AS rrf_score
    FROM semantic s
    FULL OUTER JOIN lexical l ON s.id = l.id
  )
SELECT id, document_id, content, original_chunk, metadata, tags,
       sem_score, lex_score, rrf_score
FROM fused
ORDER BY rrf_score DESC
LIMIT $limit;
```

For multi-query recall (formulate_queries returns N queries), the library iterates this query per formulated query, then fuses the per-query result-lists with a second RRF pass before returning. That's library-side logic, not schema-side — but the schema supports it cheaply because each per-query SQL is fast.

### Caller override mechanism

`Memory.recall(queries=[...], mode='hybrid' | 'semantic' | 'lexical', tags=[...], tags_match='any'|'all'|'any_strict'|'all_strict', limit=N)`:

- `mode='semantic'` → skip the `lexical` CTE; rank by `embedding <=> query` directly.
- `mode='lexical'` → skip the `semantic` CTE; rank by `ts_rank_cd` directly.
- `mode='hybrid'` (default for `recall_synth`; default for low-level `recall` too because the spine wants the safety net) → full RRF pattern above.

`tags_match` lifted from hindsight (`recall_request.py`) — `'any' | 'all' | 'any_strict' | 'all_strict'`. `_strict` excludes untagged rows; non-strict includes them. SQL predicate:

```sql
-- 'any'      : tags && $tags OR cardinality(tags) = 0
-- 'all'      : tags @> $tags OR cardinality(tags) = 0
-- 'any_strict': tags && $tags
-- 'all_strict': tags @> $tags
```

`tag_groups` (hindsight's mental-model trigger machinery) **dropped** per schema-context Off-Limits.

---

## Migrations

### Hand-rolled SQL, no Alembic — locked

v0.1 uses hand-rolled `.sql` files in `prospecta/db/migrations/` numbered `0001_initial.sql`, `0002_add_foo.sql`, etc. Applied by `prospecta-migrate` CLI, tracked via `prospecta_schema_version` table.

**Why not Alembic:**
1. Alembic adds a dependency, an `env.py`, autogenerate machinery the library doesn't need, and pulls in SQLAlchemy by transitive convention.
2. Schema is small and stable (8 tables, no ORM, raw psycopg). The Alembic value-prop (declarative model → migration diff) doesn't apply.
3. Hand-rolled SQL is debuggable by anyone with psql open; Alembic introduces an extra mental layer.
4. P12 (honest config surface) and P9 (thin adapter) imply minimizing surface area. Alembic is surface.

**Migration runner:**

```python
# prospecta/db/migrations.py
def run_migrations(conn: psycopg.Connection) -> None:
    """Apply all pending migrations in numbered order. Idempotent."""
    # 1. CREATE TABLE IF NOT EXISTS prospecta_schema_version (...)
    # 2. SELECT MAX(version) FROM prospecta_schema_version
    # 3. For each .sql file in migrations/ with N > current: BEGIN; exec; INSERT version; COMMIT
```

### Initial migration outline

`0001_initial.sql` ships everything in §"Table set" except the per-bank HNSW indexes (those are created on bank creation by the library, since dim is per-bank).

`prospecta/db/migrations/`:
- `0001_initial.sql` — extensions, schema_version, banks, documents, memory_items, all event tables, sweeper_state, all non-per-bank indexes.

Bank-creation helper (in library, not migration):
```python
def _create_bank_hnsw_index(conn, bank_id: str, dim: int) -> None:
    safe = re.sub(r'[^a-zA-Z0-9_]', '_', bank_id)
    conn.execute(
        f"CREATE INDEX IF NOT EXISTS memory_items_emb_hnsw_{safe} "
        f"ON memory_items USING hnsw "
        f"((embedding::vector({dim})) vector_cosine_ops) "
        f"WITH (m = 16, ef_construction = 64) "
        f"WHERE bank_id = %s", (bank_id,))
```

---

## Event tables — confirmed set + rationale

Four typed event tables (`retain_events`, `recall_events`, `formulate_events`, `llm_calls`) plus one state table (`sweeper_state`). **`sweep_passes` collapsed into `sweeper_state`** because:

- The ops query Donald cares about ("when did sweeper last run, was it healthy?") is answered by current state, not history.
- History is recoverable from `retain_events` filtered by source — when the sweeper retains, it writes retain_events; the sweeper itself doesn't need its own log.
- Five tables for what could be four is the simplicity payoff (P12).

**Why not one polymorphic `events` table with `type` discriminator:**

The proliferation cost is real, but the polymorphic cost is worse:
- Each event type has its own typed columns (`raw_llm_response` only on retain/formulate; `queries` JSONB only on recall; `prompt_name` only on llm_calls). A single events table would push all of these into one wide JSONB blob, defeating SQL queryability and P5 (full content, queryable).
- Indexes per-type are cleaner: `recall_events_bank_time_idx` is a real composite; `events_bank_time_idx WHERE type='recall'` would be a partial index with worse selectivity heuristics.
- Donald's ops queries (schema-context §6) read cleanly against typed tables; against a polymorphic table they read as JSONB extractions, which is exactly the failure mode P5 warns against.

The 4-table cost is ~200 LOC of migration + 4 index decisions. The polymorphic cost is every analytics query becoming a JSONB extraction exercise. **Typed tables win.**

---

## API JSON contracts

Mirror hindsight where transferable, diverge where prospecta is simpler. Citations to hindsight source files inline.

### `RetainResponse`

```json
{
  "document_id": "uuid-string",
  "items_count": 5,
  "duration_ms": 234,
  "index_text_caller_supplied": false
}
```

Mimics hindsight's retain response minus async/background fields (hindsight's `retain_request.py` has `var_async`; prospecta v0.1 is sync, async deferred to v0.2 per Decision 1). `index_text_caller_supplied` is prospecta-specific (P4 caller-wins observability).

### `RecallResult` (per item)

```json
{
  "id": "uuid-string",
  "content": "the index_text question form",
  "original_chunk": "the full chunk this memory_item covers",
  "source": "document_id-or-source-path",
  "score": 0.0234,
  "scores": {
    "semantic": 0.812,
    "lexical": 0.443,
    "rrf": 0.0234
  },
  "metadata": {"k": "v"},
  "tags": ["tag1"],
  "bank_id": "prospecta",
  "document_id": "uuid-string",
  "created_at": "2026-05-18T..."
}
```

`content` + `original_chunk` are both present — the spine returns the question that matched AND the chunk it points at. Caller can render either. **`original_chunk` is never truncated** (P5). Hindsight's `RecallResult` (`recall_result.py`) has `text` + `context` + `chunk_id`; we converge on `content` (the matched thing) + `original_chunk` (the source) because it's clearer for the spine. `scores` is broken out so trace consumers can see both sides; `score` is the fusion score.

### `RecallResponse`

```json
{
  "results": [RecallResult, ...],
  "trace": {
    "queries_executed": ["q1", "q2"],
    "mode": "hybrid",
    "per_query_results_count": [5, 3],
    "fusion": "rrf",
    "rrf_k": 60
  }
}
```

Hindsight's `RecallResponse` (`recall_response.py`) carries `entities`, `chunks`, `source_facts` — all dropped (KG, Decision 5). The remaining shape — `results` + `trace` — is prospecta's minimal honest contract.

### `RAGResult` (from `recall_synth`, plan §3.4)

```json
{
  "synthesis": "LLM-synthesized answer text",
  "sources": [RecallResult, ...],
  "queries": ["formulated query 1", "formulated query 2"],
  "queries_to_results": {
    "formulated query 1": [RecallResult, ...],
    "formulated query 2": [RecallResult, ...]
  }
}
```

`queries_to_results` is the auditability handle — caller can see which formulated query produced which results before fusion. Diverges from hindsight (which doesn't expose this) because prospecta's spine makes the per-query attribution load-bearing.

---

## Test strategy — how tests get a clean DB

**Locked: per-test bank scoping on a shared DB, with a session-scoped testcontainer for CI.**

Three layers:

1. **CI / `pytest`:** `pytest-postgresql` or `testcontainers-python` spins up a Postgres+pgvector container once per test session. Migrations run once at session start.
2. **Per-test isolation:** Each test creates a unique `bank_id` (UUID-suffixed). All assertions are bank-scoped via `WHERE bank_id = ...`. Teardown is a single `DELETE FROM banks WHERE bank_id = $test_bank_id` (cascades through documents, memory_items, events, sweeper_state).
3. **Local dev:** Same docker-compose Postgres from Decision 3. Test fixture detects `PROSPECTA_TEST_DATABASE_URL` env var; falls back to spinning a container.

**Why not transaction-rollback isolation:** the HNSW index build inside a transaction is suspended until commit; tests that depend on vector-search results require committed writes. Transaction-rollback breaks the recall side of the bilateral integration test.

**Why not per-test schema (`CREATE SCHEMA test_xxx`):** migrations would run per-test; multi-second overhead. Per-bank cleanup is sub-ms.

The 2×2 bilateral integration test (plan §7) runs in this scaffold and verifies both spine directions land correctly through the full Postgres path, including hybrid retrieval (the test grows to include a lexical-side assertion per Decision 5 amendments).

---

## Hindsight-shape lineage — lift vs. diverge

### What prospecta lifts (transferable shape)

- **`banks` table identity.** `bank_id` as TEXT PK, `config` JSONB, `mission` / `retain_mission` TEXT (`bank_config_response.py`).
- **`documents` shape.** `id`, `bank_id`, `original_text`, `content_hash`, `tags`, `document_metadata`, `retain_params`, timestamps (`document_response.py` minus `nodes_by_fact_type`).
- **`memory_items` core columns.** `content`, `context`, `metadata`, `document_id`, `tags`, `update_mode` (`memory_item.py` minus `entities`, `observation_scopes`).
- **`tags_match` enum.** `'any' | 'all' | 'any_strict' | 'all_strict'` (`recall_request.py`).
- **Bank-ID multi-tenancy primitive.** Decision 4.
- **`bank_id_template` placeholder pattern.** Lifted from hermes-hindsight plugin for `hermes-prospecta`.
- **JSON response shape conventions** for retain/recall (ecosystem familiarity is real — operators familiar with hindsight transfer mental model instantly, per Decision 4 rationale).

### What prospecta diverges on (intentional)

- **No KG.** `entities`, `entity_observations`, `fact_types`, `nodes_by_fact_type`, `mental_models`, `directives`, `dispositions`, `webhooks`, `observation_scopes` — all dropped. Prospecta is a hybrid-retrieval RAG library, not a knowledge graph. (Decision 5 + schema-context Off-Limits.)
- **`embedding` column exposed.** Hindsight hides vectors behind its API; prospecta surfaces `embedding` on `memory_items` because the library is one tier closer to the storage substrate (no daemon, Decision 1).
- **`content_tsv` exposed.** Lexical side of hybrid retrieval; hindsight doesn't ship this because hindsight's recall is observation-based, not BM25-fused.
- **`original_chunk` as distinct column.** Hindsight stores chunks separately (`ChunkData` model); prospecta inlines the chunk on the memory_item because the spine's read-side returns content+chunk together every time. Joining on every recall would be wasted I/O.
- **`llm_generated` boolean.** Tracks whether `index_text` was caller-supplied (P4) or library-generated (P1). Useful for the bilateral integration test and for operators auditing spine behavior. Hindsight doesn't have this concept.
- **`banks.embedding_dim` + `banks.embedding_model_id`.** Per-bank embedder identity, enabling Donald's home/cloud split (Decision 2).
- **Synchronous v0.1.** Hindsight's `RetainRequest.var_async` deferred to v0.2.
- **`sweep_passes` collapsed into `sweeper_state`.** Hindsight has no equivalent; prospecta gains it for P14 safety-net observability.

---

## PRINCIPLES audit (compact)

| # | Principle | Honored by |
|---|---|---|
| P1 | Bilateral synthesis is the spine | `memory_items.content` = LLM-anticipated question; `llm_generated` boolean makes the spine auditable |
| P2 | Standalone first | Schema has zero Hermes dependency; `bank_id` is a TEXT field, not a hermes concept |
| P3 | LLM as injected callable | Schema is silent on LLM provider; `llm_calls` records prompt_name only, never provider |
| P4 | Caller wins on overrides | `llm_generated` boolean records whether spine fired or caller-supplied |
| P5 | Full content, no truncation | `original_chunk`, `raw_llm_response`, `raw_response` (formulate), `original_text` — all TEXT, never truncated |
| P6 | Let the LLM cook | Spine column is the question form; lexical safety net does not heuristic the LLM's job, it complements it |
| P7 | Single write path | Schema has no hot/cold-path distinction; retain and sweeper both upsert `memory_items` identically |
| P8 | Embedded-chroma zero-config | **Superseded** by Decision 1+3: zero-config path is `docker compose up` |
| P9 | Hermes plugin is thin | Schema is library-owned; plugin only sets `bank_id` via template |
| P10 | Surface adjacent mechanisms | Migration runner reuses psycopg directly; no parallel-build |
| P11 | Tests over prose | Per-bank test isolation against real pgvector container; 2×2 bilateral integration test |
| P12 | Honest config surface | 8 tables, no Alembic, `k=60` not exposed, per-side limit not exposed; `mode`/`tags`/`limit` exposed |
| P13 | Prompts with library | Schema is silent on prompts; `llm_calls.prompt_name` records the prompt name only |
| P14 | Sweeper is safety net | `sweeper_state` table makes drift observable; lexical-side of hybrid is the secondary safety net |
| P15 | Spine documented | README §"What makes prospecta different" cites `memory_items.content = question form` |

**No principle violated.** P8 is superseded by Decision 1 + 3 (not a violation; the principle as written named chroma, which is gone — the *spirit* of P8 ("zero-config first run") is now honored by `docker compose up`).

---

## Concrete answers to the 10 contested questions

1. **Granularity of `memory_items`** → One row per `index_text` (per question), multiple rows per document. Per-chunk-only would defeat P1.
2. **Vector dimensionality** → Per-bank. `banks.embedding_dim` locked at bank creation. Donald's BGE/OpenAI split = different banks.
3. **HNSW vs IVFFlat** → HNSW. Modern default; better for 100K–1M rows with quality-sensitive recall and moderate writes.
4. **FTS dictionary** → `english` default, `simple` as v0.2 escape hatch. Per-bank `fts_config` column exists but the generated `content_tsv` hardcodes `english` in v0.1 (Postgres generated-column limitation).
5. **RRF `k`** → Locked at 60. Library config `hybrid_rrf_k` exists but not exposed per-call.
6. **Event tables** → Four typed tables (`retain_events`, `recall_events`, `formulate_events`, `llm_calls`) + `sweeper_state`. `sweep_passes` collapsed into `sweeper_state`.
7. **Migrations infrastructure** → Hand-rolled SQL in numbered files, applied by `prospecta-migrate` CLI, tracked via `prospecta_schema_version` table. No Alembic.
8. **`content_hash` uniqueness** → `UNIQUE (bank_id, content_hash)` on `documents`. Same content with different `index_text` = one document, multiple memory_items. Same content with same index_text via `update_mode='replace'` = delete-then-insert.
9. **Sweeper state in DB** → Yes, `sweeper_state` table keyed by `(bank_id, corpus_path)`. One row per corpus, upserted per pass. Observable, resumable across restart.
10. **API JSON contracts** → Mirror hindsight's `RetainResponse` / `RecallResponse` shape conventions for ecosystem familiarity; diverge intentionally on (a) dropping KG fields, (b) adding `original_chunk` + `scores` to `RecallResult`, (c) adding `queries_to_results` to `RAGResult`.

---

## Open questions (legitimately deferred to Critic / R2)

1. **Cross-bank reflection in v0.2.** Decision 4 explicitly puts cross-bank queries out of v0.1 scope, but the schema does not preclude a future `views` layer that unions across banks for a "reflection" persona. Worth naming in §"Future extensions" of plan-v2.md but not designing now.
2. **`pg_trgm` index value.** I created the extension but no index uses it yet. Trigram could be a tertiary fallback when both semantic and lexical return zero — but adds complexity. Lean: leave extension on, no index, revisit if v0.1 testing surfaces zero-result cases.
3. **Generated column hardcoding `'english'`.** v0.2 may need per-bank FTS via trigger. Worth naming as known limitation.
4. **`vector_cosine_ops` vs `vector_l2_ops` vs `vector_ip_ops`.** Cosine is the right default for normalized embeddings (sentence-transformers, OpenAI), but BGE outputs may differ. Locked cosine for v0.1; document in operator notes.
5. **Should `llm_calls.prompt_name` be an ENUM?** It's bounded (P13 names four prompts). ENUM gives type safety; TEXT gives flexibility for caller-overridden prompts. Lean TEXT for P4.
6. **`bank_id` as TEXT (not UUID).** Convention from hindsight; lets operators write `bank_id = 'prospecta-forge'` directly. Cost: typo-tolerant FK target. Acceptable.

---

## Verdict

**APPROVE.** The stance:

- Locks all 10 contested questions with rationale.
- Honors PRINCIPLES.md P1–P15 (P8 superseded by Decision 1+3, not violated).
- Stays inside the Off-Limits list — does not re-open Postgres, daemon, embed-as-callable, bank_id, spine, or KG.
- Produces SQL DDL outline, indexing strategy, FTS configuration, hybrid retrieval scoring with example RRF query, migration plan, event-table set, JSON contracts, and test strategy.
- Cites required-reading paths inline at every load-bearing decision.
- Names six legitimately open questions for R2 / Critic without padding them as contests.

Hand off to Architect (R1) and Critic (R1) for review. If Critic surfaces a META-level concern that reframes a sub-dimension, R2 with revised stance. Otherwise → distill into `schema-stance.md` and fold into `plan-v2.md` §5.

⚒️ Forge — Planner R1, prospecta schema stance, 2026-05-18.
