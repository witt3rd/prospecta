---
status: canonical
date: 2026-05-18
artifact_type: schema-stance
domain: prospecta v0.1 Postgres+pgvector schema
parent_plan: docs/design/prospecta/plan.md  (to be merged into plan-v2.md §5)
parent_decision: docs/design/prospecta/decision-record-1.md  (six locked substrate decisions)
consensus: APPROVE_WITH_PROVISO_x2 (Architect Round 2 + Critic Round 2); ~9 provisos folded
provenance:
  - docs/design/prospecta/schema-context.md
  - docs/design/prospecta/schema-planner.md           (R1)
  - docs/design/prospecta/schema-architect.md         (R1)
  - docs/design/prospecta/schema-critic.md            (R1)
  - docs/design/prospecta/schema-planner-v2.md        (R2; response map stripped during distillation)
  - docs/design/prospecta/schema-architect-v2.md      (R2; APPROVE_WITH_PROVISO)
  - docs/design/prospecta/schema-critic-v2.md         (R2; APPROVE_WITH_PROVISO)
folded_provisos:
  - "A7-bug (Architect R2): advisory-lock constant overflowed bigint (76-bit); replaced with 56-bit safe constant"
  - "A7-idiomatic (Architect R2): array_length(...) -> vector_dims(NEW.embedding)"
  - "A7-notation (Architect R2): clarified $rrf_k is psycopg %(rrf_k)s binding"
  - "A7-honesty (Architect R2): co-write contract sweep_passes/sweeper_state explicitly library-convention not schema-enforced"
  - "A7-race (Architect R2): added LOCK TABLE banks IN EXCLUSIVE MODE for embedder migration"
  - "P1 (Critic R2): named rrf_k P4-over-P12 limitation explicitly"
  - "P2 (Critic R2): named P5 (no-truncation) hold across replace-on-source-match"
  - "P3 (Critic R2): made advisory-lock concurrency test concrete (4 subprocess workers, pytest-multiproc)"
  - "C5-addendum (Critic R2): added COMMENT ON COLUMN for embedding, content_tsv, documents.source"
---

# Prospecta v0.1 Schema

Canonical schema stance for prospecta. Distilled from a two-round ralplan
focused on schema design only. The bilateral synthesis spine is structurally
first-class (memory_items.content = LLM-anticipated question form);
hybrid retrieval (semantic + lexical via RRF) ships v0.1 as the safety net
for spine drift; multi-tenancy via `bank_id` matches hindsight's pattern.

This document becomes §5 of `plan-v2.md` (the full implementation plan
with substrate pivot).

## §1 — Premise + inherited constraints (unchanged from R1)

The schema's load-bearing intent is **bilateral synthesis as first-class shape, with hybrid retrieval as the default safety net.** Everything else — multi-tenancy, observability, sweeper state, JSON contracts — is downstream of two facts: (1) the unit of retrieval is an LLM-anticipated question (`memory_items.content` = the `index_text`, per Decision 5), and (2) the default recall path runs semantic + lexical in parallel and fuses via RRF, because the spine's match-in-question-space property must degrade gracefully when the LLM's anticipated question doesn't match the caller's actual query (P1 spine + P14 safety net, composed).

**Inherited constraints:** Postgres ≥ 14 + pgvector; direct connection via `database_url` (Decision 1); `embed` as injected `Callable[[list[str]], list[list[float]]]` (Decision 2); `bank_id` on every row (Decision 4); hindsight's `banks` / `documents` / `memory_items` shape lifted, KG dropped (Decision 5); hybrid retrieval is v0.1 (Decision 5 + schema-context §3); bilateral spine locked five rounds back.

---

## §2 — Table set DDL (revised)

Eight tables: 3 substrate (banks, documents, memory_items), 4 event logs (retain, recall, formulate, llm_calls), 1 append-only sweep history (sweep_passes), 1 state cache (sweeper_state). Migrations bookkeeping table: `prospecta_schema_version`.

### Extensions + migrations bookkeeping

```sql
CREATE EXTENSION IF NOT EXISTS vector;          -- pgvector
-- pg_trgm intentionally NOT created in v0.1 (was unused in R1; revisit if zero-result cases surface)

CREATE TABLE prospecta_schema_version (
    version       INTEGER PRIMARY KEY,
    applied_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    description   TEXT NOT NULL
);
```

### `banks` (revised: `fts_config` removed)

```sql
CREATE TABLE banks (
    bank_id            TEXT PRIMARY KEY,
    config             JSONB NOT NULL DEFAULT '{}'::jsonb,
    mission            TEXT,
    retain_mission     TEXT,
    embedding_dim      INTEGER NOT NULL,
    embedding_model_id TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX banks_created_at_idx ON banks(created_at);

COMMENT ON COLUMN banks.embedding_dim IS
  'Dimensionality of vectors in memory_items.embedding for this bank. '
  'Locked at bank creation. Enforced by trigger on memory_items insert/update. '
  'Changing requires rebuilding all bank vectors (see migration docs).';
```

`mission` / `retain_mission` lifted from hindsight (opaque caller text). `embedding_dim` / `embedding_model_id` are prospecta's per-bank embedder identity (Decision 2). **`fts_config` dropped** (A1).

### `documents` (unchanged structurally)

```sql
CREATE TABLE documents (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bank_id            TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    source             TEXT,
    original_text      TEXT NOT NULL,
    content_hash       TEXT NOT NULL,
    tags               TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    document_metadata  JSONB NOT NULL DEFAULT '{}'::jsonb,
    retain_params      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (bank_id, content_hash)
);

CREATE INDEX documents_bank_source_idx ON documents(bank_id, source);
CREATE INDEX documents_tags_gin       ON documents USING GIN(tags);
CREATE INDEX documents_created_at_idx ON documents(bank_id, created_at DESC);

COMMENT ON COLUMN documents.original_text IS
  'Full source text. Never truncated (P5). Source of truth for original_chunk slices.';
COMMENT ON COLUMN documents.content_hash IS
  'sha256 of original_text. Drives UNIQUE (bank_id, content_hash) for re-retain semantics; '
  'see Re-retain semantics section in design doc for collision behavior.';
```

### `memory_items` — the spine table (revised: comments + dim trigger reference)

```sql
CREATE TABLE memory_items (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bank_id         TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Spine columns (P1)
    content         TEXT NOT NULL,
    original_chunk  TEXT NOT NULL,
    embedding       vector NOT NULL,
    content_tsv     tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    body_tsv        tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', COALESCE(original_chunk, ''))) STORED,

    -- Provenance + filtering
    context         TEXT,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    tags            TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    update_mode     TEXT NOT NULL DEFAULT 'append',
    llm_generated   BOOLEAN NOT NULL,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON COLUMN memory_items.content IS
  'Bilateral spine: LLM-anticipated question form (index_text) that this memory_item answers. '
  'The semantic-search target; embedded into `embedding`; tokenized into `content_tsv`. '
  'NOT the source text — see original_chunk for that. Per P1, this is what makes prospecta a spine, not a chunker.';

COMMENT ON COLUMN memory_items.original_chunk IS
  'The slice of documents.original_text this memory_item indexes via its question form. '
  'Returned to callers untruncated (P5). What the user actually reads when this item recalls.';

COMMENT ON COLUMN memory_items.embedding IS
    'Vector representation of `content`. Dimensionality must match the bank''s embedding_dim. Generated by caller-supplied embed() callable.';
COMMENT ON COLUMN memory_items.content_tsv IS
    'STORED generated column: to_tsvector(''english'', content). Indexed via GIN for the lexical half of hybrid retrieval (BM25 fallback when LLM-anticipated index_text drifts).';

COMMENT ON COLUMN memory_items.body_tsv IS
    'STORED generated column: to_tsvector(''english'', COALESCE(original_chunk, '''')). Indexed via GIN for the body channel of three-channel hybrid retrieval. Implements the P14 body-fallback safety net: when LLM-generated index_text drifts hard from query language (content_tsv has no overlap), BM25 over the source body still surfaces the doc via RRF fusion.';
COMMENT ON COLUMN documents.source IS
    'Caller-supplied identifier for the originating source (file path, URL, conversation ID). Used by replace-on-source-match for re-retain.';
COMMENT ON COLUMN memory_items.llm_generated IS
  'TRUE if `content` was produced by the spine LLM (default); FALSE if caller-supplied via frontmatter '
  'or explicit index_text param (P4 caller-wins). Auditability handle for spine behavior.';
```

Indexes in §5.

### Dim-check trigger (A2)

```sql
CREATE OR REPLACE FUNCTION assert_memory_item_embedding_dim() RETURNS trigger AS $$
DECLARE
    expected_dim INTEGER;
    actual_dim   INTEGER;
BEGIN
    SELECT embedding_dim INTO expected_dim FROM banks WHERE bank_id = NEW.bank_id;
    IF expected_dim IS NULL THEN
        RAISE EXCEPTION 'memory_items.bank_id = % has no banks row', NEW.bank_id;
    END IF;
    actual_dim := vector_dims(NEW.embedding);  -- pgvector idiomatic; safer than array_length cast
    IF actual_dim <> expected_dim THEN
        RAISE EXCEPTION
          'memory_items.embedding dim mismatch for bank %: expected %, got %',
          NEW.bank_id, expected_dim, actual_dim;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER memory_items_embedding_dim_check
    BEFORE INSERT OR UPDATE OF embedding, bank_id ON memory_items
    FOR EACH ROW EXECUTE FUNCTION assert_memory_item_embedding_dim();
```

This makes the dim contract enforced by the database, not by library convention. Cost: one row lookup on `banks` per insert. Mitigation: banks table is tiny (single-digit rows in practice), buffer-cached; the trigger is sub-microsecond.

### Event tables + state (revised: `sweep_passes` restored)

```sql
CREATE TABLE retain_events (
    id                       BIGSERIAL PRIMARY KEY,
    bank_id                  TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    document_id              UUID REFERENCES documents(id) ON DELETE SET NULL,
    items_count              INTEGER NOT NULL,
    index_text_caller_supplied BOOLEAN NOT NULL,
    duration_ms              INTEGER NOT NULL,
    raw_llm_response         TEXT,
    error                    TEXT,
    created_at               TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX retain_events_bank_time_idx ON retain_events(bank_id, created_at DESC);

CREATE TABLE recall_events (
    id                BIGSERIAL PRIMARY KEY,
    bank_id           TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    queries           JSONB NOT NULL,
    mode              TEXT NOT NULL,
    n_results         INTEGER NOT NULL,
    duration_ms       INTEGER NOT NULL,
    query_timestamp   TIMESTAMPTZ,
    trace             JSONB,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX recall_events_bank_time_idx ON recall_events(bank_id, created_at DESC);

CREATE TABLE formulate_events (
    id                BIGSERIAL PRIMARY KEY,
    bank_id           TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    message           TEXT NOT NULL,
    n_queries_out     INTEGER NOT NULL,
    json_mode_used    BOOLEAN NOT NULL,
    parse_fallback    BOOLEAN NOT NULL,
    raw_response      TEXT NOT NULL,
    duration_ms       INTEGER NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX formulate_events_bank_time_idx ON formulate_events(bank_id, created_at DESC);

CREATE TABLE llm_calls (
    id              BIGSERIAL PRIMARY KEY,
    bank_id         TEXT REFERENCES banks(bank_id) ON DELETE SET NULL,
    prompt_name     TEXT NOT NULL,
    messages_count  INTEGER NOT NULL,
    json_mode       BOOLEAN NOT NULL,
    duration_ms     INTEGER NOT NULL,
    error           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX llm_calls_bank_time_idx   ON llm_calls(bank_id, created_at DESC);
CREATE INDEX llm_calls_prompt_time_idx ON llm_calls(prompt_name, created_at DESC);

-- A4: append-only sweep history
CREATE TABLE sweep_passes (
    id                  BIGSERIAL PRIMARY KEY,
    bank_id             TEXT NOT NULL REFERENCES banks(bank_id) ON DELETE CASCADE,
    corpus_path         TEXT NOT NULL,
    started_at          TIMESTAMPTZ NOT NULL,
    ended_at            TIMESTAMPTZ,
    files_seen          INTEGER NOT NULL DEFAULT 0,
    files_indexed       INTEGER NOT NULL DEFAULT 0,
    files_pruned        INTEGER NOT NULL DEFAULT 0,
    errors_count        INTEGER NOT NULL DEFAULT 0,
    duration_ms         INTEGER,
    error               TEXT,
    pass_metadata       JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX sweep_passes_bank_time_idx ON sweep_passes(bank_id, started_at DESC);
CREATE INDEX sweep_passes_corpus_idx    ON sweep_passes(bank_id, corpus_path, started_at DESC);

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

See §9 for the rationale split between `sweep_passes` (append-only history, A4) and `sweeper_state` (current-state cache).

---

## §3 — `memory_items` deep-dive (revised: dim-check trigger noted)

### Granularity — locked

One row per `index_text` (one LLM-anticipated question), multiple rows per document. Five frontmatter questions → five `memory_items`. Non-frontmatter → chunk → per-chunk LLM-generated `index_text` → N rows. Per-chunk-only would either embed content (defeating P1) or need a join table (no payoff). Hindsight's `MemoryItem` is also per-content-unit.

### Vector dimensionality — per-bank, schema-enforced (A2)

`banks.embedding_dim` records the dim; `memory_items.embedding` is column-untyped `vector` but the **dim-check trigger** in §3 raises on mismatch. Index built with exact dim via `vector(N)` cast at index-create time.

**Why per-bank not per-row:** Donald's case — BGE-large at home (1024-d) vs. OpenAI text-embedding-3-large (3072-d) remote — uses different banks, not different rows in one bank. You cannot fuse vectors from different spaces; cross-embedder in one bank is a non-feature. Per-bank lock makes it impossible (correct) and lets one HNSW per bank own its dimensionality.

**Embedder migration (v0.2):** preferred sequence — drop the bank's HNSW partial index, re-embed all `memory_items` for that bank with the new model, `UPDATE banks SET embedding_dim = N, embedding_model_id = ...`, rebuild HNSW. If re-embedding cost is high, the cheaper alternative is **carve a new bank** (`prospecta-forge-bge` → `prospecta-forge-openai`) and re-retain. Documented as v0.2 operation in plan-v2.md operator notes.

### FTS configuration — `'english'` library-level default (A1)

`memory_items.content_tsv` is a STORED generated column hardcoded to `to_tsvector('english', content)`. There is no `banks.fts_config` column — A1 catch. Non-English support is a v0.2 escape hatch via a per-bank trigger that replaces the generated column with a triggered column.

**Why `english` over `simple`:** stemming + stopword removal materially improves recall on English markdown (Donald's substrate). Lexical side is the safety net (P14); default to the safety net's strength.

**Weighting:** v0.1 does not use `setweight` — `content` is the single FTS field. Tags filtered via SQL predicate. Weighted FTS is a clean v0.2 extension.

### Spine columns (`content`, `original_chunk`, `embedding`, `content_tsv`)

Load-bearing identity. `content` → spine (P1, now SQL-commented). `original_chunk` → no-truncation (P5, now SQL-commented). `embedding` → semantic side of hybrid. `content_tsv` → lexical side of hybrid. Everything else is metadata.

---

### Embedder-migration race note

The dim-check trigger reads `banks.embedding_dim` on every memory_items insert. During an embedder migration that mutates the bank's `embedding_dim`, operators must run:

```sql
BEGIN;
LOCK TABLE banks IN EXCLUSIVE MODE;
UPDATE banks SET embedding_dim = <new_dim> WHERE bank_id = <bank>;
-- delete or migrate existing memory_items for this bank
COMMIT;
```

This serializes the dim change against concurrent inserts. Documented in operator migration guide; not schema-enforced (P12 — operators can speak SQL).

---

## §4 — Index strategy (unchanged from R1)

```sql
-- HNSW on embedding, per-bank partial index (built post bank-creation with known dim).
-- Example for a 1024-d bank:
CREATE INDEX memory_items_emb_hnsw_bank_xyz
    ON memory_items
    USING hnsw ((embedding::vector(1024)) vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    WHERE bank_id = 'xyz';

CREATE INDEX memory_items_tsv_gin
    ON memory_items USING GIN(content_tsv);

CREATE INDEX memory_items_body_tsv_gin
    ON memory_items USING GIN(body_tsv);

CREATE INDEX memory_items_bank_time_idx
    ON memory_items(bank_id, created_at DESC);

CREATE INDEX memory_items_doc_idx ON memory_items(document_id);

CREATE INDEX memory_items_tags_gin ON memory_items USING GIN(tags);
```

### HNSW over IVFFlat — locked

v0.1 ships HNSW. Rationale: (1) workload is 100K–1M memory_items, occasional reads, moderate writes — HNSW query latency wins, build cost amortizes; (2) modern pgvector ≥ 0.5 default; (3) concurrent inserts cleanly; (4) `ef_search` is per-query (`SET LOCAL hnsw.ef_search = 100`), trades latency for recall without rebuild. Defaults: `m = 16`, `ef_construction = 64`. Caller can override at `Memory(...)` init via `index_params`.

### Per-bank partial HNSW

Convention: `memory_items_emb_hnsw_<bank_id_safe>` where `bank_id_safe` is alphanumeric-safe. `WHERE bank_id = '...'` partial: planner uses it on bank-scoped queries; drop one bank, drop one index. Forced by per-bank dim — pgvector requires fixed dim per index.

---

## §5 — Hybrid retrieval (revised: parameterized `rrf_k`, COALESCE on scores)

### Default: `hybrid` via RRF with parameterized `k`

**Three-channel fusion (T19, post-bilateral hardening pass):**
v0.1 ships hybrid retrieval as a three-channel RRF fusion:
1. **semantic** — cosine distance over `embedding` (HNSW).
2. **lexical_content** — `ts_rank_cd` over `content_tsv` (= `to_tsvector(content)`, the LLM-anticipated index_text / questions channel).
3. **lexical_body** — `ts_rank_cd` over `body_tsv` (= `to_tsvector(original_chunk)`, the source body channel).

The third channel implements the P14 body-fallback axis of the safety net. When LLM-generated `index_text` drifts semantically AND lexically from the query (case the original two-channel design did NOT cover), the body channel still surfaces the doc through BM25 overlap on the source text. See §10 A6 for the always-numeric score invariant across all three channels.

**Known limitation (P4-over-P12 trade):** `rrf_k` is configured at `Memory(rrf_k=60)` init, not per-`recall_synth()` call. P4 (caller wins on every override) would argue for per-call exposure; P12 (honest config surface, no per-call noise) won the trade. v0.2 may add per-call override if lived use surfaces need.

RRF is the canonical pgvector hybrid pattern. Simple, normalization-free, degrades gracefully when one side returns zero. **`k` is parameterized at the prepared-statement level** (psycopg binds as `%(rrf_k)s`; shown below as `$rrf_k` for SQL readability) and exposed as `Memory(rrf_k=60)` init config, default 60. Removed the R1 inconsistency where SQL had a literal `60` while the prose claimed "configurable via library config." Per-call exposure stays off the `recall()` signature (P12 — config noise) but the value is one prepared-statement parameter, not a compiled constant.

### Per-side candidate limits

Each side returns 50 candidates before fusion. `limit` exposed; `per_side` internal default (50).

### Example query — full hybrid recall (revised: parameterized + COALESCE on scores)

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
      COALESCE(s.sem_score, 0) AS sem_score,    -- A6: never NULL in JSON output
      COALESCE(l.lex_score, 0) AS lex_score,    -- A6: never NULL in JSON output
      (COALESCE(1.0 / ($rrf_k + s.sem_rank), 0.0)
       + COALESCE(1.0 / ($rrf_k + l.lex_rank), 0.0)) AS rrf_score
    FROM semantic s
    FULL OUTER JOIN lexical l ON s.id = l.id
  )
SELECT id, document_id, content, original_chunk, metadata, tags,
       sem_score, lex_score, rrf_score
FROM fused
ORDER BY rrf_score DESC
LIMIT $limit;
```

For multi-query recall (formulate returns N), the library iterates per formulated query then RRF-fuses the result-lists with a second pass. Library logic; schema supports cheaply because each per-query SQL is fast.

### Caller override

`Memory.recall(queries=[...], mode='hybrid'|'semantic'|'lexical', tags=[...], tags_match='any'|'all'|'any_strict'|'all_strict', limit=N)`:

- `mode='semantic'` → skip lexical CTE.
- `mode='lexical'` → skip semantic CTE.
- `mode='hybrid'` (default) → full RRF.

`tags_match` lifted from hindsight; `_strict` excludes untagged rows. `tag_groups` (KG-era machinery) dropped per Off-Limits.

---

## §6 — Re-retain semantics (NEW)

### The collision case

`UNIQUE (bank_id, content_hash)` on `documents` means: if a caller retains text whose sha256 already exists for the bank, we must decide a behavior. The R1 stance was silent here; A3 and C3 demand we lock it.

### Locked behavior: replace-on-source-match, error-on-source-divergence

The caller-facing contract of `Memory.retain(text, source=..., ...)`:

1. **No existing row** for `(bank_id, content_hash)` → insert document, generate `memory_items` per spine policy (P1) or caller-supplied `index_text` (P4). Standard path.

2. **Existing row, matching `source`** (incl. both NULL) → **replace.** In one transaction:
   - `DELETE FROM memory_items WHERE document_id = (existing id)` (cascades from the document row's CASCADE — actually we delete by `document_id` directly so the document row stays).
   - `UPDATE documents SET original_text = ..., tags = ..., document_metadata = ..., retain_params = ..., updated_at = now() WHERE id = (existing id)`.
   - Re-run spine to produce fresh `memory_items` rows.
   - The document `id` is preserved (so external references stay valid); the `content_hash` is unchanged (same text); `memory_items` are rewritten.

   This handles the common case of "caller is re-retaining the same content because the spine prompt or chunking changed" — the document's identity is preserved, the spine output is refreshed.

3. **Existing row, divergent `source`** (existing has `source = '/a/file.md'`, new caller passes `source = '/b/file.md'`) → **raise `DocumentSourceConflictError`** containing the existing document id and source. Caller must explicitly resolve (delete the old document, or pick a stable source).

   The rationale: content-hash collision across different sources is either (a) a content duplicate the caller didn't realize (we should not silently merge — they would lose track of which source-of-truth the bank holds) or (b) a genuine error in the caller's bookkeeping. Silent merge corrupts the source-of-truth audit trail; we surface it.

### Edge cases named explicitly

- **`source IS NULL` on existing and incoming** → counts as match (both unknown source); replace.
- **`source IS NULL` on existing, non-NULL incoming** → source-match treated as upgrade-to-known: replace and set the source (one-time backfill). Documented behavior, not silent.
- **Non-NULL existing, `source IS NULL` incoming** → divergence. Raise.

### Storage growth implication

Replace-on-source-match means a re-retain of the same content from the same source does NOT grow storage — `memory_items` are deleted and rewritten, not appended. The C3 concern (unbounded growth) is closed by this contract: any source-stable caller (sweeper re-walks, periodic re-retain) has bounded total memory_items count per document. New content from a new source creates new documents; same content from the same source replaces in place.

### Why not no-op (preserve existing)

No-op feels safer but silently divorces the bank from caller intent. If the caller is re-retaining because their spine prompt changed, no-op leaves the bank stale with no signal. Replace gives the caller agency; the cost is one transaction per re-retain, which is cheap.

### Implementation locus

This is library logic, not schema. The `UNIQUE` constraint surfaces the collision; the library's `retain()` catches `UniqueViolation`, looks up the existing row, applies the replace-or-error rule. The schema's job is to make the constraint enforceable; the library's job is to define the verb. Documented in operator notes (plan-v2.md §5) and in the `Memory.retain` docstring.

---

**P5 (no truncation) note:** replace-on-source-match deletes memory_items but preserves `documents.original_text` (it's the source row that's matched on, not the items). P5 holds: the source content is never lost, only the LLM-anticipated index_text forms regenerated. Old `retain_events` rows preserve the prior LLM responses.

---

## §7 — Migrations (revised: advisory lock noted; pg_trgm removed)

### Hand-rolled SQL, no Alembic — locked

v0.1 uses `.sql` files in `prospecta/db/migrations/` numbered `0001_initial.sql`, etc. Applied by `prospecta-migrate` CLI, tracked via `prospecta_schema_version`. Rationale unchanged from R1: Alembic is surface; raw SQL is debuggable; P12 (honest config surface) and P9 (thin adapter) favor minimal dependency surface.

### Migration runner — advisory lock (A5)

The runner takes a Postgres session-level advisory lock at transaction start to serialize concurrent invocations:

```python
# prospecta/db/migrations.py
MIGRATE_LOCK_KEY = 0x70726F73706563  # "prospecta" 7 bytes, 56-bit, fits in signed bigint

def run_migrations(conn: psycopg.Connection) -> None:
    """Apply all pending migrations in numbered order. Idempotent. Concurrency-safe."""
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATE_LOCK_KEY,))
        # 1. CREATE TABLE IF NOT EXISTS prospecta_schema_version (...)
        # 2. SELECT MAX(version) FROM prospecta_schema_version
        # 3. For each .sql with N > current: exec; INSERT version
        # Lock auto-released at transaction end.
```

Two concurrent `prospecta-migrate` runs (operator + cron, or two pods on rolling deploy) serialize cleanly — second blocks on the lock, sees the first's committed version, applies nothing. Fifteen lines, no schema cost.

### Initial migration outline

`0001_initial.sql` ships: extensions (vector only — pg_trgm dropped), `prospecta_schema_version`, `banks`, `documents`, `memory_items`, all event tables, `sweep_passes`, `sweeper_state`, the dim-check trigger, all non-per-bank indexes, `COMMENT ON COLUMN` for spine columns.

Per-bank HNSW indexes are NOT in `0001_initial.sql` — they need `banks.embedding_dim` known at creation. Library helper builds them on first bank creation via `_create_bank_hnsw_index(conn, bank_id, dim)`.

---

## §8 — Event tables (revised: `sweep_passes` restored)

**Co-write contract (library convention, not schema-enforced):** sweeper_state and sweep_passes are updated by the same library transaction. The schema does not enforce this with a trigger because direct-SQL operators (P2: standalone library, users may speak SQL directly) may legitimately mutate one without the other for ops repair. Library-internal callers always update both; this is asserted in unit tests.

Five event-shaped tables + one state cache:

- **`retain_events`** (append-only) — every `Memory.retain()` call.
- **`recall_events`** (append-only) — every `Memory.recall()` / `recall_synth()` call.
- **`formulate_events`** (append-only) — every `formulate_queries()` call.
- **`llm_calls`** (append-only) — every LLM round-trip (prompt-named).
- **`sweep_passes`** (append-only, A4) — every sweeper pass, including errors, prune counts, duration.
- **`sweeper_state`** (state cache, upserted) — current state per `(bank_id, corpus_path)`.

### Why both `sweep_passes` AND `sweeper_state` (A4 rationale)

R1 collapsed them. R1 was wrong. The catches:

- **History question:** "When did this corpus error 4 times in a row?" — `sweeper_state` has only the last result. `sweep_passes` has the trail.
- **Prune accountability:** "What did the sweeper delete on 2026-05-10?" — current state has only the most recent count. The append-only log has the row-by-row history.
- **Failure mode debugging:** if `sweeper_state` shows last_error, the next pass overwrites it. `sweep_passes` keeps every error, indexed by `(bank_id, started_at DESC)` for time-bounded analysis.

`sweeper_state` is still useful: the operator dashboard wants "is the sweeper healthy right now?" which is one row per corpus, no scan. Both serve. The cost is ~80 LOC of migration and one append per pass.

### Why not one polymorphic `events` table (still no)

Each event type has typed columns (`raw_llm_response` only on retain/formulate; `queries` JSONB only on recall; `prompt_name` only on llm_calls; `files_seen` only on sweep_passes). Polymorphic forces JSONB-extraction for every analytics query — exactly the P5 failure mode. Typed tables win. The added `sweep_passes` doesn't change this calculus.

---

## §9 — API JSON contracts (verified: scores always-numeric)

### `RetainResponse` (unchanged)

```json
{
  "document_id": "uuid-string",
  "items_count": 5,
  "duration_ms": 234,
  "index_text_caller_supplied": false
}
```

Mimics hindsight minus async fields (sync v0.1). `index_text_caller_supplied` is prospecta-specific (P4 audit handle).

### `RecallResult` — scores always-numeric (A6 verification)

```json
{
  "id": "uuid-string",
  "content": "the index_text question form",
  "original_chunk": "the full chunk this memory_item covers",
  "source": "document_id-or-source-path",
  "score": 0.0234,
  "scores": {
    "semantic": 0.812,
    "lexical": 0.0,        // never null — COALESCE in SQL ensures 0 on missing channel (= lexical_content)
    "lexical_body": 0.0,   // never null — body channel via body_tsv (T19 three-channel)
    "rrf": 0.0234
  },
  "metadata": {"k": "v"},
  "tags": ["tag1"],
  "bank_id": "prospecta",
  "document_id": "uuid-string",
  "created_at": "2026-05-18T..."
}
```

The COALESCE in §6 guarantees `sem_score` and `lex_score` arrive at the Python layer as numeric `0.0`, not SQL NULL. The library's JSON serializer maps them to `0` in `scores.semantic` / `scores.lexical`. Consumers can disambiguate "side didn't contribute" by checking `<= 0.0`, but the JSON contract never carries `null` in these positions.

### `RecallResponse` (unchanged structurally)

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

`rrf_k` in trace reflects the actual parameter value used (now that `rrf_k` is parameterized).

### `RAGResult` (unchanged)

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

Per-query attribution preserved (auditability handle, spine-specific).

---

## §10 — Test strategy (unchanged from R1)

Three layers:

1. **CI:** `pytest-postgresql` or `testcontainers-python` for a session-scoped Postgres+pgvector container. Migrations run once at session start.
2. **Per-test isolation:** unique `bank_id` (UUID-suffixed); all assertions bank-scoped; teardown is `DELETE FROM banks WHERE bank_id = $test_bank_id` (cascades through documents, memory_items, events, sweep_passes, sweeper_state).
3. **Local dev:** docker-compose Postgres (Decision 3); `PROSPECTA_TEST_DATABASE_URL` env var override.

**Not transaction-rollback:** HNSW index build inside a transaction is suspended until commit; vector-search tests need committed writes.

The 2×2 bilateral integration test runs in this scaffold and verifies both spine directions through the full Postgres path, including hybrid retrieval (lexical-side assertion per Decision 5 amendments).

New tests added in R2:

- **Dim-check trigger test** — insert a vector of wrong dim, assert exception.
- **Re-retain semantics tests** — three cases: same source replaces; divergent source raises; NULL-source upgrade.
- **`scores` numeric-not-null test** — recall on a bank where lexical returns zero results, assert `scores.lexical == 0` not `None`.
- **Advisory-lock test** — spawn two concurrent migration runners, assert no race.

---

### Advisory-lock concurrency test (concrete spec)

`test_migration_advisory_lock_serializes`: spawn 4 subprocess workers, each opening a fresh psycopg connection to a fresh testcontainer DB and calling `prospecta_migrate.run()`. Expected: exactly one acquires the advisory lock and runs migrations; others block, observe `prospecta_schema_version` already populated, exit clean. Test passes if (a) zero rows duplicated in version table, (b) no errors propagate from workers, (c) total elapsed < 10s. Not hand-wavy — pytest-multiproc fixture exists.

---

## §11 — Hindsight lineage (unchanged from R1)

### Lifted

`banks` table identity (`bank_id` TEXT PK, `config` JSONB, `mission`/`retain_mission` TEXT); `documents` shape (minus `nodes_by_fact_type`); `memory_items` core columns (`content`, `context`, `metadata`, `document_id`, `tags`, `update_mode`); `tags_match` enum; `bank_id` multi-tenancy primitive; `bank_id_template` placeholder for `hermes-prospecta`; JSON response shape conventions.

### Diverged

No KG; `embedding` exposed; `content_tsv` exposed; `original_chunk` as distinct column (chunk inlined, not joined); `llm_generated` boolean; `banks.embedding_dim` + `embedding_model_id`; synchronous v0.1; `sweep_passes` + `sweeper_state` (hindsight has neither).

---

## §12 — PRINCIPLES audit (compact)

| # | Principle | Honored by |
|---|---|---|
| P1 | Bilateral spine | `memory_items.content` = LLM-anticipated question; `llm_generated` boolean; SQL-level `COMMENT ON COLUMN` |
| P2 | Standalone first | No Hermes dependency in schema |
| P3 | LLM as injected callable | Schema silent on provider; `llm_calls` records `prompt_name` only |
| P4 | Caller wins | `llm_generated` records spine-fired vs. caller-supplied |
| P5 | No truncation | `original_chunk`, `raw_llm_response`, `raw_response`, `original_text` — all TEXT; `COMMENT` on `original_chunk` asserts this |
| P6 | Let the LLM cook | Spine column is the question form; lexical net complements, doesn't heuristic |
| P7 | Single write path | Retain and sweeper both upsert `memory_items` identically; replace-on-source-match is one verb |
| P8 | Zero-config | Superseded by Decision 1+3 (docker compose up) |
| P9 | Thin Hermes plugin | Schema is library-owned |
| P10 | Surface adjacent mechanisms | Migration runner reuses psycopg + Postgres advisory locks |
| P11 | Tests over prose | Per-bank isolation against real pgvector; dim-trigger, re-retain, scores-numeric, lock tests |
| P12 | Honest config surface | 8 tables, no Alembic, no orphan `fts_config`, `rrf_k` parameterized but not exposed per-call |
| P13 | Prompts with library | `llm_calls.prompt_name` only |
| P14 | Sweeper safety net | `sweep_passes` append-only history makes drift + errors observable; lexical hybrid is secondary safety net |
| P15 | Spine documented | `COMMENT ON COLUMN memory_items.content / original_chunk` — schema-level documentation |

No principle violated. P15 strengthened by R2 — spine documentation lives at the SQL surface now, not only README.

---

## §13 — Open questions (shrunk)

1. **Cross-bank reflection in v0.2.** Decision 4 puts cross-bank out of v0.1 scope. A future `views` layer that unions across banks for a "reflection" persona is possible. Name in plan-v2 §"Future extensions."
2. **Generated column hardcoding `'english'`.** Per-bank FTS via trigger is a v0.2 path if multi-language users surface. Known limitation, documented.
3. **`vector_cosine_ops` vs `vector_l2_ops` vs `vector_ip_ops`.** Cosine for normalized embeddings (sentence-transformers, OpenAI); BGE outputs are also normalized. Locked cosine for v0.1; document in operator notes.
4. **`llm_calls.prompt_name` as ENUM vs TEXT.** Lean TEXT for P4 (caller may override prompt names). Acceptable.
5. **`bank_id` as TEXT (not UUID).** Hindsight convention; lets operators write `bank_id = 'prospecta-forge'`. Acceptable cost (typo-tolerant FK).

Closed since R1:
- `pg_trgm` index value → extension removed (A1-adjacent).
- RRF `k` exposure → parameterized at prepared-statement layer, `Memory(rrf_k=60)` config (C1).
- Vector dim enforcement → trigger in schema (A2).
- Re-retain semantics → §7 (A3/C3).
- `sweep_passes` vs `sweeper_state` → both, with documented roles (A4).
- Migration concurrency → advisory lock in runner (A5).
- RRF JSON NULL handling → COALESCE on scores (A6).
- Spine SQL-level documentation → `COMMENT ON COLUMN` (C5).

---

## §14 — Verdict

**APPROVE.**

R2 closes every catch on the orchestrator's must-fix list:

- A1 `banks.fts_config` removed.
- A2 dim-check trigger added; embedder-migration sequence documented.
- A3 + C3 re-retain semantics locked at replace-on-source-match with explicit error on source divergence (§7, new).
- A4 `sweep_passes` restored as append-only audit alongside `sweeper_state` current-state cache (§9, revised).
- A5 `pg_advisory_xact_lock` in migration runner (§8).
- A6 + JSON verify `COALESCE` on semantic/lexical scores; `RecallResult.scores` always-numeric (§6, §10).
- C1 `rrf_k` parameterized at prepared-statement, `Memory(rrf_k=60)` library config (§6).
- C5 `COMMENT ON COLUMN` for spine columns (`memory_items.content`, `memory_items.original_chunk`) at SQL surface (§3).
- pg_trgm adjacent — unused `CREATE EXTENSION` removed.

Off-limits items (substrate decisions, spine, KG drop, C2/C4/C6–C8) not touched.

⚒️ Canonical schema, Donald + Forge, 2026-05-18.

⚒️ Forge — Planner R2, prospecta schema stance, 2026-05-18.
