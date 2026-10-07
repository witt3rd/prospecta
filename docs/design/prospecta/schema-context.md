# Context — Prospecta Schema Design

**Domain:** `~/src/witt3rd/prospecta/` — schema slice only
**Parent design:** `plan.md` (the implementation plan)
**Parent decision:** `decision-record-1.md` (locks substrate as Postgres+pgvector, six decisions LOCKED)
**Artifact type:** Schema stance — SQL DDL outline + retrieval scoring strategy + indexing strategy
**Authored:** 2026-05-18 by Forge ⚒️

---

## The design question

What is the Postgres+pgvector schema for prospecta v0.1 that:

- Stores documents (caller-supplied content) and memory_items (the bilateral-synthesis indexable units) with `bank_id` multi-tenancy.
- Supports hybrid retrieval (semantic via pgvector + lexical via Postgres FTS) as the default recall strategy.
- Captures the tracer events (retain_events, recall_events, llm_calls, sweep_passes, formulate_events) as queryable tables.
- Honors PRINCIPLES.md (especially P1 spine, P5 no-truncation, P11 tests-as-the-lock).
- Mimics hindsight's shape where it transfers and explicitly drops what doesn't (entities, fact_types, mental models, KG).

Scope: this run produces a **schema stance**, not the full implementation plan. The stance becomes §5 of `plan-v2.md`.

---

## Inherited from decision-record-1 (NOT contested)

1. Postgres ≥ 14 with pgvector. Connection via `database_url`. No daemon.
2. Embeddings injected as `embed: Callable[[list[str]], list[list[float]]]`. Library has no embedding model opinion.
3. `bank_id` is the multi-tenancy primitive on every memory row.
4. Hindsight's shape is the seed (`banks`, `documents`, `memory_items`, retain/recall tables). Drop hindsight's KG (entities, fact_types, mental_models, observation pipelines).
5. `index_text` (the bilateral-synthesis write-side artifact) is a `memory_item.content` shape, not a separate table.
6. Hybrid retrieval = BM25 + semantic, parallel, fused. v0.1.

---

## Seed material (from `/tmp/hindsight-src/hindsight_client-0.6.1/` — read by Forge 2026-05-18)

### Hindsight schemas (verbatim field names from `hindsight_client_api/models/`)

**BankConfigResponse:**
- `bank_id: str` (PK)
- `config: Dict[str, Any]` (JSONB)
- `overrides: Dict[str, Any]` (JSONB)

**DocumentResponse:**
- `id: str` (PK)
- `bank_id: str` (FK)
- `original_text: str`
- `content_hash: Optional[str]`
- `created_at: str`
- `updated_at: str`
- `memory_unit_count: int`
- `nodes_by_fact_type: Optional[Dict[str, int]]` — DROP (KG)
- `tags: Optional[List[str]]`
- `document_metadata: Optional[Dict[str, Any]]`
- `retain_params: Optional[Dict[str, Any]]`

**MemoryItem (input shape for retain):**
- `content: str`
- `timestamp: Optional[Timestamp]`
- `context: Optional[str]`
- `metadata: Optional[Dict[str, str]]`
- `document_id: Optional[str]`
- `entities: Optional[List[EntityInput]]` — DROP (KG)
- `tags: Optional[List[str]]`
- `observation_scopes: Optional[ObservationScopes]` — DROP (KG)
- `strategy: Optional[str]` — extraction strategy, prospecta uses 1 strategy
- `update_mode: Optional[str]` — append/replace; prospecta should adopt

**RecallRequest:**
- `query: str`
- `types: Optional[List[str]]` — DROP (KG fact types)
- `budget: Optional[Budget]` — KEEP as concept (low/mid/high recall thoroughness)
- `max_tokens: Optional[int] = 4096`
- `trace: Optional[bool] = False`
- `query_timestamp: Optional[str]`
- `include: Optional[IncludeOptions]` — DROP partially (entity options drop; trace/scores keep)
- `tags: Optional[List[str]]`
- `tags_match: Optional[str] = 'any'`
- `tag_groups` — DROP (mental-model trigger machinery)

---

## Sub-dimensions the schema must address

### 1. Tables (canonical set for v0.1)

Minimum viable:

- `banks` — multi-tenancy root. `bank_id` PK, config JSONB, mission text, retain_mission text, created_at, updated_at. (Hindsight has more — mental_models, dispositions, directives. Drop.)
- `documents` — caller-supplied content, source-of-truth. Mimic hindsight's DocumentResponse minus KG fields.
- `memory_items` — the retrievable units. **This is the spine table** — `content` is the LLM-anticipated question form (write-side spine), `embedding` is its vector, `content_tsv` is its tsvector. Per-item or per-document — design question.
- `retain_events` — append-only log: bank_id, document_id, items_count, llm_generated_index_text (bool), duration_ms, raw_llm_response (if applicable), timestamp.
- `recall_events` — append-only log: bank_id, queries (JSONB array — the formulated queries), n_results, duration_ms, query_timestamp, trace JSONB.
- `formulate_events` — append-only log: bank_id, message, n_queries_out, json_mode_used, parse_fallback, raw_response, duration_ms.
- `llm_calls` — append-only log: bank_id, prompt_name, messages_count, json_mode, duration_ms, timestamp.
- `sweep_passes` — append-only log: bank_id, files_seen, files_indexed, files_pruned, errors, duration_ms.

Open: do we need a `corpus_paths` table to track sweeper state per-bank? Or is sweep state ephemeral?

### 2. memory_items shape

The load-bearing table. Decisions:

- **Granularity** — one row per `memory_item` (the bilateral-synthesis unit, i.e., one question-shaped index_text + its source), or one row per chunk of a large document? Hindsight does items. Prospecta's `index_text` frontmatter convention says one document can have multiple index_text strings (lists in frontmatter). Lean: one row per memory_item; multiple memory_items per document.
- **Spine columns** — `content` (the question-form index_text), `original_chunk` (the chunk-of-document this index_text covers — for the spine's no-truncation property), `embedding vector(N)`, `content_tsv tsvector`.
- **N for vector dimensionality.** Per-bank or per-row? Per-bank is cleaner but locks a bank to one embedder dimension. Per-row needs `embedding_dim` to verify on query. **Trade-off.**
- **FTS configuration.** `to_tsvector('english', content)` is the simplest. Multi-language? `simple` dictionary for general use? Weighted (content vs. context vs. tags)?
- **Indexes.** HNSW or IVFFlat on embedding (pgvector supports both; HNSW is the modern default but slower to build). GIN on `content_tsv`. B-tree on `bank_id`, `document_id`, `created_at`. Composite indexes for common query patterns.

### 3. Hybrid retrieval scoring

Hybrid = semantic + lexical, parallel, fused. v0.1 default.

- **Fusion strategy.** Reciprocal Rank Fusion (RRF) is the canonical Postgres-pgvector pattern: `1/(k + rank)` for each side, sum, sort. Default `k=60`. Alternative: weighted score normalization. RRF is simpler and more robust. Lean: RRF.
- **Per-side limits.** How many candidates does each side return before fusion? 50 each is a reasonable default; library default.
- **Caller override.** Caller can request `mode="semantic"`, `mode="lexical"`, `mode="hybrid"`. Default `hybrid` on `recall_synth`; `semantic` on direct `recall(queries=[...])` since the caller has already shaped queries via formulate.
- **Tag filtering.** `tags_match: 'any' | 'all' | 'any_strict' | 'all_strict'` (lifted from hindsight). 'strict' = excludes untagged; non-strict includes them. Implemented as SQL predicate.

### 4. Migration strategy

v0.1 ships a `prospecta-migrate` CLI subcommand that runs idempotent migrations. Initial migration creates all tables. Future migrations add columns / indexes / new tables. **Embrace `Alembic` or hand-rolled raw SQL migrations?** Alembic adds a dependency and infrastructure; hand-rolled is simpler for v0.1. Lean: hand-rolled SQL files in `prospecta/db/migrations/` with a version table tracking applied migrations. `psycopg.run_migrations()` helper.

### 5. JSON contract for the API layer

`Memory.retain` / `recall` / etc. take Python objects, but the API returns dicts that mirror hindsight's response shapes where transferable. This matters for the Hermes plugin's tool schemas — they need stable JSON shapes.

- `RetainResponse` → `{document_id, items_count, duration_ms}` (mimics hindsight, minus KG fields)
- `RecallResult` → `{content, original_chunk, source, score, metadata, bank_id, document_id}` (full audit trail — P5 no truncation)
- `RAGResult` (from plan §3.4) → `{synthesis, sources: [RecallResult], queries: [Query], queries_to_results: dict}`

### 6. Ops queries the schema must support

Concrete queries Donald named:

- "How many recall events in the last 7 days?"
- "What's the median formulate_queries duration?"
- "Which banks had the most retain operations this month?"
- "Show me LLM calls that took > 5 seconds."
- "Which documents have the most memory_items?"
- "JSON-mode parse failures over time."

These are SQL aggregates; schema must support them with reasonable index plans. The append-only event tables make this natural.

### 7. PRINCIPLES alignment

- **P1 (spine).** Schema's load-bearing table is `memory_items` with `content` = LLM-anticipated question form. Spine is structurally first-class.
- **P5 (no truncation).** Raw LLM responses preserved in `formulate_events.raw_response`, `retain_events.raw_llm_response`. Tracer events go to DB unmodified.
- **P11 (tests as the lock).** Every migration shipped with a SQL test that exercises it. Test schema lives in `tests/fixtures/test_db.sql` + per-test bank scoping.
- **P14 (sweeper is safety net).** Sweeper state can live in DB (last-pass-timestamp per corpus path) or ephemeral. Design choice.

---

## Required reading

Absolute paths only:

- `/home/dt/src/witt3rd/prospecta/docs/design/prospecta/decision-record-1.md` — *the six locked decisions; constraints*
- `/home/dt/src/witt3rd/prospecta/docs/design/prospecta/plan.md` — the implementation plan being amended
- `/home/dt/src/witt3rd/prospecta/PRINCIPLES.md` — fifteen principles
- `/tmp/hindsight-src/hindsight_client-0.6.1/hindsight_client_api/models/` — hindsight's schema shapes (open the key files)
- `/home/dt/src/ext/hermes-agent/plugins/memory/hindsight/__init__.py` — plugin behavior (bank_id_template etc.)
- `/home/dt/src/ext/hermes-agent/plugins/memory/hindsight/README.md` — hindsight's mode model

For hybrid retrieval / pgvector patterns, the Planner should web_search if needed for: "pgvector reciprocal rank fusion hybrid search 2025", "Postgres pgvector tsvector hybrid 2025", "HNSW vs IVFFlat pgvector benchmark 2025".

---

## Contested questions (seeds — Critic, full audit; don't stop here)

1. **(META) Are these the right sub-dimensions for a schema stance?** Is something missing — versioning, encryption-at-rest, row-level security, archival/retention policy? Or is a sub-dimension fake?
2. **Granularity of `memory_items`** — one row per index_text (lean), or per chunk? Does the bilateral spine work cleanly when one document has 5 index_text values + a 5-chunk fallback for non-frontmatter files?
3. **Vector dimensionality** — per-bank or per-row? Locks bank to one embedder vs. flexibility cost.
4. **FTS dictionary** — `english` (works for most cases, but English-only) or `simple` (no language assumption, less powerful)? Caller-configurable per-bank?
5. **HNSW vs IVFFlat for v0.1.** HNSW is modern default, IVFFlat is simpler/older. HNSW is significantly slower to build but faster to query. Build vs query trade-off.
6. **Migrations infrastructure** — Alembic (heavier, standard) vs hand-rolled SQL (lighter, custom). Trade-off.
7. **Event tables vs single events table** — separate tables (retain_events, recall_events, ...) or one events table with a `type` discriminator? Single-table is simpler ops; separate tables let each event have its own typed columns. Hindsight does separate. Lean: separate.
8. **Hybrid fusion** — RRF (canonical), weighted-score-normalization, or learned weighting? v0.1 should be simple and explainable. Lean RRF.
9. **Sweeper state in DB or ephemeral?** Persisting last-pass timestamps per corpus path makes "incremental resume after restart" cheap and observable. Cost: another table.
10. **Bank-level scoping of retrieval results** — strictly bank-scoped, or can `bank_id IS NULL` / cross-bank queries? v0.1 strict-only? v0.2 cross-bank reflection?

---

## What "done" looks like for this run

The schema stance must:

- Enumerate the v0.1 tables with full DDL outline (column names, types, nullability, indexes, comments). SQL-syntax, not prose.
- Lock the memory_items granularity (per-index_text vs per-chunk).
- Lock the vector dimensionality strategy (per-bank vs per-row).
- Lock the FTS configuration.
- Lock HNSW vs IVFFlat for v0.1.
- Lock the hybrid fusion strategy.
- Lock the migrations infrastructure choice.
- Specify the API-layer JSON contracts for retain/recall responses.
- Honor the let-slide list below.

---

## OFF LIMITS (let-slide)

The Critic must NOT re-open:

- **Postgres-vs-other-DB.** Locked decision-record-1 Decision 1.
- **Daemon-vs-direct.** Locked decision-record-1 Decision 1.
- **Embed-as-injected-callable.** Locked decision-record-1 Decision 2.
- **`bank_id` multi-tenancy.** Locked decision-record-1 Decision 4.
- **The bilateral synthesis spine.** Decided five rounds ago with Donald.
- **Hindsight's KG (entities, fact_types).** Out of scope for prospecta.

These are constraints, not contests. Surface anything else.

---

⚒️ Forge — context package for prospecta schema ralplan, 2026-05-18.
