# Decision Record 1 — Substrate Pivot: pgvector + Injected Embeddings + Hindsight-Shape Deployment

**Date:** 2026-05-18
**Author:** Forge ⚒️, with Donald
**Supersedes:** `plan.md` §2 dependency surface (chroma), Decision #2 in assessment doc (embedded chroma default).
**Status:** Six decisions LOCKED 2026-05-18. Schema design (Decision 5 below) deferred to focused ralplan; everything else folds into `plan-v2.md`.

---

## Why this exists

The original two-round ralplan landed `plan.md` against `chromadb.PersistentClient embedded` as the storage substrate. After Donald reviewed the brief, he surfaced lived-use evidence that reshapes the substrate:

1. Animus deployment ran chroma server on RTX 4090 to host its own embedding model — embeddings-as-coupled-to-DB was awkward in practice.
2. Donald prefers Postgres with pgvector for: more DB control, richer metadata, powerful queries, **keyword search as a parallel retrieval strategy**, robust logging and ops statistics.
3. **Hindsight precedent.** Hindsight uses Postgres internally; runs in three modes (cloud, local_embedded, local_external). The local + cloud parallelism is exactly what prospecta needs.

The substrate pivot is bigger than a fold-in proviso to the existing plan. The six locked decisions below capture it; the schema-design slice gets a focused ralplan because field names, types, indexes, FTS configuration, and multi-tenancy column placement deserve Planner/Architect/Critic rigor.

---

## What's inherited from prior decisions (NOT re-litigated)

- The five locked decisions from `~/forge/wiki/2026-05-18_memory-library-and-plugin-assessment.md` §4. Names (prospecta / hermes-prospecta) hold. Animus's indexable set holds. Retain (c.3) holds. Prefetch on holds.
- The bilateral-synthesis spine (PRINCIPLES.md P1) holds. Every substrate decision below honors the spine.
- The two-artifact factoring (library + thin plugin adapter) holds.
- The three-layer API (`recall(queries=[...])` low / `formulate_queries` mid / `recall_synth` high) holds.
- The 2×2 bilateral integration test holds as v0.1 ship gate.
- PRINCIPLES.md P1–P15 hold.

The chroma-embedded default (Decision #2 in the assessment) is the **only** prior decision superseded.

---

## The six decisions

### Decision 1 — Direct-to-Postgres (no daemon)

**LOCKED:** The library imports `psycopg` (sync v0.1; `asyncpg` deferred to v0.2 async API), opens a connection to Postgres, runs SQL directly. No daemon layer, no IPC, no autostart/idle-stop logic. Caller provisions Postgres however — `docker compose up` locally, Azure Flexible Server in cloud, Neon / Supabase / RDS as alternatives.

**Why this is right for prospecta (and why hindsight needs the daemon):**

Hindsight needs `hindsight-embed` because the server holds proprietary inference (embedding model, reranker, LLM extraction). Open-sourcing the schema would require open-sourcing the stack. Prospecta has no such constraint — `llm` is injected (P3) and `embed` is now injected too (Decision 2). With both injected, the daemon layer dissolves: nothing remains for it to do.

**Consequences:**

- Library deps shift: drop `chromadb`, add `psycopg[binary]>=3.1`, `pgvector` (Python bindings).
- No daemon-supervision code. No autostart. No idle-stop. The Hermes plugin is correspondingly simpler.
- Multi-process clients sharing one Postgres works natively (Postgres handles concurrency; prospecta inherits it).
- Operator concern: caller must have a Postgres reachable. We make this easy via Decision 3 (docker-compose shipped, cloud guides documented).

### Decision 2 — Embeddings as a second injected callable

**LOCKED:** Symmetric with `llm` (P3). Caller supplies `embed: EmbedCallable = Callable[[list[str]], list[list[float]]]`. Library has zero opinion about which model, where it runs, what it costs.

```python
class EmbedCallable(Protocol):
    def __call__(self, texts: list[str]) -> list[list[float]]: ...
```

**Shipped helpers** (one-line convenience for common cases, no library coupling):

```python
from prospecta.embed import sentence_transformers, openai, openai_compatible

# Local, zero-config — caller still imports sentence-transformers themselves
embed = sentence_transformers("all-MiniLM-L6-v2")

# Managed cloud
embed = openai(api_key=..., model="text-embedding-3-large")

# Donald's 4090 case: infinity-emb on a local box
embed = openai_compatible(base_url="http://4090.local:7997/v1",
                          model="BAAI/bge-large-en-v1.5")
```

**Why this is right:**

- Honors P3 symmetrically. Prospecta doesn't import `openai` / `cohere` / `sentence-transformers` directly — caller does (or uses helpers).
- Donald's 4090 use case becomes a 5-line `openai_compatible` wrapper. Library doesn't care.
- Embedding-model migration in v0.2 is a clean operation: rebuild index against new embedder, same DB.
- Dimensionality is captured per-bank (Decision 5) so two banks can use different embedders simultaneously.

**Consequences:**

- `embedding_model` parameter on `Memory` is gone (was `NotImplementedError` in plan v1). Replaced by `embed=` callable.
- `sentence-transformers` becomes an optional dep, not transitive via chroma.
- `prospecta.embed.sentence_transformers()` helper imports lazily; the library itself doesn't require it.

### Decision 3 — One library code path, three deployment shapes

**LOCKED:** The library asks for a `database_url`. That's it. Three documented deployment shapes ship with v0.1:

| Shape | Use case | Artifact |
|---|---|---|
| **Local Docker** | Dev, single-user, zero-cloud | `docker-compose.yml` at repo root: pgvector image + sane defaults |
| **Local direct** | Operator already runs Postgres | Just docs ("install pgvector extension, set DATABASE_URL") |
| **Cloud-managed** | Production, team, multi-region | `docs/deployments/azure.md` (primary, since Donald uses Azure); `docs/deployments/neon.md` (zero-effort cloud); short notes on Supabase / RDS / GCP |

**Library requires:** Postgres ≥ 14 with pgvector extension installed. Both have been GA on Azure Flexible Server since 2024.

**Hermes plugin's setup wizard** offers three presets:
- "Local Docker (recommended for development)" — generates connection string, shows `docker compose up`
- "Local direct (you provide DATABASE_URL)"
- "Cloud (you provide DATABASE_URL)"

The wizard does NOT manage Postgres lifecycle. That's an operator concern. Hindsight's `local_embedded` autostart/idle-stop pattern is intentionally NOT adopted — it requires the daemon layer Decision 1 removed.

**Why this is right:**

- Donald gets cloud path at v0.1, not deferred to v0.2.
- The 4090 + Azure split works cleanly: 4090 hosts `infinity-emb` (caller's `embed`), Azure hosts Postgres+pgvector (caller's `database_url`).
- No single-deployment-mode bias in the library. Same code talks to any Postgres.

### Decision 4 — `bank_id` multi-tenancy primitive (mimics hindsight)

**LOCKED:** Every memory row carries a `bank_id` column. Single Postgres instance can serve many prospecta banks (Cookie's substrate, Forge's substrate, project-specific banks). Hermes plugin sets `bank_id` from profile/workspace via a template pattern (lifted directly from hindsight's `bank_id_template`).

**Concrete shape** (matches hindsight's pattern):

- Default `bank_id`: `"prospecta"` (static fallback).
- `bank_id_template`: optional template like `"prospecta-{profile}"`, `"{workspace}-{user}"`. Placeholders: `{profile}`, `{workspace}`, `{platform}`, `{user}`, `{session}`. Empty placeholders collapse cleanly.
- Bank rows live in a `banks` table with config / mission / retain_mission / created_at. Caller can scope all operations to a bank.

**Why this is right:**

- Multi-tenancy from day one. Avoids the "we'll add it in v0.2 and need a migration" failure mode.
- Hindsight pattern is proven; lifting field names + config shape is convergent-conventions cheap-and-valuable.
- Hermes plugin's bank-scoping logic mirrors hindsight's, so operators familiar with hindsight transfer mental model instantly.

**Consequences:**

- Every SQL query carries `WHERE bank_id = $1`. Indexed.
- The bilateral-synthesis spine operates per-bank: index_text generation happens per-bank; formulate_queries scoped to bank.
- A "system" or "global" bank pattern (cross-bank reflection) is **out of v0.1 scope**; v0.2 question.

### Decision 5 — Schema design deferred to focused ralplan (with hindsight as seed)

**LOCKED:** A short Planner/Architect/Critic ralplan dispatched immediately after this decision record lands, scoped only to schema design. Other substrate decisions above lock without ralplan rigor (they're "match hindsight pattern except simpler" calls). Schema benefits from rigor because field names, types, indexes, FTS configuration, multi-tenancy column placement, hybrid retrieval scoring strategy are dense decisions with downstream cost.

**Seed material from hindsight-client v0.6.1** (read by Forge 2026-05-18 from `/tmp/hindsight-src/hindsight_client-0.6.1/`):

- **`banks`** table — `bank_id PK, config JSONB, mission TEXT, retain_mission TEXT, created_at, updated_at`. Hindsight's `BankConfigResponse` has `bank_id, config, overrides`.
- **`documents`** table — hindsight's `DocumentResponse`: `id PK, bank_id FK, original_text TEXT, content_hash TEXT, created_at, updated_at, tags TEXT[], document_metadata JSONB, retain_params JSONB`. Source-of-truth for retained content.
- **`memory_items`** table — hindsight's `MemoryItem`: `content TEXT, timestamp, context TEXT, metadata JSONB, document_id FK, tags TEXT[]`. The unit of retrieval. **Prospecta's `index_text` is a memory_item's content-shape question form** (bilateral synthesis is here).
- **`retain_events`**, **`recall_events`**, **`llm_calls`** tables — derived from the tracer events in plan §11. Append-only, queryable, dashboardable.
- **What we DROP from hindsight's schema:** `entities`, `entity_observations`, `fact_types`, `nodes_by_fact_type`, `mental_models`, `directives`, `dispositions`, `webhooks`. Those serve hindsight's KG + dialectic recall + observation pipelines; prospecta doesn't do those.
- **What we ADD beyond hindsight:** pgvector `embedding vector(N)` column on `memory_items` (hindsight hides the vector behind its API); Postgres FTS `tsvector` column on `memory_items.content` for the **hybrid retrieval** Donald named (BM25 + semantic, parallel). Indexes for both.

The ralplan's job is to lock the exact field types, indexes, FTS configuration (which dictionary, which weights), how the vector column dimensionality is captured (per-bank or per-row), and hybrid-retrieval scoring (RRF / reciprocal rank fusion is the default Postgres-pgvector hybrid pattern, but other choices exist).

### Decision 6 — Path forward: hybrid (γ)

**LOCKED:**

1. **Inline revise `plan.md` v1 → `plan-v2.md`** marking §2 chroma as superseded. Substrate addendum (this document's six decisions) folded into `plan-v2.md`. ~30 minutes.
2. **Dispatch focused schema ralplan** (Decision 5). 1 Planner + 1 Architect + 1 Critic, single round expected unless schema is contested. ~1-1.5 hours.
3. **Distill schema ralplan output into `schema-stance.md`**, fold into `plan-v2.md` §5 (Schema).
4. **Brief to Donald for ratification.** Once `plan-v2.md` ratified, hand off to omh-ralph-driver for execution.

---

## What this changes in the existing `plan.md`

| Section | Change |
|---|---|
| §1 Premise | Update: substrate is Postgres+pgvector, not chroma |
| §2 Dependency surface | Replace `chromadb` with `psycopg[binary]`, `pgvector`. Drop hardcoded `sentence-transformers` (now optional via helper) |
| §3 Module structure | Add `prospecta/embed/` (helpers), `prospecta/db/` (psycopg pool + queries), `prospecta/observability/` (tracer-to-postgres sink) |
| §3.4 Public API | `Memory(llm=..., embed=..., database_url=..., bank_id=..., ...)`. `embedding_model` arg gone (replaced by embed callable). |
| §5 Port order | Insert M-1 between M0 and M1: "Schema migration + smoke DB connectivity test." M2 (vertical slice) rewrites against Postgres instead of chroma. |
| §6 Milestones | Budget grows; estimate +2 days for Postgres wiring + schema migration + hybrid retrieval. New total: ~14.5 working days. |
| §7 2×2 bilateral test | Unchanged in shape, ported to Postgres. Hybrid retrieval (Decision 5) is a NEW dimension: the test grows to verify BM25-side also fires when applicable. |
| §8 LLMCallable | Unchanged. EmbedCallable added in §3.4. |
| §9 v0.1 cut lines | Hybrid retrieval moves to v0.1 (load-bearing for spine's safety net). Multi-process scope unchanged (single library instance per process; Postgres handles cross-process via its own ACID). |
| §10 hermes-prospecta section | (f) inheritances updated: Postgres setup wizard preset replaces chroma config; embed callable inherits Hermes's pattern of caller-provided callable (mirroring `ctx.llm`). |
| §11 README | Spine paragraph unchanged. Install/quickstart updated to docker-compose + DATABASE_URL. "What makes prospecta different" updated to include hybrid retrieval and pgvector. |
| §12 Tracer | Unchanged in shape. NEW: `prospecta.observability.postgres_sink` ships as the default tracer-to-DB sink, writing the six events to `retain_events`, `recall_events`, etc. P3-shaped (caller can swap). |
| §13 Concurrency | Updated: Postgres handles ACID. Per-instance `threading.RLock` retired (was a chroma-workaround). Multi-process: now supported natively. |
| §14 LLM failure-mode spec | Add `embed` callable failure modes (mirrors `llm` failures). |

⚒️ Six decisions locked, 2026-05-18.

---

## ADDENDUM — Revision to indexable file set (2026-05-19, mid-implementation)

**Context:** Reviewing the T5 port-decision sheet for `animus/memory/index.py`, Donald surfaced that we were carrying animus residue into the file-set decision. Animus indexed its own substrate including code files (`.py`, `.yaml`, `.yml`) and conversation logs in animus's specific `log.jsonl` shape (per-line entries with `index_text:` per-entry). For a generic OSS semantic memory library, those choices were animus's, not prospecta's.

### Revision: markdown-native + caller-supplied parser plugins

**LOCKED 2026-05-19.** Supersedes the assessment doc's "exactly animus's set at v1" decision.

- **Markdown native.** `.md` files with optional `index_text:` frontmatter are the library's first-class corpus shape. The bilateral spine pivots around this.
- **All other file types via caller-registered parser plugins.** The library exposes a `ParserPlugin` Protocol; callers register parsers for `.jsonl`, `.pdf`, `.txt`, or whatever their corpus contains. The library walks directories, applies ignore patterns, dispatches to the right parser by file pattern, embeds, indexes.
- **No bundled non-markdown parsers in v0.1.** `.txt`, `.pdf`, and others may ship as optional bundled plugins later. v0.1 ships markdown native plus the plugin extension point.

### Architectural symmetry — P3 extends

The library asks the caller for three things:

- `llm: LLMCallable` (already P3)
- `embed: EmbedCallable` (Decision 2)
- `parsers: list[ParserPlugin]` (this revision)

Library has no opinions about any of them. Each is caller-supplied, swappable, testable in isolation.

### ParserPlugin Protocol (file-walk-and-yield shape, locked)

```python
@dataclass(frozen=True)
class ParsedDocument:
    original_text: str
    index_text: str | list[str] | None  # optional spine-write override
    metadata: dict[str, Any]
    tags: list[str]

class ParserPlugin(Protocol):
    file_patterns: list[str]  # e.g., ["*.jsonl", "*.log.jsonl"]

    def parse(self, path: Path) -> Iterator[ParsedDocument]: ...
```

**Library owns:** directory walk, `.memoryignore` application, dispatch to parser by `file_patterns` match, embed + index.

**Plugin owns:** turn one file into one or more ParsedDocuments. That's it.

The bundled markdown parser handles `.md` files (with `index_text:` frontmatter detection per the bilateral spine convention).

### Consequences for the plan

- **`_index.py` (T9):** drops the jsonl walker entirely (was animus `_index_log_jsonl`). Drops the `.py/.yaml` chunker dispatch (was animus indexing its own codebase). Adds a small parser-dispatch layer that finds the right plugin per file pattern.
- **`prospecta._parser.py` (T8 already shipped):** the markdown-frontmatter parsing it does is the *bundled markdown ParserPlugin's* implementation. The shape is already right; we just expose it as a Plugin in T9.
- **`Memory.__init__` gets `parsers: list[ParserPlugin] | None = None`** (defaults to `[BundledMarkdownParser()]`). Callers extend.
- **Hermes plugin** can ship its own ParserPlugin for whatever conversation-log shape it consumes — animus's `log.jsonl` shape, Cookie's substrate, whatever. Not prospecta's concern.

### What this does NOT change

- The bilateral spine. Untouched.
- The schema. Untouched (`memory_items.content` is still the LLM-anticipated index_text; `documents.original_text` still preserves source).
- The `bank_id` multi-tenancy. Untouched.
- The 2×2 bilateral integration test. Test fixtures use markdown files (which were always the right shape for the test).
- T1–T7 shipped work. The chunker / parser / ignore / template / types / pool / migrate / create_bank are all generic OSS-shaped. They remain valid.

### What this DOES change in the plan-v2 / schema

- **plan-v2.md §3.9 v0.1 cut lines** — drop "exactly animus's indexable set." Replace with "markdown native + ParserPlugin extension point. No bundled non-markdown parsers in v0.1."
- **plan-v2.md §M2** — T9's vertical-slice port now drops more code than before (~250 LOC of animus-domain parsing). _index.py lands closer to 400-500 LOC, not 600.
- **PRINCIPLES.md** — consider adding P16: "Parsers as injected plugins (P3 extends)." Not strictly necessary; the existing P3 (LLM as injected callable) already gestures at the pattern.

⚒️ Addendum locked 2026-05-19. T5 port-decision sheet writes against this.

