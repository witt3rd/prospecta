---
status: canonical
date: 2026-05-18
artifact_type: implementation-plan
domain: prospecta + hermes-prospecta
supersedes: plan.md (chroma substrate)
substrate: Postgres + pgvector (per decision-record-1.md)
schema: schema.md (canonical, ralplan-blessed)
consensus_lineage:
  - prospecta-impl ralplan (Round 2 APPROVE_WITH_PROVISO, 2026-05-18, plan.md distilled)
  - decision-record-1.md (substrate pivot, 6 decisions locked 2026-05-18 with Donald)
  - prospecta-schema ralplan (Round 2 APPROVE_WITH_PROVISO, 2026-05-18, schema.md distilled)
total_estimate: ~14.5 working days (was 12.5; +2 for Postgres wiring + schema migration + hybrid retrieval)
---

# Prospecta v0.1 Implementation Plan (v2 — Postgres substrate)

This supersedes `plan.md`. The substrate pivot from chroma-embedded to Postgres+pgvector (per `decision-record-1.md`) reshapes specific sections; the spine, the API layers, the bounded plugin section, and most milestones carry forward intact.

**The bilateral synthesis spine is unchanged** — both write-side `index_text` generation and read-side `formulate_queries` are non-negotiable. What changed underneath:

- Storage: chroma → Postgres+pgvector
- Embeddings: bundled (chroma onnx) → injected callable (`embed`)
- Retrieval: vector-only → hybrid (semantic + lexical via RRF)
- Lifecycle: embedded process → external Postgres (docker-compose local / cloud-managed)
- Multi-tenancy: implicit (single collection) → explicit `bank_id` per hindsight pattern

---

## 1. Premise

Build `prospecta` v0.1 as a translated port of animus's `memory/` package re-shaped around (a) injected `llm` callable for LLM work, (b) injected `embed` callable for vector generation, (c) Postgres+pgvector as the storage substrate, (d) hybrid retrieval (semantic + lexical RRF) as the default safety net for spine drift. Bilateral synthesis (write-side `index_text` generation, read-side `formulate_queries` fan-out) is the load-bearing pattern that everything exists to support (P1, P15). v0.1 ships a pip-installable library, a small CLI, three translated prompts (renamed and generalized from Cookie-shaped originals), a README that leads with the spine, a 2×2 bilateral-synthesis integration test, and the canonical Postgres schema (`schema.md`). The Hermes plugin is derived in a bounded ≤2-page section.

---

## 2. Premises verified

1. **`animus/memory/index.py` ports with bounded surgical changes.** R1 said 3 sites; A1 corrected to 7. Now drives M0 disposition sheet. Unchanged from plan v1.
2. **`animus/rel/queries.py` `formulate_queries` is a translation, not a port.** Strengthened in plan v1. Unchanged.
3. **The three load-bearing prompts exist in animus substrate but are Cookie-shaped.** Strengthened in plan v1: M1 does **translation**, not mechanical copy. Unchanged.
4. **`ctx.llm.complete()` and `complete_structured()` give the Hermes plugin everything it needs** for both `llm` and embed-callable wrapping (the embed callable wraps caller-supplied HTTP / SDK; Hermes provides `ctx.llm` for LLM work, but **embedding** is a separate caller concern — the plugin's setup wizard configures embed via env or wizard prompt). Verified.
5. **`pgvector` ≥ 0.5 + Postgres ≥ 14** is the substrate. GA on Azure Flexible Server since 2024 (Decision 3). Verified via schema ralplan.

---

## 3. Sub-dimensions

### 3.1 Repository scaffolding

Adds: `prospecta/embed/` (helper callables — `sentence_transformers`, `openai`, `openai_compatible`), `prospecta/db/` (psycopg connection pool + queries + migrations), `prospecta/observability/` (tracer + Postgres sink). Drops: chroma server entry point.

```
prospecta/
├── __init__.py
├── _types.py                  # Memory, Query, RecalledMemory, RAGResult, LLMCallable, EmbedCallable, Tracer
├── memory.py                  # Memory class — the public API
├── _index.py                  # core indexing logic (ported from animus/memory/index.py)
├── _chunker.py                # ported from animus
├── _ignore.py                 # ported from animus (.memoryignore)
├── _parser.py                 # ported from animus (frontmatter)
├── _index_text.py             # ported from animus — half the spine
├── _formulate.py              # translated from animus/rel/queries.py — other half of spine
├── _rag.py                    # ported from animus
├── _retain.py                 # write path (file + immediate index_single_file)
├── _sweeper.py                # daemon thread, RLock-free now (Postgres handles concurrency)
├── _tracer.py                 # observability primitive (default no-op)
├── _test_helpers.py           # _debug_use_index_text monkeypatch, _build_memory_with_spine_config
├── cli/                       # argparse-based CLI (prospecta index/search/retain/config/migrate)
├── db/
│   ├── __init__.py
│   ├── pool.py                # psycopg connection pool wrapper
│   ├── queries.py             # all SQL lives here, parameterized
│   ├── migrate.py             # migration runner with pg_advisory_xact_lock
│   └── migrations/
│       ├── 0001_initial.sql
│       ├── 0002_per_bank_indexes.sql  # helper-applied, not initial
│       └── ...
├── embed/
│   ├── __init__.py
│   ├── sentence_transformers.py   # lazy-import helper
│   ├── openai.py                  # lazy-import helper
│   └── openai_compatible.py       # for infinity-emb / 4090 / etc.
├── observability/
│   ├── __init__.py
│   └── postgres_sink.py           # tracer-to-DB default sink
└── prompts/
    ├── generate-index-text.md     # was: log-index.md (translated)
    ├── formulate-queries.md       # translated
    └── rag-synthesize.md          # lightly translated
```

Adds at repo root: `docker-compose.yml` (pgvector image + sane defaults), `docs/deployments/azure.md`, `docs/deployments/neon.md`, `docs/deployments/local-direct.md`.

### 3.2 Dependency surface

**LOCKED:**

| Dep | Reason |
|---|---|
| `psycopg[binary] >= 3.1` | Postgres client (sync v0.1; async deferred) |
| `pgvector` | pgvector Python type adapters |
| `jinja2` | Prompt templating |
| `pyyaml` | Frontmatter parsing |
| `pytest` (dev) | Test framework |
| `testcontainers` (dev) | Postgres test fixtures |

**NOT a dep:** Hermes, LlamaIndex, chroma, OpenAI SDK, Anthropic SDK, Alembic, `tiktoken`, `sentence-transformers` (lazy-imported via embed helper only when used).

### 3.3 Module structure

See §3.1 tree. Net new code beyond animus port: ~150 LOC (`db/`, `embed/`, `observability/`).

### 3.4 Public API surface — REVISED for Postgres + embed callable

```python
from typing import Any, Callable, Literal, Protocol
from pathlib import Path
from dataclasses import dataclass

class LLMCallable(Protocol):
    def __call__(self, messages: list[dict], *, json_mode: bool = False) -> str: ...

class EmbedCallable(Protocol):
    def __call__(self, texts: list[str]) -> list[list[float]]: ...

@dataclass(frozen=True)
class Query:
    text: str
    metadata_filter: dict[str, Any] | None = None
    tags: list[str] | None = None
    tags_match: Literal["any", "all", "any_strict", "all_strict"] = "any"

@dataclass(frozen=True)
class RecalledMemory:
    content: str                # the LLM-anticipated index_text (spine write-side)
    original_chunk: str         # the source text (P5 no-truncation)
    source: str                 # documents.source — file path / URL / conv ID
    score: float                # RRF or single-mode score
    scores: dict[str, float]    # {"semantic": x, "lexical": y, "rrf": z} — always-numeric (COALESCE)
    metadata: dict[str, Any]
    bank_id: str
    document_id: str

@dataclass(frozen=True)
class RAGResult:
    synthesis: str
    sources: list[RecalledMemory]
    queries: list[Query]
    queries_to_results: dict[str, list[RecalledMemory]]  # per-query trace (A9 fold)

Tracer = Callable[[str, dict[str, Any]], None]
# Six events: retain, recall, formulate_queries, index_single_file, sweep_pass, llm_call

class Memory:
    def __init__(
        self,
        *,
        llm: LLMCallable,
        embed: EmbedCallable,
        database_url: str,
        bank_id: str = "prospecta",
        corpus_paths: list[str | Path] | None = None,
        retain_dir: str | Path | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        sweeper_interval_s: int = 86400,
        sweeper_enabled: bool = True,
        prompts_dir: str | Path | None = None,
        tracer: Tracer | None = None,
        rrf_k: int = 60,                  # hybrid retrieval RRF parameter
        retrieval_mode: Literal["hybrid", "semantic", "lexical"] = "hybrid",
    ) -> None: ...

    # Layer 1 — caller pre-formulated queries
    def recall(self, queries: list[Query], *, limit: int = 5,
               mode: Literal["hybrid", "semantic", "lexical"] | None = None) -> list[RecalledMemory]: ...
    def search(self, text: str, *, limit: int = 5,
               metadata_filter: dict[str, Any] | None = None,
               tags: list[str] | None = None,
               mode: Literal["hybrid", "semantic", "lexical"] | None = None) -> list[RecalledMemory]: ...

    # Layer 2 — library formulates from message
    def formulate_queries(self, message: str, *, context: str = "",
                          prompt_override: str | None = None) -> list[Query]: ...

    # Layer 3 — one call
    def recall_synth(self, message: str, *, context: str = "", limit: int = 5,
                     synth_prompt_override: str | None = None,
                     formulate_prompt_override: str | None = None) -> RAGResult: ...

    # Write side
    def retain(self, content: str, *,
               index_text: str | list[str] | None = None,
               index_text_prompt_override: str | None = None,
               tags: list[str] | None = None,
               source: str | None = None,
               path: str | Path | None = None,
               metadata: dict[str, Any] | None = None) -> str: ...
    # Returns document_id (UUID string)

    # Bank lifecycle
    def create_bank(self, bank_id: str, *, embedding_dim: int,
                    mission: str | None = None,
                    retain_mission: str | None = None) -> None: ...
    def bank_stats(self, bank_id: str | None = None) -> dict: ...

    # Maintenance
    def index_directory(self, path: str | Path, *, resume: bool = True) -> dict: ...
    def index_single_file(self, path: str | Path) -> dict: ...
    def prune_stale(self) -> dict: ...
    def rebuild(self) -> dict: ...
    def start_sweeper(self) -> None: ...
    def stop_sweeper(self) -> None: ...
    def shutdown(self) -> None: ...
```

**Changes from plan v1:**
- `embed: EmbedCallable` added (Decision 2)
- `database_url: str` replaces `index_dir` / `chroma` config
- `bank_id` added (Decision 4)
- `rrf_k`, `retrieval_mode` for hybrid retrieval
- `RecalledMemory` carries `scores: dict` and `original_chunk` (spine + P5)
- `RAGResult.queries_to_results` (A9 from plan v1 carried forward)
- `embedding_dim` gone from `__init__`; specified per-bank at `create_bank()`
- `chunk_size`, `chunk_overlap` retained (for fallback chunker on non-frontmatter files)
- Concurrency lock removed (Postgres handles)

### 3.5 Port order with test gates

Same as plan v1's vertical-slice approach, plus M-1 (schema migration) before M0:

**M-1** (Postgres bring-up, NEW) → **M0** (Port Decision Sheet) → **M1** (scaffolding + prompts + types) → **M2** (vertical slice: classical RAG end-to-end over Postgres) → **M3** (write-side spine: generate_index_text + retain) → **M4** (read-side: recall_synth) → **M5** (read-side spine: formulate_queries with JSON mode) → **M6** (sweeper + CLI + tracer + Postgres sink) → **M7** (2×2 bilateral gate) → **M8** (README + ship gate).

### 3.6 Bilateral-synthesis integration test (unchanged in shape; Postgres-backed)

2×2 matrix: OFF/OFF, ON/OFF (write-only), OFF/ON (read-only), ON/ON (full). Same corpus design, same assertions (both-on > write-only AND both-on > read-only AND both-on > off/off). Postgres makes the test cleaner: `bank_id="test_bilateral"` isolation, transactional fixture setup. **Hybrid retrieval is implicitly tested:** all four cells use hybrid mode by default; the on/off semantics refer to the spine, not to the retrieval mode.

### 3.7 Background sweeper threading

Daemon thread, `threading.Event` for shutdown. **No more RLock** — Postgres handles concurrency via row-level locks and MVCC. Multi-process safe natively (multiple library instances sharing one Postgres works; verified in §12).

### 3.8 CLI surface

`prospecta index --resume/--rebuild/--prune/--status`, `prospecta search QUERY`, `prospecta retain CONTENT`, `prospecta config`, **`prospecta migrate`** (NEW — run pending migrations), **`prospecta create-bank --id NAME --embedding-dim N`** (NEW), **`prospecta stats [--bank NAME]`** (NEW — ops queries).

### 3.9 v0.1 vs v0.2 cut lines

**v0.1 ships:**
- Everything from plan v1's v0.1 list
- **Postgres+pgvector substrate** with all 9 tables (`schema.md`)
- **Hybrid retrieval** (semantic + lexical via RRF) as default mode
- **`embed` callable injection** with 3 shipped helpers
- **`bank_id` multi-tenancy**
- **Postgres-backed tracer sink** (default observability)
- **docker-compose.yml + 3 deployment docs**
- **Bank stats / ops query CLI** (`prospecta stats`)

**v0.2 deferrals:**
- Async API
- Cross-bank queries / reflection
- Row-level security
- Sharding / multi-region
- Per-call `rrf_k` override (currently per-instance via `Memory(rrf_k=N)`)
- Caller-registered indexable file-set extensions
- Multi-language FTS dictionaries (v0.1 locks English)
- Soft-delete with `deleted_at`
- `corpus_path` column on `documents` (sweep-time scoping enhancement)

### 3.10 Milestones — REVISED

See §5.

### 3.11 `hermes-prospecta` derived section — REVISED for Postgres

See §9.

---

## 4. Port order with test gates — REVISED

**M-1 (NEW)** → **M0** → **M1** → **M2** (substantial rewrite — now SQL-backed not chroma-backed) → **M3** → **M4** → **M5** → **M6** → **M7** → **M8**.

Each milestone closes with a runnable demo + test gate. M-1 closes with `prospecta migrate` running cleanly against a docker-compose Postgres + the bilateral schema tests passing.

---

## 5. Schema

**Defer to `/home/dt/src/witt3rd/prospecta/docs/design/prospecta/schema.md`** — the canonical schema stance (ralplan-blessed, 2 rounds, ~9 provisos folded).

Headline shape: 9 tables (`prospecta_schema_version`, `banks`, `documents`, `memory_items`, `retain_events`, `recall_events`, `formulate_events`, `llm_calls`, `sweep_passes`, `sweeper_state`). `memory_items.content` is the LLM-anticipated question form (spine); `memory_items.original_chunk` preserves source text (P5 no-truncation); `memory_items.embedding vector(N)` + `memory_items.content_tsv tsvector` enable hybrid. Per-bank HNSW partial indexes, GIN on tsvector. RRF fusion at `k=60` (configurable per-instance). Hand-rolled SQL migrations with `pg_advisory_xact_lock` serialization.

Read `schema.md` for: full DDL, index strategy, hybrid retrieval SQL, re-retain semantics, migrations machinery, API JSON contracts, test strategy.

---

## 6. Milestones — REVISED (10 milestones, ~14.5 days)

### M-1 — Postgres bring-up + schema migration (1.5 days, NEW)

- Author `docker-compose.yml` at repo root: pgvector image (`pgvector/pgvector:pg16` or equivalent), volume, sane port, healthcheck.
- Implement `prospecta/db/pool.py`: psycopg connection pool wrapper, `database_url` from env / config.
- Implement `prospecta/db/migrate.py`: migration runner with `pg_advisory_xact_lock(0x70726F73706563)` serialization (`schema.md` §7).
- Author `0001_initial.sql`: all 9 tables, indexes, triggers per `schema.md` §1 (the canonical DDL).
- Implement `Memory.create_bank()`: row in `banks` + per-bank HNSW partial index (`schema.md` §3).
- **Demo:** `docker compose up -d && prospecta migrate && python -c "from prospecta import Memory; m = Memory(...); m.create_bank('test', embedding_dim=384)"` and the bank row + per-bank index exist.
- **Gate:** `tests/integration/test_migrations.py` (idempotent, advisory-lock concurrency per `schema.md` §10), `tests/integration/test_bank_lifecycle.py` (create / stats / migration of embedding_dim).

### M0 — Port Decision Sheet (0.5 day, unchanged)

Per plan v1: enumerate every `animus/memory/index.py` + `animus/rel/queries.py` entanglement site with disposition. Critic-reviewed before M1 starts.

### M1 — Scaffolding + prompt translation + types (2.5 days, unchanged)

Per plan v1: prompt **translation** (not copy) for `generate-index-text.md`, `formulate-queries.md`, `rag-synthesize.md`. `_types.py` adds `EmbedCallable` Protocol now.

### M2 — Vertical slice: classical RAG over Postgres (4 days, WAS 3.5 for chroma)

Per M0 dispositions: port `_chunker`, `_parser`, `_ignore` (mechanical, ~340 LOC). Port `_index.py` translated to **psycopg SQL via `db/queries.py`** instead of chroma calls (this is the substantial rewrite — chroma's `upsert` becomes SQL `INSERT ... ON CONFLICT`, chroma's `query` becomes the hybrid RRF SQL from `schema.md` §5). `Memory.__init__`, `Memory.index_directory`, `Memory.search` working against Postgres. **No spine yet.**

**Demo:** `prospecta index --path tests/fixtures/bilateral_corpus/ --rebuild && prospecta search "kelly birthday"` returns chunks via hybrid retrieval.

**Gate:** `tests/integration/test_index_end_to_end.py` (Postgres-backed), plus `tests/integration/test_hybrid_modes.py` (semantic-only, lexical-only, hybrid each return expected shapes; scores numeric per A6 / `schema.md` §10).

### M3 — Write-side spine: `generate_index_text` + `retain` (1 day, unchanged)

Per plan v1, Postgres-backed. `retain` writes file + creates `documents` row + `memory_items` rows + appends `retain_events`. Re-retain semantics per `schema.md` §6 (replace-on-source-match).

**Gate:** `tests/integration/test_retain_roundtrip.py` includes re-retain semantics (three cases per `schema.md` §6).

### M4 — Read-side: `recall_synth` (1 day, unchanged)

Per plan v1. Postgres-backed: synthesis uses hybrid retrieval by default.

### M5 — Read-side spine: `formulate_queries` with JSON mode (1 day, unchanged)

Per plan v1. JSON-mode + replaces M4's stub.

### M6 — Sweeper + CLI + Tracer + Postgres sink (1.5 days, WAS 1)

Per plan v1 + Postgres sink wiring: `prospecta/observability/postgres_sink.py` writes the 6 tracer events to `retain_events` / `recall_events` / `formulate_events` / `llm_calls` / `sweep_passes` / `sweeper_state`. Default tracer if `tracer=None` and `database_url` is configured (P3-shaped: caller can still inject custom tracer).

**Demo:** Start REPL with sweeper on + default Postgres sink; drop a file in another shell; observe rows appear in event tables.

**Gate:** `tests/integration/test_sweeper.py` + `tests/integration/test_postgres_sink.py`.

### M7 — The bilateral-synthesis gate (1.5 days, unchanged)

Per plan v1: 2×2 matrix, Postgres-backed. **NEW:** also tests hybrid retrieval correctness — verify that when index_text drifts (spine semi-fails), lexical fallback still returns the right document. This is the hybrid-as-safety-net validation.

### M8 — README + ship gate (0.5 day, unchanged)

Per plan v1's spec (`schema.md` informs the "what makes prospecta different" section: now includes pgvector + hybrid + bank_id_template).

**Total: ~14.5 working days for one engineer** (was 12.5 for chroma; +2 for M-1 Postgres bring-up + M2 SQL translation + M6 Postgres sink).

---

## 7. Bilateral-synthesis integration test (unchanged in shape, Postgres-backed)

See plan v1 §7 — 2×2 matrix, same assertions, same corpus design. Implementation now uses `bank_id="test_bilateral"` for isolation; testcontainers spins a fresh Postgres per test session.

The hybrid retrieval adds a related test (different from the bilateral 2×2): `test_hybrid_safety_net` — given a corpus where index_text deliberately drifts from query language, lexical (BM25 via `content_tsv`) recovers the correct document via RRF fusion. This validates the "spine + safety net" composition.

---

## 8. `llm` callable + `embed` callable signatures

**`LLMCallable`** unchanged from plan v1:

```python
class LLMCallable(Protocol):
    def __call__(self, messages: list[dict], *, json_mode: bool = False) -> str: ...
```

**`EmbedCallable`** (NEW per Decision 2):

```python
class EmbedCallable(Protocol):
    def __call__(self, texts: list[str]) -> list[list[float]]: ...
```

**Shipped helpers** (lazy-imported, no top-level deps):

```python
from prospecta.embed import sentence_transformers, openai, openai_compatible

# Local zero-config
embed = sentence_transformers("all-MiniLM-L6-v2")  # 384-dim

# Managed cloud
embed = openai(api_key="sk-...", model="text-embedding-3-large")  # 3072-dim

# Donald's 4090 case
embed = openai_compatible(
    base_url="http://4090.local:7997/v1",
    model="BAAI/bge-large-en-v1.5",
)  # 1024-dim
```

Each helper returns an `EmbedCallable` that batches and adds reasonable error handling. The library never imports the underlying SDK directly.

---

## 9. v0.1 vs v0.2 cut lines

See §3.9. Headline: hybrid retrieval, embed-as-injected, bank_id, Postgres sink, three deployment shapes all v0.1. Cross-bank, async, soft-delete, multi-language FTS deferred.

---

## 10. `hermes-prospecta` derived section (≤2 pages, 6 items)

Items (a)–(e) unchanged from plan v1 except (a) and (d) updated for Postgres.

### (a) `__init__.py` skeleton outline (Postgres-backed)

```python
from agent.memory_provider import MemoryProvider
from prospecta import Memory
from prospecta.embed import sentence_transformers, openai, openai_compatible

class ProspectaMemoryProvider(MemoryProvider):
    def __init__(self):
        self._memory: Memory | None = None

    def initialize(self, session_id, **kwargs):
        hermes_home = kwargs["hermes_home"]
        cfg = self._load_config(hermes_home)
        llm = self._build_llm_callable(self._ctx)        # wraps ctx.llm.complete / .complete_structured
        embed = self._build_embed_callable(cfg)          # resolves "embed_provider" config to helper
        self._memory = Memory(
            llm=llm,
            embed=embed,
            database_url=cfg["database_url"],
            bank_id=self._resolve_bank_id(cfg, kwargs),  # hindsight-style template
            tracer=None,  # uses default Postgres sink
            ...
        )

    def prefetch(self, query, *, session_id=""):
        return self._memory.recall_synth(message=query).synthesis

    def get_tool_schemas(self): return [TOOL_RECALL, TOOL_RETAIN, TOOL_SEARCH]

    def handle_tool_call(self, name, args, **kwargs):
        if name == "recall":   return self._memory.recall_synth(**args).synthesis
        if name == "retain":   return self._memory.retain(**args)
        if name == "search":   return [_to_dict(m) for m in self._memory.search(**args)]

    def sync_turn(self, user, assistant, *, session_id=""): pass  # agent-driven retain
    def shutdown(self): self._memory.shutdown()
```

### (b) `plugin.yaml`

```yaml
name: prospecta
description: Bilateral LLM-mediated memory with hybrid retrieval over Postgres+pgvector.
hooks:
  - prefetch
  - shutdown
config_schema_version: 1
```

### (c) `cli.py` subcommand map

```python
def register_cli(subparser):
    # hermes prospecta migrate
    # hermes prospecta create-bank --id NAME --embedding-dim N
    # hermes prospecta index [--rebuild|--prune|--status]
    # hermes prospecta search QUERY
    # hermes prospecta stats [--bank NAME]
    # hermes prospecta config
```

Each subcommand delegates to `prospecta.cli.<name>(args)`.

### (d) `get_config_schema()` field list

≤6 fields prompted at `hermes memory setup`:

1. `deployment_mode`: choice — `local_docker` (recommended) / `local_direct` / `cloud`.
2. `database_url`: secret — Postgres connection string (the wizard generates a default for `local_docker`).
3. `embed_provider`: choice — `sentence_transformers` / `openai` / `openai_compatible`.
4. `embed_config`: shape depends on provider (`model_name` for st; `api_key + model` for openai; `base_url + model` for openai_compatible).
5. `bank_id_template`: optional — defaults to `"prospecta-{profile}"`. Hindsight-style placeholders.
6. `prefetch_enabled`: bool, default `True`.

All other settings (chunk_size, rrf_k, sweeper_interval_s, etc.) live in `~/.hermes/prospecta/config.json`.

### (e) Integration test approach

Mirror `tests/agent/test_memory_plugin_e2e.py` (hindsight's pattern). Spin testcontainer Postgres, instantiate `ProspectaMemoryProvider` with a mock `ctx.llm`, walk through: initialize → prefetch (no memory yet) → retain via tool → prefetch (recalls retained) → shutdown.

### (f) Decisions inherited from settled context

The plugin does NOT invent these; they fall out of the locked decisions:

- **Prefetch is on by default** (inherited from Decision 5 / plan v1 P5).
- **`sync_turn` is a no-op** (inherited from agent-driven retain model).
- **Error semantics propagate**: `retain` failures bubble; `prefetch` failures caught and logged (Hermes contract).
- **Per-session scoping inherits hindsight's pattern** — `MemoryManager` handles session isolation; prospecta is session-agnostic.
- **Agent-context awareness** (`primary` / `subagent` / `cron`) inherits hindsight's gating: prefetch enabled in `primary` only.
- **Tool schemas** follow hindsight's JSON-schema shape.
- **Setup wizard inherits the standard `hermes memory setup` flow** — six questions matching the schema in (d).
- **CLI delegations** are one-line per subcommand.
- **`bank_id_template`** inherits hindsight's template syntax verbatim (`{profile}`, `{workspace}`, `{platform}`, `{user}`, `{session}` placeholders).
- **`embed_provider` setup parallels hindsight's `llm_provider` setup** — same wizard UX shape.

**Implementer instruction:** read hindsight's `__init__.py` once for lifecycle patterns. The inheritances above name what carries forward without redesign. Plugin LOC budget: 300–500 LOC. Hard ceiling.

---

## 11. README.md content spec — UPDATED for Postgres

Same structure as plan v1 §11, with the following content updates:

- **Opening (200 words):** unchanged framing of the spine.
- **Install + Quickstart:** now leads with `docker compose up -d`, then `pip install -e . && prospecta migrate && prospecta create-bank --id my-bank --embedding-dim 384`, then three lines of Python.
- **API examples:** the 3-layer API code snippets, with `embed=` injected.
- **What makes prospecta different:** updated to include — bilateral LLM-mediated retrieval (unique), hybrid retrieval (semantic + lexical via RRF), Postgres + pgvector (more powerful than chroma for ops queries), embed-as-injected-callable (no provider lock-in symmetrical with `llm`).
- **Deployment options:** docker / cloud (Azure / Neon / Supabase / RDS) / local-direct.
- **Caveats:** v0.1 sync only, single-language FTS (English), embedding migration via bank rebuild.

---

## 12. Tracer / observability — REVISED (Postgres sink as default)

The tracer primitive (six named events, default no-op) is unchanged in shape from plan v1. **NEW:** `prospecta.observability.postgres_sink.PostgresSink(database_url, bank_id)` ships as the default sink when no caller-supplied tracer is provided and `database_url` is configured.

The Postgres sink writes:
- `"retain"` → `retain_events`
- `"recall"` → `recall_events`
- `"formulate_queries"` → `formulate_events`
- `"index_single_file"` → `sweep_passes` (per-file row) + `sweeper_state` (cache update)
- `"sweep_pass"` → `sweep_passes` (pass-level row) + `sweeper_state`
- `"llm_call"` → `llm_calls`

Schema reflects `schema.md` §1 tables. Operator queries (e.g., `prospecta stats`, ad-hoc SQL) read from these.

P3 honored: caller can override with custom tracer; default Postgres sink does not import any non-prospecta-internal libraries.

---

## 13. Concurrency contract — REVISED (Postgres handles it)

Plan v1's per-Memory `threading.RLock` (workaround for chroma's single-process embedded mode) is **retired**. Postgres handles concurrency via row-level locks, MVCC, and the per-statement transaction semantics.

**Contract:**
- A `Memory` instance is safe to call from multiple threads. Chroma's lock is gone.
- **Multiple library instances in different processes sharing one Postgres** is supported natively. Postgres ACID handles it.
- **Reads are concurrent.** Hybrid retrieval SQL runs at READ COMMITTED isolation.
- **Writes (retain, index_single_file)** wrap in transactions. Concurrent writes to the same `(bank_id, content_hash)` row resolve via `INSERT ... ON CONFLICT` semantics per `schema.md` §6 (replace-on-source-match).
- **Sweeper-vs-retain race:** Postgres's row-level locks serialize at the `documents` / `memory_items` level. If `retain` and sweeper try to upsert the same row, one wins; the other re-checks and may no-op or replace.
- **Migration concurrency:** `pg_advisory_xact_lock` serializes per `schema.md` §7.

---

## 14. `llm` + `embed` callable failure-mode spec

`llm` failure modes unchanged from plan v1 (8 paths, malformed-JSON + schema-mismatch both preserve raw response per A9/P1/P2 folds).

**`embed` callable failure modes (NEW):**

| Operation | `embed` failure | Behavior |
|---|---|---|
| `retain` | `embed` raises | Exception propagates. Document NOT written. Caller decides retry. |
| `index_single_file` | `embed` raises | Same as `retain` for hot path. For sweeper path: skip + log warning + record in `sweep_passes.errors`. |
| `formulate_queries` returns queries → embed each for recall | `embed` raises | Exception propagates to `recall_synth` caller. |

Documented in `Memory` class docstring and README caveats.

---

## 15. Open questions

1. **Wizard UX for `embed_provider`.** Six top-level providers (st, openai, openai_compatible, plus future cohere/voyage/etc.). v0.1 ships three helpers; wizard prompts cascade based on choice. Risk: too many sub-questions in setup. Mitigation: defaults sensible for each.
2. **`bank_id` resolution for shared Postgres** when multiple Hermes profiles point at one DB. Hindsight's `bank_id_template` handles this; prospecta inherits the template syntax. Donald's case: one Postgres serving Cookie's bank + Forge's bank simultaneously. Test this case in M-1.
3. **`pgvector` minimum version pin.** pgvector 0.5 added HNSW support; 0.7 added some perf improvements. Pin: `pgvector >= 0.5`. Verify against Azure Flexible Server's shipped version.
4. **Schema test fixtures across milestones.** M2/M3/M5/M6/M7 all need a clean Postgres + migrations + a bank. Helper: `tests/conftest.py` provides a session-scoped testcontainer + `bank_id` per test. Per `schema.md` §10.
5. **The 7th `index.py` entanglement site** — carry forward from plan v1. M0 sheet surfaces it.
6. **Azure deployment doc accuracy.** Donald uses Azure; needs the deployment doc to be live-tested against Flexible Server during M8.

---

## 16. Verdict

**APPROVE_WITH_DEFERENCE-TO-USER-RATIFICATION.**

Plan v2 closes the substrate pivot per `decision-record-1.md` + schema ralplan + 5-round prior work. Two ralplans behind it (prospecta-impl, prospecta-schema), both Round-2 consensus. The total budget is 14.5d, honestly grown +2d from chroma plan v1's 12.5d. The spine, the bounded plugin section, and the bilateral integration test all carry forward intact.

Ready for omh-ralph-driver execution on user sign-off.

⚒️ Forge — canonical plan-v2, 2026-05-18.
