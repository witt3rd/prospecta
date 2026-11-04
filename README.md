<p align="center"><img src="docs/assets/prospecta-logo.png" alt="Prospecta" width="320"></p>

# Prospecta

**Long-term memory for agents, on Postgres.** Prospecta stores what an agent (or a person) writes as verbatim notes cut into chunks, finds them again with one native Postgres query plus a cross-encoder, and hands back **cited evidence**: every hit says where it is from, when, why it matched, and quotes the matching words.

- **Write path: chunks, no model.** `retain` cuts a note into chunks of at most 1000 characters, embeds them and writes them, in one transaction. The note is kept verbatim. Nothing calls a model at write time.
- **Stage 1: native Postgres.** One query runs three retrievers over the chunks, each with its own reserved slots: semantic (pgvector HNSW, top 25), stemmed lexical (top 20) and exact / phrase lexical (top 20; `"quoted phrases"`, identifiers, `v1.14`). They fuse into the top 50 with an `exact_snippet` (`ts_headline`) per hit, with the date window, anti-echo list and scope pushed into every retriever.
- **Stage 2: cross-encoder.** `qwen/qwen3-reranker-8b` (through OpenRouter) re-scores every candidate against the query with a calibrated 0..1 score; a floor, a near-copy collapse and a timeout fallback to the Stage 1 order keep it safe.
- **Stage 0 and Stage 3 (optional gates).** A decision model skips acknowledgements and live-state questions before a chat message costs a recall (Stage 0); an optional sufficiency check lets `ask` widen its search once (Stage 3, off by default).
- **The cited-evidence contract.** `source`, `created_on`, `kind`, `match_kind`, `relevance_score`, `exact_snippet`, and the full chunk, on every hit and in every block handed to a model (quoted material, never instructions).

The design is `docs/postgres-hybrid.md`; it is the one normative description.

## Three ways to ask

| | mode | entry points | what runs |
|---|---|---|---|
| **Mode 1** | automatic recall for a chat message or a task brief | `Memory.expand_and_recall`, `Memory.recall_for_task`, `POST /lookup {"task": ...}` | Stage 0 gate (chat only) → a query writer turns the prompt into N structured searches (`query`, verbatim `terms`, a period) → every search through Stage 1 and Stage 2 in parallel → one merged, cited block, inside a 7 s deadline |
| **Mode 2A** | direct lookup: keywords, phrases, filters | `Memory.search`, `Memory.recall`, `prospecta search`, `POST /lookup {"questions": ...}` | Stage 1 + Stage 2, no gate, no writer |
| **Mode 2B** | a question answered from memory | `Memory.ask` | Stage 1 + Stage 2, optional Stage 3, then an answer with `[n]` citations tied to `[source \| date] "verbatim snippet"` |

## Status

**v1.0.** Release notes: `CHANGELOG.md`. A bank created fresh runs all of the above with no configuration: `Memory.create_bank` (so `prospecta init`, `create-bank`, the service and `migrate-bank`) stores `channel_config = [hybrid_chunk]` and a `cross_encoder` rerank (`prospecta/bank_defaults.py`). There is one design and one code path: a bank whose `channel_config` is empty runs `hybrid_chunk` with its defaults, and anything a stored config names that the library no longer has is ignored. Tests: `tests/` (Postgres-backed tests use testcontainers, or `PROSPECTA_TEST_PG_URL`: `docs/testing.md`).

Substrate: Postgres with the `pgvector` extension, **0.8 or newer** (Stage 1 uses iterative index scans). Models: an embedder and an `llm` are injected callables; the Stage 2 reranker and the decision model are reached over HTTP with an `OPENROUTER_API_KEY`. Without that key retrieval still works: the Stage 1 order stands instead of a rerank and a chat message is not gated (the result says so).

---

## Install

```bash
pip install prospecta                       # the library and the CLI (this also installs litellm)

# Capable defaults via LiteLLM (any provider it supports)
pip install 'prospecta[defaults]'

# Or single-provider minimal-deps embedders
pip install 'prospecta[embedder]'                      # local sentence-transformers, no API key
                                                       # (embed-sentence-transformers is a backward-compatible alias)
pip install 'prospecta[embed-openai]'                  # direct OpenAI HTTP
pip install 'prospecta[embed-openai-compatible]'       # infinity-emb, vLLM, etc.
```

The library code imports no provider SDK; `prospecta.defaults` and `prospecta.embed.*` do (`PRINCIPLES.md`, P3, P17).

## Quickstart

Three commands from `pip install` to the first useful recall, once the environment is set (run from a checkout that has the bundled `docker-compose.yml`; `.env.example` has the two database variables):

```bash
export OPENAI_API_KEY="sk-..."
export POSTGRES_PASSWORD=changeme
export DATABASE_URL="postgresql://prospecta:changeme@localhost:5432/prospecta"
prospecta init                    # docker compose up + migrate + create the default bank
prospecta retain "Kelly was born March 4th 1990" --source kelly.md
prospecta search "Kelly birthday"
```

`prospecta init` is idempotent — re-runs detect already-up containers, already-applied migrations, and the already-created default bank. For a database you already have, `prospecta init --no-substrate` with its `DATABASE_URL`. `docs/deployment.md` covers Docker, a local Postgres, Neon and Azure.

### Offline / keyless retain

The CLI embedder is selectable via `PROSPECTA_EMBEDDER`. `retain` makes no model call, so the write path needs **no API key** with local sentence-transformers embeddings:

```bash
pip install 'prospecta[embedder]'
prospecta create-bank --id local --embedding-dim 384   # all-MiniLM-L6-v2 is 384-dim
PROSPECTA_EMBEDDER=sentence-transformers \
PROSPECTA_BANK=local \
prospecta retain "Kelly was born March 4th 1990" --source kelly.md
```

`PROSPECTA_EMBEDDER` accepts `default`/`litellm` (the LiteLLM multi-provider default) or `sentence-transformers`/`st` (offline). `PROSPECTA_EMBED_MODEL` selects the model id (sentence-transformers default `all-MiniLM-L6-v2`).

**One default embedder:** the LiteLLM default is `openai/text-embedding-3-large` cut to **1536** dimensions, also what `migrate-bank` writes and the width of new banks. The CLI reads the bank's `embedding_dim` and embeds queries at that width (`PROSPECTA_EMBED_DIM` overrides); `prospecta health` prints `PROSPECTA DOWN` if the embedder's width differs from the bank's. A bank's `embedding_dim` is fixed at creation and must match the embedder's output dimensionality: a sentence-transformers bank is **not** interchangeable with an OpenAI (1536-dim) bank.

With `OPENROUTER_API_KEY` set the CLI also builds the Stage 2 rerank transport, the decision model (Jev, System One) and the accounted LLM that writes the `ask` answer. `PROSPECTA_JEV=off` opts out of Jev.

**Note dates:** every retained note gets `documents.created_on`: its own date (frontmatter `created` / `created_on` / `date`, or the `metadata=` kwarg), else today's date in the zone named by `PROSPECTA_TIMEZONE` (an IANA name such as `America/Los_Angeles`; UTC when unset).

### Health check

`prospecta health` probes the database (`DATABASE_URL`) and the configured embedder (including that `sentence_transformers` imports when selected). It prints one line and exits 0 with `PROSPECTA OK: ...` (naming the embedder probed), or exits 1 with `PROSPECTA DOWN: ...`.

---

## The library

```python
from prospecta import Memory
from prospecta.defaults import make_default_embedder, make_default_llm
from prospecta.stages import JevScore, openrouter_jev_transport, openrouter_rerank_transport

m = Memory(
    database_url="postgres://localhost:5432/prospecta",
    bank_id="default",
    embed=make_default_embedder(),                    # or your own: (list[str]) -> list[list[float]]
    llm=make_default_llm(),                           # or your own: (messages, *, json_mode, model=None) -> str
    jev=JevScore(openrouter_jev_transport()),         # Stage 0 gate and Stage 3 (optional)
    rerank_transport=openrouter_rerank_transport(),   # Stage 2 (optional)
)

m.create_bank("default", embedding_dim=1536)   # once (prospecta init does it); a new bank needs no more configuration

# Write: the note is cut into chunks and embedded; no model call
m.retain("Kelly was born March 4th 1990. Her birthday party in 2026 was epic.",
         source="kelly.md", tags=["family", "kelly"], metadata={"created_on": "2026-03-05"})

# Mode 2A: search. Phrases, OR, -exclude and identifiers work; the filters apply inside Stage 1
hits = m.search('"electric dreams" 0x80070005', date_from="2024-01-01", excluded_sources=["turn:41"])
hits[0].source, hits[0].created_on, hits[0].match_kind, hits[0].relevance_score, hits[0].exact_snippet

# Mode 2B: ask. Every statement carries [n]; optional Stage 3 widens the search once if the evidence falls short
res = m.ask("how are deletes handled?", sufficiency=True)
print(res.rendered)       # the answer, then   [n] [source | date] "verbatim supporting snippet"

# Mode 1: automatic recall, a chat message or a task brief
got = m.expand_and_recall("do you remember who I saw it with?", agent_last="You mentioned the Spring show.",
                          excluded_sources=["turn:41", "turn:42"], date_from="2024-01-01")
got = m.recall_for_task(title="Library tree view", ask="Build it", done="folders page in", project="atlas")
print(got.block)          # the searches that ran, "cite [n]", then the numbered items
got.searches, got.items, got.fallbacks, got.timings_ms, got.accounting    # what ran, what was found, what degraded, ms, tokens and cost
```

**Mode 1 in five lines** (`docs/postgres-hybrid.md`, section 1, is the design; `docs/recall-stages.md` has every setting):

1. **Gate** (chat only; a task is never gated): the decision model's `needs_memory` skips acknowledgements, go-aheads, greetings and live-state questions (at or above 0.5 it goes on). `decision=callable` injects your own; `gate=False` switches it off.
2. **Writer**: about 3 searches for a short message, up to 10 to 20 for an elaborate one, at most `max_searches` (20). Each is `{query, terms, date_from, date_to}`: `query` feeds the semantic and stemmed retrievers and the reranker; `terms` (**only** proper names, identifiers, versions, paths, error codes and "quoted phrases", verbatim from the prompt, never common words: a deterministic filter drops a plain English word the prompt did not quote) feed the exact retriever; a period is pushed into Stage 1. A name, identifier or quoted phrase of the prompt that no search carries is put back into the first one. A bad reply falls back to one search: the prompt itself. If the filter drops every search, nothing is asked (a skipped, empty result).
3. **Stage 1 and 2 in parallel**, each search against its own query, with the same filters; at most `max_concurrency` (8) searches in the database at once.
4. **Merge**: one record once (best score, `found_by` the searches), near-copies collapsed (by wording, then by meaning: chunk-embedding cosine of 0.90 or more; the kept record lists what collapsed into it), best first, a task's own project first; sized by relevance, never by a count.
5. **Never raises on a provider failure and stays inside the deadline** (7 s): a step that fails or runs late degrades and is named in `fallbacks`.

Mode 1 runs on every bank: it needs Stage 1's `hybrid_chunk`, which a bank with an empty `channel_config` runs with its defaults. (Only a config that disables it, `{"name": "hybrid_chunk", "enabled": false}`, makes `expand_and_recall` return an `error`.)

Other entry points: `Memory.index_directory` (walks a corpus; frontmatter `created`, `person`, `source_kind` become the note's date, person and kind), `Memory.start_sweeper` (drift safety net), `Memory.set_channel_config` / `set_recall_config` (`docs/recall-stages.md`), `Memory.bank_stats`. The `SourceIndexer` indexes a host's records in place from database views (`docs/source-contract.md`).

## The CLI

```bash
prospecta init                           # docker compose up + migrate + create the default bank
prospecta migrate                        # apply pending migrations
prospecta create-bank --id alice --embedding-dim 1536
prospecta migrate-bank --source-bank alice   # re-embed alice into alice-v2: another embedder or dimension (docs/embedding-migration.md)
prospecta retain "content..." --source X --tags a,b      # or "@path/to/file"
prospecta index ./corpus                 # walk a directory; chunk + embed
prospecta search "query" [--limit N] [--depth standard|deep] [--json]
prospecta eval questions.md [--ablate]   # score recall; docs/eval.md
prospecta import hindsight postgres://...   # docs/import-hindsight.md
prospecta sweep ./corpus --watch         # background sweeper for filesystem drift
prospecta stats                          # bank / document / event counters
prospecta health                         # database + embedder; non-zero if down
prospecta config                         # show loaded config (passwords redacted)
prospecta backup FILE / restore FILE     # docs/backup-restore.md
```

The global options are `--database-url` (else `DATABASE_URL`) and `--bank` (else `PROSPECTA_BANK`, default `prospecta`).

## The HTTP service

`python -m prospecta.service` (or the `Dockerfile`): stdlib HTTP, JSON.

- `POST /lookup {"questions": [...], "limit": 10, ...}` — Mode 2A for each question: ranked records, each with its citation fields (`created_on`, `match_kind`, `exact_snippet`, `relevance_score`).
- `POST /lookup {"task": {"title", "ask", "done"?, "project"?}}` — Mode 1: the cited context block plus `searches`, `results`, `fallbacks` and `error`. No `limit`: the block is sized by relevance.
- `POST /index`, `/remove`, `/reconcile`, `GET /coverage`, `GET /healthz` — in-place indexing of a host's records.

Filters on a lookup: `date_from`, `date_to`, `excluded_sources` (the anti-echo list), `kind`, `project`, `person`, `allowed_audience_keys`, `allowed_projects`; per-call `reranker_model` and `decision_model`. The contract, shared-database mode and every environment variable are in `docs/source-contract.md`.

## Observability

Every load-bearing event is recorded in Postgres event tables, a durable audit trail rather than ephemeral stderr: `retain_events`, `recall_events` (the queries, the surfaced notes, the per-stage trace with the rerank's cuts and fallback reason, cost and tokens) with `recall_event_candidates`, `llm_calls` (every model call with its purpose, model, tokens, cost and, opt-out, the verbatim prompt and response), `sweep_passes` and `sweeper_state`. `docs/observability.md` is the inspection guide with the `psql` cheat sheet, the tracer, and the opt-outs.

[`tui/`](tui/README.md) is a Rust + Ratatui terminal UI that makes the trace legible without SQL: bank and document browsing, a live event stream with end-to-end thread views, a per-bank health dashboard, lexical search and a manual retain form. Install with `cargo install --path tui`.

---

## Documentation

| | |
|---|---|
| [`docs/postgres-hybrid.md`](docs/postgres-hybrid.md) | **The architecture**: the modes, the stages, the lexical engine, the SQL, the call sequence |
| [`docs/recall-stages.md`](docs/recall-stages.md) | `channel_config` / `recall_config`: Stage 1 params, Stage 2, the models, Mode 1 settings, Stage 3, depth, the cited-evidence contract |
| [`docs/schema.md`](docs/schema.md) | the tables, indexes, write semantics and migrations 0001 to 0020 |
| [`docs/limits.md`](docs/limits.md) | every numeric limit and why it is there |
| [`docs/observability.md`](docs/observability.md) | the event tables, the `psql` cheat sheet, tracers |
| [`docs/bank-isolation.md`](docs/bank-isolation.md) | why one bank's size no longer slows another: the per-bank HNSW and GIN indexes |
| [`docs/exact-tsv.md`](docs/exact-tsv.md) | the stored exact vector (migration 0020, which rewrites `memory_items`), its operator runbook, the cost of plain questions |
| [`docs/deployment.md`](docs/deployment.md) | Docker, local Postgres, Neon, Azure; pgvector 0.8 |
| [`docs/source-contract.md`](docs/source-contract.md) | shared-database mode, source views, the HTTP service |
| [`docs/backup-restore.md`](docs/backup-restore.md), [`docs/embedding-migration.md`](docs/embedding-migration.md), [`docs/import-hindsight.md`](docs/import-hindsight.md), [`docs/testing.md`](docs/testing.md), [`docs/eval.md`](docs/eval.md) | operations |
| [`docs/hybrid-implementation-plan.md`](docs/hybrid-implementation-plan.md) | the plan that built this and the status of its slices (removed when slices 5 and 6 finish) |
| [`PRINCIPLES.md`](PRINCIPLES.md) | the load-bearing principles |
| [`AGENTS.md`](AGENTS.md) and the skills | the practice, with its reasons and evidence: [recall](skills/recall/SKILL.md), [retain](skills/retain/SKILL.md), [search-and-ask](skills/search-and-ask/SKILL.md), [tuning-and-testing](skills/tuning-and-testing/SKILL.md), [operating](skills/operating/SKILL.md); other projects adopt them, and lessons come back as PRs |

## Caveats (1.0)

Honest about what is not there:

- **Sync only.** All `retain` / `search` / `ask` calls block; Mode 1 runs its searches on threads inside one call. An async surface is post-1.0.
- **English-only stemming.** The stemmed retriever uses `to_tsvector('english', ...)`; the exact retriever (unstemmed `simple`) and the semantic retriever are language-neutral.
- **pgvector 0.8 or newer.** Stage 1 uses iterative index scans; there is no fallback for an older pgvector.
- **Embedding-dim change requires a new bank** (`prospecta migrate-bank`).
- **HNSW build cost on large banks.** Each bank has one partial HNSW index (over its chunks); the first bulk index can be slow.
- **A very long plain question over common words is slow** in the stemmed retriever (seconds at 20 common words on a 73,000-chunk bank): measured in `docs/exact-tsv.md`; Mode 1's per-search `query` is short.
- **Stage 2 and the gates need a key.** With no rerank transport the Stage 1 order stands; with no decision transport there is no gate. Both are recorded in the result.
- **Migration 0020 rewrites `memory_items`** under an `ACCESS EXCLUSIVE` lock: read `docs/exact-tsv.md` before upgrading a database that holds a bank.

## License

See `LICENSE`. ⚒️
