# Changelog

All notable changes to Prospecta are recorded here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

The default cross-encoder rerank timeout (`rerank.timeout_s`) is now 4 s, up from 2.5 s. Banks that store their own value keep it.

The cross-encoder reranker now retries once on an HTTP 503 or 429 inside the same time budget (`rerank.retries`, default 1, 0 turns it off); the fallback to the Stage 1 order is unchanged.

The cited block now indents every line of a snippet and a note and flattens header values, so a newline in a note cannot forge a numbered header.

## [1.0.0]

Prospecta 1.0 is one design with one read path and one write path. The design is
[`docs/postgres-hybrid.md`](docs/postgres-hybrid.md); the settings are in
[`docs/recall-stages.md`](docs/recall-stages.md).

### What 1.0 is

- **One write path: chunks, no model.** `retain` cuts a note into chunks of at most 1000
  characters, embeds them and writes them in one transaction. The note is kept verbatim. No
  model is called at write time.
- **One read path.**
  - **Stage 1, `hybrid_chunk`:** one SQL query over the chunks runs three retrievers: semantic
    (pgvector HNSW), stemmed text, and exact / phrase text. The exact retriever reads the stored
    `exact_tsv` vector (migration 0020). They fuse into the top 50, each hit with an
    `exact_snippet`; the date window, anti-echo list and scope apply inside every retriever.
  - **Stage 2, cross-encoder:** `qwen/qwen3-reranker-8b` scores every candidate against the
    query (0 to 1). A score floor, a near-copy collapse and a timeout fall back to the Stage 1
    order, so a reranker failure never loses results.
  - **Stage 0 gate and Mode 1 query expansion:** for a chat message a decision model skips
    acknowledgements and live-state questions; a query writer turns the prompt into several
    structured searches that each run Stages 1 and 2, merged into one cited block inside a 7 s
    deadline.
  - **Mode 2A** (`search`, `recall`, `prospecta search`, `POST /lookup {"questions"}`): Stage 1 + 2,
    no gate, no writer. **Mode 2B** (`ask`): Stage 1 + 2, then an answer with `[n]` citations
    tied to verbatim snippets.
  - **Stage 3 sufficiency** is optional and off by default: `ask` widens its search once when the
    evidence falls short.
- **The cited-evidence contract.** Every hit and every block handed to a model carries `source`,
  `created_on`, `kind`, `match_kind`, `relevance_score`, `exact_snippet` and the full chunk.

### Removed since 0.1

- The question-indexing write path and the question lane (the write path is chunk-only).
- The graph, linker and entities, and the link CLI.
- Map-reduce.
- The in-process BM25 and keyword channels.
- `recall_synth` and `formulate_queries`.
- The legacy Sonnet / Jev rerank, gate, reader and hops.
- The `dense_chunk` and `meta` channels, and `floor_union`.
- The legacy three-channel read path.

### What an operator must know

- **No migration beyond 0020 (schema version 20).** 0019 adds `banks.link_config` (unused now);
  0020 adds the stored `memory_items.exact_tsv` column and its GIN index.
  **0020 rewrites `memory_items` under an `ACCESS EXCLUSIVE` lock** held for the whole rewrite
  (172 s on a synthetic 73,000-chunk bank): reads and writes of that table wait. Read
  [`docs/exact-tsv.md`](docs/exact-tsv.md) before applying it to a database that holds a bank.
- **Existing banks keep working, unchanged.** A bank whose `channel_config` is empty or unset runs
  `hybrid_chunk` with its defaults. Legacy `recall_config` keys the library no longer reads are
  ignored, and stored channel entries naming removed channels are skipped. Old tables and rows
  (entities, links, question items and the like) stay as history and are never read. Nothing
  rewrites a stored bank config.
- **A newly created bank** (`prospecta init`, `create-bank`, the service, `migrate-bank`,
  `import hindsight`) stores `channel_config = [{"name": "hybrid_chunk", "enabled": true,
  "weight": 1, "params": {}}]` and `recall_config = {"rerank": {"enabled": true, "stage":
  "cross_encoder"}}` (`prospecta/bank_defaults.py`). The rerank model, floor, `top_n` and timeout
  are not stored; they come from the code defaults and the environment.
- **Models and environment.** Resolution is per-call, then the bank's `recall_config`, then the
  environment, then the default:

  | setting | environment variable | default |
  |---|---|---|
  | decision model (Stage 0 gate, Stage 3) | `PROSPECTA_DECISION_MODEL` (older alias `PROSPECTA_JEV_MODEL`) | `typesafe/jev-1.14` |
  | reranker model (Stage 2) | `PROSPECTA_RERANKER_MODEL` | `qwen/qwen3-reranker-8b` |
  | query-writer model (Mode 1) | `PROSPECTA_QUERY_WRITER_MODEL` | `openrouter/google/gemini-2.5-flash-lite` |

  The reranker and the decision model are reached through OpenRouter with `OPENROUTER_API_KEY`;
  without it retrieval still works (Stage 1 order stands, chat messages are not gated).
  `PROSPECTA_JEV=off` disables the decision model in the CLI.
- **pgvector 0.8 or newer** is required (Stage 1 uses iterative index scans).
- **Upgrading from 0.x:**
  1. Stop everything that uses the database (gateways, the service, sweepers). Back it up
     (`prospecta backup`; [`docs/backup-restore.md`](docs/backup-restore.md)).
  2. Install 1.0.0 into the venv (`pip install -U prospecta`).
  3. With nothing else running, run `prospecta migrate` by hand. Any process that calls
     `run_migrations` (`prospecta migrate`, `prospecta init`, the service at start-up) applies
     0020, so a service must not be the first to start on 1.0.
  4. Restart.
- **Rollback.** Restore the pre-upgrade backup to be safe; the migrations are not undone. An older
  library can still read the database, but a bank created by 1.0 stores `link_config` as `{}`,
  which the older library reads as "entities on" (and links off), so it will start extracting
  entities for that bank.

### Version

`pyproject.toml` is the only place the library version is stated (the TUI under `tui/` has its
own version). There is no `__version__` and no `--version` flag.

## Earlier history

Versions before 1.0 were development releases (tag `v0.1.0`) that had the question-indexing write
path, the entity graph and a three-channel read path; the `git log` is their history. The
pre-hybrid explainer is kept at the tag `archive/explainer-v1`.
