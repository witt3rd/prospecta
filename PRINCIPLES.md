# Prospecta — Principles

Load-bearing principles for the **prospecta** library. Each names a failure mode it prevents. At most 15 entries by design — the anchor, not the manual. The numbers are stable identifiers (code and docs cite them: `P4`, `P5`, `P14`, `P19`); a retired principle's number is not reused, so the list has gaps. The architecture these principles serve is `docs/postgres-hybrid.md`.

---

## P2 — Standalone first, plugin second

The library has no agent-framework dependency. A Hermes plugin, a venue server calling the HTTP service, a script or a notebook are adapters over the same `Memory` API. If a design move requires host-specific knowledge inside the library, that is a code smell: the move belongs in the adapter.

**Prevents:** coupling that stops other agents, scripts and notebooks from using prospecta directly.

## P3 — Models are injected callables, not providers

The library takes an `embed` callable and an `llm` callable from the caller; the decision model and the reranker are reached through injected transports (`Memory(jev=...)`, `Memory(rerank_transport=...)`). The library code imports no provider SDK: `prospecta.defaults` (LiteLLM) and `prospecta.embed.*` (OpenAI, sentence-transformers) are the modules that do. (`litellm` is nevertheless listed in the core `dependencies` of `pyproject.toml`, so `pip install prospecta` installs it.)

**Prevents:** locking the library to one provider; turning prospecta into a provider-routing layer it is not.

## P4 — Caller wins on every override

Per-call arguments beat the bank's `recall_config`, which beats the environment, which beats the default: the reranker, decision and query-writer models, `deadline`, `max_searches`, `date_from` / `date_to`, `excluded_sources`, `depth`. `expand_and_recall(decision=...)` replaces the gate with the caller's own; `retain(index_text=...)` supplies the question items instead of the chunks. The library fills in what is absent and never overrides what the caller supplied.

**Prevents:** the library out-thinking callers who have more context (a venue that already decided a message needs no memory; a caller that knows the period it means).

## P5 — Full content, no truncation

The note is stored verbatim (`documents.original_text`); a chunk reaches the caller whole (`RecalledMemory.original_chunk`) and goes whole to the reranker. The library does not pre-judge what is "needed": no "first 200 characters". The only clip is a physical provider limit, and then the full untruncated text is logged in a warning (`docs/limits.md`).

**Prevents:** load-bearing context thrown away at API boundaries; debugging by spelunking lost data.

## P7 — One write path, deterministic

`retain` is the primary write path: it cuts the note into chunks, embeds them and writes the rows, in one transaction, with no model call. The background sweeper is the drift safety net for files that change outside the library; both write chunk rows. Writing the same content under the same source replaces its items; the same content under another source is an error, not a silent re-bind (`docs/schema.md`).

**Prevents:** a write that costs a model call and can fail or drift; two different "what gets indexed" implementations.

## P8 — Postgres + pgvector is the substrate; the cost is owned, not hidden

The substrate is Postgres with pgvector 0.8 or newer. It buys HNSW, full-text search, one-query hybrid candidate generation, multi-tenancy by bank and transactional writes. The cost is that the client must reach a running Postgres before the first `search`. We do not pretend that cost away: a one-command bootstrap (`prospecta init` brings up the compose file, migrates and creates the default bank), a bank that needs no configuration, and a quickstart of three commands. The trade is named in `docs/deployment.md`, not buried.

**Prevents:** drifting back toward embedded-store "convenience" that would forfeit the substrate; and the opposite failure, a powerful library nobody can stand up in five minutes.

## P11 — Tests over prose specifications

The library API is locked by tests, not docstrings. The public surface — `retain`, `search`, `recall`, `ask`, `expand_and_recall`, `recall_for_task`, `index_directory`, the sweeper, the service — has end-to-end tests against a real Postgres with pgvector (a testcontainer, or `PROSPECTA_TEST_PG_URL`: `docs/testing.md`). The docs are checked the same way where they can be: a test verifies every relative link and every doc path a source file cites.

**Prevents:** API drift, "we changed the return shape and forgot three callers", confident-sounding docs that do not match code.

## P14 — The background sweeper is the safety net, not the primary path

The sweeper handles drift (files added, modified or deleted outside the library). It does not handle the hot path: `retain` does. Its cadence is conservative (default 24 h, configurable) and a sweeper failure must not block the hot path.

**Prevents:** waiting-on-sweeper latency in the agent's recall; "sweeper down" meaning new retains are not searchable.

## P16 — Client lift is a first-class design surface

Every change that touches the path from `pip install` to the first successful `search` answers one question: *does this add a step?* If yes, it justifies the step or compensates for it (a default, a one-command bootstrap, an environment fallback). The step count is a tracked number: `prospecta init`, `prospecta retain ...`, `prospecta search ...`. A bank created fresh runs the whole architecture without configuration, so there is no fourth step.

**Prevents:** the substrate's power being inaccessible because the on-ramp is six manual steps; the plugin shipping convenience the library lacks.

## P17 — Capable defaults ship as an extras module

Provider code lives in `prospecta.defaults` (`pip install 'prospecta[defaults]'`: LiteLLM, `make_default_embedder()`, `make_default_llm()`) and `prospecta.embed.*` (`[embedder]`, `[embed-openai]`, `[embed-openai-compatible]`). They read environment variables (`OPENAI_API_KEY`, `PROSPECTA_EMBED_MODEL`, `PROSPECTA_EMBEDDER`) and return callables that satisfy the injection contract; callers that pass their own `embed=` / `llm=` always win (P4). The pattern is "capable defaults, bring your own API key, the caller can always override the abstraction."

**Prevents:** provider code spreading into the core ("just one little import"); and the opposite, no defaults, which forces every caller to hand-roll a provider client.

## P19 — Observability is durable, not ephemeral

Every event a user might want to inspect after the fact is persisted in Postgres, not flushed to stderr: `retain_events`, `recall_events` with `recall_event_candidates`, `llm_calls` (every model call with its purpose, model, tokens, cost, and the verbatim prompt and response), `formulate_events` (the legacy question path), `sweep_passes` and `sweeper_state`. The tables are append-only except `sweeper_state`, and they carry the payload, not summaries: the notes that surfaced with each channel's rank and score, the stage records (the rerank's cuts and fallback reason), the accounting.

The tracer fan-out (`Tracer` is an injected Protocol, `PostgresSink` the default; callers can supply `NoOpTracer`, `RecordingTracer`, `CompositeTracer` or their own) means persistence is one consumer of the event stream, not the only one, but the *default* must be durable. When a feature produces a load-bearing event, the default action is to **add a column** to the event table (or a new table), not to log a line. Logs are for transient breadcrumbs.

**Prevents:** "what just happened" being unanswerable once the session ends; the inspection surface drifting from the architecture because nobody touched the documented schema when shipping a feature.

## P20 — The chunk is the unit

What is stored is the note's own text in chunks, embedded as written and indexed lexically twice (stemmed and exact). There is no model-written proxy for a note at write time. A match is a match on what the note says, and the citation quotes it.

**Prevents:** retrieval depending on what a model once guessed a note would be asked; a write path whose quality moves with a model version.

## P21 — Every output is cited

Neither search nor automatic recall nor an answer returns unattributed prose: each hit says where it is from (`source`), when (`created_on`), what kind of record it is, why it matched (`match_kind`, the `exact_snippet`) and how relevant it is; an answer cites `[n]` for every statement and each `[n]` ties to a verbatim sentence of the chunk. Recalled material is quoted, never instructions.

**Prevents:** an agent acting on a memory it cannot trace; unverifiable answers; injected text in a note steering the agent.

## P22 — Recall degrades, it never raises into the caller

A reranker that times out, a gate that fails, a writer that returns garbage, a search that runs late: each degrades (the Stage 1 order stands, the message goes through, one search with the prompt itself, the late search is dropped) and says so in the result (`fallbacks`, `fallback_reason`, the event). A provider failure in `expand_and_recall` comes back as an error field, not an exception (a call with invalid arguments still raises `ValueError`); a failed Stage 3 gate never blocks the answer. The cost of one unremembered turn is larger than the cost of one degraded recall.

**Prevents:** an optional stage taking down the agent's turn; failures that leave no record of why the result was worse.
