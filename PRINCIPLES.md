# Prospecta — Principles

Load-bearing principles for the **prospecta** library and **hermes-prospecta** adapter. Each principle names a failure mode it prevents. ≤15 entries by design — the anchor, not the manual.

---

## P1 — Bilateral synthesis is the spine, not a feature

Both write-side (`index_text` frontmatter / generated index_text for indexed-as-line entries) and read-side (`formulate_queries` fan-out) are LLM-mediated. The embedding match happens in question-space, not content-space. Any design that lets the spine become optional, an add-on, or a flag — collapses prospecta back into a worse LlamaIndex.

**Prevents:** drifting into classical content-space RAG and losing the only thing worth building.

## P2 — Standalone first, plugin second

The library has no Hermes dependency. The Hermes plugin is one of N possible adapters. If a design move requires Hermes-specific knowledge inside the library, that's a code smell: the move belongs in the adapter.

**Prevents:** coupling that prevents Cookie, animus, scripts, notebooks, future agents from using prospecta directly.

## P3 — LLM as injected callable, not provider

The library takes a single `llm` callable from the caller. No `openai`, `anthropic`, `litellm` imports. The Hermes adapter passes `ctx.llm.complete`-wrapping; an animus caller passes `invoke_prompt`-wrapping; a notebook passes its own.

**Prevents:** locking the library to one provider; turning prospecta into a provider-routing layer it's not.

## P4 — Caller wins on every override

`retain(index_text=...)` overrides auto-generation. `recall(queries=[...])` skips formulation. `Memory(corpus_paths=..., embedding_model=...)` overrides defaults. Library auto-generates *if absent*; it never overrides what the caller supplied.

**Prevents:** the library out-thinking callers who have more context. Animus's `pre_engage` already classifies and may want different formulator prompts — the low-level entry has to be exposed and honored.

## P5 — Full content, no truncation (from animus CODING.md)

`SearchResult → RetrievedChunk → RAGResult → RecalledMemory` carries content through unfiltered. The library does not pre-judge what's "needed." The caller decides what to use. No "we'll return the first 200 chars."

**Prevents:** load-bearing context being thrown away at API boundaries; debugging by spelunking lost data.

## P6 — Let the LLM cook (from animus CODING.md)

Where prospecta delegates judgment to an LLM — index_text generation, query formulation, RAG synthesis, episode summarization-if-indexable — Python orchestrates and routes; Python does not heuristic the LLM's job. No regex parsing where a prompt suffices. No keyword filters where an LLM-decision belongs.

**Prevents:** brittle heuristics rotting in place; subtle bias from rule-based content judgment.

## P7 — Single write path through `index_single_file`

`retain` writes the file AND upserts to chroma. The background sweeper also calls `index_single_file` on changed paths. Two callers, one function. The sweeper is the drift safety net; `retain` is the primary write path. They do not diverge.

**Prevents:** code paths that drift apart, retain-then-no-search-for-24h surprise, two different "what gets indexed" implementations.

## P8 — Postgres+pgvector is the substrate; the cost is owned, not hidden

The substrate is Postgres + pgvector (locked in `decision-record-1.md`). This buys HNSW + FTS + RRF + multi-tenancy + transactional writes — capabilities embedded chroma cannot match for the production case. The cost is that the client must reach a running Postgres before first `recall`.

We do **not** pretend that cost away. We compensate for it: a one-command bootstrap (`prospecta init` → docker-compose + migrate + default-bank in one shot), sane defaults at every layer, and a README quickstart that is three lines after `init`. The trade is named in the docs, not buried.

**Prevents:** drift back toward embedded-chroma "convenience" that would forfeit the substrate; AND the opposite failure — shipping a powerful library nobody can stand up in under five minutes.

## P9 — Hermes plugin is a thin adapter, not a fork

`hermes-prospecta`'s `__init__.py` constructs `Memory`, delegates every method, owns nothing of substance. Plugin LOC budget: ~300-500 LOC including CLI. If the plugin grows beyond that, the library is missing a feature — push it down.

**Prevents:** the plugin reimplementing library logic for Hermes-flavored reasons; divergence between standalone and plugin behavior.

## P10 — Surface adjacent mechanisms; don't reinvent them

If hermes-agent ships a way to do something the plugin needs — credential pool, vision routing, structured JSON, async — use it via `ctx.llm` / `ctx`. Don't reimplement. Animus's existing patterns that fit prospecta's scope (chunker, ignore, parser) port as-is rather than getting rewritten.

**Prevents:** parallel-build; the plugin shipping its own provider client because the integrator missed `ctx.llm`.

## P11 — Tests over prose specifications (from animus CODING.md)

Library API is locked by tests, not by docstrings. Public surface — `retain`, `recall`, `recall_synth`, `search`, `formulate_queries`, `index_directory`, sweeper lifecycle — has end-to-end tests with a real chroma instance (embedded, ephemeral dir). Plugin has integration tests against `MemoryManager` like `tests/agent/test_memory_plugin_e2e.py`.

**Prevents:** API drift, "we changed the return shape and forgot to update three callers," confident-sounding docs that don't match code.

## P12 — Honest config surface

`get_config_schema()` returns ≤6 fields for `hermes memory setup`. Anything else lives in a `prospecta.json` at `hermes_home`. Library defaults are sane; setup wizard never asks 12 questions.

**Prevents:** setup-wizard fatigue; users abandoning the plugin because configuration felt heavyweight.

## P13 — Prompts live with the library, are overridable by the caller

The three load-bearing prompts (`rag-synthesize`, `formulate-queries`, `log-index` / `generate-index-text`) ship inside the package at `prospecta/prompts/`. They're loaded by default. Callers can override per-call (animus, who has refined prompts already, will). Adding a fourth prompt requires a design reason; the four-prompt surface is intentionally small.

**Prevents:** prompt sprawl; callers having to fork the library to change prompt language.

## P14 — Background sweeper is the safety net, not the primary path

The sweeper handles drift (files added/modified/deleted outside the library). It does NOT handle the hot path — `retain` does that directly via `index_single_file`. Sweeper cadence is conservative (default 24h, configurable). Sweeper failure must not block the hot path.

**Prevents:** waiting-on-sweeper latency in the agent's recall; sweeper-down meaning new retains aren't searchable.

## P15 — Spine is documented in the README

Prospecta's README leads with bilateral synthesis. It says *this is what makes us different from LlamaIndex / Khoj / mem0*. Not buried in §4 of an architecture doc. The pattern is the value proposition; the implementation is in service of it.

**Prevents:** the project being mistaken for "yet another markdown indexer" by drive-by readers; contributors not internalizing why the LLM calls on the write side matter.

## P16 — Client lift is a first-class design surface

Every change that touches the path from `pip install` to first successful `recall` answers one question: *does this add a step?* If yes, it justifies the step or compensates for it (default, wizard, one-command bootstrap, env-var fallback). The "steps to first recall" count is a tracked number, not a side effect.

Concretely: substrate-up, migrate, bank-create, embedder-config, llm-config, first `retain`, first `recall` are the seven sites where lift accumulates. The library and CLI work together to keep the *user-visible* step count at three or fewer for the happy path. Power users get the explicit surface; first-time users get `prospecta init && prospecta retain ... && prospecta recall ...`.

**Prevents:** the substrate's real power being inaccessible because the on-ramp is six manual steps; "I'll try it tomorrow" becoming "I never tried it"; the plugin shipping convenience the library lacks (which would violate P9).

## P17 — Capable defaults ship as an extras module, not as core dependencies

The core library has zero provider imports (P3 enforces the injection contract). Capable defaults live in `prospecta.defaults`, installed via `pip install prospecta[defaults]`, which pulls in LiteLLM and exposes `make_default_embedder()` and `make_default_llm()`. These factories read env vars (`OPENAI_API_KEY`, `PROSPECTA_EMBED_MODEL`, `PROSPECTA_LLM_MODEL`) and return callables that conform to the injection contract. Callers that supply their own `embed=` / `llm=` always win (P4).

The pattern is **"capable defaults, bring your own API key, caller can always override the abstraction."** It serves three audiences with one architecture:

| Audience | Embed | LLM |
|---|---|---|
| Notebook / script user | `prospecta.defaults` | `prospecta.defaults` |
| Hermes plugin | `prospecta.defaults` | `ctx.llm.complete` (host-provided, free) |
| Animus / Cookie / advanced caller | own embedder | own LLM router |

The Hermes plugin case is load-bearing in this design: `ctx.llm` covers the LLM half of the spine without any provider deps in the plugin or library, but it does not cover embeddings — so `prospecta.defaults` exists *specifically* to fill that gap cleanly without forcing core to import LiteLLM.

**Prevents:** core dependency creep ("just one little import"); alternately, no-defaults rigidity that forces every caller — including the Hermes plugin — to hand-roll provider clients; and the plugin growing its own embedding routing in violation of P9.

---

*Authored 2026-05-18 by Forge ⚒️ as PRINCIPLES.md for the prospecta ralplan. Distilled from the assessment doc (`~/forge/wiki/2026-05-18_memory-library-and-plugin-assessment.md`) and animus's CODING.md operating philosophy.*
